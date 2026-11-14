"""Console access. One extra access role, "admin", on top of the door roles (student, staff,
contractor, visitor): an admin can open every door at any time and has every console power.
Everyone else only ever sees the kiosk decision.

The first admin is enrolled during first-run setup (open until an admin exists); after that only
an admin can enroll people or make someone an admin.

Sign-in = identify yourself EITHER with a live face check (recognised among admins, with liveness
and a random cue) OR by typing your email - then, either way, a code from your authenticator app.
Policy edits and admin grants/removals need a face check from the last 5 minutes (a one-time
recovery code can stand in for it if a webcam fails), so an email sign-in can't change anything
sensitive. Every sign-in, failure, grant and policy change is audited.
"""
from __future__ import annotations

import secrets
import threading
import time

import pyotp

import store

ADMIN = "admin"
ROLES = {ADMIN: "Admin"}
CAN = {perm: {ADMIN} for perm in (
    "edit_policy", "grant_roles", "view_policy", "view_audit",
    "manage_people",      # enroll, edit access, record clips
    "run_eval", "view_results",
    "kiosk_controls",     # demo clock, zone override, simulate network loss
)}
SESSION_S = 15 * 60
MAX_FAILS, LOCK_S = 5, 300
FACE_TOKEN_S = 120          # a passed face check must be used for sign-in within 2 minutes
REAUTH_S = 5 * 60           # sensitive actions need a face check this recent

_lock = threading.Lock()
_sessions: dict[str, dict] = {}
_fails: dict[str, list[float]] = {}
_face_tokens: dict[str, dict] = {}


def _locked_out(user_id: str, now: float) -> bool:
    with _lock:
        recent = [t for t in _fails.get(user_id, []) if now - t < LOCK_S]
        _fails[user_id] = recent
        return len(recent) >= MAX_FAILS


def _fail(user_id: str) -> None:
    with _lock:
        _fails.setdefault(user_id, []).append(time.time())


def is_staff(user_id: str) -> bool:
    u = store.get_user(user_id)
    return bool(u and u.get("role") == ADMIN)


# ------------------------------------------------------------------ factor 1: face + liveness
def issue_face_token(user_id: str) -> str:
    """Called after a passed 1:1 face + liveness + random-cue check for this user."""
    t = secrets.token_urlsafe(24)
    with _lock:
        _face_tokens[t] = {"user_id": user_id, "expires": time.time() + FACE_TOKEN_S}
    return t


def take_face_token(token: str | None, user_id: str) -> bool:
    """Single use, bound to the user it was issued for, short-lived."""
    if not token:
        return False
    with _lock:
        rec = _face_tokens.pop(token, None)
    return bool(rec and rec["user_id"] == user_id and rec["expires"] >= time.time())


# ------------------------------------------------------------------ factor 2: authenticator
def _totp_ok(u: dict, code: str) -> bool:
    return bool(u.get("totp_secret")) and pyotp.TOTP(u["totp_secret"]).verify(code.strip().replace(" ", ""), valid_window=1)


def new_recovery_codes(user_id: str) -> list[str]:
    """One-time codes that stand in for the face check if a webcam fails (shown once)."""
    import hashlib
    codes = ["-".join(secrets.token_hex(3) for _ in range(2)) for _ in range(8)]
    store.set_recovery_codes(user_id, [hashlib.sha256(c.encode()).hexdigest() for c in codes])
    return codes


def login(user_id: str, code: str, face_token: str | None = None) -> tuple[str | None, str]:
    """Staff sign-in = (face check OR email) + authenticator code. Returns (session token, message)."""
    now = time.time()
    if _locked_out(user_id, now):
        return None, "Too many failed attempts - try again in a few minutes"
    u = store.get_user(user_id)
    if not (u and u.get("role") == ADMIN and _totp_ok(u, code)):
        _fail(user_id)
        return None, "Not authorised: needs the admin role and a valid authenticator code"
    via = "email"
    if face_token:
        if not take_face_token(face_token, user_id):
            _fail(user_id)
            return None, "Face check expired - verify your face again"
        via = "face"
    token = secrets.token_urlsafe(32)
    with _lock:
        _fails.pop(user_id, None)
        _sessions[token] = {"user_id": user_id, "role": ADMIN, "expires": now + SESSION_S,
                            "face_at": now if via == "face" else 0.0, "via": via}
    return token, f"Signed in ({via} + authenticator)"


def reauth(sess_token: str | None, face_token: str | None, recovery_code: str | None = None) -> bool:
    """Refresh the 'recent face check' mark on a session (step-up before sensitive actions).
    A one-time recovery code is accepted instead of a face check if the webcam fails."""
    import hashlib
    s = session(sess_token)
    if not s:
        return False
    ok = take_face_token(face_token, s["user_id"]) or bool(
        recovery_code and store.use_recovery_code(s["user_id"], hashlib.sha256(recovery_code.strip().lower().encode()).hexdigest()))
    if not ok:
        return False
    with _lock:
        _sessions[sess_token]["face_at"] = time.time()
    return True


def recent_face(sess: dict | None) -> bool:
    return bool(sess) and time.time() - sess.get("face_at", 0) <= REAUTH_S


def session(token: str | None) -> dict | None:
    if not token:
        return None
    with _lock:
        s = _sessions.get(token)
        if s and s["expires"] < time.time():
            _sessions.pop(token, None)
            s = None
    if s:  # role may have been revoked since sign-in
        u = store.get_user(s["user_id"])
        if not u or u.get("role") != ADMIN:
            logout(token)
            return None
    return s


def logout(token: str | None) -> None:
    with _lock:
        _sessions.pop(token or "", None)


def allowed(sess: dict | None, permission: str) -> bool:
    return bool(sess) and sess["role"] in CAN[permission]


def bootstrap_open() -> bool:
    """First-run setup: until an admin exists, anyone signed in to the site may enroll the
    first admin. After that only an admin can enroll people or grant the admin role."""
    return not any(u.get("role") == ADMIN for u in store.list_users())
