"""Qualified roles for sensitive functions (separation of duties).

  policy_admin  everything, incl. editing the campus access schedule and granting roles
  operator      enrollment desk: enroll people, record test clips, run the evaluation, demo controls
  auditor       read-only: audit log, policy, results

Everyone else (students, visitors) only ever sees the kiosk decision.

Sign-in = enrolled user ID + a code from that person's authenticator app (something they have),
giving a short admin session. Every sign-in, failure, grant and policy change is audited.
"""
from __future__ import annotations

import secrets
import threading
import time

import pyotp

import store

ROLES = {"policy_admin": "Policy admin", "operator": "Operator", "auditor": "Auditor"}
ALL = set(ROLES)
CAN = {
    "edit_policy": {"policy_admin"},
    "grant_roles": {"policy_admin"},
    "view_policy": ALL,
    "view_audit": {"policy_admin", "auditor"},
    "manage_people": {"policy_admin", "operator"},   # enroll, edit access, record clips
    "run_eval": {"policy_admin", "operator"},
    "view_results": ALL,
    "kiosk_controls": {"policy_admin", "operator"},  # demo clock, zone override, simulate network loss
}
SESSION_S = 15 * 60
MAX_FAILS, LOCK_S = 5, 300

_lock = threading.Lock()
_sessions: dict[str, dict] = {}
_fails: dict[str, list[float]] = {}


def login(user_id: str, code: str) -> tuple[str | None, str]:
    """Returns (session token, message)."""
    now = time.time()
    with _lock:
        recent = [t for t in _fails.get(user_id, []) if now - t < LOCK_S]
        _fails[user_id] = recent
        if len(recent) >= MAX_FAILS:
            return None, "Too many failed attempts - try again in a few minutes"
    u = store.get_user(user_id)
    ok = bool(u and u.get("admin_role") in ROLES and u.get("totp_secret")
              and pyotp.TOTP(u["totp_secret"]).verify(code.strip().replace(" ", ""), valid_window=1))
    if not ok:
        with _lock:
            _fails.setdefault(user_id, []).append(now)
        return None, "Not authorised: needs a staff role (policy admin, operator or auditor) and a valid authenticator code"
    token = secrets.token_urlsafe(32)
    with _lock:
        _fails.pop(user_id, None)
        _sessions[token] = {"user_id": user_id, "role": u["admin_role"], "expires": now + SESSION_S}
    return token, "Signed in"


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
        if not u or u.get("admin_role") != s["role"]:
            logout(token)
            return None
    return s


def logout(token: str | None) -> None:
    with _lock:
        _sessions.pop(token or "", None)


def allowed(sess: dict | None, permission: str) -> bool:
    return bool(sess) and sess["role"] in CAN[permission]


def bootstrap_open() -> bool:
    """First-run setup: until someone holds the policy-admin role, anyone signed in to the
    site may appoint the first policy admin. After that only a policy admin can grant roles."""
    return not any(u.get("admin_role") == "policy_admin" for u in store.list_users())
