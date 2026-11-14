"""In-memory challenge and OTP sessions for the REQUEST_VERIFICATION flow."""
from __future__ import annotations

import secrets
import threading
import time
import uuid

from pipeline.config import CFG

ACTIONS = {
    "blink": "Blink twice",
    "turn_left": "Turn your head to your LEFT",
    "turn_right": "Turn your head to your RIGHT",
}

_lock = threading.Lock()
_challenges: dict[str, dict] = {}
_otps: dict[str, dict] = {}
OUTBOX: list[dict] = []   # mock "SMS" messages shown on the demo phone panel


def new_challenge(candidate: str | None, claimed_id: str | None, avoid: str | None = None) -> dict:
    """avoid='blink' when the eyes can't be seen (e.g. sunglasses): a head turn is just as random."""
    ch = {
        "id": uuid.uuid4().hex,
        "action": secrets.choice([a for a in ACTIONS if a != avoid] or list(ACTIONS)),
        "candidate": candidate,
        "claimed_id": claimed_id,
        "expires": time.time() + CFG["challenge"]["ttl_s"],
    }
    ch["instruction"] = ACTIONS[ch["action"]]
    with _lock:
        _challenges[ch["id"]] = ch
    return ch


def take_challenge(cid: str) -> dict | None:
    """One-time use: a recorded answer can't be replayed against the same challenge."""
    with _lock:
        ch = _challenges.pop(cid, None)
    if ch and ch["expires"] >= time.time():
        return ch
    return None


def new_otp(user_id: str, name: str, phone: str | None, sms: bool = True,
            totp_secret: str | None = None, context: dict | None = None) -> dict:
    """Second-factor session. Accepts the SMS code (online only) or the user's authenticator
    app code (TOTP - works with no network, used by the offline fail-safe flow)."""
    code = "".join(secrets.choice("0123456789") for _ in range(CFG["challenge"]["otp_digits"])) if sms else None
    sid = uuid.uuid4().hex
    with _lock:
        _otps[sid] = {"user_id": user_id, "code": code, "totp": totp_secret, "tries": 0,
                      "expires": time.time() + 120, "context": context or {}}
        if sms:
            OUTBOX.append({"ts": time.time(), "to": phone or f"{name}'s registered phone",
                           "text": f"HID kiosk access code: {code}"})
            del OUTBOX[:-10]
    methods = (["sms"] if sms else []) + (["authenticator"] if totp_secret else [])
    return {"session_id": sid, "sent_to": (phone or "registered phone") if sms else None, "methods": methods}


def _code_ok(rec: dict, code: str) -> bool:
    code = code.strip().replace(" ", "")
    if rec["code"] and secrets.compare_digest(rec["code"], code):
        return True
    if rec["totp"]:
        import pyotp
        return pyotp.TOTP(rec["totp"]).verify(code, valid_window=1)
    return False


def check_otp(sid: str, code: str) -> tuple[bool, str | None, str, dict]:
    with _lock:
        rec = _otps.get(sid)
        if not rec:
            return False, None, "Verification session not found", {}
        ctx = rec["context"]
        if rec["expires"] < time.time():
            _otps.pop(sid, None)
            return False, rec["user_id"], "Code expired", ctx
        rec["tries"] += 1
        if _code_ok(rec, code):
            _otps.pop(sid, None)
            return True, rec["user_id"], "Second factor confirmed", ctx
        if rec["tries"] >= CFG["challenge"]["max_otp_tries"]:
            _otps.pop(sid, None)
            return False, rec["user_id"], "Too many wrong codes", ctx
        return False, rec["user_id"], "WRONG_CODE", ctx
