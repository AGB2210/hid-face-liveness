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


def new_challenge(candidate: str | None, claimed_id: str | None) -> dict:
    ch = {
        "id": uuid.uuid4().hex,
        "action": secrets.choice(list(ACTIONS)),
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


def new_otp(user_id: str, name: str, phone: str | None) -> dict:
    code = "".join(secrets.choice("0123456789") for _ in range(CFG["challenge"]["otp_digits"]))
    sid = uuid.uuid4().hex
    with _lock:
        _otps[sid] = {"user_id": user_id, "code": code, "tries": 0,
                      "expires": time.time() + 120}
        OUTBOX.append({"ts": time.time(), "to": phone or f"{name}'s registered phone",
                       "text": f"HID kiosk access code: {code}"})
        del OUTBOX[:-10]
    return {"session_id": sid, "sent_to": phone or "registered phone"}


def check_otp(sid: str, code: str) -> tuple[bool, str | None, str]:
    with _lock:
        rec = _otps.get(sid)
        if not rec:
            return False, None, "Verification session not found"
        if rec["expires"] < time.time():
            _otps.pop(sid, None)
            return False, rec["user_id"], "Code expired"
        rec["tries"] += 1
        if secrets.compare_digest(rec["code"], code.strip()):
            _otps.pop(sid, None)
            return True, rec["user_id"], "Code accepted"
        if rec["tries"] >= CFG["challenge"]["max_otp_tries"]:
            _otps.pop(sid, None)
            return False, rec["user_id"], "Too many wrong codes"
        return False, rec["user_id"], "WRONG_CODE"
