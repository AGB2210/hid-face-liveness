"""HID kiosk liveness + identity API.

Run (from backend/):  ../.venv/Scripts/python -m uvicorn app:app --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import sessions  # noqa: E402
import store  # noqa: E402
from pipeline import engine  # noqa: E402
from pipeline.config import CFG, ROOT, reload_config  # noqa: E402
from pipeline.frames import analyse_frame, decode  # noqa: E402
from pipeline.identity import get_provider  # noqa: E402
from pipeline.models import get_models, models_lock  # noqa: E402

LABELS = ["live", "print", "screen_photo", "video_replay", "mask"]
CLIPS = ROOT / "data" / "clips"

app = FastAPI(title="HID Liveness Kiosk")


@app.on_event("startup")
def _warm() -> None:
    get_models()  # load models once so the first user isn't slow


async def _read(frames: list[UploadFile]) -> list[bytes]:
    if not frames:
        raise HTTPException(400, "no frames")
    return [await f.read() for f in frames]


@app.get("/api/config")
def config() -> dict:
    return {"capture": CFG["capture"], "labels": LABELS, "actions": sessions.ACTIONS,
            "identity_provider": CFG["identity"]["provider"]}


@app.post("/api/config/reload")
def config_reload() -> dict:
    reload_config()
    return {"ok": True}


@app.get("/api/users")
def users() -> list[dict]:
    return store.list_users()


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str) -> dict:
    store.delete_user(user_id)
    return {"ok": True}


@app.post("/api/enroll")
async def enroll(user_id: str = Form(...), name: str = Form(...), phone: str = Form(""),
                 frames: list[UploadFile] = File(...)) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", user_id):
        raise HTTPException(400, "user_id: letters, digits, _ . - only")
    imgs = decode(await _read(frames))
    with models_lock():
        fr = [analyse_frame(i, im) for i, im in enumerate(imgs)]
        good = sorted([f for f in fr if f.good], key=lambda f: f.rank, reverse=True)[:5]
        if len(good) < 2:
            issues = sorted({i for f in fr for i in f.issues})
            raise HTTPException(422, f"Not enough good-quality frames ({len(good)}). Issues: {issues}")
        n = get_provider().enroll(user_id, name, phone or None, good)
    return {"ok": True, "user_id": user_id, "templates": n}


@app.post("/api/verify")
async def verify(frames: list[UploadFile] = File(...), claimed_id: str = Form(""),
                 challenge_id: str = Form(""), capture_ms: float = Form(0)) -> dict:
    jpegs = await _read(frames)
    challenge = None
    if challenge_id:
        challenge = sessions.take_challenge(challenge_id)
        if challenge is None:
            raise HTTPException(410, "Challenge expired or already used - start again")
        claimed_id = claimed_id or challenge.get("claimed_id") or ""

    res = engine.analyse(jpegs, claimed_id or None, challenge)
    res["stage"] = "challenge" if challenge else "passive"
    res["capture_ms"] = capture_ms

    if res["decision"] == engine.VERIFY and res["next"] == "challenge":
        cand = res["identity"]["user_id"] if res.get("identity") else None
        ch = sessions.new_challenge(cand, claimed_id or None)
        res["challenge_request"] = {
            "id": ch["id"], "action": ch["action"], "instruction": ch["instruction"],
            "duration_ms": CFG["capture"]["challenge_duration_ms"],
            "frames": CFG["capture"]["challenge_frames"]}
    elif res["decision"] == engine.VERIFY and res["next"] == "otp":
        ident = res.get("identity") or {}
        user = store.get_user(ident["user_id"]) if ident.get("user_id") else None
        if user:
            res["otp_request"] = sessions.new_otp(user["id"], user["name"], user.get("phone"))
        else:
            res.update(decision=engine.DENY, reasons=["Identity could not be established"], next=None)

    ident = res.get("identity") or {}
    store.log_attempt(res["decision"], res["stage"], ident.get("user_id"), ident.get("similarity"),
                      (res.get("liveness") or {}).get("score"), res["timing_ms"]["processing"],
                      None, res["reasons"])
    return res


@app.post("/api/otp/verify")
def otp_verify(session_id: str = Form(...), code: str = Form(...)) -> dict:
    ok, user_id, msg = sessions.check_otp(session_id, code)
    if msg == "WRONG_CODE":
        return {"decision": None, "retry": True, "reasons": ["Wrong code - try again"]}
    user = store.get_user(user_id) if user_id else None
    decision = "ALLOW" if ok else "DENY"
    reasons = ["Identity confirmed by one-time code"] if ok else [msg]
    store.log_attempt(decision, "otp", user_id, None, None, 0, None, reasons)
    return {"decision": decision, "reasons": reasons,
            "identity": {"user_id": user_id, "name": user["name"] if user else None}}


@app.get("/api/otp/outbox")
def otp_outbox() -> list[dict]:
    return list(reversed(sessions.OUTBOX))


@app.post("/api/timing")
def timing(total_ms: float = Form(...)) -> dict:
    store.update_total_ms(total_ms)
    return {"ok": True}


@app.get("/api/attempts")
def attempts(limit: int = 50) -> list[dict]:
    return store.recent_attempts(limit)


def _clip_counts() -> dict:
    return {lab: len(list((CLIPS / lab).glob("*/*/meta.json"))) for lab in LABELS}


@app.get("/api/clips/counts")
def clip_counts() -> dict:
    return _clip_counts()


@app.post("/api/record")
async def record(label: str = Form(...), subject: str = Form(...), capture_ms: float = Form(0),
                 note: str = Form(""), frames: list[UploadFile] = File(...)) -> dict:
    """Save a labelled test clip for eval/run_eval.py, and show what the kiosk would decide."""
    if label not in LABELS:
        raise HTTPException(400, f"label must be one of {LABELS}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", subject):
        raise HTTPException(400, "subject: letters, digits, _ . - only")
    jpegs = await _read(frames)
    clip = CLIPS / label / subject / time.strftime("%Y%m%d-%H%M%S")
    clip.mkdir(parents=True, exist_ok=True)
    for i, b in enumerate(jpegs):
        (clip / f"frame_{i:02d}.jpg").write_bytes(b)
    (clip / "meta.json").write_text(json.dumps(
        {"label": label, "subject": subject, "capture_ms": capture_ms, "note": note,
         "frames": len(jpegs), "created": time.time()}, indent=2))
    res = engine.analyse(jpegs, None, None)
    return {"saved": str(clip.relative_to(ROOT)), "result": res, "counts": _clip_counts()}


app.mount("/", StaticFiles(directory=ROOT / "kiosk", html=True), name="kiosk")
