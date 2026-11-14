"""HID kiosk liveness + identity API.

Run (from backend/):  ../.venv/Scripts/python -m uvicorn app:app --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import bursts  # noqa: E402
import sessions  # noqa: E402
import store  # noqa: E402
from pipeline import engine  # noqa: E402
from pipeline.config import CFG, DATA, ROOT, reload_config  # noqa: E402
from pipeline.frames import analyse_frame, decode  # noqa: E402
from pipeline.identity import get_provider, provider_name  # noqa: E402
from pipeline.models import models_lock, warmup  # noqa: E402

LABELS = ["live", "print", "screen_photo", "video_replay", "mask"]
CLIPS = DATA / "clips"

app = FastAPI(title="HID Liveness Kiosk")
PASSWORD = os.environ.get("HID_PASSWORD")  # set on the public deployment
AUTH_TOKEN = hashlib.sha256(f"hid:{PASSWORD}".encode()).hexdigest() if PASSWORD else ""
LOGIN_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>HID Kiosk Login</title>
<link rel="stylesheet" href="/styles.css"></head><body><main style="max-width:380px;margin:12vh auto">
<div class="card stack"><div class="row"><span class="logo">HID</span><h1 style="margin:0;font-size:18px">Liveness Kiosk</h1></div>
<form method="post" action="/login" class="stack"><label>Access password
<input type="password" name="password" autofocus required></label>
<button type="submit">Enter</button>{msg}</form></div></main></body></html>"""


@app.middleware("http")
async def access_gate(request: Request, call_next):
    """Simple shared-password gate for the public deployment (HID_PASSWORD)."""
    path = request.url.path
    if PASSWORD and path not in ("/login", "/styles.css") and             not secrets.compare_digest(request.cookies.get("hid_auth", ""), AUTH_TOKEN):
        if path.startswith("/api/"):
            return Response('{"detail":"login required"}', status_code=401, media_type="application/json")
        return HTMLResponse(LOGIN_HTML.format(msg=""), status_code=401)
    response = await call_next(request)
    # pages/scripts change with every update: make browsers revalidate instead of reusing stale copies
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


@app.post("/login")
def login(password: str = Form(...)):
    if not PASSWORD or not secrets.compare_digest(password, PASSWORD):
        return HTMLResponse(LOGIN_HTML.format(msg='<p class="small" style="color:var(--deny)">Wrong password</p>'),
                            status_code=401)
    r = RedirectResponse("/", status_code=303)
    r.set_cookie("hid_auth", AUTH_TOKEN, max_age=7 * 24 * 3600, httponly=True, samesite="lax")
    return r


@app.on_event("startup")
def _warm() -> None:
    warmup()  # load and run every model once so the first user isn't slow


async def _read(frames: list[UploadFile]) -> list[bytes]:
    if not frames:
        raise HTTPException(400, "no frames")
    return [await f.read() for f in frames]


@app.get("/api/config")
def config() -> dict:
    return {"capture": CFG["capture"], "labels": LABELS, "actions": sessions.ACTIONS,
            "identity_provider": provider_name()}


@app.post("/api/config/reload")
def config_reload() -> dict:
    reload_config()
    return {"ok": True}


@app.get("/api/users")
def users() -> list[dict]:
    return store.list_users()


@app.delete("/api/users/{user_id}")
def delete_user(user_id: str) -> dict:
    get_provider().delete(user_id)
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
        try:
            n = get_provider().enroll(user_id, name, phone or None, good)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:  # AWS credentials / network
            raise HTTPException(502, f"Identity provider error: {type(e).__name__}: {e}"[:300])
    return {"ok": True, "user_id": user_id, "templates": n, "provider": provider_name()}


@app.post("/api/verify")
async def verify(frames: list[UploadFile] = File(...), claimed_id: str = Form(""),
                 challenge_id: str = Form(""), capture_ms: float = Form(0),
                 burst_id: str = Form(""), n_frames: int = Form(0)) -> dict:
    """frames: the whole burst, or (streaming) just the last frame(s) with burst_id set
    and n_frames = total frames in the burst."""
    jpegs = await _read(frames)
    return await run_in_threadpool(_verify, jpegs, claimed_id, challenge_id, capture_ms, burst_id, n_frames)


@app.post("/api/burst/{burst_id}/frame")
async def burst_frame(burst_id: str, idx: int = Form(...), frame: UploadFile = File(...)) -> dict:
    """Streaming capture: one frame, analysed as soon as it arrives."""
    if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", burst_id):
        raise HTTPException(400, "bad burst id")
    return await run_in_threadpool(bursts.add_frame, burst_id, idx, await frame.read())


def _verify(jpegs, claimed_id, challenge_id, capture_ms, burst_id, n_frames) -> dict:
    pre, pre_ms = [], 0.0
    if burst_id:
        if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", burst_id):
            raise HTTPException(400, "bad burst id")
        pre, pre_ms = bursts.collect(burst_id, expected=max(0, n_frames - len(jpegs)))
    challenge = None
    if challenge_id:
        challenge = sessions.take_challenge(challenge_id)
        if challenge is None:
            raise HTTPException(410, "Challenge expired or already used - start again")
        claimed_id = claimed_id or challenge.get("claimed_id") or ""

    res = engine.analyse(jpegs, claimed_id or None, challenge, pre_frames=pre,
                         first_idx=(n_frames - len(jpegs)) if burst_id else None)
    res["stage"] = "challenge" if challenge else "passive"
    res["capture_ms"] = capture_ms
    if burst_id:
        res["timing_ms"]["streamed_frames"] = len(pre)
        res["timing_ms"]["analysed_during_capture"] = round(pre_ms, 1)

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
    return {"saved": str(clip.relative_to(DATA.parent)), "result": res, "counts": _clip_counts()}


@app.get("/api/report")
def report() -> dict:
    import evaluation
    res = evaluation.load_results()
    if res is None:
        raise HTTPException(404, "No evaluation yet")
    return res


@app.post("/api/eval/run")
def eval_run() -> dict:
    """Replays every recorded clip (takes ~0.4 s per clip; the kiosk waits meanwhile)."""
    import evaluation
    return evaluation.evaluate()


app.mount("/", StaticFiles(directory=ROOT / "kiosk", html=True), name="kiosk")
