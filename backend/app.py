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

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, Response, UploadFile  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import HTMLResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import admin  # noqa: E402
import bursts  # noqa: E402
import node  # noqa: E402
import policy  # noqa: E402
import sessions  # noqa: E402
import store  # noqa: E402
from pipeline import engine  # noqa: E402
from pipeline.config import CFG, DATA, ROOT, reload_config  # noqa: E402
from pipeline.frames import analyse_frame, decode  # noqa: E402
from pipeline.identity import get_provider, provider_name  # noqa: E402
from pipeline.models import models_lock, warmup  # noqa: E402

LABELS = ["live", "live_glasses", "live_sunglasses", "live_mask", "print", "screen_photo", "video_replay", "mask"]
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
    if path.startswith("/api/sync/"):
        key = os.environ.get("HID_SYNC_KEY", "")
        if not key or not secrets.compare_digest(request.headers.get("x-hid-sync-key", ""), key):
            return Response('{"detail":"sync key required"}', status_code=401, media_type="application/json")
        return await call_next(request)
    if PASSWORD and path not in ("/login", "/styles.css") and             not secrets.compare_digest(request.cookies.get("hid_auth", ""), AUTH_TOKEN):
        if path.startswith("/api/"):
            return Response('{"detail":"login required"}', status_code=401, media_type="application/json")
        return HTMLResponse(LOGIN_HTML.format(msg=""), status_code=401)
    response = await call_next(request)
    # pages/scripts change with every update: make browsers revalidate instead of reusing stale copies
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


def _sess(request: Request) -> dict | None:
    return admin.session(request.cookies.get("hid_admin"))


def _allow(request: Request, permission: str, first_run: bool = False) -> dict | None:
    """Staff permission check. first_run=True also allows it during first-run setup
    (before any policy admin exists) so the first admin can be enrolled."""
    s = _sess(request)
    if admin.allowed(s, permission) or (first_run and admin.bootstrap_open()):
        return s
    raise HTTPException(403, "Staff only - sign in with a qualified role")


def _kiosk_zone() -> str:
    return os.environ.get("HID_KIOSK_ZONE") or CFG["access"]["default_zone"]


def _public(res: dict) -> dict:
    """What a student/visitor at the kiosk gets back: the decision and what to do next -
    no match scores for other people, no internal liveness numbers."""
    ident, ctx, ch, occ = res.get("identity") or {}, res.get("context") or {}, res.get("challenge"), res.get("occlusion") or {}
    out = {k: res.get(k) for k in ("decision", "reasons", "next", "challenge_request", "otp_request", "mode",
                                   "capture_ms", "stage", "confidence") if k in res}
    out["identity"] = {"name": ident.get("name"), "user_id": ident.get("user_id")} if ident.get("name") else {}
    if ctx:
        out["context"] = {k: ctx.get(k) for k in ("zone_name", "verdict", "time", "role", "security")}
    if ch:
        out["challenge"] = {"action": ch.get("action"), "passed": ch.get("passed")}
    out["occlusion"] = {"sunglasses": occ.get("sunglasses"), "mask": occ.get("mask")}
    out["timing_ms"] = {"processing": (res.get("timing_ms") or {}).get("processing")}
    return out


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
    node.start()  # central-server monitor + audit sync (edge kiosks)


async def _read(frames: list[UploadFile]) -> list[bytes]:
    if not frames:
        raise HTTPException(400, "no frames")
    return [await f.read() for f in frames]


@app.get("/assets/face_landmarker.task")
def landmarker_model():
    """Same MediaPipe model the server uses, for live guidance in the browser."""
    from fastapi.responses import FileResponse
    return FileResponse(ROOT / "models" / "face_landmarker.task", media_type="application/octet-stream")


@app.get("/api/config")
def config() -> dict:
    pol = node.current_policy()
    return {"capture": CFG["capture"], "labels": LABELS, "actions": sessions.ACTIONS,
            "identity_provider": provider_name(), "zones": policy.zones(pol), "roles": policy.roles(pol),
            "default_zone": _kiosk_zone(), "node": {"mode": node.mode()}}


@app.post("/api/config/reload")
def config_reload(request: Request) -> dict:
    _allow(request, "edit_policy")
    reload_config()
    return {"ok": True}


@app.get("/api/users")
def users(request: Request) -> list[dict]:
    _allow(request, "view_policy", first_run=True)
    return store.list_users()


@app.delete("/api/users/{user_id}")
def delete_user(request: Request, user_id: str) -> dict:
    _allow(request, "manage_people")
    get_provider().delete(user_id)
    return {"ok": True}


@app.post("/api/enroll")
async def enroll(request: Request, user_id: str = Form(...), name: str = Form(...), phone: str = Form(""),
                 role: str = Form("student"), valid_until: str = Form(""), poses: str = Form(""),
                 frames: list[UploadFile] = File(...)) -> dict:
    _allow(request, "manage_people", first_run=True)
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", user_id):
        raise HTTPException(400, "user_id: letters, digits, _ . - only")
    jpegs = await _read(frames)
    return await run_in_threadpool(_enroll, user_id, name, phone, role, valid_until, jpegs, poses)


def _enroll(user_id, name, phone, role, valid_until, jpegs, poses: str = "") -> dict:
    """Guided enrollment: frames cover straight, left and right head poses. The turns make the
    3-D flatness test conclusive, so a photo or screen can't be enrolled in someone's name,
    and templates from several angles make later matching more robust."""
    from pipeline import liveness as lv
    ec = CFG["enrollment"]
    imgs = decode(jpegs)
    with models_lock():
        fr = [analyse_frame(i, im) for i, im in enumerate(imgs)]
        labels = [x.strip() for x in poses.split(",")] if poses else []
        pose_of = {id(f): (labels[i] if len(labels) == len(fr) else None) for i, f in enumerate(fr)}
        good = [f for f in fr if f.good]
        face_frames = [f for f in fr if f.box is not None and f.mesh is not None]
        if len(good) < 2:
            issues = sorted({i for f in fr for i in f.issues})
            raise HTTPException(422, f"Not enough good-quality frames ({len(good)}). Issues: {issues}")
        yaws = [f.yaw for f in face_frames]
        turned = {"left": max(yaws) if yaws else 0.0, "right": -min(yaws) if yaws else 0.0}
        if ec["require_turns"] and min(turned.values()) < ec["min_turn_deg"]:
            raise HTTPException(422, f"Please turn your head left and right during enrollment "
                                     f"(measured {turned['left']:.0f} deg left, {turned['right']:.0f} deg right)")
        top = sorted(good, key=lambda f: f.rank, reverse=True)[:3]
        live = lv.run_all(face_frames, top)
        fused, vetoes = engine.fuse(live)
        if ec["require_liveness"] and (vetoes or fused < CFG["liveness"]["deny"]):
            raise HTTPException(422, "Enrollment rejected: liveness check failed (" +
                                ("; ".join(vetoes) or f"score {fused:.2f}") + "). Enroll in person, not from a photo or screen.")
        # templates: best frontal frame, best left and right views, then (if the person wears
        # glasses) the best frontal frame without them, then more frontal frames
        bare_frames = [f for f in good if pose_of[id(f)] == "bare"]
        worn = [f for f in good if pose_of[id(f)] != "bare"]
        front = sorted([f for f in worn if abs(f.yaw) < 10], key=lambda f: f.rank, reverse=True)
        left = max((f for f in worn if f.yaw >= 8), key=lambda f: f.rank, default=None)
        right = max((f for f in worn if f.yaw <= -8), key=lambda f: f.rank, default=None)
        bare = max(bare_frames, key=lambda f: f.rank, default=None)
        chosen = front[:1] + [x for x in (left, right, bare) if x] + front[1:3]
        for f in sorted(good, key=lambda f: f.rank, reverse=True):  # top up to 5 with the best others
            if len(chosen) >= 5:
                break
            if all(f is not c for c in chosen):
                chosen.append(f)
        try:
            n = get_provider().enroll(user_id, name, phone or None, chosen)
        except ValueError as e:
            raise HTTPException(422, str(e))
        except Exception as e:  # AWS credentials / network
            raise HTTPException(502, f"Identity provider error: {type(e).__name__}: {e}"[:300])
    store.set_access(user_id, role, valid_until or None)
    poses = {"front": bool(front), "left": left is not None, "right": right is not None}
    if bare_frames:
        poses["without glasses"] = bare is not None
    store.append_audit(node.KIOSK_ID, None, user_id, "ENROLLED", "enroll", node.mode(), round(fused, 3), None, None,
                       [f"Enrollment liveness {fused:.2f}", f"poses: {', '.join(k for k, v in poses.items() if v)}"])
    return {"ok": True, "user_id": user_id, "templates": n, "provider": provider_name(),
            "liveness": {"score": round(fused, 3), "checks": live},
            "turn_deg": {k: round(v, 1) for k, v in turned.items()}, "poses": poses,
            "authenticator": _new_totp(user_id, name)}


def _new_totp(user_id: str, name: str) -> dict:
    """Authenticator-app second factor: works with no network, so the offline flow can use it."""
    import io

    import pyotp
    import qrcode
    import qrcode.image.svg
    secret = pyotp.random_base32()
    store.set_totp(user_id, secret)
    uri = pyotp.TOTP(secret).provisioning_uri(name=f"{name} ({user_id})", issuer_name="HID Kiosk")
    buf = io.BytesIO()
    qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2).save(buf)
    return {"uri": uri, "svg": buf.getvalue().decode()}


@app.post("/api/users/{user_id}/access")
def set_access(request: Request, user_id: str, role: str = Form(...), valid_until: str = Form("")) -> dict:
    _allow(request, "manage_people")
    if not store.get_user(user_id):
        raise HTTPException(404, "unknown user")
    if role not in policy.roles(node.current_policy()):
        raise HTTPException(400, "unknown role")
    store.set_access(user_id, role, valid_until or None)
    return {"ok": True}


@app.post("/api/users/{user_id}/authenticator")
def reset_authenticator(request: Request, user_id: str) -> dict:
    _allow(request, "manage_people", first_run=True)
    u = store.get_user(user_id)
    if not u:
        raise HTTPException(404, "unknown user")
    return _new_totp(user_id, u["name"])


# ---------------------------------------------------------------- policy, node, audit
@app.get("/api/policy")
def get_policy(request: Request) -> dict:
    _allow(request, "view_policy")
    pol = node.current_policy()
    return {"zones": pol["zones"], "roles": pol["roles"], "timezone": pol.get("timezone"),
            "grace_minutes": pol.get("grace_minutes"), "holidays": [str(h) for h in pol.get("holidays", [])],
            "source": pol.get("_source"), "now": policy.now(pol).strftime("%a %d %b %Y %H:%M")}


@app.post("/api/policy/test")
def policy_test(request: Request, zone: str = Form(...), user_id: str = Form(""), role: str = Form(""),
                valid_until: str = Form(""), at: str = Form("")) -> dict:
    _allow(request, "view_policy")
    user = store.get_user(user_id) if user_id else {"role": role or "visitor", "valid_until": valid_until or None}
    return policy.evaluate(user, zone, at or None, node.current_policy())


def cue_required(zone: str, offline: bool) -> bool:
    """Risk-based friction: a random live cue (blink / head turn) only where it matters."""
    z = node.current_policy()["zones"].get(zone, {})
    return offline or z.get("security") == "high"


@app.post("/api/cue")
def cue(zone: str = Form(""), avoid: str = Form("")) -> dict:
    zone = zone or CFG["access"]["default_zone"]
    ch = sessions.new_challenge(None, None, avoid=avoid if avoid == "blink" else None)
    return {"required": cue_required(zone, node.offline()), "id": ch["id"], "action": ch["action"],
            "instruction": ch["instruction"], "duration_ms": CFG["capture"]["challenge_duration_ms"],
            "frames": CFG["capture"]["challenge_frames"]}


@app.get("/api/node/status")
def node_status(request: Request) -> dict:
    st = node.status()
    if admin.allowed(_sess(request), "view_policy"):
        return st
    return {"mode": st["mode"], "role": st["role"]}


@app.post("/api/node/simulate_offline")
def simulate_offline(request: Request, on: bool = Form(...)) -> dict:
    _allow(request, "kiosk_controls")
    node.simulate_offline(on)
    return node.status()


def _admin(request: Request) -> dict | None:
    return _sess(request)


def _require(request: Request, permission: str) -> dict:
    return _allow(request, permission)


@app.get("/api/audit")
def audit(request: Request, limit: int = 200) -> dict:
    _require(request, "view_audit")
    return {"records": store.list_audit(limit), "chains": store.verify_chains(), "node": node.status()}


# ---------------------------------------------------------------- qualified roles
@app.post("/api/admin/login")
def admin_login(user_id: str = Form(...), code: str = Form(...)):
    from fastapi.responses import JSONResponse
    token, msg = admin.login(user_id.strip(), code)
    store.append_audit(node.KIOSK_ID, None, user_id.strip() or None, "ADMIN_SIGNIN" if token else "ADMIN_DENIED",
                       "admin", node.mode(), None, None, None, [msg])
    if not token:
        raise HTTPException(403, msg)
    r = JSONResponse({"ok": True, **admin_me_for(admin.session(token))})
    r.set_cookie("hid_admin", token, max_age=admin.SESSION_S, httponly=True, samesite="strict")
    return r


@app.post("/api/admin/logout")
def admin_logout(request: Request):
    from fastapi.responses import JSONResponse
    admin.logout(request.cookies.get("hid_admin"))
    r = JSONResponse({"ok": True})
    r.delete_cookie("hid_admin")
    return r


def admin_me_for(s: dict | None) -> dict:
    import time as _t
    u = store.get_user(s["user_id"]) if s else None
    return {"user_id": s["user_id"] if s else None, "name": u["name"] if u else None,
            "role": s["role"] if s else None, "role_name": admin.ROLES.get(s["role"]) if s else None,
            "expires_in": int(s["expires"] - _t.time()) if s else 0,
            "can_edit_policy": admin.allowed(s, "edit_policy") and node.ROLE != "edge",
            "can_view_audit": admin.allowed(s, "view_audit"),
            "can": {perm: admin.allowed(s, perm) for perm in admin.CAN},
            "edge": node.ROLE == "edge", "bootstrap": admin.bootstrap_open(), "roles": admin.ROLES}


@app.get("/api/admin/me")
def admin_me(request: Request) -> dict:
    return admin_me_for(_admin(request))


@app.post("/api/users/{user_id}/admin_role")
def set_admin_role(request: Request, user_id: str, role: str = Form("")) -> dict:
    s = _admin(request)
    if role and role not in admin.ROLES:
        raise HTTPException(400, "unknown admin role")
    if not store.get_user(user_id):
        raise HTTPException(404, "unknown user")
    first = admin.bootstrap_open() and role == "policy_admin"
    if not (admin.allowed(s, "edit_policy") or first):
        raise HTTPException(403, "Only a policy admin can grant or revoke admin roles")
    if not store.get_user(user_id).get("totp_secret"):
        raise HTTPException(400, "This person needs an authenticator app first (New QR on the Enroll page)")
    store.set_admin_role(user_id, role or None)
    by = s["user_id"] if s else "first-run setup"
    store.append_audit(node.KIOSK_ID, None, user_id, "ROLE_CHANGED", "admin", node.mode(), None, None, None,
                       [f"{admin.ROLES.get(role, 'no admin role')} granted by {by}"])
    return {"ok": True}


@app.put("/api/policy")
def put_policy(request: Request, payload: dict = Body(...)) -> dict:
    s = _require(request, "edit_policy")
    if node.ROLE == "edge":
        raise HTTPException(409, "This kiosk follows the central server's policy - edit it there")
    current = policy.load()
    new = {"timezone": current.get("timezone"), "grace_minutes": payload.get("grace_minutes", current.get("grace_minutes")),
           "zones": current["zones"], "roles": current["roles"], "holidays": payload.get("holidays", current.get("holidays", []))}
    for zid, sec in (payload.get("zone_security") or {}).items():
        if zid in new["zones"]:
            new["zones"][zid] = {**new["zones"][zid], "security": sec}
    for rid, zones in (payload.get("windows") or {}).items():
        if rid in new["roles"]:
            new["roles"][rid] = {**new["roles"][rid], "zones": {z: [w.strip() for w in ws if w.strip()] for z, ws in zones.items()}}
    try:
        summary = policy.save_edit(new, s["user_id"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    if summary:
        store.append_audit(node.KIOSK_ID, None, s["user_id"], "POLICY_CHANGED", "admin", node.mode(),
                           None, None, None, summary)
    return {"ok": True, "changes": summary}


# ---------------------------------------------------------------- central-server sync API (shared key)
@app.get("/api/sync/health")
def sync_health() -> dict:
    return {"ok": True, "role": node.ROLE}


@app.post("/api/sync/audit")
def sync_audit(payload: dict = Body(...)) -> dict:
    recs = payload.get("records", [])
    stored = store.ingest_audit(recs)
    valid = [r["id"] for r in recs if r.get("hash") == store._hash(r)]
    return {"stored": stored, "ids": valid}


@app.get("/api/sync/policy")
def sync_policy() -> Response:
    return Response(policy.policy_text(), media_type="text/yaml")


@app.post("/api/verify")
async def verify(request: Request, frames: list[UploadFile] = File(...), claimed_id: str = Form(""),
                 challenge_id: str = Form(""), capture_ms: float = Form(0),
                 burst_id: str = Form(""), n_frames: int = Form(0),
                 zone: str = Form(""), at: str = Form("")) -> dict:
    """frames: the whole burst, or (streaming) just the last frame(s) with burst_id set
    and n_frames = total frames in the burst."""
    jpegs = await _read(frames)
    staff = admin.allowed(_sess(request), "kiosk_controls")
    if not staff:  # students can't pick another zone or fake the time
        zone, at = _kiosk_zone(), ""
    res = await run_in_threadpool(_verify, jpegs, claimed_id, challenge_id, capture_ms, burst_id, n_frames,
                                  zone or _kiosk_zone(), at)
    return res if admin.allowed(_sess(request), "view_policy") else _public(res)


@app.post("/api/burst/{burst_id}/frame")
async def burst_frame(burst_id: str, idx: int = Form(...), frame: UploadFile = File(...)) -> dict:
    """Streaming capture: one frame, analysed as soon as it arrives."""
    if not re.fullmatch(r"[A-Za-z0-9-]{8,64}", burst_id):
        raise HTTPException(400, "bad burst id")
    return await run_in_threadpool(bursts.add_frame, burst_id, idx, await frame.read())


def _verify(jpegs, claimed_id, challenge_id, capture_ms, burst_id, n_frames, zone, at) -> dict:
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

    offline = node.offline()
    res = engine.analyse(jpegs, claimed_id or None, challenge, pre_frames=pre,
                         first_idx=(n_frames - len(jpegs)) if burst_id else None,
                         # offline: the cloud matcher is unreachable -> on-device template cache
                         provider_name="local_arcface" if offline else None)
    res["stage"] = "challenge" if challenge else "passive"
    res["capture_ms"] = capture_ms
    if burst_id:
        res["timing_ms"]["streamed_frames"] = len(pre)
        res["timing_ms"]["analysed_during_capture"] = round(pre_ms, 1)

    cue_passed = bool(challenge) and bool((res.get("challenge") or {}).get("passed"))
    _apply_context(res, zone, at or None, offline, cue_passed)

    if res["decision"] == engine.VERIFY and res["next"] == "challenge":
        cand = res["identity"]["user_id"] if res.get("identity") else None
        eyes_hidden = bool((res.get("occlusion") or {}).get("sunglasses"))
        ch = sessions.new_challenge(cand, claimed_id or None, avoid="blink" if eyes_hidden else None)
        res["challenge_request"] = {
            "id": ch["id"], "action": ch["action"], "instruction": ch["instruction"],
            "duration_ms": CFG["capture"]["challenge_duration_ms"],
            "frames": CFG["capture"]["challenge_frames"]}
    elif res["decision"] == engine.VERIFY and res["next"] in ("otp", "second_factor"):
        ident = res.get("identity") or {}
        user = store.get_user(ident["user_id"]) if ident.get("user_id") else None
        if user:
            res["otp_request"] = sessions.new_otp(
                user["id"], user["name"], user.get("phone"), sms=not offline, totp_secret=user.get("totp_secret"),
                context={"zone": zone, "schedule": (res.get("context") or {}).get("verdict"), "mode": res["mode"],
                         "liveness": (res.get("liveness") or {}).get("score"), "similarity": ident.get("similarity")})
        else:
            res.update(decision=engine.DENY, reasons=["Identity could not be established"], next=None)

    ident = res.get("identity") or {}
    store.log_attempt(res["decision"], res["stage"], ident.get("user_id"), ident.get("similarity"),
                      (res.get("liveness") or {}).get("score"), res["timing_ms"]["processing"],
                      None, res["reasons"])
    store.append_audit(node.KIOSK_ID, zone, ident.get("user_id"), res["decision"], res["stage"], res["mode"],
                       (res.get("liveness") or {}).get("score"), ident.get("similarity"),
                       (res.get("context") or {}).get("verdict"), res["reasons"])
    return res


def _apply_context(res: dict, zone: str, at: str | None, offline: bool, cue_passed: bool = False) -> None:
    """Biometrics say who is there and that they are live; the schedule says whether they may
    enter here, now. Offline, the kiosk never fails open: high-security zones close, and every
    other entry needs the authenticator-app code on top of liveness + the cached template."""
    res["mode"] = "offline" if offline else "online"
    ident = res.get("identity") or {}
    user = store.get_user(ident["user_id"]) if ident.get("user_id") else None
    if res["decision"] == engine.VERIFY and res.get("next") == "recapture" and not user:
        return  # guidance first (e.g. remove sunglasses), identity is checked on the retry
    if res["decision"] == engine.DENY or not user:
        if offline and res["decision"] != engine.DENY:
            res.update(decision=engine.DENY, next=None,
                       reasons=["Offline fail-safe: identity could not be confirmed on this device"])
        return
    ctx = policy.evaluate(user, zone, at, node.current_policy())
    res["context"] = ctx
    if ctx["verdict"] == "outside":
        res.update(decision=engine.DENY, next=None, reasons=[ctx["reason"]])
        return
    if res["decision"] == engine.VERIFY and res.get("next") == "recapture":
        return  # guidance (remove sunglasses / mask, better light) comes before cues and codes
    # enforced server-side: a modified client can't skip the cue in high-risk situations
    if cue_required(zone, offline) and not cue_passed and not (offline and ctx["security"] == "high"):
        why = "offline fail-safe" if offline else f"{ctx['zone_name']} is high-security"
        res.update(decision=engine.VERIFY, next="challenge",
                   reasons=[f"Live cue required ({why})", ctx["reason"]])
        return
    if offline:
        age = store.template_age_days(user["id"])
        if ctx["security"] == "high":
            why = f"{ctx['zone_name']} is high-security: closed while the central server is unreachable"
        elif age is None or age > CFG["access"]["offline_template_ttl_days"]:
            why = "Offline template cache expired - re-enroll when back online"
        elif not user.get("totp_secret"):
            why = "Offline: no authenticator app enrolled for the second factor"
        else:
            why = None
        if why:
            res.update(decision=engine.DENY, next=None, reasons=[f"Offline fail-safe: {why}"])
        else:
            res.update(decision=engine.VERIFY, next="second_factor",
                       reasons=["Offline fail-safe: live cue passed - now enter the code from your authenticator app",
                                ctx["reason"]])
        return
    if res["decision"] == engine.ALLOW:
        if ctx["verdict"] == "grace":
            res.update(decision=engine.VERIFY, next="second_factor", reasons=[f"Step-up: {ctx['reason']}"])
        elif ctx["security"] == "high":
            res.update(decision=engine.VERIFY, next="second_factor",
                       reasons=[f"Step-up: live cue passed; {ctx['zone_name']} is high-security - second factor required"])
        else:
            res["reasons"] = res["reasons"] + [ctx["reason"]]


@app.post("/api/otp/verify")
def otp_verify(session_id: str = Form(...), code: str = Form(...)) -> dict:
    ok, user_id, msg, ctx = sessions.check_otp(session_id, code)
    if msg == "WRONG_CODE":
        return {"decision": None, "retry": True, "reasons": ["Wrong code - try again"]}
    user = store.get_user(user_id) if user_id else None
    decision = "ALLOW" if ok else "DENY"
    mode = ctx.get("mode", node.mode())
    reasons = ([f"Second factor confirmed{' (offline)' if mode == 'offline' else ''}"] if ok else [msg])
    store.log_attempt(decision, "second_factor", user_id, None, None, 0, None, reasons)
    store.append_audit(node.KIOSK_ID, ctx.get("zone"), user_id, decision, "second_factor", mode,
                       ctx.get("liveness"), ctx.get("similarity"), ctx.get("schedule"), reasons)
    return {"decision": decision, "reasons": reasons, "mode": mode,
            "identity": {"user_id": user_id, "name": user["name"] if user else None}}


@app.get("/api/otp/outbox")
def otp_outbox(request: Request) -> list[dict]:
    _allow(request, "kiosk_controls")  # demo "phone": staff screen only
    return list(reversed(sessions.OUTBOX))


@app.post("/api/timing")
def timing(total_ms: float = Form(...)) -> dict:
    store.update_total_ms(total_ms)
    return {"ok": True}


@app.get("/api/attempts")
def attempts(request: Request, limit: int = 50) -> list[dict]:
    _require(request, "view_audit")
    return store.recent_attempts(limit)


def _clip_counts() -> dict:
    return {lab: len(list((CLIPS / lab).glob("*/*/meta.json"))) for lab in LABELS}


@app.get("/api/clips/counts")
def clip_counts(request: Request) -> dict:
    _allow(request, "manage_people")
    return _clip_counts()


@app.post("/api/record")
async def record(request: Request, label: str = Form(...), subject: str = Form(...), capture_ms: float = Form(0),
                 note: str = Form(""), frames: list[UploadFile] = File(...)) -> dict:
    _allow(request, "manage_people")
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
def report(request: Request) -> dict:
    _allow(request, "view_results")
    import evaluation
    res = evaluation.load_results()
    if res is None:
        raise HTTPException(404, "No evaluation yet")
    return res


@app.post("/api/eval/run")
def eval_run(request: Request) -> dict:
    _allow(request, "run_eval")
    """Replays every recorded clip (takes ~0.4 s per clip; the kiosk waits meanwhile)."""
    import evaluation
    return evaluation.evaluate()


# Two apps: "/" is the public kiosk; "/console/" is the staff back office (own sign-in screen).
for _old in ("enroll", "policy", "audit", "record", "report"):
    app.add_api_route(f"/{_old}.html", (lambda name=_old: RedirectResponse(f"/console/{name}.html", status_code=308)),
                      methods=["GET"], include_in_schema=False)
app.add_api_route("/staff.html", lambda: RedirectResponse("/console/", status_code=308), methods=["GET"], include_in_schema=False)
app.add_api_route("/console", lambda: RedirectResponse("/console/", status_code=308), methods=["GET"], include_in_schema=False)

app.mount("/", StaticFiles(directory=ROOT / "kiosk", html=True), name="kiosk")
