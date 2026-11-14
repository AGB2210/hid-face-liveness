"""Runs a capture burst through quality -> liveness -> identity -> decision."""
from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import liveness as lv
from .config import CFG
from .frames import ISSUE_TEXT, Frame, analyse_frame, decode
from .identity import get_provider, identity_thresholds
from .models import models_lock

ALLOW, VERIFY, DENY = "ALLOW", "REQUEST_VERIFICATION", "DENY"


def _ramp(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="identity")


def _safe_identify(provider, frames, claimed_id) -> dict:
    try:
        return provider.identify(frames, claimed_id)
    except Exception as e:  # network / credentials / service errors must not crash the kiosk
        return {"user_id": None, "name": None, "similarity": 0.0, "scores": {},
                "mode": "1:1" if claimed_id else "1:N", "error": f"{type(e).__name__}: {e}"[:300]}


def fuse(live: dict, extra_bonus: float = 0.0) -> tuple[float, list[str]]:
    c = CFG["liveness"]
    num = den = 0.0
    for k, w in c["weights"].items():
        s = live.get(k, {}).get("score")
        if s is not None:
            num += w * s
            den += w
    fused = num / den if den else 0.5
    if live.get("blink", {}).get("blinked"):
        fused += c["blink_bonus"]
    fused = float(np.clip(fused + extra_bonus, 0, 1))

    vetoes = []
    v = c["veto"]
    if (live["minifas"]["score"] is not None) and live["minifas"]["score"] < v["minifas_below"]:
        vetoes.append("Anti-spoof model: presentation attack detected")
    if (live["moire"]["score"] is not None) and live["moire"]["score"] < v["moire_below"]:
        vetoes.append("Screen pixel pattern (moire) detected")
    if (live["flatness"]["score"] is not None) and live["flatness"]["score"] < v["flatness_below"]:
        vetoes.append("Face moved like a flat surface (photo/screen)")
    return fused, vetoes


def action_done(frames: list[Frame], action: str) -> tuple[bool, dict]:
    c = CFG["challenge"]
    if action == "blink":
        s = lv.blink_series(frames)
        return lv.blinked(frames, c["blink_thresh"]), {"blink_max": round(max(s), 3) if s else None}
    yaws = [f.yaw for f in frames if f.mesh is not None]
    if len(yaws) < 3:
        return False, {"reason": "face lost"}
    start = float(np.median(yaws[:2]))
    delta = (max(yaws) - start) if action == "turn_left" else (start - min(yaws))
    return delta >= c["turn_deg"], {"yaw_delta": round(delta, 1)}


def analyse(jpegs: list[bytes], claimed_id: str | None = None,
            challenge: dict | None = None, pre_frames: list[Frame] | None = None,
            first_idx: int | None = None, provider_name: str | None = None) -> dict:
    """pre_frames: frames already analysed while the burst was still being captured
    (streaming upload). jpegs are the remaining frames, numbered from first_idx."""
    pre_frames = pre_frames or []
    start = len(pre_frames) if first_idx is None else first_idx
    t0 = time.perf_counter()
    imgs = decode(jpegs)
    t_dec = time.perf_counter()
    with models_lock():
        new = [analyse_frame(start + i, im) for i, im in enumerate(imgs)]
        frames = sorted(pre_frames + new, key=lambda f: f.idx)
        t_frames = time.perf_counter()

        face_frames = [f for f in frames if f.box is not None and f.mesh is not None]
        good = [f for f in frames if f.good]
        q = CFG["quality"]
        issue_counts = Counter(i for f in frames for i in f.issues)
        quality = {
            "frames": len(frames), "good_frames": len(good),
            "issues": dict(issue_counts),
            "face_px": round(float(np.median([f.face_px for f in face_frames])), 1) if face_frames else 0,
            "blur": round(float(np.median([f.blur for f in face_frames])), 1) if face_frames else 0,
            "brightness": round(float(np.median([f.brightness for f in face_frames])), 1) if face_frames else 0,
        }
        quality_ok = len(good) >= min(q["min_good_frames"], len(frames))

        result = {"quality": quality, "liveness": None, "identity": None, "challenge": None}
        if not face_frames:
            result.update(_decision(VERIFY, ["No face found - please look at the camera"], 0.0,
                                    next_step="recapture"))
            return _finish(result, t0, t_dec, t_frames, t_frames, t_frames)

        top = sorted(good or face_frames, key=lambda f: f.rank, reverse=True)[:3]
        provider = get_provider(provider_name)
        # A remote provider (Rekognition) is a network call: run it while liveness computes.
        fut = _POOL.submit(_safe_identify, provider, top, claimed_id) if provider.remote else None
        live = lv.run_all(face_frames, top)
        t_live = time.perf_counter()
        if fut is None:
            ident = _safe_identify(provider, top, claimed_id)
    if fut is not None:
        ident = fut.result()
    t_id = time.perf_counter()

    action_ok = None
    if challenge:
        action_ok, adetail = action_done(face_frames, challenge["action"])
        result["challenge"] = {"action": challenge["action"], "passed": action_ok, **adetail}

    bonus = CFG["challenge"]["pass_bonus"] if action_ok else 0.0
    fused, vetoes = fuse(live, bonus)
    result["liveness"] = {"score": round(fused, 3), "checks": live, "vetoes": vetoes}
    result["identity"] = ident

    ic = identity_thresholds(provider_name)
    sim = ident["similarity"] if ident["user_id"] else 0.0
    id_conf = _ramp(sim, ic["borderline"] - 0.1, min(1.0, ic["match"] + 0.08))
    lc = CFG["liveness"]
    worst_issue = issue_counts.most_common(1)[0][0] if issue_counts else None

    if challenge and not action_ok:
        d = _decision(DENY, [f"Challenge not completed ({challenge['action'].replace('_', ' ')})"], 0.9)
    elif vetoes:
        d = _decision(DENY, vetoes, max(0.9, 1 - fused))
    elif fused < lc["deny"]:
        d = _decision(DENY, ["Liveness check failed - possible spoof"], 1 - fused)
    elif ident.get("error"):
        d = _decision(VERIFY, ["Identity service unavailable - please try again"], 0.0, next_step="recapture")
    elif not ident["user_id"] or sim < ic["borderline"]:
        d = _decision(DENY, ["Identity not recognised"], 1 - id_conf)
    elif challenge and challenge.get("candidate") and ident["user_id"] != challenge["candidate"]:
        d = _decision(DENY, ["Different person from first attempt"], 0.9)
    elif fused >= lc["allow"] and sim >= ic["match"] and quality_ok:
        d = _decision(ALLOW, ["Live person and identity match"], min(fused, id_conf))
    else:
        reasons = []
        if not quality_ok and worst_issue:
            reasons.append(ISSUE_TEXT.get(worst_issue, worst_issue))
        if fused < lc["allow"]:
            reasons.append("Liveness confidence is low")
        if sim < ic["match"]:
            reasons.append("Identity match is borderline")
        nxt = "otp" if challenge else "challenge"
        d = _decision(VERIFY, reasons or ["Low confidence"], min(fused, id_conf), next_step=nxt)
    result.update(d)
    return _finish(result, t0, t_dec, t_frames, t_live, t_id)


def _decision(decision, reasons, confidence, next_step=None):
    return {"decision": decision, "reasons": reasons,
            "confidence": round(float(np.clip(confidence, 0, 1)), 3), "next": next_step}


def _finish(result, t0, t_dec, t_frames, t_live, t_id):
    t_end = time.perf_counter()
    result["timing_ms"] = {
        "decode": round((t_dec - t0) * 1000, 1),
        "detect_landmarks_quality": round((t_frames - t_dec) * 1000, 1),
        "liveness": round((t_live - t_frames) * 1000, 1),
        "identity": round((t_id - t_live) * 1000, 1),
        "processing": round((t_end - t0) * 1000, 1),
    }
    return result
