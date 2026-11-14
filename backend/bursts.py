"""Streaming capture: the kiosk uploads each frame while the camera is still capturing,
and the server analyses it on arrival. When the final request comes in, only the last
frame and the decision are left to do, so network time overlaps with the capture."""
from __future__ import annotations

import threading
import time

from pipeline.frames import Frame, analyse_frame, decode
from pipeline.models import models_lock

TTL_S = 60
_lock = threading.Lock()
_bursts: dict[str, dict] = {}


def _get(burst_id: str) -> dict:
    with _lock:
        now = time.time()
        for k in [k for k, b in _bursts.items() if now - b["created"] > TTL_S]:
            del _bursts[k]
        b = _bursts.get(burst_id)
        if b is None:
            b = _bursts[burst_id] = {"created": now, "frames": {}, "cond": threading.Condition(), "ms": 0.0}
        return b


def add_frame(burst_id: str, idx: int, jpeg: bytes, action: str | None = None) -> dict:
    """action: the live cue this burst is for. The reply then says whether it is already done,
    so the kiosk can stop capturing early (the final decision checks all frames again)."""
    b = _get(burst_id)
    t = time.perf_counter()
    imgs = decode([jpeg])
    if not imgs:
        return {"ok": False, "reason": "bad image"}
    with models_lock():
        f = analyse_frame(idx, imgs[0])
    ms = (time.perf_counter() - t) * 1000
    with b["cond"]:
        b["frames"][idx] = f
        b["ms"] += ms
        b["cond"].notify_all()
        so_far = [b["frames"][k] for k in sorted(b["frames"])]
    out = {"ok": True, "idx": idx, "ms": round(ms, 1)}
    if action:
        from pipeline.engine import action_done
        out["done"] = action_done([x for x in so_far if x.mesh is not None], action)[0]
    return out


def collect(burst_id: str, expected: int, timeout: float = 2.0) -> tuple[list[Frame], float]:
    """Wait (briefly) for frames still in flight, then hand them over and forget the burst."""
    b = _get(burst_id)
    with b["cond"]:
        b["cond"].wait_for(lambda: len(b["frames"]) >= expected, timeout=timeout)
        frames = [b["frames"][k] for k in sorted(b["frames"])]
        ms = b["ms"]
    with _lock:
        _bursts.pop(burst_id, None)
    return frames, ms
