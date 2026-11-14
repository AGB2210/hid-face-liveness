"""Identity providers. Local ArcFace or Amazon Rekognition, chosen in config.yaml
(identity.provider) or with the HID_IDENTITY environment variable."""
from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from functools import lru_cache

import cv2
import numpy as np

from .config import CFG, DATA
from .frames import Frame
from .models import get_models

ENROLL_DIR = DATA / "enroll"


def provider_name() -> str:
    return os.environ.get("HID_IDENTITY") or CFG["identity"]["provider"]


def identity_thresholds() -> dict:
    name = provider_name()
    return {"provider": name, **{k: CFG["identity"][name][k] for k in ("match", "borderline")}}


def face_jpeg(f: Frame, margin: float = 0.4, max_side: int = 640) -> bytes:
    """Face crop with some context, small enough to upload quickly."""
    x, y, w, h = f.box
    H, W = f.img.shape[:2]
    x0, y0 = int(max(0, x - margin * w)), int(max(0, y - margin * h))
    x1, y1 = int(min(W, x + w * (1 + margin))), int(min(H, y + h * (1 + margin)))
    crop = f.img[y0:y1, x0:x1]
    s = min(1.0, max_side / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def save_enrollment_crops(user_id: str, frames: list[Frame]) -> list[bytes]:
    """Keep the enrollment face crops locally so users can be re-indexed into another
    provider later (scripts/sync_rekognition.py) without re-enrolling."""
    d = ENROLL_DIR / user_id
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    crops = [face_jpeg(f) for f in frames]
    for i, b in enumerate(crops):
        (d / f"{i:02d}.jpg").write_bytes(b)
    return crops


def delete_enrollment_crops(user_id: str) -> None:
    shutil.rmtree(ENROLL_DIR / user_id, ignore_errors=True)


class IdentityProvider(ABC):
    remote = False  # True -> network call, run in parallel with liveness

    @abstractmethod
    def enroll(self, user_id: str, name: str, phone: str | None, frames: list[Frame]) -> int:
        """Store templates for a user; returns number of templates stored."""

    @abstractmethod
    def identify(self, frames: list[Frame], claimed_id: str | None = None) -> dict:
        """Returns {user_id, name, similarity (0-1), mode, scores{user: sim}}."""

    def delete(self, user_id: str) -> None:
        import store
        store.delete_user(user_id)
        delete_enrollment_crops(user_id)


def _result(scores: dict, claimed_id: str | None) -> dict:
    """Common result shape. scores holds every enrolled user (0.0 if not returned)."""
    import store
    if claimed_id:
        scores = {k: v for k, v in scores.items() if k == claimed_id}
    best_id = max(scores, key=scores.get) if scores else None
    best = scores.get(best_id, 0.0) if best_id else 0.0
    user = store.get_user(best_id) if best_id else None
    return {"user_id": best_id if user else None, "name": user["name"] if user else None,
            "similarity": round(max(best, 0.0), 3), "mode": "1:1" if claimed_id else "1:N",
            "scores": {k: round(v, 3) for k, v in scores.items()}}


class LocalArcFace(IdentityProvider):
    def _embed(self, frames: list[Frame]) -> np.ndarray:
        af = get_models()["arcface"]
        return af.embed([af.align(f.img, f.lmk5) for f in frames])

    def enroll(self, user_id, name, phone, frames):
        import store
        save_enrollment_crops(user_id, frames)
        embs = self._embed(frames)
        store.upsert_user(user_id, name, phone, embs)
        return len(embs)

    def identify(self, frames, claimed_id=None):
        import store
        probe = self._embed(frames).mean(axis=0)
        probe /= np.linalg.norm(probe) + 1e-9
        scores = {uid: float(np.max(embs @ probe)) for uid, embs in store.templates().items()}
        return _result(scores, claimed_id)


@lru_cache(maxsize=2)
def _make(name: str) -> IdentityProvider:
    if name == "rekognition":
        from .identity_rekognition import Rekognition
        return Rekognition()
    return LocalArcFace()


def get_provider() -> IdentityProvider:
    return _make(provider_name())
