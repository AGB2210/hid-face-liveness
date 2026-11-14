"""Identity providers. Local ArcFace now; Amazon Rekognition is swapped in via config."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .config import CFG
from .frames import Frame
from .models import get_models


class IdentityProvider(ABC):
    @abstractmethod
    def enroll(self, user_id: str, name: str, phone: str | None, frames: list[Frame]) -> int:
        """Store templates for a user; returns number of templates stored."""

    @abstractmethod
    def identify(self, frames: list[Frame], claimed_id: str | None = None) -> dict:
        """Returns {user_id, name, similarity, mode} for the best match (user_id None if nobody)."""


class LocalArcFace(IdentityProvider):
    def _embed(self, frames: list[Frame]) -> np.ndarray:
        af = get_models()["arcface"]
        return af.embed([af.align(f.img, f.lmk5) for f in frames])

    def enroll(self, user_id, name, phone, frames):
        import store
        embs = self._embed(frames)
        store.upsert_user(user_id, name, phone, embs)
        return len(embs)

    def identify(self, frames, claimed_id=None):
        import store
        probe = self._embed(frames).mean(axis=0)
        probe /= np.linalg.norm(probe) + 1e-9
        gallery = store.templates()
        if claimed_id:
            gallery = {k: v for k, v in gallery.items() if k == claimed_id}
        best_id, best_sim, scores = None, -1.0, {}
        for uid, embs in gallery.items():
            sim = float(np.max(embs @ probe))  # best of the user's templates
            scores[uid] = round(sim, 3)
            if sim > best_sim:
                best_id, best_sim = uid, sim
        user = store.get_user(best_id) if best_id else None
        return {"user_id": best_id, "name": user["name"] if user else None,
                "similarity": round(max(best_sim, 0.0), 3),
                "mode": "1:1" if claimed_id else "1:N", "scores": scores}


def get_provider() -> IdentityProvider:
    name = CFG["identity"]["provider"]
    if name == "rekognition":
        from .identity_rekognition import Rekognition  # added in the AWS phase
        return Rekognition()
    return LocalArcFace()
