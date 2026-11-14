"""Accessories and occlusions, from the face mesh we already compute (no extra model).

  * lens regions   - where glasses sit; excluded from the glare (spoof) check so that
                     prescription lenses reflecting the kiosk screen don't count as a spoof
  * sunglasses     - eye region much darker than the person's own cheeks
  * face mask      - lips not redder than the person's own cheeks AND lower face a different
                     colour (comparing against their own skin keeps this skin-tone independent;
                     checking the lips keeps beards from being mistaken for masks)

Detections only produce guidance ("please remove your sunglasses"), never a deny.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import CFG
from .frames import Frame

R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
CHEEKS = [50, 280, 117, 346]       # cheek surface, both sides
FOREHEAD = [151, 9]
LOWER = [164, 200, 18]             # philtrum, chin
LIPS = [13, 14, 0, 17]             # inner/outer lip midpoints


def _eye_boxes(mesh: np.ndarray, lens: bool) -> list[tuple[int, int, int, int]]:
    boxes = []
    face_w = float(np.ptp(mesh[:, 0])) + 1e-6
    for idx in (R_EYE, L_EYE):
        pts = mesh[idx]
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        ew = np.ptp(pts[:, 0]) + 1e-6
        # lenses don't foreshorten like the far eye does on a turned head: size them from the face
        hw, hh = (0.18 * face_w, 0.14 * face_w) if lens else (0.62 * ew, 0.32 * ew)
        boxes.append((int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh)))
    return boxes


def lens_mask(f: Frame, box, shape: tuple[int, int]) -> np.ndarray | None:
    """Boolean mask for the face crop (same origin and shape as frames.face_crop) that is
    False where glasses lenses would be."""
    if f.mesh is None:
        return None
    ox, oy = int(max(0, box[0])), int(max(0, box[1]))
    m = np.ones(shape, bool)
    for x0, y0, x1, y1 in _eye_boxes(f.mesh, lens=True):
        m[max(0, y0 - oy):max(0, y1 - oy), max(0, x0 - ox):max(0, x1 - ox)] = False
    return m


def _patch(img: np.ndarray, pt, half: int) -> np.ndarray:
    x, y = int(pt[0]), int(pt[1])
    return img[max(0, y - half):y + half, max(0, x - half):x + half]


def _region(img, mesh, idx, half):
    ps = [_patch(img, mesh[i], half) for i in idx]
    ps = [p.reshape(-1, img.shape[2]) if img.ndim == 3 else p.reshape(-1) for p in ps if p.size]
    return np.concatenate(ps) if ps else None


def analyse_frame(f: Frame) -> dict | None:
    if f.mesh is None or f.box is None:
        return None
    face_w = float(np.ptp(f.mesh[:, 0]))
    half = max(3, int(0.045 * face_w))
    gray = cv2.cvtColor(f.img, cv2.COLOR_BGR2GRAY)
    ycc = cv2.cvtColor(f.img, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    eye_px = np.concatenate([gray[max(0, y0):y1, max(0, x0):x1].reshape(-1)
                             for x0, y0, x1, y1 in _eye_boxes(f.mesh, lens=False)])
    cheek_g = _region(gray, f.mesh, CHEEKS, half)
    cheek_c = _region(ycc, f.mesh, CHEEKS + FOREHEAD, half)
    lower_c = _region(ycc, f.mesh, LOWER, half)
    lips_c = _region(ycc, f.mesh, LIPS, max(2, half // 2))
    if any(v is None or len(v) == 0 for v in (eye_px, cheek_g, cheek_c, lower_c, lips_c)):
        return None
    eye_ratio = float(eye_px.mean() / (cheek_g.mean() + 1e-6))
    skin = cheek_c.mean(axis=0)
    lower_dist = float(np.linalg.norm(lower_c.mean(axis=0)[1:] - skin[1:]))   # Cr/Cb distance
    lip_redness = float(lips_c[:, 1].mean() - skin[1])                        # Cr: lips vs own skin
    return {"eye_ratio": round(eye_ratio, 3), "lower_dist": round(lower_dist, 1), "lip_redness": round(lip_redness, 1)}


def analyse(frames: list[Frame]) -> dict:
    c = CFG["occlusion"]
    feats = [x for x in (analyse_frame(f) for f in frames) if x]
    if not feats:
        return {"sunglasses": False, "mask": False}
    med = {k: float(np.median([x[k] for x in feats])) for k in feats[0]}
    sunglasses = med["eye_ratio"] < c["eye_dark_ratio"]
    mask = med["lip_redness"] < c["lip_redness_min"] and med["lower_dist"] > c["lower_dist_min"]
    return {"sunglasses": bool(sunglasses), "mask": bool(mask), **{k: round(v, 3) for k, v in med.items()}}
