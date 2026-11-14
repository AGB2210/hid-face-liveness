"""Liveness checks. Each returns a score in [0, 1] (1 = live) or None when inconclusive,
plus a details dict for the debug panel."""
from __future__ import annotations

import cv2
import numpy as np

from .config import CFG
from .frames import Frame, face_crop
from .models import get_models


def _ramp(x: float, lo: float, hi: float) -> float:
    """0 at/below lo, 1 at/above hi, linear between."""
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


# --- 1. Pretrained passive anti-spoof (MiniFASNet x2) -------------------------------
def minifas_score(frames: list[Frame]) -> tuple[float | None, dict]:
    if not frames:
        return None, {}
    mf = get_models()["minifas"]
    probs = [mf.real_prob(f.img, f.box) for f in frames]
    return float(np.mean(probs)), {"per_frame": [round(p, 3) for p in probs]}


# --- 2. Flatness: does a single homography explain the face's texture motion? ---------
# Track real image corners inside the face with optical flow (model-free). A photo or
# screen is a plane, so one homography explains every track up to tracking noise. A real
# head that turns or changes expression leaves parallax that no homography can absorb.
_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _point_residuals(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """Median per-point residual after a similarity fit and after a homography fit."""
    from .geometry import fit_similarity
    m = fit_similarity(src, dst)
    sim_err = np.linalg.norm(src @ m[:, :2].T + m[:, 2] - dst, axis=1)
    h, _ = cv2.findHomography(src, dst, 0)
    if h is None:
        return float(np.median(sim_err)), float("nan")
    pred = cv2.perspectiveTransform(src.reshape(-1, 1, 2), h).reshape(-1, 2)
    return float(np.median(sim_err)), float(np.median(np.linalg.norm(pred - dst, axis=1)))


def flatness_score(frames: list[Frame], ref: Frame) -> tuple[float | None, dict]:
    c = CFG["flatness"]
    x, y, w, h = ref.box
    # work on one fixed region around the face (same crop in every frame) for speed
    H, W = ref.img.shape[:2]
    rx0, ry0 = int(max(0, x - 0.4 * w)), int(max(0, y - 0.4 * h))
    rx1, ry1 = int(min(W, x + 1.4 * w)), int(min(H, y + 1.4 * h))

    def gray(img):
        return cv2.cvtColor(img[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2GRAY)

    g_ref = gray(ref.img)
    mask = np.zeros_like(g_ref)
    mask[int(y - ry0 + 0.1 * h):int(y - ry0 + 0.95 * h), int(x - rx0 + 0.12 * w):int(x - rx0 + 0.88 * w)] = 255
    p0 = cv2.goodFeaturesToTrack(g_ref, maxCorners=200, qualityLevel=0.005, minDistance=4, mask=mask)
    if p0 is None or len(p0) < 20:
        return None, {"reason": "too little face texture"}
    sr2 = hr2 = 0.0
    used, max_motion, n_pts = 0, 0.0, []
    for f in frames:
        if f is ref:
            continue
        g = gray(f.img)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g_ref, g, p0, None, **_LK)
        pb, st2, _ = cv2.calcOpticalFlowPyrLK(g, g_ref, p1, None, **_LK)
        ok = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(pb - p0, axis=2)[:, 0] < 0.7)
        if ok.sum() < 20:
            continue
        a, b = p0[ok, 0].astype(np.float32), p1[ok, 0].astype(np.float32)
        sr, hr = _point_residuals(a, b)
        sr, hr = sr / w, hr / w
        max_motion = max(max_motion, sr)
        if sr < c["min_motion"] or not np.isfinite(hr):
            continue
        sr2 += sr * sr
        hr2 += hr * hr
        used += 1
        n_pts.append(int(ok.sum()))
    if used == 0:
        return None, {"reason": "not enough head motion", "max_motion": round(max_motion, 4)}
    ratio = float(np.sqrt(hr2 / sr2))
    return _ramp(ratio, c["flat_ratio"], c["deep_ratio"]), {
        "ratio": round(ratio, 3), "pairs": used, "max_motion": round(max_motion, 4),
        "tracked_pts": int(np.median(n_pts))}


# --- 3. Moire / screen pixel-grid pattern -------------------------------------------
def _moire_peak(img: np.ndarray, box) -> float | None:
    x, y, w, h = box
    inner = (x + 0.15 * w, y + 0.2 * h, 0.7 * w, 0.7 * h)
    g = cv2.cvtColor(face_crop(img, inner), cv2.COLOR_BGR2GRAY).astype(np.float32)
    n = min(g.shape[:2])
    if n < 64:
        return None
    oy, ox = (g.shape[0] - n) // 2, (g.shape[1] - n) // 2
    g = g[oy:oy + n, ox:ox + n]
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    mag = np.abs(np.fft.fftshift(np.fft.fft2(g)))

    fy, fx = np.indices((n, n)) - n // 2
    r = np.sqrt(fx ** 2 + fy ** 2) / (n / 2)            # 0..~1.41, 1 = Nyquist
    band = (r > 0.15) & (r < 0.95) & (np.abs(fx) > 2) & (np.abs(fy) > 2)
    for k in (1, 2, 3):                                   # JPEG 8x8 block grid lines
        b = round(k * n / 8)
        for v in (fx, fy):
            band &= np.abs(np.abs(v) - b) > 1
    # whiten by the radial median so the natural 1/f fall-off doesn't dominate
    rb = np.minimum((r * 40).astype(int), 59)
    med = np.zeros(60)
    for i in np.unique(rb[band]):
        med[i] = np.median(mag[band & (rb == i)])
    white = mag[band] / (med[rb[band]] + 1e-9)
    return float(np.percentile(white, 99.95))


def moire_score(frames: list[Frame]) -> tuple[float | None, dict]:
    c = CFG["moire"]
    peaks = [p for p in (_moire_peak(f.img, f.box) for f in frames) if p is not None]
    if not peaks:
        return None, {"reason": "face too small"}
    peak = float(np.median(peaks))
    return 1.0 - _ramp(peak, c["peak_flat"], c["peak_screen"]), {"peak": round(peak, 2)}


# --- 4. Specular glare (glossy print / screen reflections) ---------------------------
def glare_score(frames: list[Frame]) -> tuple[float | None, dict]:
    c = CFG["glare"]
    fracs = []
    from .occlusion import lens_mask
    for f in frames:
        crop = face_crop(f.img, f.box)
        if crop.size:
            sat = np.all(crop >= 250, axis=2)
            m = lens_mask(f, f.box, sat.shape)  # ignore glasses lenses: reflections there are normal
            if m is not None and m.any():
                sat = sat[m]
            fracs.append(float(np.mean(sat)))
    if not fracs:
        return None, {}
    frac = float(np.median(fracs))
    return 1.0 - _ramp(frac, c["clean_frac"], c["glare_frac"]), {"saturated_frac": round(frac, 4)}


# --- 5. Blink / eye motion (bonus only in passive mode) ------------------------------
def blink_series(frames: list[Frame]) -> list[float]:
    return [(f.blend.get("eyeBlinkLeft", 0) + f.blend.get("eyeBlinkRight", 0)) / 2
            for f in frames if f.blend]


def blinked(frames: list[Frame], thresh: float) -> bool:
    s = blink_series(frames)
    return bool(s) and max(s) >= thresh and min(s) <= thresh * 0.5


def run_all(face_frames: list[Frame], top: list[Frame]) -> dict:
    """face_frames: every frame with a face (for motion); top: best-quality frames."""
    ref = top[0]
    out = {}
    for name, fn in (("minifas", lambda: minifas_score(top)),
                     ("flatness", lambda: flatness_score(face_frames, ref)),
                     ("moire", lambda: moire_score(top)),
                     ("glare", lambda: glare_score(top))):
        score, details = fn()
        out[name] = {"score": None if score is None else round(score, 3), **details}
    series = blink_series(face_frames)
    out["blink"] = {"blinked": blinked(face_frames, CFG["challenge"]["blink_thresh"]),
                    "max": round(max(series), 3) if series else None}
    return out
