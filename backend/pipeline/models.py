"""Lazy-loaded pretrained models (all run locally)."""
from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from .config import CFG, ROOT

MODELS = ROOT / "models"
_LOCK = threading.Lock()


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    providers = [p for p in CFG["runtime"]["providers"] if p in ort.get_available_providers()]
    return ort.InferenceSession(str(path), opts, providers=providers or ["CPUExecutionProvider"])


class YuNet:
    """OpenCV Model Zoo face detector. Runs on a downscaled copy for speed."""

    MAX_W = 640

    def __init__(self) -> None:
        self.det = cv2.FaceDetectorYN.create(
            str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (320, 320), 0.6, 0.3, 50)

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Returns (N, 15): x, y, w, h, 5 landmark pairs, score - in full-res pixels."""
        h, w = bgr.shape[:2]
        s = min(1.0, self.MAX_W / w)
        small = cv2.resize(bgr, (int(w * s), int(h * s))) if s < 1 else bgr
        self.det.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.det.detect(small)
        if faces is None:
            return np.zeros((0, 15), np.float32)
        faces = faces.copy()
        faces[:, :14] /= s
        return faces


class Landmarker:
    """MediaPipe Face Landmarker: 478 points, blendshapes, head pose matrix."""

    def __init__(self) -> None:
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.lm = vision.FaceLandmarker.create_from_options(opts)

    def __call__(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))


class ArcFace:
    """InsightFace w600k_r50 recogniser -> L2-normalised 512-d embedding."""

    DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                    [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

    def __init__(self) -> None:
        self.sess = _session(MODELS / "w600k_r50.onnx")
        self.inp = self.sess.get_inputs()[0].name

    def align(self, bgr: np.ndarray, lmk5: np.ndarray) -> np.ndarray:
        from .geometry import fit_similarity
        m = fit_similarity(lmk5.astype(np.float32), self.DST)
        return cv2.warpAffine(bgr, m, (112, 112), borderValue=0)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = []
        for c in crops:  # model output is fixed to batch 1
            blob = cv2.dnn.blobFromImage(c, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
            e = self.sess.run(None, {self.inp: blob})[0][0]
            out.append(e / (np.linalg.norm(e) + 1e-9))
        return np.stack(out)


class MiniFAS:
    """Minivision Silent-Face-Anti-Spoofing, two pretrained models averaged.
    Class 1 of the softmax is 'real'."""

    SPECS = [("minifas_v2_2.7.onnx", 2.7), ("minifas_v1se_4.0.onnx", 4.0)]

    def __init__(self) -> None:
        self.models = [(_session(MODELS / f), scale) for f, scale in self.SPECS]

    @staticmethod
    def _crop(bgr: np.ndarray, box, scale: float) -> np.ndarray:
        src_h, src_w = bgr.shape[:2]
        x, y, bw, bh = box
        scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
        nw, nh = bw * scale, bh * scale
        cx, cy = x + bw / 2, y + bh / 2
        l, t, r, b = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
        if l < 0: r -= l; l = 0
        if t < 0: b -= t; t = 0
        if r > src_w - 1: l -= r - src_w + 1; r = src_w - 1
        if b > src_h - 1: t -= b - src_h + 1; b = src_h - 1
        crop = bgr[int(t):int(b) + 1, int(l):int(r) + 1]
        return cv2.resize(crop, (80, 80))

    def real_prob(self, bgr: np.ndarray, box) -> float:
        probs = []
        for sess, scale in self.models:
            x = self._crop(bgr, box, scale).transpose(2, 0, 1)[None].astype(np.float32)  # BGR, 0..255
            logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
            p = np.exp(logits - logits.max()); p /= p.sum()
            probs.append(float(p[1]))
        return float(np.mean(probs))


@lru_cache(maxsize=1)
def get_models():
    return {"yunet": YuNet(), "landmarker": Landmarker(), "arcface": ArcFace(), "minifas": MiniFAS()}


def warmup() -> None:
    """Run every model once so the first real user doesn't pay the ~0.5 s init cost."""
    m = get_models()
    img = np.full((720, 960, 3), 128, np.uint8)
    cv2.circle(img, (480, 330), 140, (150, 170, 200), -1)
    with _LOCK:
        m["yunet"](img)
        m["landmarker"](np.ascontiguousarray(img[150:520, 300:660]))
        m["arcface"].embed([np.zeros((112, 112, 3), np.uint8)])
        m["minifas"].real_prob(img, (380, 230, 200, 200))


def models_lock() -> threading.Lock:
    return _LOCK
