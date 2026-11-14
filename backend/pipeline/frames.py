"""Per-frame analysis: detection, landmarks, head pose, quality."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import CFG
from .models import get_models

# Well-defined anatomical mesh points (corners, tips) - NOT silhouette/contour points,
# whose placement follows MediaPipe's 3-D prior rather than the image. Used by flatness.
FEATURE_IDX = [33, 133, 362, 263,            # eye corners
               70, 105, 107, 336, 334, 300,  # eyebrows
               168, 6, 1, 2, 98, 327,        # nose bridge, tip, base, alae
               61, 291, 0, 17,               # mouth corners, lips
               50, 280, 152]                 # cheek surface, chin


@dataclass
class Frame:
    idx: int
    img: np.ndarray
    n_faces: int = 0
    box: tuple | None = None          # x, y, w, h  (YuNet, full-res pixels)
    lmk5: np.ndarray | None = None    # 5 points for ArcFace alignment
    det_score: float = 0.0
    mesh: np.ndarray | None = None    # (478, 2) pixel coords
    blend: dict = field(default_factory=dict)
    yaw: float = 0.0                  # degrees, + = subject turned to their left
    pitch: float = 0.0
    face_px: float = 0.0
    blur: float = 0.0
    brightness: float = 0.0
    issues: list = field(default_factory=list)

    @property
    def good(self) -> bool:
        return self.box is not None and not self.issues

    @property
    def rank(self) -> float:
        q = CFG["quality"]
        return (min(self.blur / q["min_blur"], 3.0) + min(self.face_px / q["min_face_px"], 2.0)
                - abs(self.yaw) / 30.0)


def decode(jpeg_bytes: list[bytes]) -> list[np.ndarray]:
    out = []
    for b in jpeg_bytes:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


def _pose_from_matrix(m: np.ndarray) -> tuple[float, float]:
    """(yaw, pitch) in degrees from MediaPipe's facial transformation matrix.
    yaw > 0 = subject turned to their own left (raw, un-mirrored camera frame)."""
    r = np.asarray(m)[:3, :3]
    yaw = math.degrees(math.asin(float(np.clip(-r[2, 0], -1, 1))))
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    return yaw, pitch


def face_crop(img: np.ndarray, box, pad: float = 0.0) -> np.ndarray:
    x, y, w, h = box
    x0, y0 = int(max(0, x - pad * w)), int(max(0, y - pad * h))
    x1, y1 = int(min(img.shape[1], x + w * (1 + pad))), int(min(img.shape[0], y + h * (1 + pad)))
    return img[y0:y1, x0:x1]


def analyse_frame(idx: int, img: np.ndarray) -> Frame:
    m = get_models()
    q = CFG["quality"]
    f = Frame(idx=idx, img=img)

    faces = m["yunet"](img)
    f.n_faces = len(faces)
    if f.n_faces == 0:
        f.issues.append("no_face")
        return f
    best = faces[np.argmax(faces[:, 2] * faces[:, 3])]
    f.box = tuple(float(v) for v in best[:4])
    f.lmk5 = best[4:14].reshape(5, 2)
    f.det_score = float(best[14])
    f.face_px = f.box[2]

    # A second face that is reasonably large is a real problem (another person / held-up photo).
    if f.n_faces > 1:
        areas = sorted((faces[:, 2] * faces[:, 3]).tolist(), reverse=True)
        if areas[1] > 0.25 * areas[0]:
            f.issues.append("multiple_faces")

    # Run the landmarker on a square crop around the YuNet face so the face fills the
    # model input (its own detector misses small faces) - also faster than full frame.
    x, y, bw, bh = f.box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * 2.2
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(img.shape[1], cx + side / 2)), int(min(img.shape[0], cy + side / 2))
    crop = img[y0:y1, x0:x1]
    s = min(1.0, 384 / max(crop.shape[:2]))
    if s < 1:
        crop = cv2.resize(crop, (int(crop.shape[1] * s), int(crop.shape[0] * s)))
    res = m["landmarker"](np.ascontiguousarray(crop))
    if res.face_landmarks:
        ch, cw = crop.shape[:2]
        best_i, best_d = 0, 1e18
        meshes = []
        for i, lms in enumerate(res.face_landmarks):
            pts = np.array([[x0 + p.x * cw / s, y0 + p.y * ch / s] for p in lms], np.float32)
            meshes.append(pts)
            d = (pts[:, 0].mean() - cx) ** 2 + (pts[:, 1].mean() - cy) ** 2
            if d < best_d:
                best_i, best_d = i, d
        f.mesh = meshes[best_i]
        if res.face_blendshapes:
            f.blend = {c.category_name: float(c.score) for c in res.face_blendshapes[best_i]}
        if res.facial_transformation_matrixes:
            f.yaw, f.pitch = _pose_from_matrix(res.facial_transformation_matrixes[best_i])
    else:
        f.issues.append("no_landmarks")

    gray = cv2.cvtColor(face_crop(img, f.box), cv2.COLOR_BGR2GRAY)
    if gray.size:
        g = cv2.resize(gray, (160, 160))
        f.blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
        f.brightness = float(g.mean())

    if f.face_px < q["min_face_px"]:
        f.issues.append("face_too_small")
    if f.blur < q["min_blur"]:
        f.issues.append("blurry")
    if f.brightness < q["min_brightness"]:
        f.issues.append("too_dark")
    if f.brightness > q["max_brightness"]:
        f.issues.append("too_bright")
    if abs(f.yaw) > q["max_yaw_deg"] or abs(f.pitch) > q["max_pitch_deg"]:
        f.issues.append("not_facing_camera")
    return f


ISSUE_TEXT = {
    "no_face": "No face found - please look at the camera",
    "multiple_faces": "More than one face in view",
    "no_landmarks": "Face not clear enough",
    "face_too_small": "Please move closer to the camera",
    "blurry": "Image is blurry - please hold still",
    "too_dark": "Lighting is too dark",
    "too_bright": "Lighting is too bright / backlit",
    "not_facing_camera": "Please face the camera directly",
}
