"""Download the pretrained models into models/.

Run:  .venv/Scripts/python scripts/get_models.py
Re-running skips files that already exist.
"""
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
THIRD_PARTY = ROOT / "third_party"

SF = "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/raw/master"

FILES = {
    # Face detector - OpenCV Model Zoo (MIT)
    MODELS / "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    # 478-pt landmarks + blendshapes + head pose - Google MediaPipe (Apache-2.0)
    MODELS / "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    # Passive anti-spoof - Minivision Silent-Face-Anti-Spoofing (Apache-2.0)
    THIRD_PARTY / "minifas" / "2.7_80x80_MiniFASNetV2.pth":
        f"{SF}/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
    THIRD_PARTY / "minifas" / "4_0_0_80x80_MiniFASNetV1SE.pth":
        f"{SF}/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth",
    THIRD_PARTY / "minifas" / "MiniFASNet.py":
        f"{SF}/src/model_lib/MiniFASNet.py",
}

# ArcFace recogniser - InsightFace buffalo_l pack (weights: non-commercial research use)
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
ARCFACE = MODELS / "w600k_r50.onnx"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "model-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def main() -> int:
    for path, url in FILES.items():
        if path.exists() and path.stat().st_size > 0:
            print(f"ok    {path.relative_to(ROOT)}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"get   {path.relative_to(ROOT)}")
        path.write_bytes(fetch(url))

    if not ARCFACE.exists():
        print("get   models/w600k_r50.onnx (from buffalo_l.zip, ~280 MB)")
        data = fetch(BUFFALO_URL)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.endswith("w600k_r50.onnx"))
            ARCFACE.write_bytes(z.read(name))
    else:
        print("ok    models/w600k_r50.onnx")

    print("done. Next: python scripts/export_minifas_onnx.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
