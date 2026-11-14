"""Import a zip of recorded clips into data/clips/ so the Record tab counts them and the
evaluation replays them.

Expected zip layout:  clips/<clip_id>/00.jpg ... 15.jpg + metadata.json
  metadata.json: participant, scenario (live | phone_photo | phone_video | ...), split,
                 condition, consent, timestamps, capture_span_ms

Run:  .venv/Scripts/python scripts/import_clips.py path/to/clips.zip [--replace]
Frames are stored encrypted (vault), like clips recorded on the Record page.
"""
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import vault  # noqa: E402
from pipeline.config import DATA  # noqa: E402

SCENARIO_LABEL = {
    "live": "live", "live_glasses": "live_glasses", "glasses": "live_glasses",
    "live_sunglasses": "live_sunglasses", "sunglasses": "live_sunglasses",
    "live_mask": "live_mask", "print": "print", "printed_photo": "print",
    "phone_photo": "screen_photo", "screen_photo": "screen_photo", "laptop_photo": "screen_photo",
    "phone_video": "video_replay", "video_replay": "video_replay", "laptop_video": "video_replay",
    "mask": "mask", "3d_mask": "mask",
}
SAFE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 1
    replace = "--replace" in sys.argv
    clips_root = DATA / "clips"
    added = skipped = bad = 0
    with zipfile.ZipFile(args[0]) as z:
        groups: dict[str, dict] = {}
        for name in z.namelist():
            parts = name.split("/")
            if len(parts) != 3 or not parts[2] or not SAFE.fullmatch(parts[1]) or not SAFE.fullmatch(parts[2]):
                continue  # directories and anything unexpected (no path traversal possible)
            groups.setdefault(parts[1], {})[parts[2]] = name
        for clip_id, files in sorted(groups.items()):
            if "metadata.json" not in files:
                print(f"skip  {clip_id}: no metadata.json"); bad += 1; continue
            meta = json.loads(z.read(files["metadata.json"]))
            label = SCENARIO_LABEL.get(str(meta.get("scenario", "")).lower())
            subject = str(meta.get("participant", ""))
            if not label or not SAFE.fullmatch(subject):
                print(f"skip  {clip_id}: unknown scenario {meta.get('scenario')!r} / participant {subject!r}"); bad += 1; continue
            if meta.get("consent") is False:
                print(f"skip  {clip_id}: no consent"); bad += 1; continue
            jpgs = sorted(n for n in files if n.lower().endswith(".jpg"))
            frames = []
            for n in jpgs:
                b = z.read(files[n])
                if cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR) is None:
                    break
                frames.append(b)
            if len(frames) != len(jpgs) or not frames:
                print(f"skip  {clip_id}: unreadable frame"); bad += 1; continue
            dest = clips_root / label / subject / clip_id
            if (dest / "meta.json").exists() and not replace:
                skipped += 1; continue
            dest.mkdir(parents=True, exist_ok=True)
            for i, b in enumerate(frames):
                vault.write_file(dest / f"frame_{i:02d}.jpg", b)
            (dest / "meta.json").write_text(json.dumps({
                "label": label, "subject": subject,
                "capture_ms": meta.get("capture_span_ms") or 0,
                "note": meta.get("condition", ""), "frames": len(frames),
                "created": meta.get("captured_at"), "split": meta.get("split"),
                "session": meta.get("session"), "source": "import", "source_clip": clip_id,
                "scenario": meta.get("scenario"),
            }, indent=2))
            added += 1
    print(f"imported {added}, already there {skipped}, rejected {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
