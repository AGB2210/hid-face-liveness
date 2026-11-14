"""Push every locally enrolled user into the Amazon Rekognition collection, using the
face crops saved at enrollment (data/enroll/<user>/). No need to re-enroll people.

Run:  .venv/Scripts/python scripts/sync_rekognition.py          (needs AWS credentials)
      .venv/Scripts/python scripts/sync_rekognition.py --check  (only test access)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import store  # noqa: E402
from pipeline.identity import ENROLL_DIR  # noqa: E402
from pipeline.identity_rekognition import Rekognition  # noqa: E402


def main() -> int:
    rek = Rekognition()
    try:
        rek.ensure_collection()
    except Exception as e:
        print(f"Cannot reach Rekognition: {type(e).__name__}: {e}")
        print("Check aws-credentials.bat (see README) - Workshop credentials expire after a few hours.")
        return 1
    desc = rek.client.describe_collection(CollectionId=rek.collection)
    print(f"Collection '{rek.collection}' OK - {desc.get('FaceCount', 0)} faces indexed")
    if "--check" in sys.argv:
        return 0
    users = store.list_users()
    if not users:
        print("Nobody enrolled yet.")
    for u in users:
        crops = [p.read_bytes() for p in sorted((ENROLL_DIR / u["id"]).glob("*.jpg"))]
        if not crops:
            print(f"skip  {u['id']}: no saved face crops - re-enroll this user")
            continue
        ids = rek.index_crops(u["id"], crops)
        print(f"sync  {u['id']:<16} {len(ids)} face(s) indexed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
