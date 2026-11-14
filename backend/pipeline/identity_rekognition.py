"""Amazon Rekognition identity provider (face collection + SearchFacesByImage).

Credentials come from the normal AWS chain: environment variables (start.bat loads
aws-credentials.bat if present), ~/.aws, or the EC2 instance role.
"""
from __future__ import annotations

import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from .config import CFG
from .frames import Frame
from .identity import IdentityProvider, _result, face_jpeg, save_enrollment_crops


class Rekognition(IdentityProvider):
    remote = True

    def __init__(self) -> None:
        c = CFG["identity"]["rekognition"]
        self.collection = c["collection_id"]
        self.n_enroll = int(c.get("enroll_faces", 3))
        self.client = boto3.client(
            "rekognition", region_name=c["region"],
            config=Config(connect_timeout=2, read_timeout=4, retries={"max_attempts": 2, "mode": "standard"}))
        self._ready = False

    # -- collection -----------------------------------------------------------------
    def ensure_collection(self) -> None:
        if self._ready:
            return
        try:
            self.client.describe_collection(CollectionId=self.collection)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise
            self.client.create_collection(CollectionId=self.collection)
        self._ready = True

    def index_crops(self, user_id: str, crops: list[bytes]) -> list[str]:
        """(Re)index a user's face crops. Old faces for the user are removed first."""
        import store
        self.ensure_collection()
        old = store.get_rek_faces(user_id, self.collection)
        if old:
            self.client.delete_faces(CollectionId=self.collection, FaceIds=old)
        ids = []
        for b in crops[: self.n_enroll]:
            r = self.client.index_faces(CollectionId=self.collection, Image={"Bytes": b},
                                        ExternalImageId=user_id, MaxFaces=1, QualityFilter="AUTO")
            ids += [fr["Face"]["FaceId"] for fr in r.get("FaceRecords", [])]
        store.set_rek_faces(user_id, self.collection, ids)
        return ids

    # -- IdentityProvider -------------------------------------------------------------
    def enroll(self, user_id, name, phone, frames):
        import store
        crops = save_enrollment_crops(user_id, frames)
        store.upsert_user_meta(user_id, name, phone)
        ids = self.index_crops(user_id, crops)
        if not ids:
            raise ValueError("Rekognition rejected every face crop (quality too low) - try better light")
        return len(ids)

    def identify(self, frames: list[Frame], claimed_id=None):
        import store
        self.ensure_collection()
        t0 = time.perf_counter()
        scores = {uid: 0.0 for uid in store.user_ids()}
        try:
            r = self.client.search_faces_by_image(
                CollectionId=self.collection, Image={"Bytes": face_jpeg(frames[0])},
                MaxFaces=50, FaceMatchThreshold=0)
            for m in r.get("FaceMatches", []):
                uid = m["Face"].get("ExternalImageId")
                if uid in scores:
                    scores[uid] = max(scores[uid], m["Similarity"] / 100.0)
        except ClientError as e:
            # InvalidParameterException = Rekognition found no face in the crop
            if e.response["Error"]["Code"] != "InvalidParameterException":
                raise
        out = _result(scores, claimed_id)
        out["api_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return out

    def delete(self, user_id):
        import store
        ids = store.get_rek_faces(user_id, self.collection)
        if ids:
            try:
                self.client.delete_faces(CollectionId=self.collection, FaceIds=ids)
            except ClientError:
                pass
        super().delete(user_id)
