"""SQLite storage: enrolled users, face templates, attempt log and the tamper-evident audit log.

The audit log never contains images or face templates - only decision metadata."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
import threading
import time

import numpy as np

from pipeline.config import DATA

DB_PATH = DATA / "hid.db"
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


_db = _conn()
_db.executescript("""
CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY, name TEXT NOT NULL, phone TEXT, created REAL);
CREATE TABLE IF NOT EXISTS templates (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, emb BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS rek_faces (
  face_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, collection TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, decision TEXT, stage TEXT,
  user_id TEXT, similarity REAL, liveness REAL, processing_ms REAL, total_ms REAL,
  reasons TEXT);
CREATE TABLE IF NOT EXISTS recovery_codes (
  user_id TEXT NOT NULL, hash TEXT NOT NULL, used REAL, PRIMARY KEY (user_id, hash));
CREATE TABLE IF NOT EXISTS audit (
  id TEXT PRIMARY KEY, seq INTEGER, ts REAL, kiosk TEXT, zone TEXT, user_id TEXT, decision TEXT,
  stage TEXT, mode TEXT, liveness REAL, similarity REAL, schedule TEXT, reasons TEXT,
  prev_hash TEXT, hash TEXT, synced INTEGER DEFAULT 0, received REAL);
""")
for _col in ("role", "valid_until", "totp_secret", "admin_role", "email"):
    if _col not in [r["name"] for r in _db.execute("PRAGMA table_info(users)")]:
        _db.execute(f"ALTER TABLE users ADD COLUMN {_col} TEXT")
_db.commit()

AUDIT_FIELDS = ("id", "seq", "ts", "kiosk", "zone", "user_id", "decision", "stage", "mode",
                "liveness", "similarity", "schedule", "reasons", "prev_hash")


def upsert_user(user_id: str, name: str, phone: str | None, embs: np.ndarray) -> None:
    with _lock:
        _db.execute("""INSERT INTO users (id, name, phone, created) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE
                       SET name=excluded.name, phone=excluded.phone, created=excluded.created""",
                    (user_id, name, phone, time.time()))
        _db.execute("DELETE FROM templates WHERE user_id=?", (user_id,))
        _db.executemany("INSERT INTO templates (user_id, emb) VALUES (?,?)",
                        [(user_id, e.astype(np.float32).tobytes()) for e in embs])
        _db.commit()


def upsert_user_meta(user_id: str, name: str, phone: str | None) -> None:
    """Create/update a user without touching local templates (remote providers)."""
    with _lock:
        _db.execute("""INSERT INTO users (id, name, phone, created) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE
                       SET name=excluded.name, phone=excluded.phone""", (user_id, name, phone, time.time()))
        _db.commit()


def set_rek_faces(user_id: str, collection: str, face_ids: list[str]) -> None:
    with _lock:
        _db.execute("DELETE FROM rek_faces WHERE user_id=? AND collection=?", (user_id, collection))
        _db.executemany("INSERT OR REPLACE INTO rek_faces VALUES (?,?,?)",
                        [(fid, user_id, collection) for fid in face_ids])
        _db.commit()


def get_rek_faces(user_id: str, collection: str) -> list[str]:
    return [r["face_id"] for r in _db.execute(
        "SELECT face_id FROM rek_faces WHERE user_id=? AND collection=?", (user_id, collection))]


def user_ids() -> list[str]:
    return [r["id"] for r in _db.execute("SELECT id FROM users")]


def delete_user(user_id: str) -> None:
    with _lock:
        _db.execute("DELETE FROM rek_faces WHERE user_id=?", (user_id,))
        _db.execute("DELETE FROM templates WHERE user_id=?", (user_id,))
        _db.execute("DELETE FROM users WHERE id=?", (user_id,))
        _db.commit()


def list_users() -> list[dict]:
    rows = _db.execute("""SELECT u.id, u.name, u.phone, u.email, u.created, u.role, u.valid_until, u.admin_role,
                            (u.totp_secret IS NOT NULL) has_totp,
                            (SELECT COUNT(*) FROM templates t WHERE t.user_id=u.id) n,
                            (SELECT COUNT(*) FROM rek_faces r WHERE r.user_id=u.id) n_rekognition
                          FROM users u ORDER BY u.created""").fetchall()
    return [dict(r) for r in rows]


def get_user(user_id: str) -> dict | None:
    r = _db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(r) if r else None


def templates() -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    for r in _db.execute("SELECT user_id, emb FROM templates"):
        out.setdefault(r["user_id"], []).append(np.frombuffer(r["emb"], np.float32))
    return {k: np.stack(v) for k, v in out.items()}


def log_attempt(decision: str, stage: str, user_id, similarity, liveness,
                processing_ms, total_ms, reasons) -> None:
    with _lock:
        _db.execute("""INSERT INTO attempts (ts, decision, stage, user_id, similarity, liveness,
                       processing_ms, total_ms, reasons) VALUES (?,?,?,?,?,?,?,?,?)""",
                    (time.time(), decision, stage, user_id, similarity, liveness,
                     processing_ms, total_ms, json.dumps(reasons)))
        _db.commit()


def update_total_ms(total_ms: float) -> None:
    with _lock:
        _db.execute("UPDATE attempts SET total_ms=? WHERE id=(SELECT MAX(id) FROM attempts)", (total_ms,))
        _db.commit()


def recent_attempts(limit: int = 50) -> list[dict]:
    rows = _db.execute("SELECT * FROM attempts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"reasons": json.loads(r["reasons"] or "[]")} for r in rows]


# --------------------------------------------------------------------- access + 2nd factor
def set_access(user_id: str, role: str | None, valid_until: str | None) -> None:
    with _lock:
        _db.execute("UPDATE users SET role=?, valid_until=? WHERE id=?", (role or None, valid_until or None, user_id))
        _db.commit()


def normalise_email(email: str | None) -> str | None:
    e = (email or "").strip().lower()
    return e or None


def get_user_by_email(email: str) -> dict | None:
    r = _db.execute("SELECT * FROM users WHERE email=?", (normalise_email(email),)).fetchone()
    return dict(r) if r else None


def set_email(user_id: str, email: str | None) -> None:
    with _lock:
        _db.execute("UPDATE users SET email=? WHERE id=?", (normalise_email(email), user_id))
        _db.commit()


def set_admin_role(user_id: str, role: str | None) -> None:
    with _lock:
        _db.execute("UPDATE users SET admin_role=? WHERE id=?", (role or None, user_id))
        _db.commit()


def set_totp(user_id: str, secret: str | None) -> None:
    with _lock:
        _db.execute("UPDATE users SET totp_secret=? WHERE id=?", (secret, user_id))
        _db.commit()


def template_age_days(user_id: str) -> float | None:
    r = _db.execute("SELECT created FROM users WHERE id=?", (user_id,)).fetchone()
    return (time.time() - r["created"]) / 86400 if r and r["created"] else None


# ------------------------------------------------------------------------------ audit log
def _hash(rec: dict) -> str:
    body = json.dumps({k: rec.get(k) for k in AUDIT_FIELDS}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def append_audit(kiosk: str, zone, user_id, decision: str, stage: str, mode: str,
                 liveness, similarity, schedule, reasons: list) -> dict:
    """Append a decision to this node's hash chain (each record commits to the previous one)."""
    with _lock:
        last = _db.execute("SELECT seq, hash FROM audit WHERE kiosk=? AND received IS NULL ORDER BY seq DESC LIMIT 1",
                           (kiosk,)).fetchone()
        rec = {"id": uuid.uuid4().hex, "seq": (last["seq"] + 1) if last else 1, "ts": round(time.time(), 3),
               "kiosk": kiosk, "zone": zone, "user_id": user_id, "decision": decision, "stage": stage,
               "mode": mode, "liveness": liveness, "similarity": similarity, "schedule": schedule,
               "reasons": json.dumps(reasons), "prev_hash": last["hash"] if last else "0" * 64}
        rec["hash"] = _hash(rec)
        cols = ", ".join(AUDIT_FIELDS) + ", hash"
        _db.execute(f"INSERT INTO audit ({cols}) VALUES ({', '.join('?' * (len(AUDIT_FIELDS) + 1))})",
                    [rec[k] for k in AUDIT_FIELDS] + [rec["hash"]])
        _db.commit()
    return rec


def unsynced_audit(limit: int = 200) -> list[dict]:
    rows = _db.execute("SELECT * FROM audit WHERE synced=0 AND received IS NULL ORDER BY ts LIMIT ?", (limit,))
    return [dict(r) for r in rows]


def mark_synced(ids: list[str]) -> None:
    with _lock:
        _db.executemany("UPDATE audit SET synced=1 WHERE id=?", [(i,) for i in ids])
        _db.commit()


def ingest_audit(records: list[dict]) -> int:
    """Central server: store records pushed by edge kiosks. Idempotent (same id is ignored);
    records whose hash doesn't match their content are rejected."""
    n = 0
    cols = ", ".join(AUDIT_FIELDS) + ", hash, synced, received"
    with _lock:
        for r in records:
            if r.get("hash") != _hash(r):
                continue
            cur = _db.execute(f"INSERT OR IGNORE INTO audit ({cols}) VALUES ({', '.join('?' * (len(AUDIT_FIELDS) + 3))})",
                              [r.get(k) for k in AUDIT_FIELDS] + [r["hash"], 1, time.time()])
            n += cur.rowcount
        _db.commit()
    return n


def list_audit(limit: int = 200) -> list[dict]:
    rows = _db.execute("SELECT * FROM audit ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) | {"reasons": json.loads(r["reasons"] or "[]")} for r in rows]


def verify_chains() -> dict:
    """Recompute every kiosk's hash chain. An edited record fails its hash; a removed one leaves a gap."""
    out = {}
    for k in [r["kiosk"] for r in _db.execute("SELECT DISTINCT kiosk FROM audit")]:
        rows = [dict(r) for r in _db.execute("SELECT * FROM audit WHERE kiosk=? ORDER BY seq", (k,))]
        prev, intact, gaps = "0" * 64, True, 0
        for r in rows:
            if r["hash"] != _hash(r):
                intact = False
            if r["prev_hash"] != prev:
                gaps += 1
            prev = r["hash"]
        out[k] = {"records": len(rows), "intact": intact, "gaps": gaps}
    return out


def audit_counts() -> dict:
    r = _db.execute("""SELECT SUM(synced=0 AND received IS NULL) pending,
                              SUM(mode='offline' AND received IS NULL) offline_total FROM audit""").fetchone()
    return {"pending": r["pending"] or 0, "offline_total": r["offline_total"] or 0}


# ------------------------------------------------------------------ staff recovery codes
def set_recovery_codes(user_id: str, hashes: list[str]) -> None:
    """Replace a person's recovery codes (only SHA-256 hashes are stored)."""
    with _lock:
        _db.execute("DELETE FROM recovery_codes WHERE user_id=?", (user_id,))
        _db.executemany("INSERT INTO recovery_codes (user_id, hash) VALUES (?,?)", [(user_id, h) for h in hashes])
        _db.commit()


def use_recovery_code(user_id: str, code_hash: str) -> bool:
    with _lock:
        cur = _db.execute("UPDATE recovery_codes SET used=? WHERE user_id=? AND hash=? AND used IS NULL",
                          (time.time(), user_id, code_hash))
        _db.commit()
        return cur.rowcount == 1


def recovery_codes_left(user_id: str) -> int:
    return _db.execute("SELECT COUNT(*) FROM recovery_codes WHERE user_id=? AND used IS NULL", (user_id,)).fetchone()[0]
