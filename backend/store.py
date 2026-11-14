"""SQLite storage: enrolled users, face templates, attempt log and the tamper-evident audit log.

Face templates, authenticator secrets and phone numbers are encrypted at rest (see vault.py).
The audit log never contains images or face templates - only decision metadata."""
from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
import uuid
import threading
import time

import numpy as np

import vault
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
for _col in ("role", "valid_until", "totp_secret", "admin_role", "email", "source"):  # source='central': mirrored
    if _col not in [r["name"] for r in _db.execute("PRAGMA table_info(users)")]:
        _db.execute(f"ALTER TABLE users ADD COLUMN {_col} TEXT")
# one admin role now: former policy admins become admins; the old operator/auditor roles are gone
_db.execute("UPDATE users SET role='admin' WHERE admin_role='policy_admin'")
_db.execute("UPDATE users SET admin_role=NULL WHERE admin_role IS NOT NULL")
_db.commit()

ENCRYPTED_USER_FIELDS = ("phone", "totp_secret")


def _encrypt_existing() -> None:
    """Startup: check the key opens the stored data, then encrypt anything still in plaintext
    (data written before encryption was added). Safe to run every time."""
    sample = _db.execute("SELECT emb FROM templates LIMIT 1").fetchone()
    vault.check(sample["emb"] if sample else None)
    sample = _db.execute("SELECT totp_secret FROM users WHERE totp_secret IS NOT NULL LIMIT 1").fetchone()
    vault.check(sample["totp_secret"] if sample else None)
    for r in _db.execute("SELECT id, emb FROM templates").fetchall():
        if not vault.is_token(r["emb"]):
            _db.execute("UPDATE templates SET emb=? WHERE id=?", (vault.enc(r["emb"]), r["id"]))
    for r in _db.execute("SELECT id, phone, totp_secret FROM users").fetchall():
        for col in ENCRYPTED_USER_FIELDS:
            if r[col] is not None and not vault.is_token(r[col]):
                _db.execute(f"UPDATE users SET {col}=? WHERE id=?", (vault.enc_str(r[col]), r["id"]))
    _db.commit()
    # face crops and recorded test clips on disk
    vault.encrypt_tree(DATA / "enroll")
    vault.encrypt_tree(DATA / "clips")


def _user(r) -> dict | None:
    if r is None:
        return None
    u = dict(r)
    for col in ENCRYPTED_USER_FIELDS:
        if col in u:
            u[col] = vault.dec_str(u[col])
    return u


_encrypt_existing()

AUDIT_FIELDS = ("id", "seq", "ts", "kiosk", "zone", "user_id", "decision", "stage", "mode",
                "liveness", "similarity", "schedule", "reasons", "prev_hash")


def upsert_user(user_id: str, name: str, phone: str | None, embs: np.ndarray) -> None:
    with _lock:
        _db.execute("""INSERT INTO users (id, name, phone, created) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE
                       SET name=excluded.name, phone=excluded.phone, created=excluded.created""",
                    (user_id, name, vault.enc_str(phone), time.time()))
        _db.execute("DELETE FROM templates WHERE user_id=?", (user_id,))
        _db.executemany("INSERT INTO templates (user_id, emb) VALUES (?,?)",
                        [(user_id, vault.enc(e.astype(np.float32).tobytes())) for e in embs])
        _db.commit()


def upsert_user_meta(user_id: str, name: str, phone: str | None) -> None:
    """Create/update a user without touching local templates (remote providers)."""
    with _lock:
        _db.execute("""INSERT INTO users (id, name, phone, created) VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE
                       SET name=excluded.name, phone=excluded.phone""", (user_id, name, vault.enc_str(phone), time.time()))
        _db.commit()


# ------------------------------------------------------------------ people sync (central -> edge)
PEOPLE_FIELDS = ("id", "name", "phone", "email", "role", "valid_until", "created", "totp_secret")


def export_people() -> list[dict]:
    """Central: everyone an edge kiosk needs to decide offline - details, authenticator secret
    and on-device face templates. No photos. Encrypted for transport by the caller."""
    tpl = {}
    for r in _db.execute("SELECT user_id, emb FROM templates ORDER BY id"):
        tpl.setdefault(r["user_id"], []).append(base64.b64encode(vault.dec(r["emb"])).decode())
    rows = _db.execute(f"SELECT {', '.join(PEOPLE_FIELDS)} FROM users").fetchall()
    return [{**_user(r), "templates": tpl.get(r["id"], [])} for r in rows]


def mirror_people(people: list[dict]) -> dict:
    """Edge: make the local copy of central's people match exactly. People enrolled on this
    kiosk itself are left alone; central wins if the same ID exists in both."""
    incoming = {p["id"]: p for p in people}
    with _lock:
        mirrored = {r["id"] for r in _db.execute("SELECT id FROM users WHERE source='central'")}
        removed = mirrored - set(incoming)
        for uid in removed:
            _db.execute("DELETE FROM templates WHERE user_id=?", (uid,))
            _db.execute("DELETE FROM users WHERE id=?", (uid,))
        for uid, p in incoming.items():
            _db.execute("""INSERT INTO users (id, name, phone, email, role, valid_until, created, totp_secret, source)
                           VALUES (?,?,?,?,?,?,?,?, 'central') ON CONFLICT(id) DO UPDATE SET name=excluded.name,
                           phone=excluded.phone, email=excluded.email, role=excluded.role, valid_until=excluded.valid_until,
                           created=excluded.created, totp_secret=excluded.totp_secret, source='central'""",
                        (uid, p["name"], vault.enc_str(p.get("phone")), p.get("email"), p.get("role"),
                         p.get("valid_until"), p.get("created"), vault.enc_str(p.get("totp_secret"))))
            _db.execute("DELETE FROM templates WHERE user_id=?", (uid,))
            _db.executemany("INSERT INTO templates (user_id, emb) VALUES (?,?)",
                            [(uid, vault.enc(base64.b64decode(t))) for t in p.get("templates", [])])
        _db.commit()
    return {"people": len(incoming), "removed": len(removed)}


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
    rows = _db.execute("""SELECT u.id, u.name, u.phone, u.email, u.created, u.role, u.valid_until,
                            (u.totp_secret IS NOT NULL) has_totp,
                            (SELECT COUNT(*) FROM templates t WHERE t.user_id=u.id) n,
                            (SELECT COUNT(*) FROM rek_faces r WHERE r.user_id=u.id) n_rekognition
                          FROM users u ORDER BY u.created""").fetchall()
    return [_user(r) for r in rows]


def get_user(user_id: str) -> dict | None:
    return _user(_db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())


def templates() -> dict[str, np.ndarray]:
    out: dict[str, list] = {}
    for r in _db.execute("SELECT user_id, emb FROM templates"):
        out.setdefault(r["user_id"], []).append(np.frombuffer(vault.dec(r["emb"]), np.float32))
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
    return _user(_db.execute("SELECT * FROM users WHERE email=?", (normalise_email(email),)).fetchone())


def set_email(user_id: str, email: str | None) -> None:
    with _lock:
        _db.execute("UPDATE users SET email=? WHERE id=?", (normalise_email(email), user_id))
        _db.commit()


def set_totp(user_id: str, secret: str | None) -> None:
    with _lock:
        _db.execute("UPDATE users SET totp_secret=? WHERE id=?", (vault.enc_str(secret), user_id))
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
