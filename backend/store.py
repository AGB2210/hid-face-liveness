"""SQLite storage for enrolled users, face templates and the attempt log."""
from __future__ import annotations

import json
import sqlite3
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
CREATE TABLE IF NOT EXISTS attempts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, decision TEXT, stage TEXT,
  user_id TEXT, similarity REAL, liveness REAL, processing_ms REAL, total_ms REAL,
  reasons TEXT);
""")
_db.commit()


def upsert_user(user_id: str, name: str, phone: str | None, embs: np.ndarray) -> None:
    with _lock:
        _db.execute("INSERT OR REPLACE INTO users VALUES (?,?,?,?)", (user_id, name, phone, time.time()))
        _db.execute("DELETE FROM templates WHERE user_id=?", (user_id,))
        _db.executemany("INSERT INTO templates (user_id, emb) VALUES (?,?)",
                        [(user_id, e.astype(np.float32).tobytes()) for e in embs])
        _db.commit()


def delete_user(user_id: str) -> None:
    with _lock:
        _db.execute("DELETE FROM templates WHERE user_id=?", (user_id,))
        _db.execute("DELETE FROM users WHERE id=?", (user_id,))
        _db.commit()


def list_users() -> list[dict]:
    rows = _db.execute("""SELECT u.id, u.name, u.phone, u.created, COUNT(t.id) n
                          FROM users u LEFT JOIN templates t ON t.user_id=u.id
                          GROUP BY u.id ORDER BY u.created""").fetchall()
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
