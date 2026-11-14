"""Edge / central roles and the offline fail-safe.

  HID_NODE=central   the AWS server: source of truth for the access policy, receives audit logs
  HID_NODE=edge      a kiosk: checks the central server every few seconds; when it can't reach
                     it, decisions switch to the local fail-safe flow (see app._apply_context)
  (unset)            standalone: no central server; "Simulate network loss" still works for demos

Edge settings: HID_CENTRAL_URL, HID_SYNC_KEY (shared secret), HID_KIOSK_ID.
"""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import urllib.request

import policy
import store

ROLE = os.environ.get("HID_NODE", "standalone")
KIOSK_ID = os.environ.get("HID_KIOSK_ID") or f"kiosk-{socket.gethostname().lower()}"
CENTRAL = (os.environ.get("HID_CENTRAL_URL") or "").rstrip("/")
SYNC_KEY = os.environ.get("HID_SYNC_KEY", "")
INTERVAL_S = 4

STATE = {"online": True, "simulated_offline": False, "last_ok": None, "last_check": None,
         "offline_since": None, "last_sync": None, "last_reconnect_synced": 0,
         "policy_pulled": None, "error": None}
_lock = threading.Lock()


def has_central() -> bool:
    return ROLE == "edge" and bool(CENTRAL)


def offline() -> bool:
    return STATE["simulated_offline"] or (has_central() and not STATE["online"])


def mode() -> str:
    return "offline" if offline() else "online"


def current_policy() -> dict:
    return policy.load(prefer_cache=has_central())


def _call(path: str, body: dict | None = None, timeout: float = 2.5):
    req = urllib.request.Request(CENTRAL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"X-HID-Sync-Key": SYNC_KEY, "Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
        return json.loads(raw) if r.headers.get_content_type() == "application/json" else raw.decode()


def check_once() -> None:
    STATE["last_check"] = time.time()
    if STATE["simulated_offline"]:
        _set_online(False, "simulated network loss")
        return
    if not has_central():
        _set_online(True, None)
        return
    try:
        _call("/api/sync/health")
        _set_online(True, None)
    except Exception as e:
        _set_online(False, f"{type(e).__name__}: {e}"[:160])


def _set_online(ok: bool, err: str | None) -> None:
    was_offline = not STATE["online"]
    STATE["online"], STATE["error"] = ok, err
    if ok:
        STATE["last_ok"] = time.time()
        if was_offline:
            STATE["offline_since"] = None
            n = sync_once()
            STATE["last_reconnect_synced"] = n
    elif STATE["offline_since"] is None:
        STATE["offline_since"] = time.time()


def sync_once() -> int:
    """Push pending audit records, pull the latest policy. Returns records pushed."""
    if offline() or not has_central():
        return 0
    with _lock:
        pushed = 0
        try:
            while True:
                recs = store.unsynced_audit(200)
                if not recs:
                    break
                res = _call("/api/sync/audit", {"kiosk": KIOSK_ID, "records": recs}, timeout=10)
                store.mark_synced(res.get("ids", []))
                pushed += len(res.get("ids", []))
                if len(res.get("ids", [])) < len(recs):
                    break
            text = _call("/api/sync/policy")
            policy.save_cache(text)
            STATE["policy_pulled"] = time.time()
            STATE["last_sync"] = time.time()
        except Exception as e:
            STATE["error"] = f"sync: {type(e).__name__}: {e}"[:160]
        return pushed


def _loop() -> None:
    while True:
        try:
            check_once()
            if not offline():
                sync_once()
        except Exception as e:  # never let the monitor die
            STATE["error"] = f"{type(e).__name__}: {e}"[:160]
        time.sleep(INTERVAL_S)


def start() -> None:
    threading.Thread(target=_loop, name="central-monitor", daemon=True).start()


def simulate_offline(on: bool) -> None:
    STATE["simulated_offline"] = on
    check_once()


def status() -> dict:
    c = store.audit_counts()
    return {"role": ROLE, "kiosk": KIOSK_ID, "central": CENTRAL or None, "mode": mode(),
            "simulated": STATE["simulated_offline"], "offline_since": STATE["offline_since"],
            "last_ok": STATE["last_ok"], "last_sync": STATE["last_sync"],
            "pending": c["pending"] if has_central() else 0, "offline_decisions": c["offline_total"],
            "last_reconnect_synced": STATE["last_reconnect_synced"],
            "policy_source": current_policy().get("_source"), "error": STATE["error"]}
