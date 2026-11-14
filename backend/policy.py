"""Context-aware access: cross-reference a recognised person with the campus schedule.

evaluate() answers: is this role allowed in this zone at this moment?
  inside  - within an allowed window
  grace   - within grace_minutes of a window (step-up verification)
  outside - no window, wrong zone, expired access or holiday (deny)
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from pipeline.config import DATA

POLICY_FILE = Path(__file__).resolve().parent / "access_policy.yaml"
CACHE_FILE = DATA / "policy_cache.yaml"   # copy pulled from the central server (edge kiosks)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_WIN = re.compile(r"^\s*([A-Za-z,\-]+)\s+(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def load(prefer_cache: bool = False) -> dict:
    src = CACHE_FILE if prefer_cache and CACHE_FILE.exists() else POLICY_FILE
    with open(src, encoding="utf-8") as f:
        pol = yaml.safe_load(f)
    pol["_source"] = "central cache" if src == CACHE_FILE else "local file"
    return pol


def policy_text() -> str:
    return POLICY_FILE.read_text(encoding="utf-8")


def save_cache(text: str) -> None:
    yaml.safe_load(text)  # validate before replacing the cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(text, encoding="utf-8")


def _days(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            i, j = DAYS.index(a.strip()[:3].title()), DAYS.index(b.strip()[:3].title())
            out.update(range(i, j + 1) if i <= j else list(range(i, 7)) + list(range(0, j + 1)))
        else:
            out.add(DAYS.index(part.strip()[:3].title()))
    return out


def parse_window(w: str) -> tuple[set[int], int, int]:
    m = _WIN.match(w)
    if not m:
        raise ValueError(f"bad window '{w}'")
    days = _days(m.group(1))
    return days, int(m.group(2)) * 60 + int(m.group(3)), int(m.group(4)) * 60 + int(m.group(5))


def now(pol: dict, at: str | None = None) -> datetime:
    tz = ZoneInfo(pol.get("timezone", "UTC"))
    if at:
        dt = datetime.fromisoformat(at)
        return dt.replace(tzinfo=tz) if dt.tzinfo is None else dt.astimezone(tz)
    return datetime.now(tz)


def _minutes_to_window(t: datetime, windows: list[str]) -> tuple[bool, float]:
    """(inside, distance in minutes to the nearest window edge when outside)."""
    best = float("inf")
    for w in windows:
        days, start, end = parse_window(w)
        for offset in (-1, 0, 1):  # neighbouring days, so 23:50 vs a 00:00 start works
            d = t.date() + timedelta(days=offset)
            if d.weekday() not in days:
                continue
            ws = datetime(d.year, d.month, d.day, tzinfo=t.tzinfo) + timedelta(minutes=start)
            we = datetime(d.year, d.month, d.day, tzinfo=t.tzinfo) + timedelta(minutes=end)
            if ws <= t < we:
                return True, 0.0
            best = min(best, abs((t - ws).total_seconds()) / 60, abs((t - we).total_seconds()) / 60)
    return False, best


def evaluate(user: dict | None, zone_id: str, at: str | None = None, pol: dict | None = None) -> dict:
    pol = pol or load()
    t = now(pol, at)
    zone = pol["zones"].get(zone_id)
    base = {"zone": zone_id, "zone_name": (zone or {}).get("name", zone_id),
            "security": (zone or {}).get("security", "standard"),
            "time": t.strftime("%a %d %b %H:%M"), "source": pol.get("_source")}
    if zone is None:
        return {**base, "verdict": "outside", "reason": f"Unknown zone '{zone_id}'"}
    if not user:
        return {**base, "verdict": "outside", "reason": "No identified person"}
    role_id = user.get("role") or "visitor"
    role = pol["roles"].get(role_id)
    base["role"] = (role or {}).get("name", role_id)
    if role is None:
        return {**base, "verdict": "outside", "reason": f"Unknown role '{role_id}'"}
    valid_until = user.get("valid_until")
    if valid_until and date.fromisoformat(valid_until) < t.date():
        return {**base, "verdict": "outside", "reason": f"Access expired on {valid_until}"}
    if role.get("holidays_apply", True) and t.date().isoformat() in [str(h) for h in pol.get("holidays", [])]:
        return {**base, "verdict": "outside", "reason": "Campus holiday - entry closed"}
    windows = role.get("zones", {}).get(zone_id)
    if not windows:
        return {**base, "verdict": "outside", "reason": f"{base['role']} has no access to {base['zone_name']}"}
    base["windows"] = windows
    inside, dist = _minutes_to_window(t, windows)
    if inside:
        return {**base, "verdict": "inside", "reason": f"Within access window ({'; '.join(windows)})"}
    if dist <= pol.get("grace_minutes", 0):
        return {**base, "verdict": "grace",
                "reason": f"{int(round(dist))} min outside access window ({'; '.join(windows)})"}
    return {**base, "verdict": "outside",
            "reason": f"Outside access window for {base['zone_name']} ({'; '.join(windows)})"}


def zones(pol: dict | None = None) -> dict:
    pol = pol or load()
    return {k: v for k, v in pol["zones"].items()}


def roles(pol: dict | None = None) -> dict:
    pol = pol or load()
    return {k: v.get("name", k) for k, v in pol["roles"].items()}
