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

POLICY_FILE = Path(__file__).resolve().parent / "access_policy.yaml"   # shipped default
EDITED_FILE = DATA / "access_policy.yaml"   # admin edits (kept in data/, so redeploys don't overwrite them)
HISTORY_DIR = DATA / "policy_history"       # every previous version, timestamped
CACHE_FILE = DATA / "policy_cache.yaml"     # copy pulled from the central server (edge kiosks)
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_WIN = re.compile(r"^\s*([A-Za-z,\-]+)\s+(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*$")


def active_file() -> Path:
    return EDITED_FILE if EDITED_FILE.exists() else POLICY_FILE


def load(prefer_cache: bool = False) -> dict:
    src = CACHE_FILE if prefer_cache and CACHE_FILE.exists() else active_file()
    with open(src, encoding="utf-8") as f:
        pol = yaml.safe_load(f)
    pol["_source"] = {CACHE_FILE: "central cache", EDITED_FILE: "edited by admin"}.get(src, "default file")
    return pol


def policy_text() -> str:
    return active_file().read_text(encoding="utf-8")


# ----------------------------------------------------------------------- admin editing
def validate(pol: dict) -> list[str]:
    """Every problem with a proposed policy (empty list = valid). Nothing is saved unless valid."""
    errs = []
    zones, roles = pol.get("zones") or {}, pol.get("roles") or {}
    try:
        ZoneInfo(pol.get("timezone", ""))
    except Exception:
        errs.append(f"Unknown timezone '{pol.get('timezone')}'")
    g = pol.get("grace_minutes")
    if not isinstance(g, int) or not 0 <= g <= 240:
        errs.append("Grace period must be a whole number of minutes between 0 and 240")
    for zid, z in zones.items():
        if (z or {}).get("security") not in ("standard", "high"):
            errs.append(f"Zone {zid}: security must be 'standard' or 'high'")
    for rid, r in roles.items():
        for zid, wins in ((r or {}).get("zones") or {}).items():
            if zid not in zones:
                errs.append(f"{rid}: unknown zone '{zid}'")
            for w in wins or []:
                try:
                    days, start, end = parse_window(w)
                    if not days:
                        raise ValueError
                    if not (0 <= start < end <= 24 * 60):
                        errs.append(f"{rid} / {zid}: '{w}' - start must be before end (00:00-24:00)")
                except (ValueError, IndexError):
                    errs.append(f"{rid} / {zid}: can't read '{w}' - use e.g. 'Mon-Fri 09:00-17:00'")
    for h in pol.get("holidays") or []:
        try:
            date.fromisoformat(str(h))
        except ValueError:
            errs.append(f"Holiday '{h}' is not a date (YYYY-MM-DD)")
    return errs


def changes(old: dict, new: dict) -> list[str]:
    """Human-readable summary of what an edit changed (for the audit log)."""
    out = []
    for zid, z in new["zones"].items():
        a, b = (old["zones"].get(zid) or {}).get("security"), z.get("security")
        if a != b:
            out.append(f"{z.get('name', zid)}: security {a} -> {b}")
    for rid, r in new["roles"].items():
        oz = (old["roles"].get(rid) or {}).get("zones") or {}
        for zid in sorted(set(oz) | set(r.get("zones") or {})):
            a, b = oz.get(zid) or [], (r.get("zones") or {}).get(zid) or []
            if a != b:
                out.append(f"{r.get('name', rid)} / {zid}: {'; '.join(a) or 'no access'} -> {'; '.join(b) or 'no access'}")
    if old.get("grace_minutes") != new.get("grace_minutes"):
        out.append(f"grace period {old.get('grace_minutes')} -> {new.get('grace_minutes')} min")
    if [str(h) for h in old.get("holidays") or []] != [str(h) for h in new.get("holidays") or []]:
        out.append(f"holidays -> {', '.join(str(h) for h in new.get('holidays') or []) or 'none'}")
    return out


def save_edit(new: dict, admin_id: str) -> list[str]:
    """Validate, keep the previous version, write the new one. Returns the change summary."""
    import hashlib
    import time
    errs = validate(new)
    if errs:
        raise ValueError("; ".join(errs[:6]))
    old = load()
    clean = {k: v for k, v in new.items() if not k.startswith("_")}
    for r in clean["roles"].values():  # drop zones with no windows
        r["zones"] = {z: w for z, w in (r.get("zones") or {}).items() if w}
    summary = changes(old, clean)
    if not summary:
        return []
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    (HISTORY_DIR / time.strftime("%Y%m%d-%H%M%S.yaml")).write_text(policy_text(), encoding="utf-8")
    body = yaml.safe_dump(clean, sort_keys=False, allow_unicode=True)
    header = (f"# Campus access schedule - edited by {admin_id} on {time.strftime('%Y-%m-%d %H:%M')}\n"
              "# Window format: \"<days> HH:MM-HH:MM\" (e.g. Mon-Fri 09:00-17:00). "
              "Previous versions: data/policy_history/\n")
    EDITED_FILE.parent.mkdir(parents=True, exist_ok=True)
    EDITED_FILE.write_text(header + body, encoding="utf-8")
    summary.append("policy sha256 " + hashlib.sha256(body.encode()).hexdigest()[:12])
    return summary


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
