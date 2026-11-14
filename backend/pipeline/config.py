import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "backend" / "config.yaml"
# Enrolled users, test clips and eval output. Override with HID_DATA (e.g. for tests).
DATA = Path(os.environ.get("HID_DATA", ROOT / "data")).resolve()


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config() -> dict:
    """config.yaml, with an optional profile applied on top (HID_PROFILE=cloud)."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    profile = os.environ.get("HID_PROFILE")
    if profile:
        _merge(cfg, cfg.get("profiles", {}).get(profile, {}))
    return cfg


CFG = load_config()


def reload_config() -> dict:
    CFG.clear()
    CFG.update(load_config())
    return CFG
