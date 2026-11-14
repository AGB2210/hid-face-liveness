import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "backend" / "config.yaml"
# Enrolled users, test clips and eval output. Override with HID_DATA (e.g. for tests).
DATA = Path(os.environ.get("HID_DATA", ROOT / "data")).resolve()


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()


def reload_config() -> dict:
    CFG.clear()
    CFG.update(load_config())
    return CFG
