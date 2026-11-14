"""Offline evaluation over recorded clips (data/clips/<label>/<subject>/<ts>/).

Every clip is replayed through exactly the same pipeline as the kiosk. Metrics:
  * Liveness (ISO/IEC 30107-3): APCER per attack type, BPCER, ACER
  * Face authentication: FAR / FRR over genuine and impostor comparisons
  * Headline numbers in the organisers' wording: liveness detection accuracy,
    face authentication accuracy, FAR, FRR, decision time
  * Ablation: how well each individual check separates live from attack
"""
from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import vault
from pipeline import engine
from pipeline.config import CFG, DATA
from pipeline.identity import identity_thresholds

ATTACKS = ["print", "screen_photo", "video_replay", "mask"]
BONA_FIDE = ["live", "live_glasses"]              # how people normally present themselves
OCCLUDED = ["live_sunglasses", "live_mask"]       # real people, expected outcome: guidance
LIVE_VARIANTS = BONA_FIDE + OCCLUDED
ALL_LABELS = LIVE_VARIANTS + ATTACKS
LABEL_NAMES = {"live": "Live person", "live_glasses": "Live, clear glasses", "live_sunglasses": "Live, sunglasses",
               "live_mask": "Live, face mask", "print": "Printed photo", "screen_photo": "Screen replay (photo)",
               "video_replay": "Video replay", "mask": "Mask"}
CHECKS = ["minifas", "flatness", "moire", "glare"]
DECISIONS = ["ALLOW", "REQUEST_VERIFICATION", "DENY"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (honest error bars for small n)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}


def auc(pos: list[float], neg: list[float]) -> float | None:
    """P(score_live > score_attack); 1.0 = perfect separation, 0.5 = useless."""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(np.mean((p > n) + 0.5 * (p == n)))


def find_clips(root: Path) -> list[Path]:
    return sorted(root.glob("*/*/*/meta.json"))


def run_clip(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text())
    frames = [vault.read_file(p) for p in sorted(meta_path.parent.glob("frame_*.jpg*"))]
    res = engine.analyse(frames, None, None)
    live = res.get("liveness") or {}
    ident = res.get("identity") or {}
    scores = ident.get("scores") or {}
    subject = meta["subject"]
    return {
        "clip": str(meta_path.parent.relative_to(DATA)),
        "label": meta["label"], "subject": subject, "note": meta.get("note", ""),
        "decision": res["decision"], "reasons": res["reasons"],
        "liveness": live.get("score"), "vetoes": live.get("vetoes", []),
        "checks": {k: (live.get("checks", {}).get(k) or {}).get("score") for k in CHECKS},
        "blink": (live.get("checks", {}).get("blink") or {}).get("blinked"),
        "best_user": ident.get("user_id"), "best_sim": ident.get("similarity"),
        "genuine_sim": scores.get(subject),
        "impostor_sims": [v for k, v in scores.items() if k != subject],
        "good_frames": res["quality"]["good_frames"], "frames": res["quality"]["frames"],
        "occlusion": {k: (res.get("occlusion") or {}).get(k) for k in ("sunglasses", "mask")},
        "quality_issues": res["quality"]["issues"],
        "processing_ms": res["timing_ms"]["processing"],
        "capture_ms": meta.get("capture_ms") or CFG["capture"]["duration_ms"],
    }


def liveness_pass(r: dict, allow: float) -> bool:
    return r["liveness"] is not None and not r["vetoes"] and r["liveness"] >= allow


def liveness_reject(r: dict, deny: float) -> bool:
    return r["liveness"] is None or bool(r["vetoes"]) or r["liveness"] < deny


def summarise(rows: list[dict]) -> dict:
    lc, ic = CFG["liveness"], identity_thresholds()
    live = [r for r in rows if r["label"] in BONA_FIDE]
    attacks = [r for r in rows if r["label"] in ATTACKS]

    # ---- liveness, per attack type ------------------------------------------------
    bpcer = rate(sum(liveness_reject(r, lc["deny"]) for r in live), len(live))
    per_attack = {}
    for a in ATTACKS:
        rs = [r for r in rows if r["label"] == a]
        if not rs:
            continue
        apcer_sys = rate(sum(r["decision"] == "ALLOW" for r in rs), len(rs))
        apcer_live = rate(sum(liveness_pass(r, lc["allow"]) for r in rs), len(rs))
        per_attack[a] = {
            "name": LABEL_NAMES[a], "clips": len(rs),
            "apcer_system": apcer_sys,       # attack got ALLOW (the number that matters)
            "apcer_liveness": apcer_live,    # liveness alone called it live
            "acer": round(((apcer_live["rate"] or 0) + (bpcer["rate"] or 0)) / 2, 4) if live else None,
            "outcomes": dict(Counter(r["decision"] for r in rs)),
            "top_reasons": Counter(x for r in rs for x in r["reasons"]).most_common(3),
        }

    # ---- headline liveness accuracy (binary at the ALLOW threshold) ----------------
    correct = sum(liveness_pass(r, lc["allow"]) for r in live) + \
        sum(not liveness_pass(r, lc["allow"]) for r in attacks)
    liveness_acc = rate(correct, len(live) + len(attacks))

    # ---- face authentication (live clips only) -------------------------------------
    gen = [r["genuine_sim"] for r in live if r["genuine_sim"] is not None]
    imp = [s for r in live for s in r["impostor_sims"]]
    fr = sum(s < ic["match"] for s in gen)
    fa = sum(s >= ic["match"] for s in imp)
    rank1 = sum(r["best_user"] == r["subject"] for r in live if r["genuine_sim"] is not None)
    face_acc = rate((len(gen) - fr) + (len(imp) - fa), len(gen) + len(imp))

    # ---- decision time ------------------------------------------------------------
    total = np.array([r["capture_ms"] + r["processing_ms"] for r in rows]) if rows else np.zeros(0)
    proc = np.array([r["processing_ms"] for r in rows]) if rows else np.zeros(0)

    def pct(a, q):
        return round(float(np.percentile(a, q)), 1) if len(a) else None

    timing = {"total_p50": pct(total, 50), "total_p95": pct(total, 95),
              "total_max": round(float(total.max()), 1) if len(total) else None,
              "processing_p50": pct(proc, 50), "processing_p95": pct(proc, 95),
              "under_2s": rate(int((total < 2000).sum()), len(total)),
              "histogram": np.histogram(total, bins=np.arange(800, 2200, 100))[0].tolist() if len(total) else [],
              "bins_ms": list(range(800, 2200, 100))}

    # ---- live user experience --------------------------------------------------------
    live_outcomes = {d: sum(r["decision"] == d for r in live) for d in DECISIONS}

    # ---- confusion matrix ----------------------------------------------------------
    labels_present = [l for l in ALL_LABELS if any(r["label"] == l for r in rows)]

    # ---- real people with accessories: who gets let in, guided, or wrongly turned away
    accessories = {}
    for v in LIVE_VARIANTS:
        rs = [r for r in rows if r["label"] == v]
        if not rs:
            continue
        n = len(rs)
        accessories[v] = {
            "name": LABEL_NAMES[v], "clips": n,
            "allowed": rate(sum(r["decision"] == "ALLOW" for r in rs), n),
            "guided": rate(sum(r["decision"] == "REQUEST_VERIFICATION" for r in rs), n),
            "denied": rate(sum(r["decision"] == "DENY" for r in rs), n),
            "bpcer": rate(sum(liveness_reject(r, lc["deny"]) for r in rs), n),
            "detected": rate(sum(bool((r.get("occlusion") or {}).get("sunglasses") or (r.get("occlusion") or {}).get("mask"))
                                 for r in rs), n),
            "expected": "guidance" if v in OCCLUDED else "allow",
        }
    confusion = {l: {d: sum(r["label"] == l and r["decision"] == d for r in rows) for d in DECISIONS}
                 for l in labels_present}

    # ---- ablation: each check on its own ---------------------------------------------
    ablation = {}
    for c in CHECKS + ["fused"]:
        get = (lambda r: r["liveness"]) if c == "fused" else (lambda r, c=c: r["checks"][c])
        lv = [get(r) for r in live if get(r) is not None]
        at = [get(r) for r in attacks if get(r) is not None]
        per = {a: auc(lv, [get(r) for r in rows if r["label"] == a and get(r) is not None]) for a in per_attack}
        ablation[c] = {"auc_all": auc(lv, at), "auc_per_attack": per,
                       "coverage": round((len(lv) + len(at)) / max(1, len(live) + len(attacks)), 3),
                       "live_median": round(float(np.median(lv)), 3) if lv else None,
                       "attack_median": round(float(np.median(at)), 3) if at else None}

    # ---- threshold sweep for calibration ---------------------------------------------
    sweep = []
    for t in np.round(np.arange(0.20, 0.96, 0.05), 2):
        sweep.append({"threshold": float(t),
                      "apcer": rate(sum(liveness_pass(r, t) for r in attacks), len(attacks))["rate"],
                      "bpcer": rate(sum(not liveness_pass(r, t) for r in live), len(live))["rate"]})
    id_sweep = []
    lo, hi = max(0.0, ic["match"] - 0.25), min(1.0, ic["match"] + 0.25)
    for t in np.round(np.arange(lo, hi + 1e-6, 0.05), 2):
        id_sweep.append({"threshold": float(t),
                         "far": (sum(s >= t for s in imp) / len(imp)) if imp else None,
                         "frr": (sum(s < t for s in gen) / len(gen)) if gen else None})

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "counts": {l: sum(r["label"] == l for r in rows) for l in ALL_LABELS},
        "accessories": accessories,
        "subjects": sorted({r["subject"] for r in rows}),
        "thresholds": {"liveness_allow": lc["allow"], "liveness_deny": lc["deny"],
                       "identity_match": ic["match"], "identity_borderline": ic["borderline"],
                       "identity_provider": ic["provider"]},
        "headline": {
            "liveness_accuracy": liveness_acc,
            "face_auth_accuracy": face_acc,
            "far": rate(fa, len(imp)), "frr": rate(fr, len(gen)),
            "rank1": rate(rank1, len(gen)),
            "decision_time_p95_ms": timing["total_p95"],
            "attacks_allowed": rate(sum(r["decision"] == "ALLOW" for r in attacks), len(attacks)),
        },
        "bpcer": bpcer, "per_attack": per_attack, "live_outcomes": live_outcomes,
        "confusion": confusion, "timing": timing, "ablation": ablation,
        "sweep": sweep, "id_sweep": id_sweep,
        "identity_scores": {"genuine": [round(s, 3) for s in gen], "impostor": [round(s, 3) for s in imp]},
        "clips": rows,
    }


def evaluate(progress=None) -> dict:
    from pipeline.models import warmup
    warmup()
    paths = find_clips(DATA / "clips")
    rows = []
    for i, p in enumerate(paths):
        rows.append(run_clip(p))
        if progress:
            progress(i + 1, len(paths), rows[-1])
    out = summarise(rows)
    dest = DATA / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def load_results() -> dict | None:
    f = DATA / "eval" / "results.json"
    return json.loads(f.read_text()) if f.exists() else None
