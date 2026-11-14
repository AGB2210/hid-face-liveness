"""Offline evaluation over recorded clips (data/clips/<label>/<subject>/<ts>/).

Every clip is replayed through exactly the same pipeline as the kiosk. Metrics:
  * Liveness (ISO/IEC 30107-3): APCER per attack type, BPCER, ACER
  * Face authentication: FAR / FRR over genuine and impostor comparisons
  * Headline numbers in the organisers' wording: liveness detection accuracy,
    face authentication accuracy, FAR, FRR, decision time
  * Ablation: how well each individual check separates live from attack

Test subjects who aren't enrolled kiosk users (e.g. imported clips) get an evaluation-only
face gallery: each one is enrolled from their first live clip, held in memory only (never
written to the user database). That enrollment clip is left out of the identity metrics.
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
from pipeline.frames import analyse_frame, decode
from pipeline.identity import IdentityProvider, LocalArcFace, identity_thresholds
from pipeline.models import models_lock

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


def _clip_frames(meta_path: Path) -> list[bytes]:
    return [vault.read_file(p) for p in sorted(meta_path.parent.glob("frame_*.jpg*"))]


class EvalGallery(IdentityProvider):
    """Local ArcFace over the enrolled users plus in-memory templates for test subjects."""

    def __init__(self, extra: dict[str, np.ndarray]):
        self.extra, self.arc = extra, LocalArcFace()

    def enroll(self, user_id, name, phone, frames):
        raise NotImplementedError("evaluation gallery is read-only")

    def identify(self, frames, claimed_id=None):
        import store
        probe = self.arc._embed(frames).mean(axis=0)
        probe /= np.linalg.norm(probe) + 1e-9
        scores = {uid: float(np.max(e @ probe)) for uid, e in {**store.templates(), **self.extra}.items()}
        best = max(scores, key=scores.get) if scores else None
        return {"user_id": best, "name": best, "similarity": round(max(scores.get(best, 0.0), 0.0), 3),
                "mode": "1:N", "scores": {k: round(v, 3) for k, v in scores.items()}}


def build_gallery(paths: list[Path]) -> tuple[dict[str, np.ndarray], set[Path]]:
    """Enroll every subject who isn't a kiosk user from their first live clip (best 5 frames)."""
    import store
    enrolled = set(store.templates())
    first: dict[str, tuple] = {}
    for p in paths:
        meta = json.loads(p.read_text())
        if meta["label"] != "live" or meta["subject"] in enrolled:
            continue
        key = (meta.get("created") or 0, str(p))
        if meta["subject"] not in first or key < first[meta["subject"]][0]:
            first[meta["subject"]] = (key, p)
    arc, gallery, used = LocalArcFace(), {}, set()
    for subject, (_, p) in sorted(first.items()):
        with models_lock():
            frames = [analyse_frame(i, im) for i, im in enumerate(decode(_clip_frames(p)))]
            frames = sorted((f for f in frames if f.box is not None and f.lmk5 is not None),
                            key=lambda f: f.rank, reverse=True)[:5]
            if frames:
                gallery[subject] = arc._embed(frames)
                used.add(p)
    return gallery, used


# A test subject whose face matches an enrolled kiosk user this strongly is that same person
# (e.g. a tester who is also an admin). Well above the match threshold and every score between
# different people seen in testing (<= 0.32), so a real impostor isn't hidden this way.
SAME_PERSON = 0.60


def same_people(gallery: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """test subject -> {kiosk user id: similarity} for kiosk users who are the same person."""
    import store
    out = {}
    for subject, embs in gallery.items():
        probe = embs.mean(axis=0)
        probe /= np.linalg.norm(probe) + 1e-9
        hits = {uid: round(float(np.max(t @ probe)), 3) for uid, t in store.templates().items()}
        hits = {k: v for k, v in hits.items() if v >= SAME_PERSON}
        if hits:
            out[subject] = hits
    return out


def run_clip(meta_path: Path, provider=None, enrollment: bool = False, aliases: dict | None = None) -> dict:
    meta = json.loads(meta_path.read_text())
    frames = _clip_frames(meta_path)
    res = engine.analyse(frames, None, None, provider=provider,
                         provider_name="local_arcface" if provider else None)
    live = res.get("liveness") or {}
    ident = res.get("identity") or {}
    scores = ident.get("scores") or {}
    subject = meta["subject"]
    same = {subject, *((aliases or {}).get(subject) or {})}
    return {
        "clip": str(meta_path.parent.relative_to(DATA)),
        "label": meta["label"], "subject": subject, "note": meta.get("note", ""),
        "split": meta.get("split"), "enrollment": enrollment,
        "decision": res["decision"], "reasons": res["reasons"],
        "liveness": live.get("score"), "vetoes": live.get("vetoes", []),
        "checks": {k: (live.get("checks", {}).get(k) or {}).get("score") for k in CHECKS},
        "blink": (live.get("checks", {}).get("blink") or {}).get("blinked"),
        "best_user": ident.get("user_id"), "best_sim": ident.get("similarity"),
        "best_is_subject": ident.get("user_id") in same,
        "genuine_sim": max((v for k, v in scores.items() if k in same), default=None),
        "impostor_sims": [v for k, v in scores.items() if k not in same],
        "good_frames": res["quality"]["good_frames"], "frames": res["quality"]["frames"],
        "occlusion": res.get("occlusion") or {},
        "quality_issues": res["quality"]["issues"], "blur": res["quality"]["blur"],
        "processing_ms": res["timing_ms"]["processing"],
        "capture_ms": meta.get("capture_ms") or CFG["capture"]["duration_ms"],
    }


def liveness_pass(r: dict, allow: float) -> bool:
    return r["liveness"] is not None and not r["vetoes"] and r["liveness"] >= allow


def liveness_reject(r: dict, deny: float) -> bool:
    return r["liveness"] is None or bool(r["vetoes"]) or r["liveness"] < deny


def summarise(rows: list[dict], provider: str | None = None, with_splits: bool = True,
              same_person: dict | None = None) -> dict:
    lc, ic = CFG["liveness"], identity_thresholds(provider)
    live = [r for r in rows if r["label"] in BONA_FIDE]
    attacks = [r for r in rows if r["label"] in ATTACKS]
    # decisions and identity for live people: leave out the clip a test subject was enrolled from
    live_id = [r for r in live if not r.get("enrollment")]

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
    gen = [r["genuine_sim"] for r in live_id if r["genuine_sim"] is not None]
    imp = [s for r in live_id for s in r["impostor_sims"]]
    fr = sum(s < ic["match"] for s in gen)
    fa = sum(s >= ic["match"] for s in imp)
    rank1 = sum(r.get("best_is_subject", r["best_user"] == r["subject"])
                for r in live_id if r["genuine_sim"] is not None)
    face_acc = rate((len(gen) - fr) + (len(imp) - fa), len(gen) + len(imp))

    # ---- decision time ------------------------------------------------------------
    # decision time = getting the decision, not the camera capture before it (replayed clips have
    # no network, so that is the processing time)
    total = np.array([r["processing_ms"] for r in rows]) if rows else np.zeros(0)
    proc = np.array([r["processing_ms"] for r in rows]) if rows else np.zeros(0)

    def pct(a, q):
        return round(float(np.percentile(a, q)), 1) if len(a) else None

    timing = {"total_p50": pct(total, 50), "total_p95": pct(total, 95),
              "total_max": round(float(total.max()), 1) if len(total) else None,
              "processing_p50": pct(proc, 50), "processing_p95": pct(proc, 95),
              "under_2s": rate(int((total < 2000).sum()), len(total)),
              "histogram": np.histogram(total, bins=np.arange(0, 2200, 100))[0].tolist() if len(total) else [],
              "bins_ms": list(range(0, 2200, 100))}

    # ---- live user experience --------------------------------------------------------
    live_outcomes = {d: sum(r["decision"] == d for r in live_id) for d in DECISIONS}

    # ---- confusion matrix ----------------------------------------------------------
    labels_present = [l for l in ALL_LABELS if any(r["label"] == l for r in rows)]

    # ---- real people with accessories: who gets let in, guided, or wrongly turned away
    accessories = {}
    for v in LIVE_VARIANTS:
        rs = [r for r in rows if r["label"] == v and not r.get("enrollment")]
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

    # ---- per split (train / validation / test), when the clips carry one -------------------
    splits = {}
    if with_splits:
        for sp in sorted({r.get("split") for r in rows if r.get("split")}):
            part = [r for r in rows if r.get("split") == sp]
            sub = summarise(part, provider, with_splits=False)
            splits[sp] = {"clips": len(part), "subjects": sub["subjects"], "headline": sub["headline"],
                          "bpcer": sub["bpcer"], "live_outcomes": sub["live_outcomes"]}

    return {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": splits,
        "counts": {l: sum(r["label"] == l for r in rows) for l in ALL_LABELS},
        "accessories": accessories,
        "subjects": sorted({r["subject"] for r in rows}),
        "thresholds": {"liveness_allow": lc["allow"], "liveness_deny": lc["deny"],
                       "identity_match": ic["match"], "identity_borderline": ic["borderline"],
                       "identity_provider": ic["provider"]},
        "eval_gallery": sorted({r["subject"] for r in rows if r.get("enrollment")}),
        "same_person": same_person or {},
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
    gallery, enroll_clips = build_gallery(paths)
    aliases = same_people(gallery)
    # with test subjects in the gallery, score everyone with local ArcFace (the gallery's model)
    provider = EvalGallery(gallery) if gallery else None
    rows = []
    for i, p in enumerate(paths):
        rows.append(run_clip(p, provider, p in enroll_clips, aliases))
        if progress:
            progress(i + 1, len(paths), rows[-1])
    out = summarise(rows, "local_arcface" if provider else None, same_person=aliases)
    dest = DATA / "eval"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "results.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def load_results() -> dict | None:
    f = DATA / "eval" / "results.json"
    return json.loads(f.read_text()) if f.exists() else None
