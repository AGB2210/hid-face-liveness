"""Replay every recorded clip through the pipeline and write data/eval/results.json.

Run:  .venv/Scripts/python eval/run_eval.py
Then open http://localhost:8000/report.html (or press "Run evaluation" there).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from evaluation import LABEL_NAMES, evaluate  # noqa: E402


def pct(r):
    return "n/a" if r["rate"] is None else f"{100 * r['rate']:.1f}%  ({r['k']}/{r['n']}, 95% CI {100 * r['ci95'][0]:.1f}-{100 * r['ci95'][1]:.1f}%)"


def main() -> int:
    def progress(i, n, row):
        print(f"[{i:3d}/{n}] {row['label']:<13} {row['subject']:<12} -> {row['decision']:<21} "
              f"live={row['liveness']}  sim={row['genuine_sim']}  {row['processing_ms']:.0f} ms")

    out = evaluate(progress)
    if not out["clips"]:
        print("No clips found. Record some at http://localhost:8000/record.html")
        return 1
    h = out["headline"]
    print("\n=== Headline ===")
    print("Liveness detection accuracy :", pct(h["liveness_accuracy"]))
    print("Face authentication accuracy:", pct(h["face_auth_accuracy"]))
    print("False acceptance rate (FAR) :", pct(h["far"]))
    print("False rejection rate (FRR)  :", pct(h["frr"]))
    print("Attacks that got ALLOW      :", pct(h["attacks_allowed"]))
    print("Decision time p95           :", h["decision_time_p95_ms"], "ms")
    print("\n=== Per attack (ISO 30107-3) ===")
    print("BPCER (live rejected as spoof):", pct(out["bpcer"]))
    for a, r in out["per_attack"].items():
        print(f"{LABEL_NAMES[a]:<22} APCER(system) {pct(r['apcer_system'])}   ACER {r['acer']}")
    print("\nWrote data/eval/results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
