"""Summarize where the local ViT classifier fails based on eval_vision.csv.

Reports failures bucketed by confidence band, the most common (expected,
predicted) confusion pairs, and species with the most misclassifications.

Usage:
    python src/analyze_failures.py
    python src/analyze_failures.py --csv eval_vision.csv --top 20
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CSV = PROJECT_ROOT / "eval_vision.csv"


def analyze(csv_path: Path, top: int = 15):
    """Print failure-band, confusion-pair, and per-species summaries from the eval CSV."""
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    wrong = [r for r in rows if r["top1_correct"] == "False"]
    print(f"Total images: {len(rows)}")
    print(f"Misclassified: {len(wrong)} ({len(wrong)/len(rows):.1%})\n")

    bands = {"<40%": 0, "40-60%": 0, "60-80%": 0, ">=80%": 0}
    for r in wrong:
        c = float(r["top1_confidence"])
        if c < 0.4:
            bands["<40%"] += 1
        elif c < 0.6:
            bands["40-60%"] += 1
        elif c < 0.8:
            bands["60-80%"] += 1
        else:
            bands[">=80%"] += 1
    print("Misclassifications by top-1 confidence band:")
    for k, v in bands.items():
        print(f"  {k:<8} {v}")
    print()

    confusions = Counter((r["expected_name"], r["predicted_name"]) for r in wrong)
    print(f"Top {top} confusion pairs (expected -> predicted):")
    for (exp, pred), n in confusions.most_common(top):
        print(f"  {exp:<32} -> {pred:<32} x{n}")
    print()

    per_species_total = Counter(r["expected_name"] for r in rows)
    per_species_wrong = Counter(r["expected_name"] for r in wrong)
    print(f"Species with most misclassifications (top {top}):")
    for name, n in per_species_wrong.most_common(top):
        total = per_species_total[name]
        print(f"  {name:<35} {n}/{total}  ({n/total:.0%} error)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                   help=f"Path to eval_vision.csv (default: {DEFAULT_CSV})")
    p.add_argument("--top", type=int, default=15,
                   help="Number of rows to show in confusion and per-species tables")
    args = p.parse_args()
    analyze(args.csv, args.top)


if __name__ == "__main__":
    main()
