"""
Aggregate evaluation results from sweep and ablation directories into
a single comparison.json file.

Scans for *_results.json files under --sweep-dir and --ablation-dir,
categorises them by strategy/dataset, and writes a unified JSON.

Usage:
    python compare_results.py \
        --sweep-dir /gpfs/scratch/gs4342/grpo-sweep \
        --ablation-dir /gpfs/scratch/gs4342/grpo-ablation \
        --output comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def gather_results(base_dir: Path, pattern: str = "*_results.json") -> dict:
    """Walk subdirectories and collect all result JSONs."""
    collected = {}
    if not base_dir.exists():
        return collected
    for results_file in sorted(base_dir.rglob(pattern)):
        with open(results_file) as f:
            data = json.load(f)
        label = data.get("label") or results_file.stem.replace("_results", "")
        collected[label] = data
    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--sweep-dir", type=str, required=True,
                        help="Root directory containing sweep strategy results")
    parser.add_argument("--ablation-dir", type=str, required=True,
                        help="Root directory containing single-dataset ablation results")
    parser.add_argument("--output", type=str, default="comparison.json",
                        help="Output path for the aggregated comparison file")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    ablation_dir = Path(args.ablation_dir)

    sweep_results = gather_results(sweep_dir)
    ablation_results = gather_results(ablation_dir)

    # Separate baseline from sweep strategies (baseline label starts with "baseline")
    sweep = {}
    for label, data in sweep_results.items():
        sweep[label] = data

    ablation = {}
    for label, data in ablation_results.items():
        ablation[label] = data

    comparison = {
        "sweep": sweep,
        "ablation": ablation,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    n_sweep = len(sweep)
    n_ablation = len(ablation)
    print(f"Wrote {out_path}: {n_sweep} sweep + {n_ablation} ablation results")

    # Quick summary
    for section_name, section in [("Sweep", sweep), ("Ablation", ablation)]:
        if section:
            print(f"\n{section_name}:")
            for label in sorted(section):
                mixed = section[label].get("overall_mixed", "?")
                print(f"  {label:<25} mixed={mixed}")


if __name__ == "__main__":
    main()
