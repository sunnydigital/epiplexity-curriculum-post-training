"""
Aggregate and compare evaluation results across curriculum strategies.

Reads *_results.json files from multiple evaluation directories and produces
a comparison table for the paper. Run after sweep and ablation jobs complete.

Usage:
    python compare_results.py \\
        --sweep-dir /gpfs/scratch/gs4342/grpo-sweep \\
        --ablation-dir /gpfs/scratch/gs4342/grpo-ablation \\
        [--output comparison.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare evaluation results")
    parser.add_argument("--sweep-dir", type=str, default=None)
    parser.add_argument("--ablation-dir", type=str, default=None)
    parser.add_argument("--output", type=str, default="comparison.json")
    return parser.parse_args()


def collect_results(base_dir: str) -> dict:
    """Recursively find and load all *_results.json files."""
    results = {}
    for path in Path(base_dir).rglob("*_results.json"):
        with open(path) as f:
            data = json.load(f)
        label = data.get("label") or path.stem.replace("_results", "")
        results[label] = data
    return results


def print_comparison_table(results: dict, title: str) -> None:
    """Print a comparison table across strategies/ablations."""
    if not results:
        return

    # Collect all dataset names
    all_datasets = set()
    for r in results.values():
        all_datasets.update(r.get("per_dataset", {}).keys())
    datasets = sorted(all_datasets)

    # Header
    strategies = sorted(results.keys())
    col_w = 12
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")

    header = f"{'Dataset':<15}" + "".join(f"{s:>{col_w}}" for s in strategies)
    print(header)
    print("-" * len(header))

    # Per-dataset rows
    for ds in datasets:
        row = f"{ds:<15}"
        for strat in strategies:
            r = results[strat].get("per_dataset", {}).get(ds, {})
            acc = r.get("accuracy")
            if acc is not None:
                row += f"{acc:>{col_w}.4f}"
            else:
                err = "error" if "error" in r else "—"
                row += f"{err:>{col_w}}"
        print(row)

    # Category means
    print("-" * len(header))
    categories = set()
    for r in results.values():
        categories.update(r.get("category_means", {}).keys())
    for cat in sorted(categories):
        row = f"[{cat}]"
        row = f"{row:<15}"
        for strat in strategies:
            mean = results[strat].get("category_means", {}).get(cat)
            if mean is not None:
                row += f"{mean:>{col_w}.4f}"
            else:
                row += f"{'—':>{col_w}}"
        print(row)

    # Overall
    row = f"{'[OVERALL]':<15}"
    for strat in strategies:
        overall = results[strat].get("overall_accuracy")
        if overall is not None:
            row += f"{overall:>{col_w}.4f}"
        else:
            row += f"{'—':>{col_w}}"
    print(row)
    print()


def main() -> None:
    args = parse_args()

    all_results = {}

    if args.sweep_dir:
        sweep = collect_results(args.sweep_dir)
        print_comparison_table(sweep, "Curriculum Strategy Comparison")
        all_results["sweep"] = sweep

    if args.ablation_dir:
        ablation = collect_results(args.ablation_dir)
        print_comparison_table(ablation, "Single-Dataset Ablation Results")
        all_results["ablation"] = ablation

        # If we have both sweep and ablation, print transfer matrix
        if ablation:
            print("=" * 80)
            print("  Transfer Matrix: trained on (row) → evaluated on (col)")
            print("=" * 80)
            datasets = sorted(set().union(
                *(r.get("per_dataset", {}).keys() for r in ablation.values())
            ))
            col_w = 10
            train_label = "train\\eval"
            header = f"{train_label:<15}" + "".join(f"{d:>{col_w}}" for d in datasets)
            print(header)
            print("-" * len(header))
            for train_ds in sorted(ablation.keys()):
                row = f"{train_ds.replace('ablation_', ''):<15}"
                for eval_ds in datasets:
                    acc = ablation[train_ds].get("per_dataset", {}).get(eval_ds, {}).get("accuracy")
                    if acc is not None:
                        row += f"{acc:>{col_w}.4f}"
                    else:
                        row += f"{'—':>{col_w}}"
                print(row)
            print()

    # Save combined results
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Combined results saved to {args.output}")


if __name__ == "__main__":
    main()
