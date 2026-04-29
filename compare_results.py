"""
Aggregate evaluation results from sweep and ablation directories into
a single comparison.json file.

Scans for *_results.json files under --sweep-dir and --ablation-dir,
categorises them by strategy/dataset, and writes a unified JSON.

When --predictors-dir is supplied, also computes Spearman correlations
between each predictor (epiplexity 1.5B/3B, reward variance 1.5B/3B,
rollout epiplexity 1.5B/3B) and the per-source ablation MIXED transfer
score, producing the predictor-comparison table consumed by findings/RESULTS.md.

Usage:
    python compare_results.py \
        --sweep-dir /gpfs/scratch/gs4342/grpo-sweep \
        --ablation-dir /gpfs/scratch/gs4342/grpo-ablation \
        --predictors-dir data \
        --output comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


# Datasets, in canonical order, used for both transfer extraction and
# predictor alignment. Mirrors data/registry.py.
DATASET_ORDER = [
    "gsm8k", "math", "humaneval", "mbpp",
    "mmlu", "arc", "triviaqa", "boolq",
]


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


# ---------------------------------------------------------------------------
# Predictor loading + correlation
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_predictors(predictors_dir: Path) -> dict[str, dict[str, float]]:
    """
    Load all available predictor JSONs and normalize to
        {predictor_name: {dataset: scalar}}.

    Looks for the files written by:
        probe_epiplexity.py            -> epiplexity_scores{,_3b}.json
        measure_reward_variance.py     -> reward_variance_{1.5b,3b}.json
        measure_rollout_epiplexity.py  -> rollout_epiplexity_{1.5b,3b}.json
    """
    candidates = {
        "epiplexity_1.5b":         (predictors_dir / "epiplexity_scores.json",         "flat"),
        "epiplexity_3b":           (predictors_dir / "epiplexity_scores_3b.json",      "flat"),
        "reward_variance_1.5b":    (predictors_dir / "reward_variance_1.5b.json",      "per_dataset.mean_reward_variance"),
        "reward_variance_3b":      (predictors_dir / "reward_variance_3b.json",        "per_dataset.mean_reward_variance"),
        "rollout_epiplexity_1.5b": (predictors_dir / "rollout_epiplexity_1.5b.json",   "per_dataset.rollout_epiplexity_per_token"),
        "rollout_epiplexity_3b":   (predictors_dir / "rollout_epiplexity_3b.json",     "per_dataset.rollout_epiplexity_per_token"),
    }

    out: dict[str, dict[str, float]] = {}
    for name, (path, schema) in candidates.items():
        data = _load_json(path)
        if data is None:
            continue
        if schema == "flat":
            # epiplexity_scores.json: {dataset: score, "_metadata": {...}}
            out[name] = {
                k: float(v) for k, v in data.items()
                if k in DATASET_ORDER and isinstance(v, (int, float))
            }
        else:
            # "per_dataset.<field>" — drill down into the nested dict.
            field = schema.split(".", 1)[1]
            per_ds = data.get("per_dataset", {})
            out[name] = {
                ds: float(per_ds[ds][field])
                for ds in DATASET_ORDER
                if ds in per_ds and field in per_ds[ds]
            }
    return out


def extract_ablation_transfer(ablation: dict) -> dict[str, float]:
    """
    Map source-dataset -> MIXED transfer score from ablation results.

    ablation labels look like 'ablation_math', 'ablation_gsm8k', ...
    """
    transfer: dict[str, float] = {}
    for label, data in ablation.items():
        # Strip 'ablation_' prefix if present.
        ds = label[len("ablation_"):] if label.startswith("ablation_") else label
        if ds in DATASET_ORDER and "overall_mixed" in data:
            transfer[ds] = float(data["overall_mixed"])
    return transfer


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """
    Spearman rank correlation. Pure-Python (no scipy dep), uses average
    ranks for ties and a tie-corrected formula via Pearson on ranks.
    """
    n = len(xs)
    if n < 2 or n != len(ys):
        return None

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    vx = sum((rx[i] - mx) ** 2 for i in range(n))
    vy = sum((ry[i] - my) ** 2 for i in range(n))
    denom = (vx * vy) ** 0.5
    if denom == 0:
        return None
    return cov / denom


def build_predictor_table(
    predictors: dict[str, dict[str, float]],
    transfer: dict[str, float],
) -> dict:
    """
    Build the predictor comparison block:
        {
            "transfer_scores": {dataset: mixed},
            "predictor_values": {predictor_name: {dataset: value}},
            "spearman_with_transfer": {predictor_name: rho_or_null},
            "n_datasets": int,
        }
    Aligned datasets are the intersection of (dataset has transfer score)
    and (dataset has predictor value), evaluated independently per predictor.
    """
    spearman = {}
    for pname, pvals in predictors.items():
        common = [ds for ds in DATASET_ORDER if ds in pvals and ds in transfer]
        xs = [pvals[ds] for ds in common]
        ys = [transfer[ds] for ds in common]
        spearman[pname] = spearman_rho(xs, ys)
    return {
        "transfer_scores": {ds: transfer[ds] for ds in DATASET_ORDER if ds in transfer},
        "predictor_values": predictors,
        "spearman_with_transfer": spearman,
        "n_datasets_in_transfer": len(transfer),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results")
    parser.add_argument("--sweep-dir", type=str, required=True,
                        help="Root directory containing sweep strategy results")
    parser.add_argument("--ablation-dir", type=str, required=True,
                        help="Root directory containing single-dataset ablation results")
    parser.add_argument("--predictors-dir", type=str, default=None,
                        help="If set, load predictor JSONs from this dir and "
                             "compute Spearman rho with ablation transfer")
    parser.add_argument("--output", type=str, default="comparison.json",
                        help="Output path for the aggregated comparison file")
    args = parser.parse_args()

    sweep_dir = Path(args.sweep_dir)
    ablation_dir = Path(args.ablation_dir)

    sweep = gather_results(sweep_dir)
    ablation = gather_results(ablation_dir)

    comparison = {
        "sweep": sweep,
        "ablation": ablation,
    }

    if args.predictors_dir:
        predictors_dir = Path(args.predictors_dir)
        predictors = load_predictors(predictors_dir)
        transfer = extract_ablation_transfer(ablation)
        if predictors and transfer:
            comparison["predictor_comparison"] = build_predictor_table(predictors, transfer)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    n_sweep = len(sweep)
    n_ablation = len(ablation)
    print(f"Wrote {out_path}: {n_sweep} sweep + {n_ablation} ablation results")

    for section_name, section in [("Sweep", sweep), ("Ablation", ablation)]:
        if section:
            print(f"\n{section_name}:")
            for label in sorted(section):
                mixed = section[label].get("overall_mixed", "?")
                print(f"  {label:<25} mixed={mixed}")

    pc = comparison.get("predictor_comparison")
    if pc:
        print(f"\nPredictor → Transfer (Spearman rho, n={pc['n_datasets_in_transfer']}):")
        for pname, rho in pc["spearman_with_transfer"].items():
            rho_str = f"{rho:+.3f}" if rho is not None else "  n/a"
            print(f"  {pname:<28} {rho_str}")


if __name__ == "__main__":
    main()
