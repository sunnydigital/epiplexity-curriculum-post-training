"""
Recompute rollout-epiplexity K_auc with alternative L_final anchoring strategies.

Hypothesis being tested
-----------------------
The published integrator anchors K_auc against `L_final = loss_curve[-1]`,
a single noisy chunk. We suspect this drives the 1.5B vs 3B sign inversion:
gsm8k 1.5B and arc 3B both have 49/50 chunks below zero and a single
positive last chunk that becomes L_final and zeros K_auc.

If the artifact diagnosis is correct, replacing `L_final` with
`mean(L_last_K)` or `min(L_last_K)` should:
  - leave the 1.5B Spearman correlation roughly intact (rankings already
    have enough net signal),
  - improve the 3B Spearman correlation toward zero or positive (rescuing
    arc and gsm8k from the K_auc=0 floor).

If smoothing barely moves either correlation, the L_final-anchoring story
is wrong and the 3B failure is substantive rather than a protocol artifact.

Inputs
------
- data/rollout_epiplexity_1.5b.json
- data/rollout_epiplexity_3b.json
- comparison.json (for per-dataset ablation MIXED transfer scores)

Outputs
-------
- recompute_script/recomputed_kauc.json     (all variants, both scales)
- recompute_script/correlations.csv         (Spearman table)

Run (from repo root):
    python recompute_script/recompute_kauc.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Callable, Iterable

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT_DIR = Path(__file__).resolve().parent

ROLLOUT_FILES = {
    "1.5b": DATA / "rollout_epiplexity_1.5b.json",
    "3b": DATA / "rollout_epiplexity_3b.json",
}
COMPARISON_FILE = REPO / "comparison.json"


# --------------------------------------------------------------------------
# Spearman (no scipy dependency — n=8, this is trivial)
# --------------------------------------------------------------------------

def _rank(values: list[float]) -> list[float]:
    """Average-rank assignment, ties get the mean of their tied positions."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx, ry = _rank(xs), _rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    dy = math.sqrt(sum((r - my) ** 2 for r in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


# --------------------------------------------------------------------------
# K_auc integrator with pluggable L_final
# --------------------------------------------------------------------------

def integrate_k_auc(
    losses: list[float],
    total_tokens: int,
    final_loss: float,
) -> float:
    """
    Trapezoidal integration of max(0, L_i - final_loss) over tokens-seen.
    Mirrors measure_rollout_epiplexity.integrate_k_auc, except that
    `final_loss` is supplied externally (instead of taking losses[-1]) so
    we can test alternative anchoring strategies.

    Per-chunk token counts aren't persisted in the JSON, so we approximate
    with uniform delta_tokens = total_tokens / num_chunks. This is exact
    when chunks have similar token counts (they should: each chunk is
    16 prompts x 8 generations of similar length).

    Returns K_auc in nats. Caller converts to bits with /ln(2).
    """
    n = len(losses)
    if n < 2 or total_tokens <= 0:
        return 0.0
    delta_tokens = total_tokens / n
    k_auc_nats = 0.0
    for i in range(1, n):
        avg_excess = ((losses[i - 1] - final_loss) + (losses[i] - final_loss)) / 2.0
        k_auc_nats += max(0.0, avg_excess) * delta_tokens
    return k_auc_nats


# --------------------------------------------------------------------------
# L_final anchoring strategies
# --------------------------------------------------------------------------

def anchor_last(losses: list[float]) -> float:
    return losses[-1]


def anchor_mean_last_k(k: int) -> Callable[[list[float]], float]:
    def fn(losses: list[float]) -> float:
        tail = losses[-k:] if len(losses) >= k else losses
        return sum(tail) / len(tail)
    fn.__name__ = f"mean_last_{k}"
    return fn


def anchor_min_last_k(k: int) -> Callable[[list[float]], float]:
    def fn(losses: list[float]) -> float:
        tail = losses[-k:] if len(losses) >= k else losses
        return min(tail)
    fn.__name__ = f"min_last_{k}"
    return fn


def anchor_min_overall(losses: list[float]) -> float:
    return min(losses)


def anchor_median_last_k(k: int) -> Callable[[list[float]], float]:
    def fn(losses: list[float]) -> float:
        tail = sorted(losses[-k:] if len(losses) >= k else losses)
        m = len(tail)
        return tail[m // 2] if m % 2 else (tail[m // 2 - 1] + tail[m // 2]) / 2.0
    fn.__name__ = f"median_last_{k}"
    return fn


ANCHORS: list[tuple[str, Callable[[list[float]], float]]] = [
    ("last (published)",   anchor_last),
    ("mean_last_3",        anchor_mean_last_k(3)),
    ("mean_last_5",        anchor_mean_last_k(5)),
    ("mean_last_10",       anchor_mean_last_k(10)),
    ("median_last_5",      anchor_median_last_k(5)),
    ("min_last_5",         anchor_min_last_k(5)),
    ("min_last_10",        anchor_min_last_k(10)),
    ("min_overall",        anchor_min_overall),
]


# --------------------------------------------------------------------------
# Transfer scores from comparison.json (ablation MIXED per training dataset)
# --------------------------------------------------------------------------

def load_transfer_scores() -> dict[str, float]:
    with open(COMPARISON_FILE) as f:
        cmp = json.load(f)
    out = {}
    for key, run in cmp["ablation"].items():
        ds = key.replace("ablation_", "")
        out[ds] = run["overall_mixed"]
    return out


def load_loss_curves(path: Path) -> dict[str, dict]:
    with open(path) as f:
        d = json.load(f)
    out = {}
    for ds, r in d["per_dataset"].items():
        if "error" in r or not r.get("loss_curve"):
            continue
        losses = [float(l) for _, l in r["loss_curve"]]
        out[ds] = {
            "losses": losses,
            "total_tokens": int(r["total_completion_tokens"]),
            "published_k_auc_bits": float(r.get("k_auc_bits", 0.0)),
            "published_per_token": float(r.get("rollout_epiplexity_per_token", 0.0)),
        }
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    transfer = load_transfer_scores()
    print(f"Transfer scores from comparison.json: {len(transfer)} datasets")
    for ds, t in sorted(transfer.items(), key=lambda x: -x[1]):
        print(f"  {ds:<10s}  {t:.4f}")

    all_results: dict[str, dict] = {}
    rows_for_csv: list[dict] = []

    for scale, path in ROLLOUT_FILES.items():
        if not path.exists():
            print(f"\n[skip] {scale}: {path} not found")
            continue
        print(f"\n{'='*70}\nScale: {scale}\n{'='*70}")
        curves = load_loss_curves(path)

        per_anchor: dict[str, dict] = {}
        for anchor_name, anchor_fn in ANCHORS:
            per_dataset_kauc_bits: dict[str, float] = {}
            per_dataset_per_token: dict[str, float] = {}
            for ds, c in curves.items():
                final = anchor_fn(c["losses"])
                k_auc_nats = integrate_k_auc(c["losses"], c["total_tokens"], final)
                k_auc_bits = k_auc_nats / math.log(2)
                per_token = k_auc_bits / max(c["total_tokens"], 1)
                per_dataset_kauc_bits[ds] = k_auc_bits
                per_dataset_per_token[ds] = per_token

            shared = sorted(set(per_dataset_kauc_bits) & set(transfer))
            xs_kauc   = [per_dataset_kauc_bits[d] for d in shared]
            xs_pertok = [per_dataset_per_token[d] for d in shared]
            ys        = [transfer[d] for d in shared]

            rho_kauc   = spearman(xs_kauc, ys)
            rho_pertok = spearman(xs_pertok, ys)

            shared_no_boolq = [d for d in shared if d != "boolq"]
            xs_kauc_nb   = [per_dataset_kauc_bits[d] for d in shared_no_boolq]
            xs_pertok_nb = [per_dataset_per_token[d] for d in shared_no_boolq]
            ys_nb        = [transfer[d] for d in shared_no_boolq]
            rho_kauc_nb   = spearman(xs_kauc_nb, ys_nb)
            rho_pertok_nb = spearman(xs_pertok_nb, ys_nb)

            per_anchor[anchor_name] = {
                "per_dataset_kauc_bits": per_dataset_kauc_bits,
                "per_dataset_per_token": per_dataset_per_token,
                "spearman_kauc_vs_transfer": rho_kauc,
                "spearman_pertok_vs_transfer": rho_pertok,
                "spearman_kauc_vs_transfer_no_boolq": rho_kauc_nb,
                "spearman_pertok_vs_transfer_no_boolq": rho_pertok_nb,
                "n_datasets": len(shared),
            }

            print(
                f"  anchor={anchor_name:<18s}  "
                f"rho(K_auc)={rho_kauc:+.3f}  rho(per_tok)={rho_pertok:+.3f}  "
                f"|  no boolq: rho(K_auc)={rho_kauc_nb:+.3f}  "
                f"rho(per_tok)={rho_pertok_nb:+.3f}"
            )

            rows_for_csv.append({
                "scale": scale,
                "anchor": anchor_name,
                "rho_kauc": f"{rho_kauc:+.4f}",
                "rho_per_token": f"{rho_pertok:+.4f}",
                "rho_kauc_no_boolq": f"{rho_kauc_nb:+.4f}",
                "rho_per_token_no_boolq": f"{rho_pertok_nb:+.4f}",
                "n_datasets": len(shared),
            })

        all_results[scale] = {
            "rollout_file": str(path.relative_to(REPO)),
            "datasets": list(curves.keys()),
            "per_anchor": per_anchor,
        }

    out_json = OUT_DIR / "recomputed_kauc.json"
    with open(out_json, "w") as f:
        json.dump({"transfer_scores": transfer, "by_scale": all_results}, f, indent=2)
    print(f"\nDetailed results: {out_json.relative_to(REPO)}")

    out_csv = OUT_DIR / "correlations.csv"
    with open(out_csv, "w", newline="") as f:
        if rows_for_csv:
            writer = csv.DictWriter(f, fieldnames=list(rows_for_csv[0].keys()))
            writer.writeheader()
            writer.writerows(rows_for_csv)
    print(f"Correlation table:  {out_csv.relative_to(REPO)}")

    print(f"\n{'='*70}\nSUMMARY: Spearman rho (K_auc raw bits vs ablation transfer)\n{'='*70}")
    print(f"{'anchor':<20s} {'1.5b':>10s} {'1.5b -boolq':>14s} {'3b':>10s} {'3b -boolq':>14s}")
    print("-" * 70)
    for anchor_name, _ in ANCHORS:
        cells = []
        for scale in ("1.5b", "3b"):
            blob = all_results.get(scale, {}).get("per_anchor", {}).get(anchor_name)
            if blob is None:
                cells.extend(["-", "-"])
            else:
                cells.append(f"{blob['spearman_kauc_vs_transfer']:+.3f}")
                cells.append(f"{blob['spearman_kauc_vs_transfer_no_boolq']:+.3f}")
        print(f"{anchor_name:<20s} {cells[0]:>10s} {cells[1]:>14s} {cells[2]:>10s} {cells[3]:>14s}")


if __name__ == "__main__":
    main()
