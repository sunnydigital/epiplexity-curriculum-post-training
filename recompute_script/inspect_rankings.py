"""
Inspect per-dataset K_auc rankings under each anchor, side-by-side with
ablation transfer scores. The aggregate Spearman rho hides which datasets
are responsible for the sign — print the rankings so we can see directly.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "recomputed_kauc.json"

with open(OUT) as f:
    d = json.load(f)

transfer = d["transfer_scores"]
ds_order = sorted(transfer, key=lambda x: -transfer[x])

def rank_lookup(values: dict[str, float]) -> dict[str, int]:
    ordered = sorted(values, key=lambda x: -values[x])
    return {ds: i + 1 for i, ds in enumerate(ordered)}

transfer_rank = rank_lookup(transfer)

ANCHORS_TO_SHOW = [
    "last (published)",
    "mean_last_5",
    "min_last_5",
    "min_overall",
]

for scale in ("1.5b", "3b"):
    blob = d["by_scale"].get(scale)
    if blob is None:
        continue
    print(f"\n{'='*100}")
    print(f"Scale: {scale}")
    print(f"{'='*100}")
    print(f"{'dataset':<10s} {'transfer':>9s} {'tRk':>4s}", end="")
    for a in ANCHORS_TO_SHOW:
        print(f"  | {a[:16]:<16s} {'rk':>3s}", end="")
    print()
    print("-" * 100)
    for ds in ds_order:
        t = transfer[ds]
        print(f"{ds:<10s} {t:>9.4f} {transfer_rank[ds]:>4d}", end="")
        for a in ANCHORS_TO_SHOW:
            kauc = blob["per_anchor"][a]["per_dataset_kauc_bits"][ds]
            r = rank_lookup(blob["per_anchor"][a]["per_dataset_kauc_bits"])[ds]
            print(f"  | {kauc:>16,.0f} {r:>3d}", end="")
        print()

print()
print("=" * 100)
print("Key: tRk = transfer rank (1=best transfer). rk = K_auc rank under that anchor.")
print("=" * 100)
