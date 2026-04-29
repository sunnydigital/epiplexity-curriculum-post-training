# Anchor Sensitivity of Rollout K_auc

This note documents how the headline rollout-K_auc → transfer correlation
depends on the choice of `L_final` anchor used to integrate the per-chunk
loss curve. It exists to give a single-page reference for the result that
was previously slated for an appendix slide in the findings deck.

## Background

K_auc is the prequential code length integrated against a chosen end-of-curve
loss `L_final`:

```
K_auc = Σ_chunks (L_chunk − L_final) × tokens_chunk
```

The published anchor takes `L_final` to be the loss measured on the **last
chunk** of the rollout. This follows the canonical prequential definition
(Finzi et al.) and is what the original `integrate_k_auc` implementation
computes. Alternative anchors (window means, medians, mins) are post-hoc
robustness checks performed *after* the published number was reported, by
re-deriving K_auc from the persisted `loss_curve` fields without rerunning
GPU work.

## Result

Spearman ρ between K_auc and ablation transfer score, computed across the
8 datasets in the suite (n=7 with boolq excluded). All numbers are
reproducible by running [`recompute_script/recompute_kauc.py`](../recompute_script/recompute_kauc.py).

| Anchor                        | 1.5B (n=8) | 1.5B − boolq (n=7) | 3B (n=8) | 3B − boolq (n=7) |
|-------------------------------|-----------:|-------------------:|---------:|-----------------:|
| **last (published)**          |  **+0.643** |        **+0.893** |   +0.048 |           −0.018 |
| mean of last 3                |     +0.452 |            +0.786 |   −0.238 |           −0.250 |
| mean of last 5                |     +0.190 |            +0.571 |   −0.357 |           −0.500 |
| mean of last 10               |     +0.167 |            +0.321 |   +0.000 |           −0.250 |
| median of last 5              |     +0.214 |            +0.607 |   −0.048 |           −0.071 |
| min of last 5                 |     +0.381 |            +0.679 |   +0.071 |           +0.000 |
| min of last 10                |     +0.214 |            +0.286 |   −0.024 |           −0.179 |
| min over full curve           |     +0.381 |            +0.500 |   +0.095 |           −0.107 |

## What this says

**At 1.5B (probe scale):**
- The sign of the correlation is positive across **all eight** anchor
  choices, both with and without boolq. The magnitude varies from +0.17 to
  +0.64 (with boolq) and from +0.29 to +0.89 (without boolq).
- Only the published `last` anchor reaches the n=7 Spearman significance
  threshold of |ρ| ≈ 0.786 cleanly. `mean_last_3` is right at the threshold.
  Window-averaged anchors are suggestive but not significant at p < 0.05.
- The published anchor is the *most extreme* of the choices. Smoothing the
  end-of-curve loss reduces the correlation, it does not preserve it.

**At 3B (target scale):**
- No anchor recovers a positive correlation. The best no-boolq value
  is exactly 0.000 (`min_last_5`).
- The probe-scale → target-scale failure is therefore **not** an artifact
  of single-chunk anchor noise. Smoothing `L_final` does not rescue 3B.

## What this means for the headline

The headline number "ρ = +0.89 at 1.5B without boolq" survives in **sign**
across every reasonable anchor, but its **magnitude** depends on the anchor
choice, and statistical significance depends on choosing the most extreme
anchor in the family.

The honest one-sentence summary: *the published correlation is real but
amplified by the choice of integrator anchor; a more conservative anchor
(window mean) shows a weaker, suggestive-but-not-significant version of the
same positive trend.*

## More importantly: this is not the binding robustness check

Anchor sensitivity is a fine-grained sensitivity test inside one estimator.
The far more important untested robustness check is **multi-seed stability**:
all numbers in the experiment are from a single seed at n=8 datasets. A
follow-up that reruns the rollout-epiplexity pipeline under three or more
seeds would dominate this analysis in informativeness.

## Source of truth

- **Canonical writeup:** [`FINAL_RESULTS.md`](../FINAL_RESULTS.md), §5.
- **Recomputation script:** [`recompute_script/recompute_kauc.py`](../recompute_script/recompute_kauc.py).
- **Persisted curves:** [`data/rollout_epiplexity_1.5b.json`](../data/rollout_epiplexity_1.5b.json),
  [`data/rollout_epiplexity_3b.json`](../data/rollout_epiplexity_3b.json).
- **Tabular output:** [`recompute_script/correlations.csv`](../recompute_script/correlations.csv).
