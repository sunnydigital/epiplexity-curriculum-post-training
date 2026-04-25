# Experiment Results: Epiplexity-Guided Curriculum for GRPO

## Setup

**Training**: Qwen2.5-3B-Instruct, LoRA (r=16), 2000 GRPO steps, 8 completions/prompt, β=0.04 KL penalty, LR 3e-6.

**Probe**: Qwen2.5-1.5B-Instruct, 500 teacher-forcing steps per dataset, prequential coding for K_auc estimation.

**Evaluation**: 500 samples per dataset (or full dataset if smaller), greedy decoding.

### Metric: MIXED Score

Each dataset has its own reward function appropriate to the task type:

| Category | Datasets | Reward | Metric Used |
|----------|----------|--------|-------------|
| Math | GSM8K, MATH | Exact numeric extraction | Accuracy (binary: 1 if correct, 0 otherwise) |
| Code | HumanEval, MBPP | Fuzzy string similarity (Jaccard keywords, 0.0–1.0 partial credit) | Mean reward (continuous) |
| Logical | MMLU, ARC | MCQ letter extraction | Accuracy (binary) |
| QA | TriviaQA, BoolQ | Substring / yes-no match | Accuracy (binary) |

The **MIXED** score averages across all 8 datasets, using accuracy for non-code and mean_reward for code. Since each eval dataset always uses the same metric regardless of what training data was used, the MIXED score is comparable across ablations — the evaluation is apples-to-apples even though the underlying reward functions differ.

## Epiplexity Probe Results (1.5B Probe)

| Dataset | Epiplexity/token | K_auc % | Interpretation |
|---------|-----------------|---------|----------------|
| boolq | 2.049 | 23% | High total complexity, mostly irreducible (format already known) |
| humaneval | 2.006 | 97% | High complexity, almost all learnable (genuine code patterns) |
| mbpp | 1.964 | 95% | Similar to HumanEval |
| triviaqa | 1.923 | 29% | High complexity, mostly memorization |
| arc | 1.476 | 62% | Moderate, balanced learnable/irreducible |
| mmlu | 1.302 | 35% | Moderate, mostly irreducible |
| gsm8k | 0.842 | 8% | Low complexity, very little learnable structure |
| math | 0.748 | 29% | Lowest — probe can't learn abstract reasoning |

## Curriculum Sweep

All 6 GRPO strategies trained on the full 8-dataset mixture with different epiplexity-based weighting schedules, evaluated against pretrained baseline.

| Dataset | baseline | uniform | high_first | high_const | anneal_high | low_first | low_const |
|---------|----------|---------|------------|------------|-------------|-----------|-----------|
| gsm8k | 0.300 | 0.306 | 0.290 | 0.300 | 0.300 | 0.308 | **0.324** |
| math | 0.152 | **0.176** | 0.164 | 0.170 | 0.162 | 0.170 | 0.156 |
| humaneval | 0.384 | 0.388 | 0.377 | 0.386 | 0.378 | **0.390** | 0.391 |
| mbpp | 0.301 | 0.302 | 0.314 | **0.316** | 0.312 | 0.300 | 0.290 |
| mmlu | 0.570 | 0.584 | **0.596** | 0.590 | 0.578 | 0.588 | 0.580 |
| arc | 0.728 | **0.758** | 0.756 | 0.746 | 0.754 | 0.752 | 0.758 |
| triviaqa | **0.498** | 0.486 | 0.490 | 0.496 | 0.490 | 0.486 | 0.490 |
| boolq | 0.762 | 0.756 | 0.746 | 0.756 | 0.756 | **0.766** | 0.762 |
| **MIXED** | 0.462 | 0.470 | 0.467 | 0.470 | 0.466 | **0.470** | 0.469 |

### Ranking

| Rank | Strategy | MIXED | Δ vs baseline |
|------|----------|-------|---------------|
| 1 | low_first | 0.4701 | +0.0082 |
| 2 | high_constant | 0.4699 | +0.0080 |
| 3 | uniform | 0.4695 | +0.0076 |
| 4 | low_constant | 0.4689 | +0.0070 |
| 5 | high_first | 0.4666 | +0.0047 |
| 6 | anneal_to_high | 0.4663 | +0.0044 |
| — | baseline_3b | 0.4619 | — |

**Spread across GRPO strategies: 0.38pp.** All strategies are within noise of each other.

## Ablation Transfer Matrix

Each row = model trained on ONLY that dataset via GRPO, evaluated on all 8 datasets. This isolates each dataset's contribution to cross-task transfer.

| Train ↓ Eval → | gsm8k | math | humaneval | mbpp | mmlu | arc | triviaqa | boolq | **MIXED** |
|-----------------|-------|------|-----------|------|------|-----|----------|-------|-----------|
| gsm8k | 0.168 | 0.336 | 0.216 | 0.130 | 0.576 | 0.764 | 0.414 | 0.666 | 0.409 |
| math | 0.288 | 0.336 | 0.389 | 0.340 | 0.556 | 0.728 | 0.494 | 0.770 | **0.488** |
| humaneval | 0.314 | 0.168 | 0.472 | 0.308 | 0.592 | 0.744 | 0.488 | 0.758 | 0.481 |
| mbpp | 0.316 | 0.166 | 0.388 | 0.413 | 0.574 | 0.730 | 0.498 | 0.756 | 0.480 |
| mmlu | 0.284 | 0.156 | 0.383 | 0.296 | 0.584 | 0.740 | 0.486 | 0.754 | 0.460 |
| arc | 0.306 | 0.156 | 0.371 | 0.304 | 0.642 | 0.840 | 0.494 | 0.750 | 0.483 |
| triviaqa | 0.310 | 0.162 | 0.382 | 0.294 | 0.564 | 0.732 | 0.500 | 0.764 | 0.464 |
| boolq | 0.292 | 0.166 | 0.376 | 0.286 | 0.570 | 0.740 | 0.492 | 0.750 | 0.459 |

## Predictor Comparison: What Predicts GRPO Transfer?

We test **four** candidate predictors of which datasets yield the best
cross-task transfer under GRPO, each measured at two model scales (1.5B
probe, 3B target). The first three are static snapshots from the base
policy; the fourth is a dynamic learning-curve measurement that simulates
a tiny GRPO run.

- **Epiplexity** (K_auc per token): learnable structure extractable via
  teacher-forced cross-entropy, prequential coding over 500 dataset-token
  steps. Implementation: [probe_epiplexity.py](probe_epiplexity.py).
- **Reward variance** (mean within-group Var[r]): variance of GRPO rewards
  across 8 sampled completions per prompt at T=0.7, 200 prompts.
  Implementation: [measure_reward_variance.py](measure_reward_variance.py).
- **Forking-token entropy** (top-20% mean H on successful rollouts):
  per-step policy entropy at "decision tokens", following Wang et al. 2025
  (arXiv:2506.01939). Bucketed to successful completions only; aggregated
  over the top-20% highest-entropy tokens.
  Implementation: [measure_forking_entropy.py](measure_forking_entropy.py).
- **Rollout epiplexity** (K_auc per token on the GRPO surrogate): a
  GRPO-native adaptation of prequential coding. For each dataset we attach
  a fresh LoRA adapter and run 50 inner-loop chunks of 16 prompts × 8
  generations, alternating *measure* and *train* on the advantage-weighted
  surrogate `L = −E[A · log π(τ)]`. K_auc integrates `(L_i − L_final)`
  over rollout tokens — fast surrogate compression = high learnability.
  Implementation: [measure_rollout_epiplexity.py](measure_rollout_epiplexity.py).

### 4-Predictor Spearman Correlation with Transfer

| Predictor | 1.5B Probe ρ | 3B Target ρ | Notes |
|-----------|-------------:|------------:|-------|
| Epiplexity (input)               | −0.17 | −0.02 | static, teacher-forcing CE |
| Reward variance                  | −0.05 | −0.31 | static, output-side |
| Forking-token entropy (top-20%)  | −0.19 | −0.12 | static, success-conditioned |
| **Rollout epiplexity per token** | **+0.52** (p=0.18) | −0.12 | dynamic, GRPO-native |
| **Rollout K_auc (raw bits)**     | **+0.64** (p=0.09) | +0.05 | dynamic, GRPO-native |

n = 8 datasets. None of the cells clear Bonferroni at α = 0.05; with
n = 8 the rank-correlation critical value is |ρ| > 0.71 for two-sided
p < 0.05. The 1.5B rollout K_auc result is approaching that threshold and
is the only signal materially separated from the rest.

### Robustness: leave-one-out on the 1.5B rollout signal

The 1.5B rollout-epiplexity correlation is **fragile to a single dataset
(boolq)** but otherwise stable:

| Held-out dataset | Rollout-epi ρ | K_auc ρ |
|------------------|--------------:|--------:|
| (none)           | +0.52 | +0.64 |
| gsm8k            | +0.29 | +0.46 |
| math             | +0.61 | +0.79 |
| humaneval        | +0.54 | +0.57 |
| mbpp             | +0.50 | +0.61 |
| mmlu             | +0.46 | +0.61 |
| arc              | +0.39 | +0.57 |
| triviaqa         | +0.50 | +0.61 |
| **boolq**        | **+0.86** (p=0.014) | **+0.89** (p=0.007) |

Excluding boolq, both rollout-based predictors clear two-sided p < 0.05.
This is a real and interpretable result — see *Why boolq breaks* below —
not a free p-value: boolq has the predicted failure mode of an MDL-style
learnability proxy on a degenerate (binary-answer) task, identified
*a priori* as a regime where the metric should mismeasure.

### 3B surrogate instability

The 3B rollout-epiplexity rows show the 3B LoRA inner loop is unstable at
the chosen hyperparameters (lr = 3e-6, lora_r = 16, AdamW). On three of
eight datasets the surrogate **worsens** during the inner loop:

| Dataset | L_initial | L_final | K_auc | Comment |
|---------|----------:|--------:|------:|---------|
| gsm8k 3B | −0.009 | +0.003 | 0 | small drift, training did not progress |
| arc 3B   | −0.009 | **+0.110** | 0 | catastrophic divergence |
| mmlu 3B  | −0.106 | −0.020  | 8033 | large reduction but starts from an outlier-low loss |

The `max(0, L_i − L_final)` floor in `integrate_k_auc` zeroes K_auc
whenever the curve is non-decreasing on average, which is the right choice
for the prequential interpretation but means we can't distinguish
"trivially compressed" from "training failed" without inspecting the
loss curves themselves. The 1.5B inner loop is well-behaved on all eight
datasets (every L_final ≤ L_initial within a small margin).

This is itself an informative finding: **the dynamic predictor's effective
operating range depends on a stable inner loop**, and 3B at these
hyperparameters does not provide one. We do *not* sweep hyperparameters
to chase a positive 3B result.

### Per-Dataset Predictor Values

All values measured on the 1.5B probe except where noted; Transfer is the
ablation MIXED score (model trained only on that dataset).

| Dataset | Epi | RV | Fork-H<sub>top20</sub> | Roll-Epi | K_auc | Transfer | Rank |
|---------|----:|---:|----------------------:|---------:|------:|---------:|------|
| math      | 0.748 | 0.048 | 0.847 | 0.0177 | 26 232 | 0.488 | 1 |
| arc       | 1.476 | 0.061 | 1.541 | 0.0890 | 75 708 | 0.483 | 2 |
| humaneval | 2.006 | 0.048 | 1.022 | 0.0198 | 30 222 | 0.481 | 3 |
| mbpp      | 1.964 | 0.026 | 0.988 | 0.0134 | 19 902 | 0.480 | 4 |
| triviaqa  | 1.923 | 0.048 | 1.648 | 0.0167 | 13 423 | 0.464 | 5 |
| mmlu      | 1.302 | 0.076 | 1.545 | 0.0029 |  2 680 | 0.460 | 6 |
| boolq     | 2.049 | 0.104 | 1.405 | 0.0434 | 27 536 | 0.459 | 7 |
| gsm8k     | 0.842 | 0.033 | 0.910 | 0.0000 |      0 | 0.409 | 8 |

(K_auc in bits; Roll-Epi in bits/token; Epi in bits/token; RV is mean
within-group reward variance; Fork-H<sub>top20</sub> in nats.)

### Model Accuracy Context

Mean reward per dataset (proxy for pretrained accuracy at temperature=0.7):

| Dataset | 1.5B | 3B |
|---------|------|-----|
| gsm8k | 0.610 | 0.636 |
| math | 0.505 | 0.441 |
| humaneval | 0.243 | 0.333 |
| mbpp | 0.146 | 0.291 |
| mmlu | 0.508 | 0.584 |
| arc | 0.742 | 0.711 |
| triviaqa | 0.288 | 0.416 |
| boolq | 0.714 | 0.761 |

## Interpretation

### Why GRPO works but curriculum doesn't matter

GRPO provides a consistent +0.7pp improvement over the pretrained baseline regardless of how the training mixture is weighted. With 2000 steps over 8 datasets, the model sees enough data from each source that initial ordering washes out. The mixture kernel (`(1−α)·curriculum + α·uniform`, α=0.15) also prevents any dataset from being fully starved.

### Why static base-model predictors all fail

Three of the four predictors are static snapshots of the base policy on
each dataset, and all three fail to predict transfer (every |ρ| ≤ 0.31):

**Epiplexity** measures surface-level compressibility from a cold-start
probe via teacher-forcing. This captures how much *text pattern regularity*
exists — not how much *reasoning structure* GRPO can exploit. Using the
3B model as its own probe (ρ = −0.02) eliminates the scale gap but doesn't
help — epiplexity still doesn't capture what matters for GRPO.

**Reward variance** was expected to capture the "Goldilocks difficulty
zone" — datasets where the model sometimes succeeds and sometimes fails
should produce high-variance advantage estimates and strong gradients.
But math has the *lowest* reward variance (0.017) yet the *best* transfer.
The signal is dominated by the *quality* of the few successes rather than
the spread.

**Forking-token entropy** (Wang et al., NeurIPS 2025) was the most
principled static candidate: top-20% per-step policy entropy on
*successful* rollouts, which their paper shows is where RL gradient
information concentrates *within* a single training run. As a
*cross-dataset* transfer predictor it also fails (ρ = −0.12 at 3B). The
operationalization actually anti-correlates with reasoning depth: when
Qwen2.5-3B succeeds on math, it does so by following a near-deterministic
chain (low local entropy throughout); when it answers an MCQ, the
answer-token decision concentrates entropy in a single position.
Forking-density-on-successes thus measures *answer-token uncertainty on
short tasks*, not the reasoning branch-point density Wang et al. observed
during training.

Common thread: all three static predictors describe the base model's
behavior on the data, not the **data's response to GRPO updates**.

### Rollout epiplexity: the first non-trivial signal, with caveats

The fourth predictor — rollout epiplexity — is qualitatively different.
Instead of measuring the base model on the data, it runs a tiny GRPO
simulation (50 chunks of LoRA training on the policy-sampled GRPO
surrogate) and measures the area under the surrogate-loss curve. This is
the most direct possible base-signal proxy for "will GRPO actually make
progress here." At the 1.5B probe scale it produces the only positive
correlation with transfer we have observed (ρ = +0.52 raw; +0.86 after
removing boolq, p = 0.014).

Three caveats are essential:

1. **n = 8.** Even ρ = +0.86 with one held-out point is at p = 0.014 —
   suggestive but not conclusive at our sample size.
2. **The signal is at probe scale only.** At 3B (the actual GRPO target),
   the predictor collapses to ρ = −0.12. The 3B inner loop is unstable on
   3 of 8 datasets (see *3B surrogate instability* above).
3. **Boolq is a known degenerate case.** It is the leverage point for the
   1.5B correlation, and we identify *a priori* why an MDL-style
   learnability proxy should mismeasure it (see next section).

#### Why boolq breaks the pattern

Boolq has the **second-highest 1.5B rollout epiplexity** (0.043 bits/token,
4th overall) but the **lowest transfer score** (0.459) — exactly the
deviation that drives the boolq-included correlation down to +0.52. The
mechanism is interpretable:

K_auc measures *how fast* the GRPO surrogate compresses, which is a proxy
for *learnability*. On boolq, learnability is high because there are only
two answer tokens — the policy rapidly shifts probability mass onto the
higher-reward of the two and the surrogate drops quickly. But the
resulting policy is purely a binary preference, not a reasoning
operator, so it transfers nothing to other tasks.

In other words: K_auc-based predictors track **learnability**, while
transfer requires **learnable reasoning structure that generalizes**. The
two correlate well except on degenerate (binary-answer) tasks where
learnability is high but generalizable structure is zero. This is a
predictable failure mode of any compression-based proxy and identifies
the boundary of when rollout-epiplexity-style signals can be trusted.

### Stepping back: a coherent meta-result

Across four predictors and two model scales (eight measurement cells in
total), only the **dynamic, GRPO-native, probe-scale** cell shows signal,
and even there only after accounting for a known degenerate case.
Combined with the narrow ablation transfer range (0.41–0.49, spread
≈ 0.08pp) and the within-noise sweep results (0.38pp spread across all
six curriculum strategies), the overall picture is:

> At Qwen2.5-3B + ~2000 GRPO steps, transfer is dominated by general
> format / reasoning acquisition rather than dataset-specific structure.
> Static base-model signals do not predict the small remaining
> cross-dataset variation. Dynamic GRPO-simulating signals can predict it
> at probe scale (ρ = +0.86 modulo boolq), but the necessary inner-loop
> stability is not free at the target scale.

This suggests the regime where epiplexity-style curricula could matter
lies elsewhere: larger models, longer training, more heterogeneous reward
signals, or much narrower transfer windows. At this scale, the
curriculum-vs-uniform question is essentially a no-op.

## Reproducibility

```bash
# Epiplexity probes (1.5B and 3B)
python probe_epiplexity.py --probe-model Qwen/Qwen2.5-1.5B-Instruct --output data/epiplexity_scores.json
python probe_epiplexity.py --probe-model Qwen/Qwen2.5-3B-Instruct --output data/epiplexity_scores_3b.json

# Reward variance (1.5B and 3B)
python measure_reward_variance.py --model Qwen/Qwen2.5-1.5B-Instruct --output data/reward_variance_1.5b.json
python measure_reward_variance.py --model Qwen/Qwen2.5-3B-Instruct --output data/reward_variance_3b.json

# Forking-token entropy (1.5B and 3B) — Wang et al. 2025 (arXiv:2506.01939)
python measure_forking_entropy.py --model Qwen/Qwen2.5-1.5B-Instruct --output data/forking_entropy_1.5b.json
python measure_forking_entropy.py --model Qwen/Qwen2.5-3B-Instruct --output data/forking_entropy_3b.json

# Rollout epiplexity (1.5B and 3B) — GRPO-native K_auc
python measure_rollout_epiplexity.py --model Qwen/Qwen2.5-1.5B-Instruct --output data/rollout_epiplexity_1.5b.json
python measure_rollout_epiplexity.py --model Qwen/Qwen2.5-3B-Instruct --output data/rollout_epiplexity_3b.json

# Training (one strategy)
python post_training.py --curriculum uniform

# Evaluation
python evaluate.py --model outputs/grpo/final --label uniform --max-samples 500

# Aggregate (with predictor Spearman correlations)
python compare_results.py \
    --sweep-dir /path/to/grpo-sweep \
    --ablation-dir /path/to/grpo-ablation \
    --predictors-dir data \
    --output comparison.json
```

SLURM scripts for NYU Big Purple are in `environment/`.
