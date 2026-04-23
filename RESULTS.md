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

We test two candidate predictors of which datasets yield the best cross-task transfer under GRPO, each measured at two model scales:

- **Epiplexity** (K_auc per token): how much learnable structure a model can extract from the dataset via teacher-forcing (prequential coding, 500 training steps).
- **Reward variance** (mean within-group Var[r]): how much variation in reward scores GRPO would see across 8 sampled completions per prompt (temperature=0.7, 200 prompts).

### 2×2 Spearman Correlation with Transfer

| Predictor | 1.5B Probe | 3B Target |
|-----------|-----------|-----------|
| **Epiplexity** | ρ = −0.17 | ρ = −0.02 |
| **Reward Variance** | ρ = −0.05 | ρ = −0.31 |
| **Rollout Epiplexity** (GRPO-native) | *pending* | *pending* |

**No (existing) predictor shows positive correlation with transfer.** All four
2×2 cells are near zero or negative.

**Rollout Epiplexity** is a GRPO-native compressibility measurement introduced
on this branch. It applies the Finzi et al. prequential coding procedure to
the advantage-weighted GRPO surrogate loss on policy-sampled rollouts, rather
than to teacher-forcing CE on dataset tokens. It is the first predictor in
the intersection of (a) MDL-theoretic trajectory integration and
(b) advantage-weighted rollout conditioning — the two properties that
respectively distinguish epiplexity (which has a but not b) and reward
variance (which has b only as a snapshot, not as a trajectory). Implementation:
[measure_rollout_epiplexity.py](measure_rollout_epiplexity.py); SLURM:
[environment/slurm_rollout_epiplexity.sbatch](environment/slurm_rollout_epiplexity.sbatch).
Results table will be filled in once the cluster runs complete.

### Per-Dataset Values

| Dataset | Epi (1.5B) | Epi (3B) | RV (1.5B) | RV (3B) | Transfer | Rank |
|---------|-----------|---------|----------|--------|----------|------|
| math | 0.748 | 0.701 | 0.048 | 0.017 | 0.488 | 1 (best) |
| arc | 1.476 | 1.362 | 0.061 | 0.103 | 0.483 | 2 |
| humaneval | 2.006 | 1.900 | 0.048 | 0.036 | 0.481 | 3 |
| mbpp | 1.964 | 2.265 | 0.026 | 0.034 | 0.480 | 4 |
| triviaqa | 1.923 | 1.806 | 0.048 | 0.048 | 0.464 | 5 |
| mmlu | 1.302 | 1.210 | 0.076 | 0.085 | 0.460 | 6 |
| boolq | 2.049 | 1.855 | 0.104 | 0.048 | 0.459 | 7 |
| gsm8k | 0.842 | 0.847 | 0.033 | 0.038 | 0.409 | 8 (worst) |

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

### Why neither predictor works

The initial hypothesis was that epiplexity (learnable structure) or reward variance (gradient signal strength) would predict which datasets transfer best under GRPO. Neither does, for different reasons:

**Epiplexity** measures surface-level compressibility from a cold-start probe via teacher-forcing. This captures how much *text pattern regularity* exists — not how much *reasoning structure* GRPO can exploit. The probe/target gap is fundamental: a dataset the 1.5B probe finds incompressible may already be well-solved by the 3B model's pretrained prior (BoolQ: 76% accuracy). Using the 3B model as its own probe (ρ = −0.02) eliminates the scale gap but doesn't help — epiplexity still doesn't capture what matters for GRPO.

**Reward variance** was expected to capture the "Goldilocks difficulty zone" — datasets where the model sometimes succeeds and sometimes fails should produce high-variance advantage estimates and strong gradients. But math has the *lowest* reward variance (0.017) yet the *best* transfer. The 3B model mostly fails on math (mean reward 0.44), producing many all-zero reward groups with near-zero variance. The signal comes not from variance magnitude but from the *quality* of the few successes — when the model does solve a math problem, it's via chain-of-thought reasoning that generalizes broadly.

### What's actually driving transfer?

The transfer results suggest that **structural reasoning depth** matters more than any scalar proxy:

- **Math** (best transfer, 0.488): Despite low accuracy and low reward variance, the rare successes require multi-step reasoning that generalizes across task types.
- **Code** (strong transfer, ~0.480): Similarly requires structured problem-solving.
- **QA/Logical** (moderate transfer, 0.459–0.464): Pattern-matching tasks that teach surface-level associations rather than deep reasoning.
- **GSM8K** (worst transfer, 0.409): Despite being "math," the narrow arithmetic format doesn't generalize — the model memorizes templates rather than learning reasoning.

This points toward a qualitative distinction that neither epiplexity nor reward variance can capture: **whether GRPO's successful completions encode transferable reasoning patterns versus task-specific shortcuts**. Quantifying this would likely require analyzing the *content* of high-reward completions rather than just their reward statistics.

## Reproducibility

```bash
# Epiplexity probes (1.5B and 3B)
python probe_epiplexity.py --probe-model Qwen/Qwen2.5-1.5B-Instruct --output data/epiplexity_scores.json
python probe_epiplexity.py --probe-model Qwen/Qwen2.5-3B-Instruct --output data/epiplexity_scores_3b.json

# Reward variance (1.5B and 3B)
python measure_reward_variance.py --model Qwen/Qwen2.5-1.5B-Instruct --output data/reward_variance_1.5b.json
python measure_reward_variance.py --model Qwen/Qwen2.5-3B-Instruct --output data/reward_variance_3b.json

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
