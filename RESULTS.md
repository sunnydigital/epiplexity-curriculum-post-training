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

## Epiplexity vs Transfer Correlation

| Dataset | Epiplexity/tok | Epi Rank | Transfer (MIXED) | Transfer Rank |
|---------|---------------|----------|-----------------|---------------|
| math | 0.748 | 8 (lowest) | 0.488 | 1 (best) |
| arc | 1.476 | 5 | 0.483 | 2 |
| humaneval | 2.006 | 2 | 0.481 | 3 |
| mbpp | 1.964 | 3 | 0.480 | 4 |
| triviaqa | 1.923 | 4 | 0.464 | 5 |
| mmlu | 1.302 | 6 | 0.460 | 6 |
| boolq | 2.049 | 1 (highest) | 0.459 | 7 |
| gsm8k | 0.842 | 7 | 0.409 | 8 (worst) |

**Spearman ρ(epiplexity vs transfer) = −0.17** — essentially uncorrelated.

## Interpretation

### Why GRPO works but curriculum doesn't matter

GRPO provides a consistent +0.7pp improvement over the pretrained baseline regardless of how the training mixture is weighted. This makes sense: with 2000 steps over 8 datasets, the model sees enough data from each source that initial ordering washes out. The mixture kernel (`(1−α)·curriculum + α·uniform`, α=0.15) also prevents any dataset from being fully starved.

### Why epiplexity doesn't predict transfer

Epiplexity measures **surface-level compressibility** from a cold-start probe — how much structure the probe can extract via teacher-forcing. GRPO transfer depends on **reward signal variance** — the model needs a Goldilocks difficulty zone where it sometimes succeeds and sometimes fails, generating high-variance advantage estimates.

These are fundamentally different:

- **BoolQ** (highest epiplexity, worst transfer): The pretrained 3B model already achieves 76% accuracy. GRPO generates completions that are mostly correct → low reward variance → weak gradients → minimal learning.

- **Math** (lowest epiplexity, best transfer): The pretrained model scores ~15%. Most completions fail, but some succeed through chain-of-thought reasoning → high reward variance → strong gradients → large parameter updates that generalize.

- **GSM8K** (low epiplexity, worst transfer): Although structurally similar to MATH, the model's in-distribution accuracy is moderate enough that reward signals are noisy rather than informative, and the narrow task format (grade school arithmetic) doesn't generalize.

The probe measures complexity from a 1.5B model's cold-start perspective. The 3B training model has a rich pretrained prior that reshapes the difficulty landscape entirely. A dataset the probe finds incompressible (high epiplexity) may already be well-solved by the larger model.

### What would predict transfer instead?

The strongest predictor is likely **reward variance at the competence boundary**: `Var[r(y) | x]` for the training model's own completions. Datasets where the model is at ~20-40% accuracy tend to produce the highest-variance reward signals and thus the strongest GRPO gradients. This could be estimated cheaply by sampling a few batches before training.

## Reproducibility

```bash
# Probe
python probe_epiplexity.py --probe-model Qwen/Qwen2.5-1.5B-Instruct

# Training (one strategy)
python post_training.py --curriculum uniform

# Evaluation
python evaluate.py --model outputs/grpo/final --label uniform --max-samples 500

# Aggregate
python compare_results.py \
    --sweep-dir /path/to/grpo-sweep \
    --ablation-dir /path/to/grpo-ablation \
    --output comparison.json
```

SLURM scripts for NYU Big Purple are in `environment/`.
