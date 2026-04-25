# Epiplexity-Guided Curriculum for GRPO Post-Training

This repository implements an epiplexity-weighted curriculum learning pipeline for Group Relative Policy Optimization (GRPO) post-training of language models. The core question: **can epiplexity scores from a small probe model predict which datasets will yield the best GRPO transfer learning?**

## Overview

We use [epiplexity](https://arxiv.org/abs/2601.03220) (Finzi et al., 2026) — a two-part description length measuring how learnable a dataset is from the perspective of a given model — to weight curriculum sampling during GRPO training. A small probe model (Qwen2.5-1.5B-Instruct) measures epiplexity across 8 benchmark datasets, and those scores guide how a larger training model (Qwen2.5-3B-Instruct) samples from the mixture during GRPO.

**Key finding**: GRPO improves performance by +0.7pp over baseline, but curriculum strategy has minimal effect (0.4pp spread across 6 strategies). Epiplexity shows essentially zero correlation with transfer (Spearman ρ = −0.17). The dominant factor is **reward variance** — datasets where the model sits at 20-40% accuracy produce high-variance reward signals and the strongest GRPO gradients.

## Pipeline

```
probe_epiplexity.py             →  data/epiplexity_scores.json
                                           ↓
post_training.py (GRPO)         ←  data/curriculum.py (scheduling)
                                           ↓
evaluate.py                     →  outputs/eval/*_results.json
                                           ↓
compare_results.py              →  comparison.json (aggregated)
                                           ↓
measure_reward_variance.py      →  data/reward_variance_3b.json
                                           ↓
measure_rollout_epiplexity.py   →  data/rollout_epiplexity_3b.json
```

## Project Structure

```
├── configs/
│   ├── datasets_config.yaml         # Dataset sampling & floor weights
│   └── training_config.yaml         # GRPO hyperparameters
├── data/
│   ├── registry.py                  # Dataset registry (HF paths, splits, formatters)
│   ├── datasets.py                  # 8 dataset formatters (prompt + answer extraction)
│   ├── curriculum.py                # CurriculumScheduler (6 weighting strategies)
│   ├── loader.py                    # EpiplexityWeightedSampler
│   └── epiplexity_scores.json       # Measured per-dataset epiplexity scores
├── rewards/
│   ├── __init__.py                  # dispatch_reward() router + RewardTracker
│   ├── math.py                      # Numeric extraction & comparison (GSM8K, MATH)
│   ├── code.py                      # Multi-tier fuzzy matching (HumanEval, MBPP)
│   └── qa.py                        # MCQ letter, substring, yes/no matching
├── environment/
│   ├── environment.yml              # Conda environment spec
│   ├── slurm_job.sbatch             # Single GRPO training run
│   ├── slurm_sweep.sbatch           # Grid sweep across all 6 strategies
│   ├── slurm_ablation.sbatch        # Single-dataset ablation runs
│   ├── slurm_probe.sbatch           # Epiplexity probe (1.5B)
│   ├── slurm_probe_3b.sbatch        # Epiplexity probe (3B)
│   ├── slurm_reward_variance.sbatch # Reward variance measurement
│   └── slurm_rollout_epiplexity.sbatch # Rollout epiplexity (GRPO-native K_auc)
├── probe_epiplexity.py              # Stage 1: measure per-dataset epiplexity via K_auc
├── post_training.py                 # Stage 2: GRPO training with curriculum scheduler
├── evaluate.py                      # Stage 3: evaluate on all 8 benchmarks
├── compare_results.py               # Stage 4: aggregate results + Spearman vs predictors
├── measure_reward_variance.py       # Reward variance analysis
├── measure_rollout_epiplexity.py    # Rollout epiplexity: prequential K_auc on GRPO surrogate
├── RESULTS.md                       # Full experiment writeup & analysis
└── pyproject.toml                   # Python dependencies (UV)
```

## Datasets

| Category | Dataset | Split | Metric |
|----------|---------|-------|--------|
| Math | GSM8K | test | Accuracy (exact numeric match) |
| Math | MATH (algebra) | test | Accuracy (exact numeric match) |
| Code | HumanEval | test | Mean reward (fuzzy string similarity) |
| Code | MBPP (sanitized) | test | Mean reward (fuzzy string similarity) |
| Logical | MMLU | validation | Accuracy (MCQ letter match) |
| Logical | ARC-Challenge | test | Accuracy (MCQ letter match) |
| QA | TriviaQA | validation | Accuracy (substring match) |
| QA | BoolQ | validation | Accuracy (yes/no match) |

## Models

- **Training target**: `Qwen/Qwen2.5-3B-Instruct` with LoRA (r=16, alpha=32, targets: q_proj/v_proj)
- **Epiplexity probe**: `Qwen/Qwen2.5-1.5B-Instruct` (teacher-forcing CLM loss, 500 steps per dataset)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| GRPO steps | 2000 |
| Batch size | 2 (x8 gradient accumulation = 16 effective) |
| Generations per prompt | 8 |
| Learning rate | 3e-6 (cosine schedule, 3% warmup) |
| KL penalty (beta) | 0.04 |
| Precision | bfloat16 |
| Max prompt / completion length | 512 / 256 |

## Curriculum Strategies

Six epiplexity-based weighting strategies tested (+ pretrained baseline). All use a mixture kernel `(1−alpha) * curriculum + alpha * uniform` (alpha=0.15) to prevent dataset starvation.

| Strategy | Description |
|----------|-------------|
| `uniform` | Equal weights across all datasets |
| `high_first` | Start with high-epiplexity, anneal toward uniform |
| `low_first` | Start with low-epiplexity, anneal toward uniform |
| `high_constant` | Static high-epiplexity weighting throughout |
| `low_constant` | Static low-epiplexity weighting throughout |
| `anneal_to_high` | Start uniform, anneal toward high-epiplexity |

## Results

See [RESULTS.md](RESULTS.md) for the full experiment writeup with tables and analysis.

### Epiplexity Scores (1.5B Probe)

| Dataset | Epiplexity/token | K_auc % |
|---------|-----------------|---------|
| boolq | 2.049 | 23% |
| humaneval | 2.006 | 97% |
| mbpp | 1.964 | 95% |
| triviaqa | 1.923 | 29% |
| arc | 1.476 | 62% |
| mmlu | 1.302 | 35% |
| gsm8k | 0.842 | 8% |
| math | 0.748 | 29% |

### Curriculum Sweep

| Rank | Strategy | MIXED Score | Delta vs baseline |
|------|----------|-------------|-------------------|
| 1 | low_first | 0.4701 | +0.82pp |
| 2 | high_constant | 0.4699 | +0.80pp |
| 3 | uniform | 0.4695 | +0.76pp |
| 4 | low_constant | 0.4689 | +0.70pp |
| 5 | high_first | 0.4666 | +0.47pp |
| 6 | anneal_to_high | 0.4663 | +0.44pp |
| -- | baseline_3b | 0.4619 | -- |

All strategies within 0.38pp of each other — curriculum ordering has negligible effect.

### Ablation Transfer (Single-Dataset Training)

| Train Dataset | MIXED Score | Interpretation |
|---------------|-------------|----------------|
| math | 0.488 | Best transfer (lowest epiplexity, highest reward variance) |
| arc | 0.483 | Strong transfer |
| humaneval | 0.481 | Strong transfer |
| mbpp | 0.480 | Strong transfer |
| triviaqa | 0.464 | Moderate |
| mmlu | 0.460 | Moderate |
| boolq | 0.459 | Worst transfer (highest epiplexity, lowest reward variance) |
| gsm8k | 0.409 | Outlier — narrow task format doesn't generalize |

**Epiplexity vs Transfer**: Spearman rho = -0.17 (uncorrelated). The strongest predictor of GRPO transfer is **reward variance at the competence boundary** — datasets where the model achieves ~20-40% accuracy produce the most informative gradient signal.

## Environment

- **Python**: >= 3.11
- **Local**: UV package manager (`pyproject.toml`)
- **Cluster**: Conda on NYU Big Purple (A100 GPUs), SLURM scripts in `environment/`

### Dependencies

`torch`, `transformers`, `trl` (GRPO trainer), `datasets`, `accelerate`, `peft` (LoRA), `sentencepiece`, `evaluate`, `PyYAML`

Install locally:
```bash
uv sync
```

## Usage

```bash
# 1. Measure epiplexity (1.5B probe, all 8 datasets)
python probe_epiplexity.py \
    --probe-model Qwen/Qwen2.5-1.5B-Instruct \
    --output data/epiplexity_scores.json \
    --train-steps 500

# 2. Run GRPO training with a curriculum strategy
python post_training.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --scores-path data/epiplexity_scores.json \
    --config configs/training_config.yaml \
    --output-dir outputs/grpo \
    --curriculum high_first \
    --temp-start 0.1 --temp-end 5.0 \
    --use-lora

# 3. Evaluate trained model on all benchmarks
python evaluate.py \
    --model outputs/grpo/final \
    --output-dir outputs/eval \
    --label high_first \
    --max-samples 500

# 4. Aggregate results across strategies
python compare_results.py \
    --sweep-dir outputs/grpo-sweep \
    --ablation-dir outputs/grpo-ablation \
    --output comparison.json

# 5. (Optional) Measure reward variance
python measure_reward_variance.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-generations 8 \
    --max-samples 200 \
    --output data/reward_variance_3b.json

# 6. (Optional) Measure rollout epiplexity — GRPO-native K_auc
python measure_rollout_epiplexity.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-chunks 50 \
    --prompts-per-chunk 16 \
    --num-generations 8 \
    --output data/rollout_epiplexity_3b.json
```

**Rollout epiplexity** applies the prequential coding procedure to the
advantage-weighted GRPO surrogate loss on policy-sampled rollouts, instead
of teacher-forcing CE on dataset tokens. It is the first measurement that
combines (a) MDL-theoretic trajectory integration with (b) advantage-weighted
rollout conditioning — closing the conceptual gap that makes vanilla
epiplexity an off-objective predictor of GRPO transfer.

For cluster runs, see the SLURM scripts in `environment/`.

## References

- Finzi, M. et al. (2026). *Epiplexity: Measuring the Two-Part Compressibility of a Dataset With Respect to a Model Class.* arXiv:2601.03220
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* (GRPO)