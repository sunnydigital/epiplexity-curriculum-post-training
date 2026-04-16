# Epiplexity-Guided Curriculum for GRPO Post-Training

This repository implements an epiplexity-weighted curriculum learning pipeline for Group Relative Policy Optimization (GRPO) post-training of language models. The core question: **can epiplexity scores from a small probe model predict which datasets will yield the best GRPO transfer learning?**

## Overview

We use [epiplexity](https://arxiv.org/abs/2601.03220) (Finzi et al., 2026) — a two-part description length measuring how learnable a dataset is from the perspective of a given model — to weight curriculum sampling during GRPO training. A small probe model (Qwen2.5-1.5B-Instruct) measures epiplexity across 8 benchmark datasets, and those scores guide how a larger training model (Qwen2.5-3B-Instruct) samples from the mixture during GRPO.

## Pipeline

```
probe_epiplexity.py          →  data/epiplexity_scores.json
                                        ↓
post_training.py (GRPO)      ←  data/curriculum.py (scheduling)
                                        ↓
evaluate.py                  →  *_results.json (per-strategy)
                                        ↓
compare_results.py           →  comparison.json (aggregated)
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

- **Training target**: `Qwen/Qwen2.5-3B-Instruct` with LoRA (r=16, q_proj/v_proj)
- **Epiplexity probe**: `Qwen/Qwen2.5-1.5B-Instruct` (teacher-forcing CLM loss)

## Curriculum Strategies

Six GRPO strategies tested (+ pretrained baseline):
- `uniform` — equal weights across all datasets
- `high_first` — start with high-epiplexity datasets, anneal toward uniform
- `low_first` — start with low-epiplexity, anneal toward uniform
- `high_constant` — static high-epiplexity weighting throughout
- `low_constant` — static low-epiplexity weighting throughout
- `anneal_to_high` — start uniform, anneal toward high-epiplexity

## Results

See [RESULTS.md](RESULTS.md) for the full experiment writeup with tables and analysis.

**TL;DR**: GRPO improves over baseline (+0.7pp avg), but curriculum ordering has negligible effect (0.4pp spread). Epiplexity does not predict GRPO transfer (ρ = −0.17). Reward signal variance is the dominant factor.

## Environment

- **Local**: UV (`pyproject.toml`)
- **Cluster**: Conda on NYU Big Purple (A100s), SLURM scripts in `environment/`

## Usage

```bash
# 1. Measure epiplexity
python probe_epiplexity.py --probe-model Qwen/Qwen2.5-1.5B-Instruct

# 2. Run GRPO training with curriculum
python post_training.py --curriculum high_first --temp-start 2.0 --temp-end 0.5

# 3. Evaluate
python evaluate.py --model outputs/grpo/final --label my_run --max-samples 500

# 4. Aggregate results
python compare_results.py --sweep-dir /path/to/sweep --ablation-dir /path/to/ablation
```

## References

- Finzi, M. et al. (2026). *Epiplexity: Measuring the Two-Part Compressibility of a Dataset With Respect to a Model Class.* arXiv:2601.03220
- Shao, Z. et al. (2024). *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* (GRPO)