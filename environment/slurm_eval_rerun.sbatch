#!/bin/bash -l
#SBATCH --job-name=eval-rerun
#SBATCH --output=/gpfs/scratch/gs4342/logs/eval-rerun_%j.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/eval-rerun_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Re-run evaluation only on all existing checkpoints (no training).
# Use this after fixing rewards/qa.py to get correct MMLU scores.

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

# Ablations
for DATASET in gsm8k math humaneval mbpp mmlu arc triviaqa boolq; do
    echo "=== Evaluating ablation: $DATASET ==="
    python evaluate.py \
        --model /gpfs/scratch/gs4342/grpo-ablation/${DATASET}/final \
        --output-dir /gpfs/scratch/gs4342/grpo-ablation/${DATASET}/eval \
        --label "ablation_${DATASET}" \
        --max-samples 500 \
        --batch-size 16
done

# Sweep strategies
for STRATEGY in uniform high_first high_constant anneal_to_high low_first low_constant; do
    echo "=== Evaluating strategy: $STRATEGY ==="
    python evaluate.py \
        --model /gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/final \
        --output-dir /gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/eval \
        --label "${STRATEGY}" \
        --max-samples 500 \
        --batch-size 16
done

# Baseline (pretrained, no GRPO)
echo "=== Evaluating baseline ==="
python evaluate.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output-dir /gpfs/scratch/gs4342/grpo-sweep/baseline/eval \
    --label baseline_3b \
    --max-samples 500 \
    --batch-size 16

echo "All evaluations complete."

# Aggregate results
python compare_results.py \
    --sweep-dir /gpfs/scratch/gs4342/grpo-sweep \
    --ablation-dir /gpfs/scratch/gs4342/grpo-ablation \
    --output /gpfs/scratch/gs4342/comparison_v2.json

echo "comparison_v2.json written."
