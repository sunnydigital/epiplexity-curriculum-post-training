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
# Re-run evaluation on all existing checkpoints and regenerate comparison.json.
# Deletes stale result files first to force fresh evaluation.

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

# Delete all stale result files so everything gets regenerated
echo "Cleaning stale result files..."
find /gpfs/scratch/gs4342/grpo-sweep -name "*_results.json" -delete 2>/dev/null || true
find /gpfs/scratch/gs4342/grpo-ablation -name "*_results.json" -delete 2>/dev/null || true

# Baseline (pretrained, no GRPO)
echo "=== Evaluating baseline ==="
mkdir -p /gpfs/scratch/gs4342/grpo-sweep/baseline/eval
python evaluate.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output-dir /gpfs/scratch/gs4342/grpo-sweep/baseline/eval \
    --label baseline_3b \
    --max-samples 500 \
    --batch-size 16

# Sweep strategies
for STRATEGY in uniform high_first high_constant anneal_to_high low_first low_constant; do
    CKPT="/gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/final"
    if [ -d "$CKPT" ]; then
        echo "=== Evaluating strategy: $STRATEGY ==="
        mkdir -p /gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/eval
        python evaluate.py \
            --model "$CKPT" \
            --output-dir /gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/eval \
            --label "${STRATEGY}" \
            --max-samples 500 \
            --batch-size 16
    else
        echo "SKIP: $CKPT not found"
    fi
done

# Ablations
for DATASET in gsm8k math humaneval mbpp mmlu arc triviaqa boolq; do
    CKPT="/gpfs/scratch/gs4342/grpo-ablation/${DATASET}/final"
    if [ -d "$CKPT" ]; then
        echo "=== Evaluating ablation: $DATASET ==="
        mkdir -p /gpfs/scratch/gs4342/grpo-ablation/${DATASET}/eval
        python evaluate.py \
            --model "$CKPT" \
            --output-dir /gpfs/scratch/gs4342/grpo-ablation/${DATASET}/eval \
            --label "ablation_${DATASET}" \
            --max-samples 500 \
            --batch-size 16
    else
        echo "SKIP: $CKPT not found"
    fi
done

# Generate comparison.json in the repo directory
echo "=== Generating comparison.json ==="
python compare_results.py \
    --sweep-dir /gpfs/scratch/gs4342/grpo-sweep \
    --ablation-dir /gpfs/scratch/gs4342/grpo-ablation \
    --output "$WORK/comparison.json"

echo "All evaluations complete. Results in $WORK/comparison.json"
