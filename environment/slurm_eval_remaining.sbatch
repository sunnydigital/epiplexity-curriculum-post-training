#!/bin/bash -l
#SBATCH --job-name=eval-remaining
#SBATCH --output=/gpfs/scratch/gs4342/logs/eval-remaining_%j.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/eval-remaining_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Run only the 4 missing evaluations (anneal_to_high, low_first, low_constant, baseline).
# The other 11 evals already completed successfully.

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

# Missing sweep strategies
for STRATEGY in anneal_to_high low_first low_constant; do
    echo "=== Evaluating strategy: $STRATEGY ==="
    mkdir -p /gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/eval
    python evaluate.py \
        --model /gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/final \
        --output-dir /gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/eval \
        --label "${STRATEGY}" \
        --max-samples 500 \
        --batch-size 16
done

# Missing baseline
echo "=== Evaluating baseline ==="
mkdir -p /gpfs/scratch/gs4342/grpo-sweep/baseline/eval
python evaluate.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output-dir /gpfs/scratch/gs4342/grpo-sweep/baseline/eval \
    --label baseline_3b \
    --max-samples 500 \
    --batch-size 16

# Regenerate comparison.json with all results
echo "=== Generating comparison.json ==="
python compare_results.py \
    --sweep-dir /gpfs/scratch/gs4342/grpo-sweep \
    --ablation-dir /gpfs/scratch/gs4342/grpo-ablation \
    --output "$WORK/comparison.json"

echo "All done. comparison.json updated."
