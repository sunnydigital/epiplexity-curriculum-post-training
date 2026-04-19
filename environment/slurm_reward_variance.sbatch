#!/bin/bash -l
#SBATCH --job-name=reward-var
#SBATCH --output=/gpfs/scratch/gs4342/logs/reward-var_%A_%a.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/reward-var_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Array job: measure reward variance with two models
#   Task 0: Qwen2.5-1.5B-Instruct (probe)
#   Task 1: Qwen2.5-3B-Instruct (target)
#
# Submit:
#   sbatch environment/slurm_reward_variance.sbatch

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

MODELS=("Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct")
LABELS=("1.5b" "3b")

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
LABEL="${LABELS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Measuring reward variance with $MODEL ==="

python measure_reward_variance.py \
    --model "$MODEL" \
    --num-generations 8 \
    --max-samples 200 \
    --temperature 0.7 \
    --output "data/reward_variance_${LABEL}.json"

echo "=== Done ==="
