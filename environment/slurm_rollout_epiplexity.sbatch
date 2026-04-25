#!/bin/bash -l
#SBATCH --job-name=rollout-epi
#SBATCH --output=/gpfs/scratch/gs4342/logs/rollout-epi_%A_%a.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/rollout-epi_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Array job: measure rollout epiplexity (GRPO-native K_auc) at two scales
#   Task 0: Qwen2.5-1.5B-Instruct (probe scale, for 2x2 with reward variance)
#   Task 1: Qwen2.5-3B-Instruct   (target scale, primary measurement)
#
# Submit:
#   sbatch environment/slurm_rollout_epiplexity.sbatch

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

echo "=== Measuring rollout epiplexity with $MODEL ==="

python measure_rollout_epiplexity.py \
    --model "$MODEL" \
    --num-chunks 50 \
    --prompts-per-chunk 16 \
    --num-generations 8 \
    --max-new-tokens 256 \
    --max-prompt-length 512 \
    --temperature 0.7 \
    --lr 3.0e-6 \
    --lora-r 16 \
    --output "data/rollout_epiplexity_${LABEL}.json"

echo "=== Done ==="
