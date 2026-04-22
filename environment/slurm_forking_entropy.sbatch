#!/bin/bash -l
#SBATCH --job-name=fork-entropy
#SBATCH --output=/gpfs/scratch/gs4342/logs/fork-entropy_%A_%a.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/fork-entropy_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Array job: measure forking-token entropy (Wang et al. 2025) for both models.
#   Task 0: Qwen2.5-1.5B-Instruct (probe)
#   Task 1: Qwen2.5-3B-Instruct (target)
#
# Hypothesis: per-dataset mean entropy of top-20% forking tokens in SUCCESSFUL
# rollouts correlates with GRPO transfer score (comparison.json ablation MIXED).
#
# Submit:
#   sbatch environment/slurm_forking_entropy.sbatch

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

echo "=== Measuring forking-token entropy with $MODEL ==="

# Lower max-samples than reward variance (100 vs 200) because entropy capture
# requires retaining per-step logits → more GPU memory & slower decode.
python measure_forking_entropy.py \
    --model "$MODEL" \
    --num-generations 8 \
    --max-samples 100 \
    --max-new-tokens 256 \
    --temperature 0.7 \
    --top-pct 0.2 \
    --output "data/forking_entropy_${LABEL}.json"

echo "=== Done ==="
