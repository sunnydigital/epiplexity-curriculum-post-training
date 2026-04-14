#!/bin/bash -l
#SBATCH --job-name=grpo-sweep
#SBATCH --output=/gpfs/scratch/gs4342/logs/grpo-sweep-%A_%a.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/grpo-sweep-%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Array job: one task per curriculum strategy
# Submit with: sbatch --array=0-5 environment/slurm_sweep.sbatch
#
# Strategies:
#   0 = uniform (baseline)
#   1 = high_first (hard-first curriculum)
#   2 = low_first (easy-first / classic curriculum)
#   3 = high_constant (always focus on high-epi)
#   4 = anneal_to_high (uniform → high-epi focus)
#   5 = low_constant (always focus on low-epi)

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

# Map array task ID to curriculum strategy
STRATEGIES=(uniform high_first low_first high_constant anneal_to_high low_constant)
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR="/gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}"
mkdir -p "$OUTPUT_DIR"

echo "=== GRPO Curriculum Sweep ==="
echo "Strategy: ${STRATEGY}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================="

python post_training.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --scores-path data/epiplexity_scores.json \
    --config configs/training_config.yaml \
    --output-dir "$OUTPUT_DIR" \
    --curriculum "$STRATEGY" \
    --temp-start 0.1 \
    --temp-end 5.0 \
    --uniform-mix 0.1 \
    --weight-update-interval 100 \
    --use-lora \
    --max-samples 2000

echo "Training complete for strategy: ${STRATEGY}"

# Run evaluation immediately after training
EVAL_DIR="/gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/eval"
mkdir -p "$EVAL_DIR"

echo "Starting evaluation..."
python evaluate.py \
    --model "${OUTPUT_DIR}/final" \
    --output-dir "$EVAL_DIR" \
    --label "${STRATEGY}" \
    --max-samples 500 \
    --batch-size 16

echo "Evaluation complete for strategy: ${STRATEGY}"
