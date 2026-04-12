#!/bin/bash -l
#SBATCH --job-name=grpo-ablation
#SBATCH --output=/gpfs/scratch/gs4342/logs/grpo-ablation-%A_%a.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/grpo-ablation-%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Single-dataset ablation: train on ONE dataset, evaluate on ALL
# Submit with: sbatch --array=0-7 environment/slurm_ablation.sbatch
#
# Array indices:
#   0 = gsm8k,  1 = math,     2 = humaneval, 3 = mbpp
#   4 = mmlu,   5 = arc,      6 = triviaqa,  7 = boolq

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

# Map array task ID to dataset name
DATASETS=(gsm8k math humaneval mbpp mmlu arc triviaqa boolq)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

OUTPUT_DIR="/gpfs/scratch/gs4342/grpo-ablation/${DATASET}"
mkdir -p "$OUTPUT_DIR"

echo "=== Single-Dataset Ablation ==="
echo "Training dataset: ${DATASET}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "================================"

python post_training.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --scores-path data/epiplexity_scores.json \
    --config configs/training_config.yaml \
    --output-dir "$OUTPUT_DIR" \
    --curriculum single \
    --single-dataset "$DATASET" \
    --use-lora \
    --max-samples 2000

echo "Training complete for dataset: ${DATASET}"

# Evaluate on all datasets
EVAL_DIR="/gpfs/scratch/gs4342/grpo-ablation/${DATASET}/eval"
mkdir -p "$EVAL_DIR"

echo "Starting evaluation on all datasets..."
python evaluate.py \
    --model "${OUTPUT_DIR}/final" \
    --output-dir "$EVAL_DIR" \
    --label "ablation_${DATASET}" \
    --max-samples 500 \
    --batch-size 16

echo "Evaluation complete for ablation: ${DATASET}"
