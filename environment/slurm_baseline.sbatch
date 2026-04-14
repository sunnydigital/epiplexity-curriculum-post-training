#!/bin/bash -l
#SBATCH --job-name=eval-baseline
#SBATCH --output=/gpfs/scratch/gs4342/logs/eval-baseline_%j.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/eval-baseline_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Evaluate the pretrained model (no GRPO) as a baseline.
# This establishes whether GRPO training actually improves anything.

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

EVAL_DIR="/gpfs/scratch/gs4342/grpo-sweep/baseline/eval"
mkdir -p "$EVAL_DIR"

echo "=== Baseline Evaluation (no GRPO) ==="
echo "Model: Qwen/Qwen2.5-3B-Instruct (pretrained, no fine-tuning)"
echo "======================================"

python evaluate.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --output-dir "$EVAL_DIR" \
    --label "baseline_3b" \
    --max-samples 500 \
    --batch-size 16

echo "Baseline evaluation complete."
