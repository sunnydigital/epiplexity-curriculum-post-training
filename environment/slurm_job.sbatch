#!/bin/bash -l
#SBATCH --job-name=grpo-train
#SBATCH --time=04:00:00
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --output=/gpfs/scratch/gs4342/logs/grpo_train_%j.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/grpo_train_%j.err

source ~/.bashrc
conda activate grpo-train
cd /gpfs/scratch/gs4342/epiplexity-curriculum-post-training

# Point HuggingFace cache to scratch to avoid home quota
export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache/transformers
export HF_DATASETS_CACHE=/gpfs/scratch/gs4342/hf_cache/datasets
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# ── Training ──────────────────────────────────────────────────────────────────
python post_training.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --scores-path data/epiplexity_scores.json \
    --config configs/training_config.yaml \
    --datasets-config configs/datasets_config.yaml \
    --output-dir /gpfs/scratch/gs4342/grpo-output \
    --max-samples 2000

echo "Training complete."
