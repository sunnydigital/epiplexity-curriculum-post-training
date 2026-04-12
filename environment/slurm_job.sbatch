#!/bin/bash
#SBATCH --job-name=grpo-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.02

conda activate grpo-train

# Point HuggingFace cache to scratch to avoid home quota (~50GB quota on BigPurple)
export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets

mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# ── Working directory ─────────────────────────────────────────────────────────
cd $HOME/epiplexity-curriculum-post-training

# ── Training ──────────────────────────────────────────────────────────────────
python post_training.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --scores-path data/epiplexity_scores.json \
    --config configs/training_config.yaml \
    --datasets-config configs/datasets_config.yaml \
    --output-dir $SCRATCH/grpo-output

echo "Training complete."
