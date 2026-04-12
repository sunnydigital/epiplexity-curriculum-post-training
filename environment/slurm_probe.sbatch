#!/bin/bash -l
#SBATCH --job-name=probe-epiplexity
#SBATCH --time=04:00:00
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/gpfs/scratch/gs4342/logs/probe_epiplexity_%j.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/probe_epiplexity_%j.err

source ~/.bashrc
conda activate grpo-train
cd /gpfs/scratch/gs4342/epiplexity-curriculum-post-training

# Point HuggingFace cache to scratch to avoid home quota
export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache/transformers
export HF_DATASETS_CACHE=/gpfs/scratch/gs4342/hf_cache/datasets
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# ── Probe ─────────────────────────────────────────────────────────────────────
# Estimates epiplexity (arXiv:2601.03220) by fine-tuning the probe on each
# dataset and computing K_auc (prequential model description length).
# This *trains* per dataset (not just inference), so takes longer than perplexity.
python probe_epiplexity.py \
    --probe-model Qwen/Qwen2.5-0.5B-Instruct \
    --output data/epiplexity_scores.json \
    --max-samples 500 \
    --batch-size 4 \
    --max-length 512 \
    --train-steps 200 \
    --lr 5e-5 \
    --log-interval 10

echo "Probe complete. Scores written to data/epiplexity_scores.json"
