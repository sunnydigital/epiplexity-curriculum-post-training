#!/bin/bash -l
#SBATCH --job-name=probe-3b
#SBATCH --output=/gpfs/scratch/gs4342/logs/probe-3b_%j.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/probe-3b_%j.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Run epiplexity probe with 3B model (to compare with existing 1.5B probe).
# Outputs to data/epiplexity_scores_3b.json (existing 1.5B scores untouched).
#
# Submit:
#   sbatch environment/slurm_probe_3b.sbatch

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

echo "=== Running epiplexity probe with Qwen2.5-3B-Instruct ==="

python probe_epiplexity.py \
    --probe-model Qwen/Qwen2.5-3B-Instruct \
    --output data/epiplexity_scores_3b.json \
    --max-samples 2000 \
    --train-steps 500

echo "=== Done ==="
