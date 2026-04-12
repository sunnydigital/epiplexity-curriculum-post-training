#!/bin/bash
#SBATCH --job-name=probe-epiplexity
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu

# ── Environment ──────────────────────────────────────────────────────────────
module purge
module load anaconda3/2024.02

conda activate grpo-train

export HF_HOME=$SCRATCH/hf_cache
export TRANSFORMERS_CACHE=$SCRATCH/hf_cache/transformers
export HF_DATASETS_CACHE=$SCRATCH/hf_cache/datasets

mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE

# ── Working directory ─────────────────────────────────────────────────────────
cd $HOME/epiplexity-curriculum-post-training

# ── Probe ─────────────────────────────────────────────────────────────────────
# Run probe first, then training can be submitted reading the output scores
python probe_epiplexity.py \
    --probe-model Qwen/Qwen2.5-0.5B-Instruct \
    --output data/epiplexity_scores.json \
    --max-samples 500 \
    --batch-size 8 \
    --max-length 1024

echo "Probe complete. Scores written to data/epiplexity_scores.json"
