#!/bin/bash
#SBATCH --job-name=reward_variance
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --account=csci_ga_3033_076-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --output=reward_variance_%j.out
#SBATCH --error=reward_variance_%j.err
#SBATCH --requeue

singularity exec --nv \
  --overlay /scratch/ap9283/epiplexity-curriculum-post-training/my_pytorch.ext3:rw \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash -c "
    source /scratch/ap9283/miniforge3/etc/profile.d/conda.sh
    conda activate cvenv
    cd /scratch/ap9283/epiplexity-curriculum-post-training
    python meaure_reward_variance.py \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --scores-path data/epiplexity_scores.json \
      --output data/reward_variance.json \
      --samples 50 \
      --temperature 1.0 \
      --device cuda
  "