#!/bin/bash -l
#SBATCH --job-name=eval-aggregate
#SBATCH --output=/gpfs/scratch/gs4342/logs/eval-aggregate_%j.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/eval-aggregate_%j.err
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Runs after slurm_eval_array.sbatch finishes (all tasks).
# Aggregates *_results.json into comparison.json.
#
# Submit (chained after eval array):
#   SWEEP_JOB=$(sbatch --parsable environment/slurm_eval_array.sbatch)
#   sbatch --dependency=afterany:$SWEEP_JOB environment/slurm_eval_aggregate.sbatch

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

echo "=== Aggregating evaluation results ==="

python compare_results.py \
    --sweep-dir /gpfs/scratch/gs4342/grpo-sweep \
    --ablation-dir /gpfs/scratch/gs4342/grpo-ablation \
    --predictors-dir "$WORK/data" \
    --output "$WORK/comparison.json"

echo "=== Done. Results in $WORK/comparison.json ==="
