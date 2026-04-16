#!/bin/bash -l
#SBATCH --job-name=eval-all
#SBATCH --output=/gpfs/scratch/gs4342/logs/eval-all_%A_%a.out
#SBATCH --error=/gpfs/scratch/gs4342/logs/eval-all_%A_%a.err
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-14
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gs4342@nyu.edu
#
# Array job: one SLURM task per evaluation (3h each, 15 total).
# Tasks 0-6:  sweep strategies + baseline
# Tasks 7-14: single-dataset ablations
#
# Submit:
#   SWEEP_JOB=$(sbatch --parsable environment/slurm_eval_array.sbatch)
#   sbatch --dependency=afterany:$SWEEP_JOB environment/slurm_eval_aggregate.sbatch

set -euo pipefail

source ~/.bashrc
conda activate grpo-train

WORK=/gpfs/scratch/gs4342/epiplexity-curriculum-post-training
cd "$WORK"

export HF_HOME=/gpfs/scratch/gs4342/hf_cache
export TRANSFORMERS_CACHE=/gpfs/scratch/gs4342/hf_cache

# --- Task mapping ---
SWEEP_STRATEGIES=(baseline uniform high_first high_constant anneal_to_high low_first low_constant)
ABLATION_DATASETS=(gsm8k math humaneval mbpp mmlu arc triviaqa boolq)

TASK_ID=${SLURM_ARRAY_TASK_ID}

if [ "$TASK_ID" -le 6 ]; then
    # Sweep strategy (or baseline)
    STRATEGY="${SWEEP_STRATEGIES[$TASK_ID]}"
    echo "=== Task $TASK_ID: Evaluating sweep strategy '$STRATEGY' ==="

    if [ "$STRATEGY" = "baseline" ]; then
        MODEL="Qwen/Qwen2.5-3B-Instruct"
        LABEL="baseline_3b"
    else
        MODEL="/gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/final"
        LABEL="${STRATEGY}"
    fi
    OUTPUT_DIR="/gpfs/scratch/gs4342/grpo-sweep/${STRATEGY}/eval"

    if [ "$STRATEGY" = "baseline" ] || [ -d "$MODEL" ]; then
        # Delete stale results for this strategy
        rm -f "${OUTPUT_DIR}/${LABEL}_results.json" 2>/dev/null || true
        mkdir -p "$OUTPUT_DIR"
        python evaluate.py \
            --model "$MODEL" \
            --output-dir "$OUTPUT_DIR" \
            --label "$LABEL" \
            --max-samples 500 \
            --batch-size 16
    else
        echo "SKIP: Checkpoint $MODEL not found"
    fi

else
    # Ablation (task 7-14 → index 0-7)
    IDX=$((TASK_ID - 7))
    DATASET="${ABLATION_DATASETS[$IDX]}"
    MODEL="/gpfs/scratch/gs4342/grpo-ablation/${DATASET}/final"
    LABEL="ablation_${DATASET}"
    OUTPUT_DIR="/gpfs/scratch/gs4342/grpo-ablation/${DATASET}/eval"

    echo "=== Task $TASK_ID: Evaluating ablation '$DATASET' ==="

    if [ -d "$MODEL" ]; then
        rm -f "${OUTPUT_DIR}/${LABEL}_results.json" 2>/dev/null || true
        mkdir -p "$OUTPUT_DIR"
        python evaluate.py \
            --model "$MODEL" \
            --output-dir "$OUTPUT_DIR" \
            --label "$LABEL" \
            --max-samples 500 \
            --batch-size 16
    else
        echo "SKIP: Checkpoint $MODEL not found"
    fi
fi

echo "=== Task $TASK_ID complete ==="
