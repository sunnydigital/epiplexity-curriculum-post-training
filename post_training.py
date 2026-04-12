"""
GRPO post-training entry point with curriculum scheduling.

Trains a causal LM with Group Relative Policy Optimization (GRPO) via TRL,
using an epiplexity-weighted curriculum sampler across 8 datasets in 4 categories.

Curriculum strategies (--curriculum):
    uniform         — Equal sampling across all datasets (baseline)
    high_first      — Start with high-epiplexity focus, anneal to uniform
    low_first       — Start with low-epiplexity focus, anneal to uniform
    high_constant   — Fixed focus on high-epiplexity datasets
    low_constant    — Fixed focus on low-epiplexity datasets
    anneal_to_high  — Start uniform, progressively focus on high-epiplexity
    single          — Train on a single dataset (--single-dataset required)

Usage:
    python post_training.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --scores-path data/epiplexity_scores.json \\
        --config configs/training_config.yaml \\
        --output-dir outputs/grpo \\
        --curriculum high_first \\
        [--use-lora] \\
        [--max-samples 1000]
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from data.registry import get_registry_with_formatters
from data.datasets import load_all_datasets
from data.loader import EpiplexityWeightedSampler, load_epiplexity_scores
from data.curriculum import CurriculumScheduler, CurriculumStrategy
from rewards import RewardTracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO curriculum post-training")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--scores-path", type=str, default="data/epiplexity_scores.json",
        help="Path to JSON file with {dataset_name: epiplexity_score}",
    )
    parser.add_argument(
        "--config", type=str, default="configs/training_config.yaml",
        help="Path to training_config.yaml",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/grpo",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--use-lora", action="store_true",
        help="Wrap model with LoRA for memory-efficient training",
    )
    parser.add_argument(
        "--lora-r", type=int, default=16,
        help="LoRA rank (only used with --use-lora)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Truncate each dataset to N samples (for smoke tests)",
    )
    parser.add_argument(
        "--datasets-config", type=str, default="configs/datasets_config.yaml",
        help="Path to datasets_config.yaml",
    )
    # Curriculum arguments
    parser.add_argument(
        "--curriculum", type=str, default="uniform",
        choices=[s.value for s in CurriculumStrategy],
        help="Curriculum scheduling strategy",
    )
    parser.add_argument(
        "--single-dataset", type=str, default=None,
        help="Dataset name for 'single' curriculum strategy",
    )
    parser.add_argument(
        "--temp-start", type=float, default=0.1,
        help="Starting temperature for annealing curriculum strategies",
    )
    parser.add_argument(
        "--temp-end", type=float, default=5.0,
        help="Ending temperature for annealing curriculum strategies",
    )
    parser.add_argument(
        "--uniform-mix", type=float, default=0.1,
        help="Fraction of uniform distribution mixed in (prevents dataset starvation)",
    )
    parser.add_argument(
        "--weight-update-interval", type=int, default=50,
        help="How often (in steps) to recompute curriculum weights for the sampler",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model_and_tokenizer(model_name: str, use_lora: bool, lora_r: int):
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    if use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# GRPOConfig builder
# ---------------------------------------------------------------------------

def build_grpo_config(training_cfg: dict, output_dir: str) -> GRPOConfig:
    training_cfg = {**training_cfg, "output_dir": output_dir}
    valid_params = set(inspect.signature(GRPOConfig.__init__).parameters.keys())
    valid_params.discard("self")
    filtered = {k: v for k, v in training_cfg.items() if k in valid_params}
    skipped = set(training_cfg) - set(filtered)
    if skipped:
        print(f"Note: skipped config keys not in GRPOConfig: {skipped}")
    return GRPOConfig(**filtered)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CurriculumCallback(TrainerCallback):
    """
    TRL-compatible callback that:
    1. Updates the curriculum sampling weights at regular intervals
    2. Logs per-dataset reward statistics
    """

    def __init__(
        self,
        sampler: EpiplexityWeightedSampler,
        reward_tracker: RewardTracker,
        update_interval: int = 50,
    ):
        self.sampler = sampler
        self.reward_tracker = reward_tracker
        self.update_interval = update_interval

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step

        # Log per-dataset rewards at the same interval as training logs
        if step % args.logging_steps == 0 and step > 0:
            stats = self.reward_tracker.get_and_reset_stats()
            if stats:
                log_dict = {}
                for ds_name, ds_stats in stats.items():
                    log_dict[f"rewards_per_dataset/{ds_name}"] = ds_stats["mean"]
                    log_dict[f"rewards_per_dataset/{ds_name}_count"] = ds_stats["count"]
                # Log curriculum weights
                weights = self.sampler.scheduler.get_weights(step)
                for ds_name, w in weights.items():
                    log_dict[f"curriculum_weights/{ds_name}"] = w
                # Use the trainer's log method via kwargs if available
                trainer = kwargs.get("model")  # TRL passes model, not trainer
                if hasattr(state, "_logging_dict"):
                    state._logging_dict.update(log_dict)
                # Print to stdout as well
                print(f"\n[Step {step}] Per-dataset rewards:")
                for ds_name in sorted(stats):
                    s = stats[ds_name]
                    w = weights.get(ds_name, 0)
                    print(f"  {ds_name}: reward={s['mean']:.4f} (n={s['count']}) weight={w:.3f}")

        # Update curriculum weights
        if step % self.update_interval == 0 and step > 0:
            new_weights = self.sampler.update_weights_for_step(step)
            # The sampler will use updated weights on next epoch/shuffle
            # TRL recreates dataloader each epoch, which picks up new weights
            print(f"\n[Step {step}] Updated curriculum weights:")
            self.sampler.scheduler.log_weights_at_step(step)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    training_cfg = load_config(args.config)
    datasets_cfg = load_config(args.datasets_config)

    # Build model and tokenizer
    model, tokenizer = build_model_and_tokenizer(
        model_name=args.model,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
    )

    # Load and format all datasets
    registry = get_registry_with_formatters()
    dataset_dict = load_all_datasets(
        registry,
        max_samples_per_dataset=args.max_samples,
    )

    # For single-dataset strategy, filter to just that dataset
    if args.curriculum == "single" and args.single_dataset:
        if args.single_dataset not in dataset_dict:
            raise ValueError(
                f"--single-dataset '{args.single_dataset}' not in loaded datasets: "
                f"{list(dataset_dict)}"
            )
        dataset_dict = {args.single_dataset: dataset_dict[args.single_dataset]}
        print(f"Single-dataset mode: training on '{args.single_dataset}' only")

    # Load epiplexity scores and build curriculum scheduler
    scores = load_epiplexity_scores(args.scores_path)
    floor_weight = datasets_cfg.get("floor_weight", 0.02)
    total_steps = training_cfg.get("max_steps", 500)

    scheduler = CurriculumScheduler(
        strategy=args.curriculum,
        raw_scores=scores,
        total_steps=total_steps,
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        floor_weight=floor_weight,
        uniform_mix=args.uniform_mix,
        single_dataset=args.single_dataset,
    )
    print(f"\nCurriculum strategy: {scheduler.describe()}")

    # Build sampler
    sampler = EpiplexityWeightedSampler(dataset_dict=dataset_dict, scheduler=scheduler)
    sampler.print_weights(step=0)

    combined_dataset = sampler.build_combined_dataset()
    print(f"Combined dataset size: {len(combined_dataset):,} examples")

    # Build reward tracker (wraps dispatch_reward with per-dataset logging)
    reward_tracker = RewardTracker()

    # Build GRPOConfig
    grpo_config = build_grpo_config(training_cfg, args.output_dir)

    # Build curriculum callback
    curriculum_cb = CurriculumCallback(
        sampler=sampler,
        reward_tracker=reward_tracker,
        update_interval=args.weight_update_interval,
    )

    # Train
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_tracker],
        args=grpo_config,
        train_dataset=combined_dataset,
        processing_class=tokenizer,
        callbacks=[curriculum_cb],
    )

    print("Starting GRPO training...")
    print(f"  Strategy: {scheduler.describe()}")
    print(f"  Total steps: {total_steps}")
    print(f"  Weight update interval: {args.weight_update_interval} steps")
    trainer.train()

    # Save final model
    output_path = Path(args.output_dir) / "final"
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Model saved to {output_path}")

    # Save final reward stats and curriculum config
    meta = {
        "curriculum_strategy": args.curriculum,
        "single_dataset": args.single_dataset,
        "temp_start": args.temp_start,
        "temp_end": args.temp_end,
        "uniform_mix": args.uniform_mix,
        "total_steps": total_steps,
        "final_weights": scheduler.get_weights(total_steps),
        "initial_weights": scheduler.get_weights(0),
    }
    meta_path = Path(args.output_dir) / "curriculum_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Curriculum metadata saved to {meta_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()