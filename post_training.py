"""
GRPO post-training entry point.

Trains a causal LM with Group Relative Policy Optimization (GRPO) via TRL,
using an epiplexity-weighted curriculum sampler across 8 datasets in 4 categories.

Usage:
    python post_training.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct \\
        --scores-path data/epiplexity_scores.json \\
        --config configs/training_config.yaml \\
        --output-dir outputs/grpo \\
        [--use-lora] \\
        [--max-samples 1000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from data.registry import get_registry_with_formatters
from data.datasets import load_all_datasets
from data.loader import EpiplexityWeightedSampler
from rewards import dispatch_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO curriculum post-training")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--scores-path",
        type=str,
        default="data/epiplexity_scores.json",
        help="Path to JSON file with {dataset_name: epiplexity_score}",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/grpo",
        help="Directory for checkpoints and logs",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Wrap model with LoRA for memory-efficient training",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (only used with --use-lora)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Truncate each dataset to N samples (for smoke tests)",
    )
    parser.add_argument(
        "--datasets-config",
        type=str,
        default="configs/datasets_config.yaml",
        help="Path to datasets_config.yaml",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_model_and_tokenizer(model_name: str, use_lora: bool, lora_r: int):
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
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


def build_grpo_config(training_cfg: dict, output_dir: str) -> GRPOConfig:
    # Override output_dir from CLI arg
    training_cfg = {**training_cfg, "output_dir": output_dir}

    # Map YAML keys to GRPOConfig fields (snake_case matches)
    # Remove any keys not accepted by GRPOConfig to avoid errors
    grpo_fields = {
        "output_dir", "num_train_epochs", "per_device_train_batch_size",
        "gradient_accumulation_steps", "num_generations", "max_prompt_length",
        "max_completion_length", "learning_rate", "lr_scheduler_type",
        "warmup_ratio", "weight_decay", "logging_steps", "save_steps",
        "save_total_limit", "beta", "bf16", "gradient_checkpointing",
        "dataloader_num_workers", "logging_dir",
    }
    filtered = {k: v for k, v in training_cfg.items() if k in grpo_fields}
    return GRPOConfig(**filtered)


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

    # Build epiplexity-weighted combined dataset
    floor_weight = datasets_cfg.get("floor_weight", 0.05)
    temperature = datasets_cfg.get("temperature", 1.0)
    sampler = EpiplexityWeightedSampler(
        dataset_dict=dataset_dict,
        scores_path=args.scores_path,
        temperature=temperature,
        floor_weight=floor_weight,
    )
    sampler.print_weights()
    combined_dataset = sampler.build_combined_dataset()
    print(f"Combined dataset size: {len(combined_dataset):,} examples")

    # Build GRPOConfig
    grpo_config = build_grpo_config(training_cfg, args.output_dir)

    # Train
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[dispatch_reward],
        args=grpo_config,
        train_dataset=combined_dataset,
        processing_class=tokenizer,
    )

    print("Starting GRPO training...")
    trainer.train()

    # Save final model
    output_path = Path(args.output_dir) / "final"
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()