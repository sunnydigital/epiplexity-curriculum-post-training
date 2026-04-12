"""
Post-training evaluation across all 8 benchmark datasets.

Loads a trained model checkpoint and evaluates on held-out splits of each
dataset using standardized metrics:
    - Math (GSM8K, MATH): solve rate (exact numeric match)
    - Code (HumanEval, MBPP): string match accuracy
    - Logical (MMLU, ARC): MCQ accuracy
    - QA (TriviaQA, BoolQ): answer accuracy

This enables apples-to-apples comparison across curriculum strategies by
measuring downstream task performance rather than comparing incompatible
reward signals.

Usage:
    python evaluate.py \\
        --model outputs/grpo/final \\
        --output-dir outputs/eval \\
        [--max-samples 200] \\
        [--batch-size 16]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.datasets import (
    format_gsm8k, format_math, format_humaneval, format_mbpp,
    format_mmlu, format_arc, format_triviaqa, format_boolq,
    _FORMATTED_FEATURES,
)
from rewards.math import math_reward
from rewards.code import code_reward
from rewards.qa import qa_reward


# ---------------------------------------------------------------------------
# Evaluation dataset registry (held-out splits)
# ---------------------------------------------------------------------------

EVAL_REGISTRY = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_config": "main",
        "split": "test",
        "formatter": format_gsm8k,
        "category": "math",
    },
    "math": {
        "hf_path": "EleutherAI/hendrycks_math",
        "hf_config": "algebra",
        "split": "test",
        "formatter": format_math,
        "category": "math",
    },
    "humaneval": {
        "hf_path": "openai/openai_humaneval",
        "hf_config": None,
        "split": "test",
        "formatter": format_humaneval,
        "category": "code",
    },
    "mbpp": {
        "hf_path": "google-research-datasets/mbpp",
        "hf_config": "sanitized",
        "split": "test",
        "formatter": format_mbpp,
        "category": "code",
    },
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_config": "all",
        "split": "validation",
        "formatter": format_mmlu,
        "category": "logical",
    },
    "arc": {
        "hf_path": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "split": "test",
        "formatter": format_arc,
        "category": "logical",
    },
    "triviaqa": {
        "hf_path": "mandarjoshi/trivia_qa",
        "hf_config": "rc",
        "split": "validation",
        "formatter": format_triviaqa,
        "category": "qa",
    },
    "boolq": {
        "hf_path": "google/boolq",
        "hf_config": None,
        "split": "validation",
        "formatter": format_boolq,
        "category": "qa",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate post-trained model on benchmarks")
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to trained model checkpoint (or HF model name for baseline)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs/eval",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max examples per dataset (for quick evaluation)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum tokens to generate per example",
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=None,
        help="Specific datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--label", type=str, default=None,
        help="Label for this evaluation run (e.g., 'high_first_500steps')",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_completions(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> list[str]:
    """
    Generate completions for a list of prompts in batches.

    Returns list of generated strings (prompt text stripped).
    """
    completions = []
    model.eval()

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs["do_sample"] = False

            outputs = model.generate(**inputs, **gen_kwargs)

        # Strip the prompt tokens from generated output
        for j, output in enumerate(outputs):
            prompt_len = inputs["input_ids"][j].shape[0]
            generated_tokens = output[prompt_len:]
            completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            completions.append(completion)

    return completions


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_dataset(
    completions: list[str],
    answers: list[str],
    dataset_name: str,
    category: str,
) -> list[float]:
    """Score completions using the appropriate reward function."""
    if category == "math":
        return math_reward(completions, answers)
    elif category == "code":
        return code_reward(completions, answers)
    else:
        datasets_list = [dataset_name] * len(completions)
        return qa_reward(completions, answers, datasets_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # important for batch generation

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    # Determine which datasets to evaluate
    eval_datasets = args.datasets or list(EVAL_REGISTRY.keys())
    results = {}

    print(f"\nEvaluating on {len(eval_datasets)} datasets...")
    print("=" * 60)

    for ds_name in eval_datasets:
        if ds_name not in EVAL_REGISTRY:
            print(f"WARNING: Unknown dataset '{ds_name}', skipping")
            continue

        cfg = EVAL_REGISTRY[ds_name]
        print(f"\n--- {ds_name} ({cfg['category']}) ---")
        start = time.time()

        # Load eval split
        load_kwargs = {"path": cfg["hf_path"], "split": cfg["split"]}
        if cfg.get("hf_config"):
            load_kwargs["name"] = cfg["hf_config"]

        try:
            raw = load_dataset(**load_kwargs)
        except Exception as e:
            print(f"  Failed to load {ds_name}: {e}")
            results[ds_name] = {"error": str(e)}
            continue

        # Format
        formatted = raw.map(
            cfg["formatter"],
            remove_columns=raw.column_names,
            desc=f"Formatting {ds_name}",
        ).cast(_FORMATTED_FEATURES)

        # Truncate if needed
        if args.max_samples:
            formatted = formatted.select(range(min(args.max_samples, len(formatted))))

        n = len(formatted)
        prompts = formatted["prompt"]
        answers = formatted["answer"]

        print(f"  Generating completions for {n} examples...")
        completions = generate_completions(
            model, tokenizer, prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )

        # Score
        rewards = score_dataset(completions, answers, ds_name, cfg["category"])
        accuracy = sum(r == 1.0 for r in rewards) / len(rewards)
        mean_reward = sum(rewards) / len(rewards)
        elapsed = time.time() - start

        results[ds_name] = {
            "accuracy": accuracy,
            "mean_reward": mean_reward,
            "n_examples": n,
            "category": cfg["category"],
            "elapsed_seconds": round(elapsed, 1),
        }

        print(f"  Accuracy: {accuracy:.4f} ({sum(r == 1.0 for r in rewards)}/{n})")
        print(f"  Mean reward: {mean_reward:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<15} {'Category':<10} {'Accuracy':>10} {'Mean Reward':>12} {'N':>6}")
    print("-" * 60)

    category_accs = {}
    overall_accs = []

    for ds_name in eval_datasets:
        if ds_name in results and "error" not in results[ds_name]:
            r = results[ds_name]
            print(f"{ds_name:<15} {r['category']:<10} {r['accuracy']:>10.4f} {r['mean_reward']:>12.4f} {r['n_examples']:>6}")
            cat = r["category"]
            category_accs.setdefault(cat, []).append(r["accuracy"])
            overall_accs.append(r["accuracy"])

    print("-" * 60)
    for cat in sorted(category_accs):
        cat_mean = sum(category_accs[cat]) / len(category_accs[cat])
        print(f"{'[' + cat + ']':<15} {'avg':<10} {cat_mean:>10.4f}")

    if overall_accs:
        overall = sum(overall_accs) / len(overall_accs)
        print(f"{'[OVERALL]':<15} {'avg':<10} {overall:>10.4f}")

    # Save results
    output = {
        "model": args.model,
        "label": args.label,
        "max_samples": args.max_samples,
        "temperature": args.temperature,
        "per_dataset": results,
        "category_means": {
            cat: sum(accs) / len(accs)
            for cat, accs in category_accs.items()
        },
        "overall_accuracy": sum(overall_accs) / len(overall_accs) if overall_accs else None,
    }

    label = args.label or "eval"
    results_path = output_dir / f"{label}_results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
