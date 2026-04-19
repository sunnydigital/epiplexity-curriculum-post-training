"""
Measure per-dataset reward variance for a given model.

For each dataset, generates N completions per prompt, scores them with the
task-appropriate reward function, and computes mean within-group variance.
This estimates how much "signal" GRPO would get from each dataset — high
variance means the model is in the Goldilocks difficulty zone where it
sometimes succeeds and sometimes fails.

Designed for the 2x2 predictor comparison:
    - Epiplexity (1.5B probe, 3B target)  vs
    - Reward variance (1.5B probe, 3B target)
    → correlate each with ablation transfer scores

Usage:
    python measure_reward_variance.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --num-generations 8 \\
        --max-samples 200 \\
        --output data/reward_variance_3b.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
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


def generate_multiple(
    model, tokenizer, prompt: str, num_generations: int,
    max_new_tokens: int = 256, temperature: float = 0.7,
) -> list[str]:
    """Generate multiple completions for a single prompt."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    completions = []
    for output in outputs:
        generated_tokens = output[prompt_len:]
        completion = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        completions.append(completion)

    return completions


def score_completions(
    completions: list[str], answer: str, dataset_name: str, category: str,
) -> list[float]:
    """Score a group of completions for one prompt."""
    answers = [answer] * len(completions)
    if category == "math":
        return math_reward(completions, answers)
    elif category == "code":
        return code_reward(completions, answers)
    else:
        datasets_list = [dataset_name] * len(completions)
        return qa_reward(completions, answers, datasets_list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure reward variance per dataset")
    parser.add_argument("--model", type=str, required=True,
                        help="HF model name or path")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="Completions per prompt (matches GRPO group size)")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Prompts per dataset")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (needs >0 for variance)")
    parser.add_argument("--output", type=str, default="data/reward_variance.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()

    results = {}

    for ds_name, cfg in EVAL_REGISTRY.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({cfg['category']})")
        print(f"{'='*60}")
        start = time.time()

        load_kwargs = {"path": cfg["hf_path"], "split": cfg["split"]}
        if cfg.get("hf_config"):
            load_kwargs["name"] = cfg["hf_config"]

        try:
            raw = load_dataset(**load_kwargs)
        except Exception as e:
            print(f"  Failed to load {ds_name}: {e}")
            results[ds_name] = {"error": str(e)}
            continue

        formatted = raw.map(
            cfg["formatter"],
            remove_columns=raw.column_names,
            desc=f"Formatting {ds_name}",
        ).cast(_FORMATTED_FEATURES)

        n = min(args.max_samples, len(formatted))
        formatted = formatted.select(range(n))
        prompts = formatted["prompt"]
        answers = formatted["answer"]

        variances = []
        mean_rewards = []

        for i in range(n):
            if i % 50 == 0:
                print(f"  Progress: {i}/{n}")

            completions = generate_multiple(
                model, tokenizer, prompts[i],
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            rewards = score_completions(
                completions, answers[i], ds_name, cfg["category"],
            )

            group_var = float(np.var(rewards))
            group_mean = float(np.mean(rewards))
            variances.append(group_var)
            mean_rewards.append(group_mean)

        elapsed = time.time() - start
        mean_var = float(np.mean(variances))
        mean_rew = float(np.mean(mean_rewards))
        std_var = float(np.std(variances))

        results[ds_name] = {
            "mean_reward_variance": mean_var,
            "std_reward_variance": std_var,
            "mean_reward": mean_rew,
            "n_prompts": n,
            "num_generations": args.num_generations,
            "category": cfg["category"],
            "elapsed_seconds": round(elapsed, 1),
        }

        print(f"  Mean reward variance: {mean_var:.4f}")
        print(f"  Mean reward: {mean_rew:.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # Save
    output = {
        "model": args.model,
        "temperature": args.temperature,
        "num_generations": args.num_generations,
        "max_samples": args.max_samples,
        "per_dataset": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("REWARD VARIANCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Category':<8} {'MeanVar':>10} {'MeanRew':>10}")
    print("-" * 45)
    for ds_name in EVAL_REGISTRY:
        if ds_name in results and "error" not in results[ds_name]:
            r = results[ds_name]
            print(f"{ds_name:<12} {r['category']:<8} {r['mean_reward_variance']:>10.4f} {r['mean_reward']:>10.4f}")


if __name__ == "__main__":
    main()
