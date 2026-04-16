"""
Measure reward variance per dataset using a frozen model.

Generates `--samples` completions per dataset at a fixed temperature (default 1.0,
matching GRPO's training distribution), then computes reward std. Higher std means
the dataset sits near the model's decision boundary — more useful gradient signal.

Also computes a combined score: K_auc × reward_std, which weights epiplexity by
how much gradient signal the dataset actually provides during RL training.

Usage:
    python measure_reward_variance.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --scores-path data/epiplexity_scores.json \
        --output data/reward_variance.json \
        [--samples 50] \
        [--max-samples 200] \
        [--temperature 1.0]
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.registry import get_registry_with_formatters
from data.datasets import load_all_datasets
from rewards import dispatch_reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--scores-path", type=str, default="data/epiplexity_scores.json")
    parser.add_argument("--output", type=str, default="data/reward_variance.json")
    parser.add_argument("--samples", type=int, default=50,
                        help="Completions per dataset for variance estimation")
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Dataset truncation passed to load_all_datasets")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (use GRPO training temperature)")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def measure_dataset_variance(
    model,
    tokenizer,
    dataset,
    dataset_name: str,
    samples: int,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> dict:
    """
    Sample `samples` completions from the frozen model on the first `samples`
    examples of the dataset, score each with the reward function, and return
    mean/std/variance of the reward distribution.
    """
    n = min(samples, len(dataset))
    subset = dataset.select(range(n))
    rewards = []

    model.eval()
    for example in subset:
        inputs = tokenizer(
            example["prompt"],
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion = tokenizer.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        reward = dispatch_reward(
            completions=[completion],
            answer=[example["answer"]],
            category=[example["category"]],
            dataset=[dataset_name],
        )[0]

        rewards.append(reward)

    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = variance ** 0.5

    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "variance": round(variance, 4),
        "count": n,
        "temperature": temperature,
        "rewards": rewards,
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    registry = get_registry_with_formatters()
    dataset_dict = load_all_datasets(registry, max_samples_per_dataset=args.max_samples)

    with open(args.scores_path) as f:
        raw = json.load(f)
    k_auc_scores = {k: v for k, v in raw.items() if not k.startswith("_")}

    variance_scores = {}
    combined_scores = {}
    details = {}

    print(f"\nMeasuring reward variance ({args.samples} samples/dataset, temp={args.temperature})...")

    for name, dataset in dataset_dict.items():
        print(f"\n{'─'*50}")
        print(f"Dataset: {name} ({len(dataset)} examples)")
        start = time.time()

        result = measure_dataset_variance(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            dataset_name=name,
            samples=args.samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )

        elapsed = time.time() - start
        variance_scores[name] = result["std"]
        details[name] = {**result, "elapsed_seconds": round(elapsed, 1)}

        k_auc = k_auc_scores.get(name, 0.0)
        combined = k_auc * result["std"]
        combined_scores[name] = round(combined, 6)

        print(f"  mean={result['mean']:.4f}  std={result['std']:.4f}  "
              f"K_auc={k_auc:.4f}  combined={combined:.6f}  time={elapsed:.1f}s")

    print(f"\n{'='*50}")
    print("Reward std ranking (higher = more GRPO signal):")
    for name, std in sorted(variance_scores.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}: std={std:.4f}  mean={details[name]['mean']:.4f}")

    print(f"\nCombined K_auc × reward_std ranking:")
    for name, score in sorted(combined_scores.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}: {score:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            **variance_scores,
            "_combined_scores": combined_scores,
            "_metadata": {
                "model": args.model,
                "samples": args.samples,
                "temperature": args.temperature,
                "k_auc_scores_path": args.scores_path,
                "details": details,
            },
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
