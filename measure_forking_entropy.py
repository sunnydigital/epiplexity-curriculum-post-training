"""
Measure forking-token entropy per dataset for a given model.

For each prompt: generate N completions, score each, compute per-token policy
entropy. Aggregate entropy statistics separately over SUCCESSFUL vs FAILED
rollouts. The key predictor is mean entropy over high-entropy "forking tokens"
in successful completions.

Based on:
    Wang et al., "Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive
    Effective Reinforcement Learning for LLM Reasoning," NeurIPS 2025,
    arXiv:2506.01939

Hypothesis: datasets whose successful completions have high decision-point
density (forking tokens) transfer better under GRPO, because forking tokens
are where RL actually imparts generalizable structure.

Usage:
    python measure_forking_entropy.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --num-generations 8 \\
        --max-samples 100 \\
        --output data/forking_entropy_3b.json
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
    "gsm8k": {"hf_path": "openai/gsm8k", "hf_config": "main", "split": "test",
              "formatter": format_gsm8k, "category": "math"},
    "math": {"hf_path": "EleutherAI/hendrycks_math", "hf_config": "algebra", "split": "test",
             "formatter": format_math, "category": "math"},
    "humaneval": {"hf_path": "openai/openai_humaneval", "hf_config": None, "split": "test",
                  "formatter": format_humaneval, "category": "code"},
    "mbpp": {"hf_path": "google-research-datasets/mbpp", "hf_config": "sanitized", "split": "test",
             "formatter": format_mbpp, "category": "code"},
    "mmlu": {"hf_path": "cais/mmlu", "hf_config": "all", "split": "validation",
             "formatter": format_mmlu, "category": "logical"},
    "arc": {"hf_path": "allenai/ai2_arc", "hf_config": "ARC-Challenge", "split": "test",
            "formatter": format_arc, "category": "logical"},
    "triviaqa": {"hf_path": "mandarjoshi/trivia_qa", "hf_config": "rc", "split": "validation",
                 "formatter": format_triviaqa, "category": "qa"},
    "boolq": {"hf_path": "google/boolq", "hf_config": None, "split": "validation",
              "formatter": format_boolq, "category": "qa"},
}


def generate_with_entropy(
    model, tokenizer, prompt: str, num_generations: int,
    max_new_tokens: int = 256, temperature: float = 0.7,
):
    """
    Generate multiple completions and return per-token entropies.

    Returns:
        completions: list of decoded strings
        entropies_per_completion: list of tensors (one per completion),
            each shape [seq_len] containing per-step entropy in nats
    """
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
            return_dict_in_generate=True,
            output_scores=True,
        )

    prompt_len = inputs["input_ids"].shape[1]

    # outputs.scores is a tuple of length max_new_tokens
    # Each element has shape [num_generations, vocab_size] — the logits at that step
    # Compute entropy from the sampling distribution (with temperature applied)
    entropies_per_step = []
    for step_logits in outputs.scores:
        scaled = step_logits / temperature
        logprobs = torch.log_softmax(scaled, dim=-1)
        probs = logprobs.exp()
        # H = -sum(p * log p), nats per token
        entropy = -(probs * logprobs).sum(dim=-1)  # [num_generations]
        entropies_per_step.append(entropy)

    # Stack to [num_generations, max_new_tokens]
    entropy_matrix = torch.stack(entropies_per_step, dim=1)

    # Decode completions + figure out actual generation length per completion
    # (generation may have ended early via EOS)
    completions = []
    entropies_per_completion = []
    eos_id = tokenizer.eos_token_id

    for i in range(num_generations):
        generated = outputs.sequences[i, prompt_len:]
        # Find first EOS to truncate entropy tensor accordingly
        eos_positions = (generated == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            actual_len = int(eos_positions[0].item()) + 1
        else:
            actual_len = generated.shape[0]

        actual_len = max(1, min(actual_len, entropy_matrix.shape[1]))
        completion_entropy = entropy_matrix[i, :actual_len].cpu().float().numpy()

        completions.append(tokenizer.decode(generated, skip_special_tokens=True))
        entropies_per_completion.append(completion_entropy)

    return completions, entropies_per_completion


def score_completions(
    completions: list[str], answer: str, dataset_name: str, category: str,
) -> list[float]:
    answers = [answer] * len(completions)
    if category == "math":
        return math_reward(completions, answers)
    elif category == "code":
        return code_reward(completions, answers)
    else:
        datasets_list = [dataset_name] * len(completions)
        return qa_reward(completions, answers, datasets_list)


def aggregate_entropy_stats(entropies_list: list[np.ndarray], top_pct: float = 0.2) -> dict:
    """
    Compute aggregate entropy statistics over a list of completion entropy arrays.
    Returns mean entropy, top-k% forking-token mean entropy, density of
    high-entropy tokens (by global threshold).
    """
    if not entropies_list:
        return {"n": 0, "mean_entropy": None, "mean_top_pct_entropy": None,
                "forking_density": None, "mean_completion_length": None}

    all_entropies = np.concatenate(entropies_list)
    if len(all_entropies) == 0:
        return {"n": 0, "mean_entropy": None, "mean_top_pct_entropy": None,
                "forking_density": None, "mean_completion_length": None}

    # Global threshold = top-pct percentile across ALL tokens in this bucket
    threshold = float(np.percentile(all_entropies, 100 * (1 - top_pct)))

    # Per-completion: mean of top-pct tokens (Wang et al.'s canonical aggregate)
    per_completion_top_means = []
    per_completion_fork_density = []
    for e in entropies_list:
        if len(e) == 0:
            continue
        k = max(1, int(np.ceil(top_pct * len(e))))
        top_vals = np.sort(e)[-k:]
        per_completion_top_means.append(float(np.mean(top_vals)))
        per_completion_fork_density.append(float(np.mean(e >= threshold)))

    return {
        "n": len(entropies_list),
        "mean_entropy": float(np.mean(all_entropies)),
        "mean_top_pct_entropy": float(np.mean(per_completion_top_means)) if per_completion_top_means else None,
        "forking_density": float(np.mean(per_completion_fork_density)) if per_completion_fork_density else None,
        "global_threshold_nats": threshold,
        "mean_completion_length": float(np.mean([len(e) for e in entropies_list])),
        "top_pct": top_pct,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure forking-token entropy per dataset on successful vs failed rollouts"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Prompts per dataset (lower than reward-variance run; "
                             "entropy computation is memory-heavier)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-pct", type=float, default=0.2,
                        help="Top-k%% tokens to consider as forking tokens (Wang et al. use 20%%)")
    parser.add_argument("--output", type=str, default="data/forking_entropy.json")
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

        # Buckets for entropies
        successful_entropies = []
        failed_entropies = []
        all_entropies = []
        rewards_all = []

        for i in range(n):
            if i % 25 == 0:
                print(f"  Progress: {i}/{n}")

            completions, entropies = generate_with_entropy(
                model, tokenizer, prompts[i],
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )

            rewards = score_completions(
                completions, answers[i], ds_name, cfg["category"],
            )

            # For code (continuous reward) treat >= 0.5 as success
            threshold_success = 0.5 if cfg["category"] == "code" else 1.0
            for r, e in zip(rewards, entropies):
                all_entropies.append(e)
                rewards_all.append(r)
                if r >= threshold_success:
                    successful_entropies.append(e)
                elif r <= 0.0:
                    failed_entropies.append(e)

        elapsed = time.time() - start

        success_stats = aggregate_entropy_stats(successful_entropies, top_pct=args.top_pct)
        failed_stats = aggregate_entropy_stats(failed_entropies, top_pct=args.top_pct)
        overall_stats = aggregate_entropy_stats(all_entropies, top_pct=args.top_pct)

        success_rate = float(np.mean([r >= (0.5 if cfg["category"] == "code" else 1.0)
                                      for r in rewards_all]))

        results[ds_name] = {
            "category": cfg["category"],
            "n_prompts": n,
            "num_generations": args.num_generations,
            "success_rate": success_rate,
            "successful_rollouts": success_stats,
            "failed_rollouts": failed_stats,
            "all_rollouts": overall_stats,
            "elapsed_seconds": round(elapsed, 1),
        }

        print(f"  Success rate: {success_rate:.3f}")
        print(f"  Successful rollouts (n={success_stats['n']}):")
        print(f"    mean top-{int(args.top_pct*100)}%% entropy: {success_stats['mean_top_pct_entropy']}")
        print(f"    forking density:                 {success_stats['forking_density']}")
        print(f"  Failed rollouts (n={failed_stats['n']}):")
        print(f"    mean top-{int(args.top_pct*100)}%% entropy: {failed_stats['mean_top_pct_entropy']}")
        print(f"  Time: {elapsed:.1f}s")

    output = {
        "model": args.model,
        "temperature": args.temperature,
        "num_generations": args.num_generations,
        "max_samples": args.max_samples,
        "top_pct": args.top_pct,
        "per_dataset": results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*75}")
    print("FORKING-TOKEN ENTROPY SUMMARY (successful rollouts)")
    print(f"{'='*75}")
    print(f"{'Dataset':<12} {'Cat':<8} {'SuccessRt':>10} {'Top20% H':>10} {'ForkDens':>10}")
    print("-" * 65)
    for ds_name in EVAL_REGISTRY:
        if ds_name in results and "error" not in results[ds_name]:
            r = results[ds_name]
            s = r["successful_rollouts"]
            top_h = s.get("mean_top_pct_entropy") or 0.0
            fd = s.get("forking_density") or 0.0
            print(f"{ds_name:<12} {r['category']:<8} {r['success_rate']:>10.3f} "
                  f"{top_h:>10.4f} {fd:>10.4f}")


if __name__ == "__main__":
    main()
