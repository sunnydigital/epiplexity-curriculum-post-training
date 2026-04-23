"""
Rollout Epiplexity — GRPO-native compressibility measurement.

Applies the prequential coding procedure (Finzi et al., 2026) to the
GRPO advantage-weighted surrogate loss on policy-sampled rollouts,
rather than to teacher-forcing CE loss on dataset tokens.

Procedure per dataset:
    1. Load the policy model (Qwen2.5-3B-Instruct) with a fresh LoRA adapter.
    2. Partition prompts into K sequential chunks.
    3. For each chunk i:
         a. Sample G completions per prompt from the *current* policy.
         b. Score completions with the dataset's reward function.
         c. Compute group-normalized advantages
                A_ij = (r_ij - mean_j) / (std_j + eps)
         d. MEASURE the GRPO surrogate L_i (no grad) on those rollouts.
         e. TRAIN: backprop L_i and take one optimizer step on this chunk.
    4. Integrate (step, L_i) into K_auc via the trapezoidal rule, then
       normalize by total rollout-token count.

Output schema mirrors data/reward_variance_3b.json so compare_results.py
can pick it up alongside the existing predictors.

Usage:
    python measure_rollout_epiplexity.py \\
        --model Qwen/Qwen2.5-3B-Instruct \\
        --num-chunks 50 \\
        --prompts-per-chunk 16 \\
        --num-generations 8 \\
        --output data/rollout_epiplexity_3b.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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


# Mirrors EVAL_REGISTRY in measure_reward_variance.py — same eval splits
# so the rollout distributions are directly comparable across predictors.
EVAL_REGISTRY = {
    "gsm8k": {
        "hf_path": "openai/gsm8k", "hf_config": "main", "split": "test",
        "formatter": format_gsm8k, "category": "math",
    },
    "math": {
        "hf_path": "EleutherAI/hendrycks_math", "hf_config": "algebra",
        "split": "test", "formatter": format_math, "category": "math",
    },
    "humaneval": {
        "hf_path": "openai/openai_humaneval", "hf_config": None,
        "split": "test", "formatter": format_humaneval, "category": "code",
    },
    "mbpp": {
        "hf_path": "google-research-datasets/mbpp", "hf_config": "sanitized",
        "split": "test", "formatter": format_mbpp, "category": "code",
    },
    "mmlu": {
        "hf_path": "cais/mmlu", "hf_config": "all", "split": "validation",
        "formatter": format_mmlu, "category": "logical",
    },
    "arc": {
        "hf_path": "allenai/ai2_arc", "hf_config": "ARC-Challenge",
        "split": "test", "formatter": format_arc, "category": "logical",
    },
    "triviaqa": {
        "hf_path": "mandarjoshi/trivia_qa", "hf_config": "rc",
        "split": "validation", "formatter": format_triviaqa, "category": "qa",
    },
    "boolq": {
        "hf_path": "google/boolq", "hf_config": None, "split": "validation",
        "formatter": format_boolq, "category": "qa",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rollout Epiplexity: prequential K_auc on GRPO surrogate loss"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Policy model (HF name or local path)")
    parser.add_argument("--output", type=str, default="data/rollout_epiplexity_3b.json")
    parser.add_argument("--num-chunks", type=int, default=50,
                        help="K = number of prequential chunks per dataset")
    parser.add_argument("--prompts-per-chunk", type=int, default=16,
                        help="Prompts sampled per chunk")
    parser.add_argument("--num-generations", type=int, default=8,
                        help="G = completions per prompt (matches GRPO group size)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature for rollouts")
    parser.add_argument("--lr", type=float, default=3.0e-6,
                        help="Inner-loop learning rate (matches training_config.yaml)")
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank for inner adapter")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for prompt shuffling and LoRA init")
    parser.add_argument("--datasets", type=str, nargs="*", default=None,
                        help="Subset of datasets (default: all 8)")
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


# ---------------------------------------------------------------------------
# Sampling + scoring (mirrors measure_reward_variance.py but returns the
# raw token IDs so we can compute the GRPO surrogate downstream)
# ---------------------------------------------------------------------------

def sample_rollouts(
    model, tokenizer, prompt: str, num_generations: int,
    max_new_tokens: int, max_prompt_length: int, temperature: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[str], int]:
    """
    Sample G completions for a single prompt.

    Returns:
        full_ids       : LongTensor [G, prompt_len + comp_len]
        completion_mask: BoolTensor [G, prompt_len + comp_len]  — True over
                         completion tokens (False over prompt + padding).
        completions    : list[str] of decoded completions
        prompt_len     : int — token length of the (left-padded) prompt
    """
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True,
        max_length=max_prompt_length,
    ).to(device)
    prompt_len = enc["input_ids"].shape[1]

    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            num_return_sequences=num_generations,
            pad_token_id=tokenizer.pad_token_id,
        )

    completions = [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in outputs
    ]

    # Build the completion mask: 1 over generated tokens that aren't pad.
    pad_id = tokenizer.pad_token_id
    completion_mask = torch.zeros_like(outputs, dtype=torch.bool)
    completion_mask[:, prompt_len:] = outputs[:, prompt_len:] != pad_id

    return outputs, completion_mask, completions, prompt_len


def score_completions(
    completions: list[str], answer: str, dataset_name: str, category: str,
) -> list[float]:
    """Dispatch to the correct reward function (matches RewardTracker logic)."""
    answers = [answer] * len(completions)
    if category == "math":
        return math_reward(completions, answers)
    if category == "code":
        return code_reward(completions, answers)
    datasets_list = [dataset_name] * len(completions)
    return qa_reward(completions, answers, datasets_list)


def group_advantages(rewards: list[float], eps: float = 1e-4) -> np.ndarray:
    """GRPO group-relative advantages: (r - mean) / (std + eps)."""
    arr = np.asarray(rewards, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()
    return (arr - mean) / (std + eps)


# ---------------------------------------------------------------------------
# GRPO surrogate loss on a batch of rollouts
# ---------------------------------------------------------------------------

def compute_grpo_surrogate(
    model,
    full_ids: torch.Tensor,           # [G, T]
    completion_mask: torch.Tensor,    # [G, T] bool
    advantages: torch.Tensor,         # [G]
) -> tuple[torch.Tensor, int]:
    """
    Per-token NLL of the *sampled* tokens, weighted by the per-completion
    advantage, averaged over completion tokens. This is the policy-gradient
    surrogate loss used by GRPO (with the KL term set to zero so the
    measurement isolates advantage compression).

    Returns (mean_loss_nats_per_token, n_completion_tokens).
    """
    attn = (full_ids != 0).long() if full_ids.min() >= 0 else torch.ones_like(full_ids)
    outputs = model(input_ids=full_ids, attention_mask=attn)
    logits = outputs.logits[:, :-1, :]            # [G, T-1, V]
    targets = full_ids[:, 1:]                     # [G, T-1]
    mask = completion_mask[:, 1:].float()         # [G, T-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [G, T-1]

    # Per-completion mean log-prob, weighted by advantage.
    n_tok_per_comp = mask.sum(dim=-1).clamp_min(1.0)                       # [G]
    comp_logp = (token_logp * mask).sum(dim=-1) / n_tok_per_comp           # [G]

    # GRPO surrogate (sign convention: minimize loss = maximize A * logp).
    surrogate = -(advantages * comp_logp).mean()

    n_completion_tokens = int(mask.sum().item())
    return surrogate, n_completion_tokens


# ---------------------------------------------------------------------------
# K_auc integration
# ---------------------------------------------------------------------------

def integrate_k_auc(loss_curve: list[tuple[int, float, int]]) -> float:
    """
    Trapezoidal integration of (loss_t - loss_final) over tokens-seen,
    in nats. Caller converts to bits with /ln(2).

    loss_curve: list of (chunk_index, loss_nats, cumulative_completion_tokens).
    """
    if len(loss_curve) < 2:
        return 0.0
    final_loss = loss_curve[-1][1]
    k_auc_nats = 0.0
    for i in range(1, len(loss_curve)):
        _, loss_prev, tok_prev = loss_curve[i - 1]
        _, loss_curr, tok_curr = loss_curve[i]
        delta_tokens = tok_curr - tok_prev
        avg_excess = ((loss_prev - final_loss) + (loss_curr - final_loss)) / 2.0
        k_auc_nats += max(0.0, avg_excess) * delta_tokens
    return k_auc_nats


# ---------------------------------------------------------------------------
# Per-dataset measurement
# ---------------------------------------------------------------------------

def estimate_rollout_epiplexity(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    answers: list[str],
    dataset_name: str,
    category: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    """Run prequential rollout-K_auc on a single dataset."""
    from peft import LoraConfig, get_peft_model, TaskType

    # Fresh LoRA adapter so each dataset starts from identical weights.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        lora_dropout=0.0,           # deterministic forward; we want clean L_i
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    policy = get_peft_model(base_model, lora_config)
    policy.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )

    rng = np.random.default_rng(args.seed)
    prompt_indices = np.arange(len(prompts))
    rng.shuffle(prompt_indices)

    needed = args.num_chunks * args.prompts_per_chunk
    if needed > len(prompt_indices):
        # Reuse with replacement if dataset is small (e.g., HumanEval = 164).
        prompt_indices = rng.choice(prompt_indices, size=needed, replace=True)
    else:
        prompt_indices = prompt_indices[:needed]

    loss_curve: list[tuple[int, float, int]] = []
    cumulative_tokens = 0
    advantage_magnitudes: list[float] = []
    zero_var_groups = 0
    total_groups = 0
    mean_rewards_per_chunk: list[float] = []

    for chunk_idx in range(args.num_chunks):
        chunk_start = chunk_idx * args.prompts_per_chunk
        chunk_end = chunk_start + args.prompts_per_chunk
        chunk_prompt_idxs = prompt_indices[chunk_start:chunk_end]

        # ----- 1. Sample rollouts + score (no grad) -----
        chunk_rollouts: list[dict] = []
        for pi in chunk_prompt_idxs:
            full_ids, comp_mask, completions, _ = sample_rollouts(
                policy, tokenizer, prompts[pi],
                num_generations=args.num_generations,
                max_new_tokens=args.max_new_tokens,
                max_prompt_length=args.max_prompt_length,
                temperature=args.temperature,
                device=device,
            )
            rewards = score_completions(
                completions, answers[pi], dataset_name, category,
            )
            adv = group_advantages(rewards)
            advantage_magnitudes.append(float(np.mean(np.abs(adv))))
            mean_rewards_per_chunk.append(float(np.mean(rewards)))
            total_groups += 1
            if float(np.var(rewards)) < 1e-8:
                zero_var_groups += 1
            chunk_rollouts.append({
                "full_ids": full_ids,
                "comp_mask": comp_mask,
                "advantages": torch.tensor(adv, dtype=torch.float32, device=device),
            })

        # ----- 2. MEASURE L_i on these rollouts (no grad) -----
        policy.eval()
        with torch.no_grad():
            chunk_loss_nats = 0.0
            chunk_tokens = 0
            for r in chunk_rollouts:
                loss, n_tok = compute_grpo_surrogate(
                    policy, r["full_ids"], r["comp_mask"], r["advantages"],
                )
                # Weight by completion-token count for token-level integration
                chunk_loss_nats += loss.item() * n_tok
                chunk_tokens += n_tok
        if chunk_tokens > 0:
            measured_loss = chunk_loss_nats / chunk_tokens
            cumulative_tokens += chunk_tokens
            loss_curve.append((chunk_idx, measured_loss, cumulative_tokens))
            print(
                f"    chunk {chunk_idx + 1}/{args.num_chunks}: "
                f"L={measured_loss:+.4f} nats/tok  tokens={cumulative_tokens:,}  "
                f"|A|={advantage_magnitudes[-1]:.3f}  "
                f"zero_var={zero_var_groups}/{total_groups}"
            )

        # ----- 3. TRAIN one step per prompt-group on this chunk -----
        policy.train()
        for r in chunk_rollouts:
            loss, _ = compute_grpo_surrogate(
                policy, r["full_ids"], r["comp_mask"], r["advantages"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0,
            )
            optimizer.step()
            optimizer.zero_grad()

        # Free per-chunk rollout tensors
        del chunk_rollouts
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ----- 4. K_auc integration -----
    k_auc_nats = integrate_k_auc(loss_curve)
    k_auc_bits = k_auc_nats / math.log(2)
    rollout_epi_per_token = k_auc_bits / max(cumulative_tokens, 1)

    initial_loss = loss_curve[0][1] if loss_curve else 0.0
    final_loss = loss_curve[-1][1] if loss_curve else 0.0

    # Detach LoRA adapter so the next dataset can attach a fresh one.
    # peft's get_peft_model wraps base_model in-place; we unwrap it.
    if hasattr(policy, "unload"):
        policy.unload()
    elif hasattr(policy, "base_model"):
        # Fallback for older peft versions
        pass
    del policy, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "rollout_epiplexity_per_token": round(rollout_epi_per_token, 6),
        "k_auc_bits": round(k_auc_bits, 2),
        "initial_surrogate_nats": round(initial_loss, 4),
        "final_surrogate_nats": round(final_loss, 4),
        "surrogate_reduction": round(initial_loss - final_loss, 4),
        "total_completion_tokens": cumulative_tokens,
        "mean_advantage_magnitude": round(float(np.mean(advantage_magnitudes)), 4),
        "fraction_zero_variance_groups": round(zero_var_groups / max(total_groups, 1), 4),
        "mean_reward": round(float(np.mean(mean_rewards_per_chunk)), 4),
        "num_chunks": args.num_chunks,
        "prompts_per_chunk": args.prompts_per_chunk,
        "num_generations": args.num_generations,
        "loss_curve": [(s, round(l, 4)) for s, l, _ in loss_curve],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Rollout Epiplexity (Finzi-style K_auc on GRPO surrogate)")

    print(f"\nLoading policy model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    selected = list(EVAL_REGISTRY.keys())
    if args.datasets:
        selected = [d for d in selected if d in args.datasets]
        missing = set(args.datasets) - set(selected)
        if missing:
            print(f"Warning: unknown datasets ignored: {missing}")

    per_dataset: dict[str, dict] = {}

    for ds_name in selected:
        cfg = EVAL_REGISTRY[ds_name]
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
            per_dataset[ds_name] = {"error": str(e)}
            continue

        formatted = raw.map(
            cfg["formatter"], remove_columns=raw.column_names,
            desc=f"Formatting {ds_name}",
        ).cast(_FORMATTED_FEATURES)

        prompts = list(formatted["prompt"])
        answers = list(formatted["answer"])

        # Reload base model fresh per dataset so LoRA adapter starts clean.
        base_model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        base_model.to(device)

        result = estimate_rollout_epiplexity(
            base_model=base_model,
            tokenizer=tokenizer,
            prompts=prompts,
            answers=answers,
            dataset_name=ds_name,
            category=cfg["category"],
            args=args,
            device=device,
        )
        result["category"] = cfg["category"]
        result["elapsed_seconds"] = round(time.time() - start, 1)
        per_dataset[ds_name] = result

        print(
            f"  rollout_epiplexity={result['rollout_epiplexity_per_token']:.4f} bits/tok  "
            f"K_auc={result['k_auc_bits']:.1f}  "
            f"L:{result['initial_surrogate_nats']:+.3f}→{result['final_surrogate_nats']:+.3f}  "
            f"|A|={result['mean_advantage_magnitude']:.3f}  "
            f"zero_var={result['fraction_zero_variance_groups']:.2f}  "
            f"time={result['elapsed_seconds']:.1f}s"
        )

        del base_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "model": args.model,
        "method": "rollout_epiplexity",
        "reference": "arXiv:2601.03220 (Finzi et al., 2026), GRPO-native adaptation",
        "temperature": args.temperature,
        "num_chunks": args.num_chunks,
        "prompts_per_chunk": args.prompts_per_chunk,
        "num_generations": args.num_generations,
        "lr": args.lr,
        "lora_r": args.lora_r,
        "max_prompt_length": args.max_prompt_length,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "per_dataset": per_dataset,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary
    print(f"\n{'='*60}")
    print("ROLLOUT EPIPLEXITY SUMMARY (bits/token)")
    print(f"{'='*60}")
    print(f"{'Dataset':<12} {'Category':<8} {'K_auc/tok':>10} {'|A|':>8} {'zero_var':>10}")
    print("-" * 55)
    rows = [
        (n, r) for n, r in per_dataset.items()
        if "error" not in r
    ]
    for n, r in sorted(rows, key=lambda x: -x[1]["rollout_epiplexity_per_token"]):
        print(
            f"{n:<12} {r['category']:<8} "
            f"{r['rollout_epiplexity_per_token']:>10.4f} "
            f"{r['mean_advantage_magnitude']:>8.3f} "
            f"{r['fraction_zero_variance_groups']:>10.2f}"
        )


if __name__ == "__main__":
    main()
