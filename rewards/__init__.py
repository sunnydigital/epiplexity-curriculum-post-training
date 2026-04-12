"""
Reward function dispatcher for GRPO training.

dispatch_reward is the single entry point passed to trl.GRPOTrainer's
reward_funcs argument. It reads the "category" field from each example's
metadata and routes to the appropriate reward function.

TRL calls reward functions with the signature:
    reward_fn(completions: list[str], **kwargs) -> list[float]

where kwargs contains the original prompt batch fields.
"""
from __future__ import annotations

from rewards.math import math_reward
from rewards.code import code_reward
from rewards.qa import qa_reward


def dispatch_reward(completions: list[str], **kwargs) -> list[float]:
    """
    Category-aware reward dispatcher compatible with trl.GRPOTrainer.

    Expects kwargs to contain:
        "answer":   list[str] — ground truth answers
        "category": list[str] — one of "math" | "code" | "logical" | "qa"
        "dataset":  list[str] — dataset name (needed for qa sub-dispatch)

    Args:
        completions: Model-generated strings for the current batch.
        **kwargs: Batch fields from the dataset (prompt, answer, category, dataset).

    Returns:
        List of float rewards, one per completion.
    """
    answers: list[str] = kwargs["answer"]
    categories: list[str] = kwargs["category"]
    datasets: list[str] = kwargs["dataset"]

    rewards: list[float] = []

    # Collect indices by category for batched reward calls
    math_idx, code_idx, qa_idx = [], [], []
    for i, cat in enumerate(categories):
        if cat == "math":
            math_idx.append(i)
        elif cat == "code":
            code_idx.append(i)
        else:  # "logical" and "qa" both use qa_reward
            qa_idx.append(i)

    # Pre-allocate results list
    result = [0.0] * len(completions)

    if math_idx:
        math_completions = [completions[i] for i in math_idx]
        math_answers = [answers[i] for i in math_idx]
        for idx, r in zip(math_idx, math_reward(math_completions, math_answers)):
            result[idx] = r

    if code_idx:
        code_completions = [completions[i] for i in code_idx]
        code_answers = [answers[i] for i in code_idx]
        for idx, r in zip(code_idx, code_reward(code_completions, code_answers)):
            result[idx] = r

    if qa_idx:
        qa_completions = [completions[i] for i in qa_idx]
        qa_answers = [answers[i] for i in qa_idx]
        qa_datasets = [datasets[i] for i in qa_idx]
        for idx, r in zip(qa_idx, qa_reward(qa_completions, qa_answers, qa_datasets)):
            result[idx] = r

    return result
