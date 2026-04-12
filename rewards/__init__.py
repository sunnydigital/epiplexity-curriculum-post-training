"""
Reward function dispatcher for GRPO training.

dispatch_reward is the single entry point passed to trl.GRPOTrainer's
reward_funcs argument. It reads the "category" field from each example's
metadata and routes to the appropriate reward function.

RewardTracker wraps dispatch_reward to accumulate per-dataset reward
statistics during training, enabling apples-to-apples logging.

TRL calls reward functions with the signature:
    reward_fn(completions: list[str], **kwargs) -> list[float]

where kwargs contains the original prompt batch fields.
"""
from __future__ import annotations

from collections import defaultdict

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


class RewardTracker:
    """
    Wraps dispatch_reward to accumulate per-dataset reward statistics.

    Usage:
        tracker = RewardTracker()
        # Pass tracker as reward_fn to GRPOTrainer
        trainer = GRPOTrainer(reward_funcs=[tracker], ...)

        # At any point, retrieve stats:
        stats = tracker.get_stats()
        # {'gsm8k': {'mean': 0.35, 'count': 120, 'sum': 42.0}, ...}

        # Log & reset (e.g., in a callback):
        stats = tracker.get_and_reset_stats()
    """

    # TRL's GRPOTrainer reads reward_func.__name__ for logging
    __name__ = "dispatch_reward"

    def __init__(self):
        self._per_dataset: dict[str, list[float]] = defaultdict(list)
        self._per_category: dict[str, list[float]] = defaultdict(list)
        self._total: list[float] = []

    def __call__(self, completions: list[str], **kwargs) -> list[float]:
        """TRL-compatible reward function that tracks per-dataset statistics."""
        rewards = dispatch_reward(completions, **kwargs)

        # Accumulate per-dataset and per-category
        datasets: list[str] = kwargs.get("dataset", [])
        categories: list[str] = kwargs.get("category", [])
        for r, ds, cat in zip(rewards, datasets, categories):
            self._per_dataset[ds].append(r)
            self._per_category[cat].append(r)
            self._total.append(r)

        return rewards

    def get_stats(self) -> dict[str, dict]:
        """Return per-dataset reward statistics without resetting."""
        stats = {}
        for name, rewards in self._per_dataset.items():
            if rewards:
                stats[name] = {
                    "mean": sum(rewards) / len(rewards),
                    "count": len(rewards),
                    "sum": sum(rewards),
                }
        return stats

    def get_category_stats(self) -> dict[str, dict]:
        """Return per-category reward statistics without resetting."""
        stats = {}
        for cat, rewards in self._per_category.items():
            if rewards:
                stats[cat] = {
                    "mean": sum(rewards) / len(rewards),
                    "count": len(rewards),
                }
        return stats

    def get_and_reset_stats(self) -> dict[str, dict]:
        """Return per-dataset stats and reset accumulators."""
        stats = self.get_stats()
        self._per_dataset.clear()
        self._per_category.clear()
        self._total.clear()
        return stats

    def summary_str(self) -> str:
        """Formatted string of current per-dataset reward means."""
        stats = self.get_stats()
        if not stats:
            return "No rewards tracked yet"
        lines = []
        for name in sorted(stats):
            s = stats[name]
            lines.append(f"  {name}: mean={s['mean']:.4f} (n={s['count']})")
        return "Per-dataset rewards:\n" + "\n".join(lines)
