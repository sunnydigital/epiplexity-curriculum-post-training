"""
Per-dataset formatter functions and combined dataset loader.

Each formatter has signature:
    (example: dict) -> {"prompt": str, "answer": str, "dataset": str, "category": str}

The "prompt" field is what gets fed to the model.
The "answer" field is the ground-truth used by reward functions.
"""
from __future__ import annotations

import re
from typing import Callable

from datasets import Dataset, concatenate_datasets, load_dataset


# ---------------------------------------------------------------------------
# Math formatters
# ---------------------------------------------------------------------------

def format_gsm8k(example: dict) -> dict:
    prompt = (
        "Solve the following math problem step by step.\n\n"
        f"Problem: {example['question']}\n\n"
        "Solution:"
    )
    # GSM8K answers end with "#### <number>"
    answer_match = re.search(r"####\s*(.+)$", example["answer"], re.MULTILINE)
    answer = answer_match.group(1).strip() if answer_match else example["answer"].strip()
    return {"prompt": prompt, "answer": answer, "dataset": "gsm8k", "category": "math"}


def format_math(example: dict) -> dict:
    prompt = (
        "Solve the following math problem. Show your work and box your final answer.\n\n"
        f"Problem: {example['problem']}\n\n"
        "Solution:"
    )
    return {
        "prompt": prompt,
        "answer": example["solution"],
        "dataset": "math",
        "category": "math",
    }


# ---------------------------------------------------------------------------
# Code formatters
# ---------------------------------------------------------------------------

def format_humaneval(example: dict) -> dict:
    prompt = (
        "Complete the following Python function:\n\n"
        f"{example['prompt']}"
    )
    return {
        "prompt": prompt,
        "answer": example["canonical_solution"],
        "dataset": "humaneval",
        "category": "code",
    }


def format_mbpp(example: dict) -> dict:
    prompt = (
        "Write a Python function to solve the following problem:\n\n"
        f"{example['text']}\n\n"
        "Your code:"
    )
    return {
        "prompt": prompt,
        "answer": example["code"],
        "dataset": "mbpp",
        "category": "code",
    }


# ---------------------------------------------------------------------------
# Logical formatters
# ---------------------------------------------------------------------------

_LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def _format_mcq(question: str, choices: list[str], answer_idx: int, dataset: str) -> dict:
    choices_text = "\n".join(f"{_LETTER_MAP[i]}. {c}" for i, c in enumerate(choices))
    prompt = (
        f"Answer the following multiple choice question by responding with only the letter "
        f"(A, B, C, or D).\n\n"
        f"Question: {question}\n\n"
        f"{choices_text}\n\n"
        "Answer:"
    )
    return {
        "prompt": prompt,
        "answer": _LETTER_MAP[answer_idx],
        "dataset": dataset,
        "category": "logical",
    }


def format_mmlu(example: dict) -> dict:
    return _format_mcq(
        question=example["question"],
        choices=example["choices"],
        answer_idx=example["answer"],
        dataset="mmlu",
    )


def format_arc(example: dict) -> dict:
    choices = example["choices"]["text"]
    labels = example["choices"]["label"]
    # Normalize labels to 0-indexed integers
    answer_key = example["answerKey"]
    if answer_key in ("A", "B", "C", "D"):
        answer_idx = ord(answer_key) - ord("A")
    elif answer_key in ("1", "2", "3", "4"):
        answer_idx = int(answer_key) - 1
    else:
        answer_idx = 0
    # Reorder choices to A/B/C/D order using labels
    ordered_choices = [None] * len(choices)
    for label, text in zip(labels, choices):
        if label in ("A", "B", "C", "D"):
            ordered_choices[ord(label) - ord("A")] = text
        elif label in ("1", "2", "3", "4"):
            ordered_choices[int(label) - 1] = text
    ordered_choices = [c for c in ordered_choices if c is not None]
    return _format_mcq(
        question=example["question"],
        choices=ordered_choices,
        answer_idx=answer_idx,
        dataset="arc",
    )


# ---------------------------------------------------------------------------
# QA formatters
# ---------------------------------------------------------------------------

def format_triviaqa(example: dict) -> dict:
    prompt = (
        "Answer the following question concisely.\n\n"
        f"Question: {example['question']}\n\n"
        "Answer:"
    )
    # TriviaQA has a nested answer structure; prefer normalized_value
    answer_data = example.get("answer", {})
    if isinstance(answer_data, dict):
        answer = answer_data.get("normalized_value") or answer_data.get("value", "")
    else:
        answer = str(answer_data)
    return {"prompt": prompt, "answer": answer, "dataset": "triviaqa", "category": "qa"}


def format_boolq(example: dict) -> dict:
    prompt = (
        "Read the passage and answer the question with 'yes' or 'no'.\n\n"
        f"Passage: {example['passage']}\n\n"
        f"Question: {example['question']}\n\n"
        "Answer:"
    )
    answer = "yes" if example["answer"] else "no"
    return {"prompt": prompt, "answer": answer, "dataset": "boolq", "category": "qa"}


# ---------------------------------------------------------------------------
# Combined loader
# ---------------------------------------------------------------------------

def load_all_datasets(
    registry: dict[str, dict],
    max_samples_per_dataset: int | None = None,
) -> dict[str, Dataset]:
    """
    Load and format all datasets in the registry.

    Args:
        registry: Output of get_registry_with_formatters() from data.registry.
        max_samples_per_dataset: If set, truncate each dataset to this many samples.
            Useful for smoke tests.

    Returns:
        Dict mapping dataset name to a formatted HuggingFace Dataset with columns:
        ["prompt", "answer", "dataset", "category"]
    """
    loaded: dict[str, Dataset] = {}

    for name, cfg in registry.items():
        print(f"Loading {name} from {cfg['hf_path']}...")
        kwargs: dict = {"path": cfg["hf_path"], "split": cfg["split"]}
        if cfg.get("hf_config"):
            kwargs["name"] = cfg["hf_config"]

        raw = load_dataset(**kwargs)

        formatter: Callable = cfg["formatter"]
        formatted = raw.map(
            formatter,
            remove_columns=raw.column_names,
            desc=f"Formatting {name}",
        )

        if max_samples_per_dataset is not None:
            formatted = formatted.select(range(min(max_samples_per_dataset, len(formatted))))

        loaded[name] = formatted
        print(f"  → {len(formatted):,} examples")

    return loaded
