"""
Dataset registry mapping dataset names to HuggingFace paths, configs, splits,
categories, and formatter functions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.datasets import (
        format_gsm8k,
        format_math,
        format_humaneval,
        format_mbpp,
        format_mmlu,
        format_arc,
        format_triviaqa,
        format_boolq,
    )


def _get_formatters():
    from data.datasets import (
        format_gsm8k,
        format_math,
        format_humaneval,
        format_mbpp,
        format_mmlu,
        format_arc,
        format_triviaqa,
        format_boolq,
    )
    return {
        "gsm8k": format_gsm8k,
        "math": format_math,
        "humaneval": format_humaneval,
        "mbpp": format_mbpp,
        "mmlu": format_mmlu,
        "arc": format_arc,
        "triviaqa": format_triviaqa,
        "boolq": format_boolq,
    }


# Registry schema per dataset:
#   hf_path:    HuggingFace dataset identifier
#   hf_config:  dataset config/subset name (None if not needed)
#   split:      train split name
#   category:   one of "math" | "code" | "logical" | "qa"
#   formatter:  callable (example dict) -> {"prompt", "answer", "dataset", "category"}
DATASET_REGISTRY: dict[str, dict] = {
    "gsm8k": {
        "hf_path": "openai/gsm8k",
        "hf_config": "main",
        "split": "train",
        "category": "math",
    },
    "math": {
        "hf_path": "lighteval/MATH",
        "hf_config": "all",
        "split": "train",
        "category": "math",
    },
    "humaneval": {
        "hf_path": "openai/openai_humaneval",
        "hf_config": None,
        "split": "test",  # HumanEval only has a test split
        "category": "code",
    },
    "mbpp": {
        "hf_path": "google-research-datasets/mbpp",
        "hf_config": "sanitized",
        "split": "train",
        "category": "code",
    },
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_config": "all",
        "split": "test",
        "category": "logical",
    },
    "arc": {
        "hf_path": "allenai/ai2_arc",
        "hf_config": "ARC-Challenge",
        "split": "train",
        "category": "logical",
    },
    "triviaqa": {
        "hf_path": "mandarjoshi/trivia_qa",
        "hf_config": "rc",
        "split": "train",
        "category": "qa",
    },
    "boolq": {
        "hf_path": "google/boolq",
        "hf_config": None,
        "split": "train",
        "category": "qa",
    },
}


def get_registry_with_formatters() -> dict[str, dict]:
    """Return DATASET_REGISTRY with formatter functions attached."""
    formatters = _get_formatters()
    registry = {}
    for name, cfg in DATASET_REGISTRY.items():
        registry[name] = {**cfg, "formatter": formatters[name]}
    return registry
