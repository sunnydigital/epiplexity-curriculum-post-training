"""
Math reward function for GSM8K and MATH datasets.

Extracts a numeric or boxed answer from the model completion and compares
it to the ground-truth answer. Returns a float reward in {0.0, 0.5, 1.0}.
"""
from __future__ import annotations

import re
import unicodedata


def _extract_number(text: str) -> str | None:
    """
    Extract the final numeric answer from generated text.
    Tries (in order):
      1. Content inside \\boxed{...}
      2. Content after "####"
      3. Last standalone number in the text
    """
    # \\boxed{...} (LaTeX)
    boxed = re.search(r"\\boxed\{([^}]+)\}", text)
    if boxed:
        return boxed.group(1).strip()

    # GSM8K-style "#### answer"
    hashes = re.search(r"####\s*(.+)$", text, re.MULTILINE)
    if hashes:
        return hashes.group(1).strip()

    # Last number-like token (handles negatives, decimals, fractions)
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?", text)
    if numbers:
        return numbers[-1]

    return None


def _normalize_number(s: str) -> float | None:
    """
    Normalize a numeric string to float.
    Handles: commas, fractions (3/4), unicode minus signs.
    """
    s = unicodedata.normalize("NFKC", s).strip()
    s = s.replace(",", "").replace(" ", "")
    try:
        if "/" in s:
            num, denom = s.split("/", 1)
            return float(num) / float(denom)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def math_reward(completions: list[str], answers: list[str]) -> list[float]:
    """
    Compute rewards for math completions.

    Args:
        completions: List of model-generated strings.
        answers: List of ground-truth answer strings (one per completion).

    Returns:
        List of floats:
          1.0 — correct numeric answer
          0.5 — valid number extracted but wrong value
          0.0 — no number found or unparseable
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        pred_str = _extract_number(completion)
        gold_str = _extract_number(answer) or answer.strip()

        if pred_str is None:
            rewards.append(0.0)
            continue

        pred_val = _normalize_number(pred_str)
        gold_val = _normalize_number(gold_str)

        if pred_val is None or gold_val is None:
            # Fall back to exact string match
            rewards.append(1.0 if pred_str.strip() == gold_str.strip() else 0.0)
        elif abs(pred_val - gold_val) < 1e-6:
            rewards.append(1.0)
        else:
            # Valid number but wrong — partial credit for format
            rewards.append(0.5)

    return rewards
