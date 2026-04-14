"""
QA reward functions for MMLU, ARC, TriviaQA, and BoolQ datasets.
"""
from __future__ import annotations

import re
import string
import unicodedata


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation/articles, and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text).lower().strip()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Strip punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _extract_letter(text: str) -> str | None:
    """
    Extract an A/B/C/D answer letter from a completion.

    Tries multiple patterns in order of specificity:
      1. "The answer is (X)" or "Answer: X"
      2. "(X)" standalone
      3. First line that is just a single letter
      4. First A-D letter at the start of the text
      5. Any standalone A-D word in the text
    """
    text = text.strip()

    # Pattern 1: "the answer is X" / "answer: X" / "answer is X"
    m = re.search(r"(?:the\s+)?answer\s*(?:is|:)\s*\(?([A-Da-d])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Pattern 2: standalone "(X)" or "[X]"
    m = re.search(r"[\(\[]\s*([A-Da-d])\s*[\)\]]", text)
    if m:
        return m.group(1).upper()

    # Pattern 3: a line that is just a single letter (possibly with period)
    for line in text.splitlines():
        line = line.strip().rstrip(".")
        if len(line) == 1 and line.upper() in "ABCD":
            return line.upper()

    # Pattern 4: starts with a letter (possibly followed by punctuation)
    m = re.match(r"\s*([A-Da-d])\s*[\.\)\:]", text)
    if m:
        return m.group(1).upper()

    # Pattern 5: first standalone A-D word
    m = re.search(r"\b([A-D])\b", text.upper())
    if m:
        return m.group(1)

    return None


def qa_reward(
    completions: list[str],
    answers: list[str],
    datasets: list[str],
) -> list[float]:
    """
    Compute rewards for QA completions, dispatching by dataset name.

    Args:
        completions: Model-generated answer strings.
        answers: Ground-truth answer strings.
        datasets: Dataset name per example ("mmlu", "arc", "triviaqa", "boolq").

    Returns:
        List of floats (1.0 correct, 0.0 incorrect).
    """
    rewards = []
    for completion, answer, dataset in zip(completions, answers, datasets):
        if dataset in ("mmlu", "arc"):
            reward = _mcq_reward(completion, answer)
        elif dataset == "triviaqa":
            reward = _triviaqa_reward(completion, answer)
        elif dataset == "boolq":
            reward = _boolq_reward(completion, answer)
        else:
            # Fallback: normalized exact match
            reward = 1.0 if _normalize_text(completion) == _normalize_text(answer) else 0.0
        rewards.append(reward)
    return rewards


def _mcq_reward(completion: str, answer: str) -> float:
    """MMLU/ARC: extract first letter and compare to expected letter."""
    pred = _extract_letter(completion)
    gold = answer.strip().upper()
    return 1.0 if (pred is not None and pred == gold) else 0.0


def _triviaqa_reward(completion: str, answer: str) -> float:
    """
    TriviaQA: normalized substring match.
    The model's answer is correct if the normalized ground-truth appears
    within the normalized completion (handles verbose completions).
    """
    pred_norm = _normalize_text(completion)
    gold_norm = _normalize_text(answer)
    return 1.0 if gold_norm in pred_norm else 0.0


def _boolq_reward(completion: str, answer: str) -> float:
    """BoolQ: map completion to yes/no, compare to ground truth."""
    comp_lower = completion.lower().strip()
    pred: str | None = None
    if re.search(r"\byes\b", comp_lower):
        pred = "yes"
    elif re.search(r"\bno\b", comp_lower):
        pred = "no"
    return 1.0 if (pred is not None and pred == answer.lower().strip()) else 0.0
