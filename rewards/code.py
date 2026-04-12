"""
Code reward function for HumanEval and MBPP datasets.

Phase 1: String-match on cleaned solution (whitespace/comment normalized).
Phase 2: Stub interface for execution-based sandbox rewards (future work).
"""
from __future__ import annotations

import re


def _clean_code(code: str) -> str:
    """Strip trailing whitespace and blank lines; normalize indentation."""
    lines = [line.rstrip() for line in code.splitlines()]
    # Drop leading/trailing blank lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _strip_markdown_fence(text: str) -> str:
    """Remove ```python ... ``` fences if the model wraps its output."""
    pattern = r"```(?:python)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text


def code_reward(completions: list[str], answers: list[str]) -> list[float]:
    """
    Compute rewards for code completions using string-match.

    Args:
        completions: List of model-generated code strings.
        answers: List of canonical solution strings.

    Returns:
        List of floats:
          1.0 — cleaned completion matches cleaned canonical solution
          0.0 — mismatch
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        pred = _clean_code(_strip_markdown_fence(completion))
        gold = _clean_code(answer)
        rewards.append(1.0 if pred == gold else 0.0)
    return rewards


def execute_code_reward(
    completion: str,
    test_cases: list[str],
    timeout: float = 5.0,
) -> float:
    """
    Stub for execution-based code reward.
    Runs the completion against test_cases in a sandboxed environment.

    NOT IMPLEMENTED — requires a secure execution sandbox (e.g., subprocess
    with restricted permissions, or a remote judge API).

    Args:
        completion: Model-generated code string.
        test_cases: List of assertion strings (e.g., from MBPP's test_list).
        timeout: Seconds before execution is killed.

    Returns:
        Float reward between 0.0 and 1.0 (fraction of passing test cases).
    """
    raise NotImplementedError(
        "Execution-based code reward requires sandbox integration. "
        "Implement with subprocess + resource limits or a remote judge API."
    )
