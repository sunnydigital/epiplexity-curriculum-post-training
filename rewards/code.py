"""
Code reward function for HumanEval and MBPP datasets.

Uses a multi-tier scoring approach:
  1.0 — Exact or near-exact match after normalization
  0.5-0.8 — Partial credit based on keyword/structure overlap
  0.0 — No meaningful overlap

Exact string match is essentially impossible for generated code, so we use
fuzzy matching that rewards structurally similar solutions.
"""
from __future__ import annotations

import re


def _clean_code(code: str) -> str:
    """Strip trailing whitespace, blank lines, comments; normalize indentation."""
    lines = []
    for line in code.splitlines():
        stripped = line.rstrip()
        # Remove inline comments (but keep strings)
        if "#" in stripped:
            # Naive: strip everything after # unless inside a string
            in_string = False
            for i, ch in enumerate(stripped):
                if ch in ('"', "'"):
                    in_string = not in_string
                elif ch == "#" and not in_string:
                    stripped = stripped[:i].rstrip()
                    break
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def _strip_markdown_fence(text: str) -> str:
    """Remove ```python ... ``` fences if the model wraps its output."""
    pattern = r"```(?:python)?\s*\n?(.*?)\n?```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text


def _extract_function_body(code: str) -> str:
    """Extract the body of the first function definition, if present."""
    lines = code.splitlines()
    in_func = False
    body_lines = []
    indent = 0
    for line in lines:
        if re.match(r"\s*def\s+", line):
            in_func = True
            indent = len(line) - len(line.lstrip()) + 4
            continue
        if in_func:
            if line.strip() == "":
                body_lines.append("")
            elif len(line) - len(line.lstrip()) >= indent:
                body_lines.append(line.strip())
            else:
                break
    return "\n".join(body_lines)


def _extract_keywords(code: str) -> set[str]:
    """Extract meaningful tokens from code: identifiers, operators, structures."""
    # Remove string literals to avoid matching on string content
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    code = re.sub(r'"[^"]*"', '""', code)
    code = re.sub(r"'[^']*'", "''", code)
    # Extract identifiers and keywords
    tokens = set(re.findall(r'\b[a-zA-Z_]\w*\b', code))
    # Add structural elements
    for op in ['for ', 'while ', 'if ', 'return ', 'yield ', 'import ',
               'try:', 'except', 'with ', 'class ', 'def ', 'lambda ',
               '==', '!=', '<=', '>=', '+=', '-=', '**', '//', 'not ',
               ' and ', ' or ', ' in ', 'True', 'False', 'None']:
        if op in code:
            tokens.add(op.strip())
    return tokens


def code_reward(completions: list[str], answers: list[str]) -> list[float]:
    """
    Compute rewards for code completions using multi-tier fuzzy matching.

    Scoring tiers:
      1.0 — Cleaned code matches exactly
      0.3-0.8 — Partial credit based on keyword/structure overlap (Jaccard)
      0.0 — < 20% keyword overlap

    Args:
        completions: List of model-generated code strings.
        answers: List of canonical solution strings.

    Returns:
        List of float rewards.
    """
    rewards = []
    for completion, answer in zip(completions, answers):
        pred_raw = _strip_markdown_fence(completion)
        pred = _clean_code(pred_raw)
        gold = _clean_code(answer)

        # Tier 1: exact match after cleaning
        if pred == gold:
            rewards.append(1.0)
            continue

        # Tier 2: function body match (ignoring def signature)
        pred_body = _extract_function_body(pred_raw)
        gold_body = _extract_function_body(answer)
        if pred_body and gold_body and pred_body == gold_body:
            rewards.append(1.0)
            continue

        # Tier 3: keyword overlap (Jaccard similarity)
        pred_kw = _extract_keywords(pred)
        gold_kw = _extract_keywords(gold)

        if not gold_kw:
            rewards.append(0.0)
            continue

        intersection = pred_kw & gold_kw
        union = pred_kw | gold_kw
        jaccard = len(intersection) / max(len(union), 1)

        # Scale: 0.2 threshold, linear 0.3-0.8 for partial credit
        if jaccard < 0.2:
            rewards.append(0.0)
        else:
            rewards.append(round(0.3 + 0.5 * min(jaccard, 1.0), 2))

    return rewards


def execute_code_reward(
    completion: str,
    test_cases: list[str],
    timeout: float = 5.0,
) -> float:
    """
    Stub for execution-based code reward.

    NOT IMPLEMENTED — requires a secure execution sandbox.

    Args:
        completion: Model-generated code string.
        test_cases: List of assertion strings.
        timeout: Seconds before execution is killed.

    Returns:
        Float reward between 0.0 and 1.0.
    """
    raise NotImplementedError(
        "Execution-based code reward requires sandbox integration. "
        "Implement with subprocess + resource limits or a remote judge API."
    )
