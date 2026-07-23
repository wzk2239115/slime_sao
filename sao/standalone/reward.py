"""Reward functions for math reasoning (AIME, etc.).

Extracts \\boxed{} answer from response and compares to ground truth.
"""
from __future__ import annotations

import re
import signal


def extract_boxed(text: str | None) -> str | None:
    """Extract the last \\boxed{...} content from text."""
    if not text:
        return None
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + 7
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i].strip()
            depth -= 1
    return None


def normalize_answer(ans: str) -> str:
    """Normalize a math answer string for comparison."""
    ans = ans.strip()
    ans = ans.replace("\\,", "").replace("\\ ", "")
    ans = ans.replace("\\text{", "").replace("}", "")
    ans = ans.replace("$", "").replace("%", "")
    ans = ans.replace(" ", "")

    try:
        val = float(ans)
        if abs(val - round(val)) < 1e-6:
            return str(int(round(val)))
        return str(val)
    except (ValueError, OverflowError):
        pass

    try:
        from fractions import Fraction
        frac = Fraction(ans)
        return str(frac)
    except Exception:
        pass

    return ans


def math_reward(response: str | None, ground_truth: str) -> float:
    """Binary reward: 1.0 if extracted answer matches ground truth, else 0.0."""
    if not response:
        return 0.0
    pred = extract_boxed(response)
    if pred is None:
        return 0.0
    return 1.0 if normalize_answer(pred) == normalize_answer(ground_truth) else 0.0


def math_reward_batch(responses: list[str], ground_truths: list[str]) -> list[float]:
    """Batch math reward."""
    return [math_reward(r, gt) for r, gt in zip(responses, ground_truths)]
