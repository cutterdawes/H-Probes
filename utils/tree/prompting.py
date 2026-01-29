"""Traversal-specific prompting utilities."""

from __future__ import annotations

import re
from typing import Optional, Sequence, Tuple, List

PATH_LINE_PATTERN = re.compile(r"PATH\s*[:|-]\s*([0-9]+(?:\s+[0-9]+)*)", re.IGNORECASE)
NUM_BLOCK_PATTERN = re.compile(r"([0-9]+(?:\s+[0-9]+){1,})")

SYSTEM_PROMPT = "You are a precise traversal assistant. You may think aloud, but ensure the final instructions are satisfied."


def extract_path(text: str) -> Tuple[List[int], Optional[str]]:
    """Extract the last PATH line block from text."""

    block: Optional[str] = None
    matches = list(PATH_LINE_PATTERN.finditer(text))
    if matches:
        block = matches[-1].group(1)
    if not block:
        return [], None
    tokens = [tok for tok in block.strip().split() if tok.strip()]
    try:
        numbers = [int(tok) for tok in tokens]
    except ValueError:
        return [], block
    return numbers, block


def grade_path(pred: Sequence[int], target: Sequence[int]) -> Tuple[bool, int, int, float]:
    """Return (exact_match, prefix_match_len, target_len, partial_score)."""

    target_len = len(target)
    match_len = 0
    for p, t in zip(pred, target):
        if p != t:
            break
        match_len += 1
    exact = match_len == target_len and len(pred) == target_len
    partial = (match_len / target_len) if target_len else 0.0
    return exact, match_len, target_len, partial
