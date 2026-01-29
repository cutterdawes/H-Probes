"""GSM8K-specific prompting and grading helpers."""

from __future__ import annotations

import re
from typing import Optional, Tuple

SYSTEM_PROMPT = "You are a careful math assistant. Provide the final answer exactly as requested."


def gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math word problem.\n"
        "Show your reasoning in clear steps.\n"
        "The last line MUST be exactly: #### <number>\n\n"
        f"Problem:\n{question}\n"
        "Answer:\n"
    )


def extract_gsm8k_final_answer_with_method(text: str) -> Tuple[Optional[str], str]:
    if not text:
        return None, "empty"

    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1).strip(), "hashes"

    tail = text[-2000:]
    cues = [
        (r"(?:final\s+answer\s*[:=]?\s*)([-+]?\d+(?:\.\d+)?)", "final_answer"),
        (r"(?:the\s+answer\s+is\s*)([-+]?\d+(?:\.\d+)?)", "the_answer_is"),
        (r"(?:so\s*,?\s*the\s+answer\s+is\s*)([-+]?\d+(?:\.\d+)?)", "so_answer_is"),
        (r"(?:therefore\s*,?\s*)([-+]?\d+(?:\.\d+)?)", "therefore"),
        (r"(?:thus\s*,?\s*)([-+]?\d+(?:\.\d+)?)", "thus"),
    ]
    for pattern, name in cues:
        matches = list(re.finditer(pattern, tail, flags=re.IGNORECASE))
        if matches:
            return matches[-1].group(1).strip(), name

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", tail)
    return (nums[-1].strip(), "last_number") if nums else (None, "none")


def grade_gsm8k_answer(pred: Optional[str], gold_text: str) -> bool:
    gold, _ = extract_gsm8k_final_answer_with_method(gold_text)
    if pred is None or gold is None:
        return False
    try:
        return float(pred) == float(gold)
    except Exception:
        return pred.strip() == gold.strip()
