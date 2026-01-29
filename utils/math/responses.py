"""GSM8K response parsing and split helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from cutter.utils.shared.basic import split_exact_only


@dataclass
class MathResponseRecord:
    example_id: int
    question: str
    answer: str
    prompt: str
    raw_response: str
    pred_answer: str | None
    pred_method: str
    format_ok: bool
    exact_match: bool
    partial_score: float

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "MathResponseRecord":
        return cls(
            example_id=int(row.get("example_id", -1)),
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            prompt=str(row.get("prompt", "")),
            raw_response=str(row.get("model_raw", row.get("raw_response", ""))),
            pred_answer=row.get("pred_answer"),
            pred_method=str(row.get("pred_method", "")),
            format_ok=bool(row.get("format_ok", False)),
            exact_match=bool(row.get("exact_match", row.get("correct", False))),
            partial_score=float(row.get("partial_score", 0.0)),
        )


def load_math_responses(path: Path) -> List[MathResponseRecord]:
    records: List[MathResponseRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            rec = MathResponseRecord.from_json(json.loads(line))
            if rec.example_id < 0:
                rec.example_id = idx
            records.append(rec)
    return records


def split_math_responses(
    records: List[MathResponseRecord],
    train_frac: float,
    seed: int,
) -> Tuple[List[MathResponseRecord], List[MathResponseRecord]]:
    """Split only exact-match responses into the train set; keep the rest in test."""
    return split_exact_only(records, train_frac, seed, exact_attr="exact_match")
