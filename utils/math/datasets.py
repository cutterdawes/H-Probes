"""GSM8K dataset helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import List

from cutter.utils.math.prompting import gsm8k_prompt


def _load_gsm8k_split(split: str):
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for GSM8K loading. "
            "Install it in the hprobes environment before running GSM8K scripts."
        ) from exc

    dataset = load_dataset("gsm8k", "main")
    if split == "train":
        return list(dataset["train"])
    if split == "test":
        return list(dataset["test"])
    if split in {"all", "train+test", "train_test"}:
        return list(dataset["train"]) + list(dataset["test"])
    raise ValueError(f"Unsupported GSM8K split '{split}'. Use train, test, or all.")


@dataclass
class GSM8KRecord:
    example_id: int
    question: str
    answer: str
    prompt: str

    def to_json(self) -> str:
        return json.dumps(
            {
                "example_id": self.example_id,
                "question": self.question,
                "answer": self.answer,
                "prompt": self.prompt,
            },
            ensure_ascii=True,
        )


def build_gsm8k_records(num_samples: int, seed: int, split: str) -> List[GSM8KRecord]:
    data = _load_gsm8k_split(split)
    rng = Random(seed)
    rng.shuffle(data)
    if num_samples > 0:
        data = data[:num_samples]

    records: List[GSM8KRecord] = []
    for idx, ex in enumerate(data):
        question = (ex.get("question") or "").strip()
        answer = (ex.get("answer") or "").strip()
        prompt = gsm8k_prompt(question)
        records.append(GSM8KRecord(example_id=idx, question=question, answer=answer, prompt=prompt))
    return records


def load_gsm8k_jsonl(path: Path) -> List[GSM8KRecord]:
    records: List[GSM8KRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                GSM8KRecord(
                    example_id=int(row.get("example_id", len(records))),
                    question=str(row.get("question", "")),
                    answer=str(row.get("answer", "")),
                    prompt=str(row.get("prompt", "")),
                )
            )
    return records
