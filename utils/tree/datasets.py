"""Traversal dataset loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TraversalRecord:
    depth: int
    source: int
    target: int
    waypoints: List[int]
    path: List[int]
    prompt: str
    num_samples: int
    num_steps: int
    label_mapping: List[int]
    canonical_source: int
    canonical_target: int
    canonical_waypoints: List[int]
    canonical_path: List[int]

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "TraversalRecord":
        return cls(
            depth=int(row["depth"]),
            source=int(row["source"]),
            target=int(row["target"]),
            waypoints=[int(x) for x in row.get("waypoints", [row["source"], row["target"]])],
            path=[int(x) for x in row["path"]],
            prompt=str(row["prompt"]),
            num_samples=int(row.get("num_samples", row.get("sample_rate", -1))),
            num_steps=int(row.get("num_steps", 1)),
            label_mapping=[int(x) for x in row.get("label_mapping", [])],
            canonical_source=int(row.get("canonical_source", row.get("source", -1))),
            canonical_target=int(row.get("canonical_target", row.get("target", -1))),
            canonical_waypoints=[int(x) for x in row.get("canonical_waypoints", row.get("waypoints", []))],
            canonical_path=[int(x) for x in row.get("canonical_path", row.get("path", []))],
        )


def load_traversal_dataset(path: Path) -> List[TraversalRecord]:
    rows: List[TraversalRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows.append(TraversalRecord.from_json(json.loads(line)))
    return rows
