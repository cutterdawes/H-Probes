"""Lightweight shared helpers."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def sanitize_tag(text: str) -> str:
    allowed = []
    for ch in text:
        if ch.isalnum():
            allowed.append(ch)
        elif ch in "-._":
            allowed.append(ch)
        elif ch in "/\\":
            allowed.append("_")
        else:
            allowed.append("-")
    tag = "".join(allowed).strip("-")
    return tag or "tag"


def parse_range_arg(values: Sequence[int], name: str, min_value: int) -> Tuple[int, int]:
    """Parse a CLI range that accepts one or two integers."""

    if len(values) == 1:
        range_min = range_max = values[0]
    elif len(values) == 2:
        range_min, range_max = values
    else:
        raise ValueError(f"{name} must have 1 or 2 integers")
    if range_min < min_value or range_max < min_value or range_max < range_min:
        raise ValueError(f"{name} must be >= {min_value} and max >= min")
    return range_min, range_max


def split_exact_only(
    records: List[object],
    train_frac: float,
    seed: int,
    exact_attr: str = "exact_match",
) -> Tuple[List[object], List[object]]:
    """Split only exact-match responses into the train set; keep the rest in test."""

    rng = np.random.default_rng(seed)
    exact_indices = np.array(
        [idx for idx, rec in enumerate(records) if bool(getattr(rec, exact_attr, False))],
        dtype=int,
    )
    rng.shuffle(exact_indices)
    split = int(len(exact_indices) * train_frac)
    train_idx = set(exact_indices[:split].tolist())
    train_records = [records[i] for i in sorted(train_idx)]
    test_records = [rec for i, rec in enumerate(records) if i not in train_idx]
    return train_records, test_records


def split_balanced(
    records: List[object],
    train_frac: float,
    seed: int,
    exact_attr: str = "exact_match",
) -> Tuple[List[object], List[object]]:
    """Split responses with exact/inexact proportions preserved."""

    rng = np.random.default_rng(seed)
    exact_indices = np.array(
        [idx for idx, rec in enumerate(records) if bool(getattr(rec, exact_attr, False))],
        dtype=int,
    )
    inexact_indices = np.array(
        [idx for idx, rec in enumerate(records) if not bool(getattr(rec, exact_attr, False))],
        dtype=int,
    )
    rng.shuffle(exact_indices)
    rng.shuffle(inexact_indices)
    exact_split = int(len(exact_indices) * train_frac)
    inexact_split = int(len(inexact_indices) * train_frac)
    train_idx = set(exact_indices[:exact_split].tolist() + inexact_indices[:inexact_split].tolist())
    train_records = [records[i] for i in sorted(train_idx)]
    test_records = [rec for i, rec in enumerate(records) if i not in train_idx]
    return train_records, test_records
