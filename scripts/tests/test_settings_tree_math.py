"""Smoke tests for tree and GSM8K utility paths."""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

# Ensure repository root importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ROOT = PROJECT_ROOT
TREE_PROMPTING = _load_module(
    "tree_prompting",
    ROOT / "cutter" / "utils" / "prompting" / "tree.py",
)
MATH_PROMPTING = _load_module(
    "math_prompting",
    ROOT / "cutter" / "utils" / "prompting" / "math.py",
)
MATH_RESPONSES = _load_module(
    "math_responses",
    ROOT / "cutter" / "utils" / "responses" / "math.py",
)


def test_tree_prompt_parsing() -> None:
    text = "Reasoning...\nPATH: 1 0 2"
    parsed, block = TREE_PROMPTING.extract_path(text)
    assert parsed == [1, 0, 2]
    assert block == "1 0 2"
    exact, prefix_len, target_len, partial = TREE_PROMPTING.grade_path(parsed, [1, 0, 2])
    assert exact is True
    assert prefix_len == target_len == 3
    assert np.isclose(partial, 1.0)


def test_gsm8k_prompt_and_grading() -> None:
    prompt = MATH_PROMPTING.gsm8k_prompt("What is 6+7?")
    assert "####" in prompt
    pred, method = MATH_PROMPTING.extract_gsm8k_final_answer_with_method("We add.\n#### 13")
    assert pred == "13"
    assert method == "hashes"
    assert MATH_PROMPTING.grade_gsm8k_answer("13", "irrelevant\n#### 13")
    assert not MATH_PROMPTING.grade_gsm8k_answer("12", "irrelevant\n#### 13")


def test_gsm8k_split_logic() -> None:
    records = [
        MATH_RESPONSES.MathResponseRecord(
            example_id=idx,
            question=f"q{idx}",
            answer="#### 1",
            prompt="p",
            raw_response="r",
            pred_answer="1" if idx % 2 == 0 else None,
            pred_method="hashes",
            format_ok=idx % 2 == 0,
            exact_match=idx % 2 == 0,
            partial_score=1.0 if idx % 2 == 0 else 0.0,
        )
        for idx in range(6)
    ]
    train, test = MATH_RESPONSES.split_math_responses(records, train_frac=0.5, seed=0)
    train_exact = [r for r in train if r.exact_match]
    test_exact = [r for r in test if r.exact_match]
    assert len(train_exact) + len(test_exact) == 3
    assert len(train_exact) == 1
    assert len(test) == 5
