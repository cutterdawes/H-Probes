#!/usr/bin/env python3
"""Quick harness to try a stricter prompt with non-reasoning chat models.

This does NOT modify any existing scripts; it reuses the shared utils to:
- load a small dataset (defaults to depth 1–4, sample 0.4)
- run a few examples through a small non-reasoning model (default Qwen1.5-1.8B-Chat)
- grade the outputs and print a short summary to stdout

The system/user prompt explicitly asks for a single PATH line to encourage
format compliance from non-reasoning models.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.tree.encoding import DEVICE, load_reasoning_model, set_global_seed
from cutter.utils.tree.datasets import TraversalRecord, load_traversal_dataset
from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_model_ids
from cutter.utils.shared.chat import render_chat
from cutter.utils.tree.prompting import extract_path, grade_path


# A concise system prompt that avoids “think as long as you like”.
SYSTEM_PROMPT = (
    "You answer traversal questions about a numbered binary tree. "
    "Be concise and follow the required output format."
)


@dataclass
class ModelResult:
    example_id: int
    raw_response: str
    parsed_path: List[int]
    parsed_text: Optional[str]
    format_ok: bool
    exact_match: bool
    prefix_match: int
    target_len: int
    partial_score: float


def build_user_prompt(record: TraversalRecord) -> str:
    """Compose a stricter user prompt on top of the stored tree diagram."""

    return (
        f"Tree depth: {record.depth} (root 0; leaves depth {record.depth}).\n"
        "Nodes are labeled breadth-first, left to right at each level.\n\n"
        f"{record.prompt}\n\n"
        "Respond with exactly one line:\n"
        "PATH: n0 n1 ... nk\n"
        "No other text."
    )


def run_batch(
    records: Iterable[TraversalRecord],
    tokenizer,
    model,
    *,
    max_new_tokens: int,
) -> Iterable[ModelResult]:
    example_id = 0
    device = model.device
    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
        ]
        prompt_text = render_chat(tokenizer, messages)
        tokens = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "output_hidden_states": False,
                "return_dict_in_generate": True,
            }
            generated = model.generate(**tokens, **gen_kwargs)
        prompt_len = tokens["input_ids"].shape[1]
        new_tokens = generated.sequences[0, prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed_path, parsed_text = extract_path(response)
        format_ok = bool(parsed_path)
        exact, prefix_len, target_len, partial = grade_path(parsed_path, record.path)
        yield ModelResult(
            example_id=example_id,
            raw_response=response,
            parsed_path=parsed_path,
            parsed_text=parsed_text,
            format_ok=format_ok,
            exact_match=exact,
            prefix_match=prefix_len,
            target_len=target_len,
            partial_score=partial,
        )
        example_id += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try a non-reasoning prompt for traversal with a small chat model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=path_utils.DEFAULT_TREE_DATASET_TAG,
        help="Traversal dataset folder tag.",
    )
    parser.add_argument(
        "--reasoning-models",
        nargs="+",
        default=DEFAULT_REASONING_SIZES,
        help="Reasoning parameter counts. Pass 'none' to skip reasoning models.",
    )
    parser.add_argument(
        "--chat-models",
        nargs="+",
        default=DEFAULT_CHAT_SIZES,
        help="Chat parameter counts. Pass 'none' to skip chat models.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to try (0 = all).")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max new tokens to generate.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    reasoning_models, chat_models = resolve_model_ids(args.reasoning_models, args.chat_models)
    if not chat_models:
        raise ValueError("This script requires a chat model. Provide --chat-models.")
    if len(chat_models) > 1 or reasoning_models:
        raise ValueError("Provide exactly one chat model for this script.")
    model_id = chat_models[0]
    dataset_path = path_utils.dataset_path_from_tag(args.dataset)
    dataset = load_traversal_dataset(dataset_path)
    if args.limit and args.limit > 0:
        dataset = dataset[: args.limit]

    print(f"Dataset: {dataset_path} (tag: {args.dataset})")
    print(f"Model:   {model_id}")
    print(f"Examples: {len(dataset)}")

    tokenizer, model = load_reasoning_model(model_id, device=DEVICE, use_half_precision=True)
    model.eval()

    exact = 0
    fmt_ok = 0
    total = 0
    for res in run_batch(
        dataset,
        tokenizer,
        model,
        max_new_tokens=args.max_new_tokens,
    ):
        total += 1
        fmt_ok += int(res.format_ok)
        exact += int(res.exact_match)
        path_str = " ".join(map(str, res.parsed_path)) if res.parsed_path else "NONE"
        print(f"[{res.example_id}] fmt={res.format_ok} exact={res.exact_match} partial={res.partial_score:.3f} | PATH: {path_str}")
        print(f"  raw: {res.raw_response[:200]}")
    if total:
        print(f"\nSummary: exact {exact}/{total} | format-ok {fmt_ok}/{total}")


if __name__ == "__main__":
    main()
