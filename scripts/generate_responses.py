#!/usr/bin/env python3
"""Generate model answers for tree traversal or GSM8K tasks."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.tree.encoding import DEVICE, load_reasoning_model, set_global_seed
from cutter.utils.shared.chat import render_chat
from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.paths import embeddings_path, embeddings_pca_path, responses_path
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_model_ids
from cutter.utils.shared.basic import split_balanced, split_exact_only
from cutter.utils.shared.embeddings_cache import build_pca_cache_from_entries, save_embedding_cache
from cutter.utils.tree.prompting import SYSTEM_PROMPT as TREE_SYSTEM_PROMPT, extract_path, grade_path
from cutter.utils.tree.datasets import TraversalRecord, load_traversal_dataset
from cutter.utils.math.datasets import GSM8KRecord, load_gsm8k_jsonl
from cutter.utils.math.prompting import (
    SYSTEM_PROMPT as MATH_SYSTEM_PROMPT,
    extract_gsm8k_final_answer_with_method,
    grade_gsm8k_answer,
)

# -----------------------------------------------------------------------------
# Data structures

@dataclass
class ModelResult:
    example_id: int
    record: TraversalRecord
    raw_response: str
    parsed_path: List[int]
    parsed_text: Optional[str]
    format_ok: bool
    exact_match: bool
    prefix_match: int
    target_len: int
    partial_score: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "depth": self.record.depth,
                "example_id": self.example_id,
                "source": self.record.source,
                "target": self.record.target,
                "waypoints": self.record.waypoints,
                "num_samples": self.record.num_samples,
                "num_steps": self.record.num_steps,
                "prompt": self.record.prompt,
                "ground_truth_path": self.record.path,
                "canonical_path": self.record.canonical_path,
                "canonical_source": self.record.canonical_source,
                "canonical_target": self.record.canonical_target,
                "canonical_waypoints": self.record.canonical_waypoints,
                "label_mapping": self.record.label_mapping,
                "model_raw": self.raw_response,
                "parsed_path": self.parsed_path,
                "parsed_text": self.parsed_text,
                "format_ok": self.format_ok,
                "exact_match": self.exact_match,
                "prefix_match": self.prefix_match,
                "target_len": self.target_len,
                "partial_score": self.partial_score,
            },
            ensure_ascii=True,
        )


@dataclass
class EmbeddingRecord:
    example_id: int
    parsed_path: List[int]
    parsed_text: Optional[str]
    token_ids: List[int]
    token_offsets: List[Tuple[int, int]]
    layers: List[int]
    hidden_dim: int
    embeddings_by_layer: Dict[int, np.ndarray]  # layer -> (tokens, hidden_dim)
    prompt_tokens: int
    model_id: str
    dataset_path: str


@dataclass
class ResponseStub:
    example_id: int
    exact_match: bool


# -----------------------------------------------------------------------------
# Core helpers

def run_batch(
    records: Iterable[TraversalRecord],
    tokenizer,
    model,
    *,
    max_new_tokens: int,
    dataset_path: Path,
) -> Iterable[Tuple[ModelResult, Optional[EmbeddingRecord]]]:
    example_id = 0
    special_ids = set(tokenizer.all_special_ids or [])
    device = model.device
    for record in records:
        messages = [
            {"role": "system", "content": TREE_SYSTEM_PROMPT},
            {"role": "user", "content": record.prompt},
        ]
        prompt_text = render_chat(tokenizer, messages)
        tokens = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }
            generated = model.generate(**tokens, **gen_kwargs)
        prompt_len = tokens["input_ids"].shape[1]
        new_tokens = generated.sequences[0, prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed_path, parsed_text = extract_path(response)
        format_ok = bool(parsed_path)
        exact, prefix_len, target_len, partial = grade_path(parsed_path, record.path)
        model_result = ModelResult(
            example_id=example_id,
            record=record,
            raw_response=response,
            parsed_path=parsed_path,
            parsed_text=parsed_text,
            format_ok=format_ok,
            exact_match=exact,
            prefix_match=prefix_len,
            target_len=target_len,
            partial_score=partial,
        )
        embedding_record: Optional[EmbeddingRecord] = None
        try:
            gen_hidden = generated.hidden_states or ()
            if len(gen_hidden) != len(new_tokens):
                raise ValueError("hidden_states do not align with generated tokens")
            filtered_tokens: List[int] = []
            filtered_layers: List[List[np.ndarray]] = []
            for step, hs in enumerate(gen_hidden):
                tok_id = int(new_tokens[step].item())
                if tok_id in special_ids:
                    continue
                filtered_tokens.append(tok_id)
                layer_vecs = []
                for layer_hidden in hs:
                    layer_vecs.append(layer_hidden[0, -1, :].detach().float().cpu().numpy())
                filtered_layers.append(layer_vecs)
            if filtered_tokens and parsed_text:
                enc = tokenizer(
                    response,
                    return_offsets_mapping=True,
                    add_special_tokens=False,
                )
                resp_ids = list(map(int, enc["input_ids"]))
                offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
                filtered_ids = [tid for tid in filtered_tokens if tid not in special_ids]
                if resp_ids != filtered_ids:
                    raise ValueError("Response tokenization mismatch with generated tokens")
                start = response.rfind(parsed_text)
                if start == -1:
                    raise ValueError("Unable to locate parsed text span in response")
                end = start + len(parsed_text)
                idx_span = [idx for idx, (s, e) in enumerate(offsets) if e > start and s < end]
                if idx_span:
                    token_offsets = [offsets[i] for i in idx_span]
                    layer_count = len(filtered_layers[0])
                    emb_by_layer: Dict[int, np.ndarray] = {}
                    for layer_idx in range(layer_count):
                        vectors = [filtered_layers[i][layer_idx] for i in idx_span]
                        emb_by_layer[layer_idx] = np.stack(vectors, axis=0).astype(np.float16)
                    # Validation: decoded span tokens must yield the parsed path numbers.
                    span_ids = [resp_ids[i] for i in idx_span]
                    span_text = tokenizer.decode(span_ids, skip_special_tokens=True)
                    parsed_nums = [str(x) for x in parsed_path]
                    decoded_nums = [tok for tok in re.findall(r"[0-9]+", span_text)]
                    if parsed_nums and decoded_nums != parsed_nums:
                        raise ValueError(f"Parsed path {parsed_nums} does not match decoded tokens {decoded_nums}")
                    embedding_record = EmbeddingRecord(
                        example_id=example_id,
                        parsed_path=list(parsed_path),
                        parsed_text=parsed_text,
                        token_ids=[resp_ids[i] for i in idx_span],
                        token_offsets=token_offsets,
                        layers=list(range(layer_count)),
                        hidden_dim=emb_by_layer[0].shape[-1] if emb_by_layer else 0,
                        embeddings_by_layer=emb_by_layer,
                        prompt_tokens=prompt_len,
                        model_id=str(getattr(getattr(model, "config", None), "_name_or_path", "")),
                        dataset_path=str(dataset_path),
                    )
        except Exception as exc:
            warnings.warn(f"Embedding extraction failed for example {example_id}: {exc}")
            embedding_record = None
        yield model_result, embedding_record
        example_id += 1


def run_math_batch(
    records: Iterable[GSM8KRecord],
    tokenizer,
    model,
    *,
    max_new_tokens: int,
) -> Iterable[Dict[str, Any]]:
    example_id = 0
    device = model.device
    for record in records:
        messages = [
            {"role": "system", "content": MATH_SYSTEM_PROMPT},
            {"role": "user", "content": record.prompt},
        ]
        prompt_text = render_chat(tokenizer, messages)
        tokens = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
                "return_dict_in_generate": True,
            }
            generated = model.generate(**tokens, **gen_kwargs)
        prompt_len = tokens["input_ids"].shape[1]
        new_tokens = generated.sequences[0, prompt_len:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_answer, pred_method = extract_gsm8k_final_answer_with_method(response)
        format_ok = pred_answer is not None
        exact_match = grade_gsm8k_answer(pred_answer, record.answer)
        partial_score = 1.0 if exact_match else 0.0
        yield {
            "example_id": example_id,
            "question": record.question,
            "answer": record.answer,
            "prompt": record.prompt,
            "model_raw": response,
            "pred_answer": pred_answer,
            "pred_method": pred_method,
            "format_ok": format_ok,
            "exact_match": exact_match,
            "partial_score": partial_score,
        }
        example_id += 1


# -----------------------------------------------------------------------------
# CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model responses for tree traversal or GSM8K tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--setting",
        choices=("tree", "math"),
        default="tree",
        help="Which setting to run: tree traversal or GSM8K math.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset folder tag (e.g., depth1-2_n1000_steps1-2 or gsm8k_all_nall_seed0).",
    )
    parser.add_argument(
        "--reasoning-models",
        nargs="+",
        default=DEFAULT_REASONING_SIZES,
        help="Reasoning (R1-distilled) parameter counts (e.g., 7B). Pass 'none' to skip reasoning models.",
    )
    parser.add_argument(
        "--chat-models",
        nargs="+",
        default=DEFAULT_CHAT_SIZES,
        help="Non-reasoning chat parameter counts (e.g., 7B). Pass 'none' to skip chat models.",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility.")
    parser.add_argument("--max-new-tokens", type=int, default=2000, help="Max new tokens to generate.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of examples (0 = all).")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Number of PCA components to store for embeddings (-1 for full embeddings).",
    )
    parser.add_argument("--train-split", type=float, default=0.5, help="Train fraction for PCA fit.")
    parser.add_argument(
        "--split-type",
        type=str,
        default="exact-only",
        choices=("random", "exact-only"),
        help="Train split strategy for PCA fitting.",
    )
    parser.add_argument(
        "--gsm8k-split",
        type=str,
        default="all",
        help="GSM8K split tag for default dataset resolution (train, test, all).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    dataset_tag = path_utils.resolve_dataset_tag(
        args.setting,
        args.dataset,
        num_samples=0,
        seed=args.seed,
        gsm8k_split=args.gsm8k_split,
    )
    dataset_path = path_utils.resolve_dataset_path(
        args.setting,
        dataset_tag,
        num_samples=0,
        seed=args.seed,
        gsm8k_split=args.gsm8k_split,
    )
    reasoning_models, chat_models = resolve_model_ids(args.reasoning_models, args.chat_models)
    all_models = reasoning_models + chat_models
    if not all_models:
        raise ValueError("No models selected. Provide --reasoning-models or --chat-models.")

    if args.setting == "math":
        for model_id in all_models:
            dataset = load_gsm8k_jsonl(dataset_path)
            print(f"Loaded {len(dataset)} GSM8K examples from {dataset_path}")
            if args.limit and args.limit > 0:
                dataset = dataset[: args.limit]
                print(f"Applying limit: using first {len(dataset)} examples")

            output_path = responses_path(dataset_tag, model_id)

            print(f"Loading model '{model_id}' on device '{DEVICE}'")
            tokenizer, model = load_reasoning_model(model_id, device=DEVICE, use_half_precision=True)
            model.eval()
            print("Model loaded; beginning generation")

            output_path.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            exact = 0
            with output_path.open("w", encoding="utf-8") as fh:
                for result in run_math_batch(
                    dataset,
                    tokenizer,
                    model,
                    max_new_tokens=args.max_new_tokens,
                ):
                    fh.write(json.dumps(result, ensure_ascii=True) + "\n")
                    total += 1
                    if result["exact_match"]:
                        exact += 1
                    if total and total % 10 == 0:
                        print(f"Processed {total} examples (exact {exact}/{total})")
            print(f"Finished {total} examples. Exact matches: {exact}/{total}")
            print(f"Wrote graded responses to {output_path}")
        return

    for model_id in all_models:
        dataset = load_traversal_dataset(dataset_path)
        print(f"Loaded {len(dataset)} traversal examples from {dataset_path}")
        if args.limit and args.limit > 0:
            dataset = dataset[: args.limit]
            print(f"Applying limit: using first {len(dataset)} examples")

        # Store outputs under dataset/model-specific subdirectories for clarity.
        output_path = responses_path(dataset_tag, model_id)
        emb_output_path = embeddings_path(dataset_tag, model_id)
        pca_output_path = embeddings_pca_path(dataset_tag, model_id, args.pca_components)

        print(f"Loading model '{model_id}' on device '{DEVICE}'")
        tokenizer, model = load_reasoning_model(model_id, device=DEVICE, use_half_precision=True)
        model.eval()
        print("Model loaded; beginning generation")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        emb_output_path.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        exact = 0
        embeddings: List[EmbeddingRecord] = []
        response_records: List[ResponseStub] = []
        with output_path.open("w", encoding="utf-8") as fh:
            for result, emb in run_batch(
                dataset,
                tokenizer,
                model,
                max_new_tokens=args.max_new_tokens,
                dataset_path=dataset_path,
            ):
                fh.write(result.to_json() + "\n")
                total += 1
                if result.exact_match:
                    exact += 1
                response_records.append(ResponseStub(example_id=result.example_id, exact_match=result.exact_match))
                if emb:
                    embeddings.append(emb)
                if total and total % 10 == 0:
                    print(f"Processed {total} examples (exact {exact}/{total})")
        # Save embeddings sidecar
        def _to_serializable(rec: EmbeddingRecord) -> Dict[str, Any]:
            return {
                "example_id": rec.example_id,
                "parsed_path": rec.parsed_path,
                "parsed_text": rec.parsed_text,
                "token_ids": rec.token_ids,
                "token_offsets": rec.token_offsets,
                "layers": rec.layers,
                "hidden_dim": rec.hidden_dim,
                "embeddings_by_layer": rec.embeddings_by_layer,
                "prompt_tokens": rec.prompt_tokens,
                "model_id": rec.model_id,
                "dataset_path": rec.dataset_path,
            }

        emb_entries = {rec.example_id: _to_serializable(rec) for rec in embeddings}
        meta = {
            "model": model_id,
            "dataset": str(dataset_path),
            "dataset_tag": dataset_tag,
            "layers": embeddings[0].layers if embeddings else [],
            "layer_count": len(embeddings[0].layers) if embeddings else 0,
            "hidden_dim": embeddings[0].hidden_dim if embeddings else 0,
            "count": len(embeddings),
        }
        if args.pca_components >= 0:
            if args.pca_components == 0:
                raise ValueError("pca_components must be -1 or a positive integer.")
            if emb_entries:
                if args.split_type == "exact-only":
                    train_records, _ = split_exact_only(response_records, args.train_split, args.seed, exact_attr="exact_match")
                else:
                    train_records, _ = split_balanced(response_records, args.train_split, args.seed, exact_attr="exact_match")
                train_ids = [rec.example_id for rec in train_records]
                reduced_entries, pca_info, hidden_dim = build_pca_cache_from_entries(
                    emb_entries,
                    train_ids,
                    args.pca_components,
                    args.seed,
                )
                meta.update(
                    {
                        "pca_components": int(args.pca_components),
                        "pca": pca_info,
                        "hidden_dim": int(hidden_dim),
                        "pca_train_split": float(args.train_split),
                        "pca_seed": int(args.seed),
                        "pca_split_type": args.split_type,
                    }
                )
                save_embedding_cache(pca_output_path, reduced_entries, meta)
                print(f"Wrote PCA embeddings to {pca_output_path}")
            else:
                meta.update(
                    {
                        "pca_components": int(args.pca_components),
                        "hidden_dim": 0,
                        "pca_train_split": float(args.train_split),
                        "pca_seed": int(args.seed),
                        "pca_split_type": args.split_type,
                    }
                )
                save_embedding_cache(pca_output_path, {}, meta)
                print(f"Warning: no embeddings extracted; wrote empty PCA cache to {pca_output_path}")
        else:
            emb_payload = np.array([emb_entries[key] for key in sorted(emb_entries.keys())], dtype=object)
            np.savez_compressed(emb_output_path, embeddings=emb_payload, meta=np.array(meta, dtype=object))
            print(f"Wrote {len(embeddings)} embedding entries to {emb_output_path}")
        print(f"Finished {total} examples. Exact matches: {exact}/{total}")
        print(f"Wrote graded responses to {output_path}")
        if args.pca_components < 0:
            print(f"Wrote {len(embeddings)} embedding entries to {emb_output_path}")


if __name__ == "__main__":
    main()
