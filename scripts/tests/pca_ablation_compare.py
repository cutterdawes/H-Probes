#!/usr/bin/env python3
"""Compare PCA ablations from node-token vs full-distribution bases."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(PROJECT_ROOT))

import cutter.scripts.intervene as intervene
from cutter.utils.tree.encoding import DEVICE, load_reasoning_model, set_global_seed
from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.paths import probe_path, repo_relative, responses_path
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_single_model_id


@dataclass
class ResponseRecord:
    example_id: int
    depth: int
    source: int
    target: int
    num_samples: int
    prompt: str
    ground_truth_path: List[int]
    parsed_path: List[int]
    parsed_text: Optional[str]
    format_ok: bool
    exact_match: bool
    partial_score: float
    raw_response: str

    @classmethod
    def from_json(cls, row: Dict[str, Any]) -> "ResponseRecord":
        return cls(
            example_id=int(row.get("example_id", -1)),
            depth=int(row["depth"]),
            source=int(row["source"]),
            target=int(row["target"]),
            num_samples=int(row.get("num_samples", row.get("sample_rate", -1))),
            prompt=str(row["prompt"]),
            ground_truth_path=[int(x) for x in row["ground_truth_path"]],
            parsed_path=[int(x) for x in row.get("parsed_path", [])],
            parsed_text=row.get("parsed_text"),
            format_ok=bool(row.get("format_ok", False)),
            exact_match=bool(row.get("exact_match", False)),
            partial_score=float(row.get("partial_score", 0.0)),
            raw_response=str(row.get("model_raw", "")),
        )


def _load_responses(path: Path) -> List[ResponseRecord]:
    records: List[ResponseRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = ResponseRecord.from_json(__import__("json").loads(line))
            if row.example_id < 0:
                row.example_id = idx
            records.append(row)
    return records


def _select_hidden_layer(
    hidden_states: Optional[Tuple[torch.Tensor, ...]],
    last_hidden_state: Optional[torch.Tensor],
    layer_idx: int,
    layer_mode: str,
) -> torch.Tensor:
    if hidden_states is None or not hidden_states:
        if last_hidden_state is None:
            raise ValueError("Model outputs did not include hidden states.")
        return last_hidden_state[0]
    if layer_mode == "mean":
        stacked = torch.stack(list(hidden_states), dim=0)
        return stacked.mean(dim=0)[0]
    if layer_mode == "layer":
        total_layers = len(hidden_states)
        idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
        idx = max(0, min(total_layers - 1, idx))
        return hidden_states[idx][0]
    return hidden_states[-1][0]


def _collect_full_embeddings(
    records: Iterable[ResponseRecord],
    tokenizer,
    model,
    *,
    layer_idx: int,
    layer_mode: str,
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for rec in records:
        prompt_text = intervene._render_prompt(tokenizer, rec.prompt)
        response_text = rec.raw_response
        full_text = prompt_text
        if response_text:
            if full_text and not full_text.endswith(" "):
                full_text += " "
            full_text += response_text
        enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = _select_hidden_layer(outputs.hidden_states, getattr(outputs, "last_hidden_state", None), layer_idx, layer_mode)
        embeddings.append(hidden.detach().float().cpu().numpy())
    if not embeddings:
        raise RuntimeError("No embeddings collected for full-distribution PCA.")
    return np.vstack(embeddings).astype(np.float32)


def _load_probe(path: Path) -> Tuple[Dict[str, Any], Dict[int, Any], Dict[int, Any]]:
    payload = np.load(path, allow_pickle=True)
    meta = payload["meta"].item()
    encodings = payload["encodings"].item()
    results = payload["results"].item()
    return meta, encodings, results


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PCA ablations for node-token vs full-distribution bases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=path_utils.DEFAULT_TREE_DATASET_TAG,
        help="Dataset folder tag (used to derive cache locations).",
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
    parser.add_argument("--layer", type=int, default=22, help="Transformer layer to ablate.")
    parser.add_argument("--layer-mode", type=str, default="layer", choices=("layer", "mean", "last"))
    parser.add_argument("--proj-dim", type=int, default=4, help="Number of PCA components to ablate.")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Number of PCA components used to train the probe (for locating the probe file).",
    )
    parser.add_argument("--probe", type=Path, default=None, help="Optional explicit probe path.")
    parser.add_argument(
        "--bucket",
        type=str,
        default="exact",
        choices=("all", "exact", "partial", "zero"),
        help="Bucket of test examples to evaluate (based on baseline responses).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2000, help="Max new tokens to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on number of test examples (0 = all).")
    parser.add_argument("--output", type=Path, default=None, help="Output npz path.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)

    model_id = resolve_single_model_id(args.reasoning_models, args.chat_models)
    dataset_tag = args.dataset
    derived_probe = probe_path(dataset_tag, model_id, args.proj_dim, False, args.pca_components)
    probe_fp = args.probe or derived_probe
    if not probe_fp.exists():
        raise FileNotFoundError(f"Probe file not found: {probe_fp}")

    meta, encodings, results = _load_probe(probe_fp)
    if args.layer not in results:
        raise ValueError(f"Layer {args.layer} not found in probe results.")
    layer_result = results[args.layer]
    pca_info = layer_result.get("pca") if isinstance(layer_result.get("pca"), dict) else None
    if not pca_info:
        raise RuntimeError("Probe file does not include PCA components.")
    components = np.asarray(pca_info.get("components", []), dtype=np.float32)
    if components.size == 0:
        raise RuntimeError("No PCA components found in probe file.")

    take = min(args.proj_dim, components.shape[0])
    if take <= 0:
        raise RuntimeError("Requested proj-dim is invalid for PCA components.")
    node_pca_basis = components[:take].T.astype(np.float32)

    enc_layer = encodings[args.layer]
    train_idx = np.asarray(enc_layer["train_idx"], dtype=int)
    test_idx = np.asarray(enc_layer["test_idx"], dtype=int)
    example_ids = np.asarray(enc_layer["example_ids"], dtype=int)
    train_examples = sorted(set(example_ids[train_idx].tolist()))
    test_examples = sorted(set(example_ids[test_idx].tolist()))

    responses_fp = responses_path(dataset_tag, model_id)
    all_responses = _load_responses(responses_fp)
    response_by_id = {rec.example_id: rec for rec in all_responses}

    selected: List[ResponseRecord] = []
    for eid in test_examples:
        rec = response_by_id.get(eid)
        if rec is not None:
            selected.append(rec)
    if not selected:
        raise RuntimeError("No responses found for the requested test split.")

    if args.bucket != "all":
        filtered: List[ResponseRecord] = []
        for rec in selected:
            if rec.exact_match:
                bucket = "exact"
            elif rec.partial_score > 0:
                bucket = "partial"
            else:
                bucket = "zero"
            if bucket == args.bucket:
                filtered.append(rec)
        selected = filtered
        if not selected:
            raise RuntimeError(f"No responses found for bucket '{args.bucket}'.")

    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    tokenizer, model = load_reasoning_model(model_id, device=DEVICE, use_half_precision=True)
    model.eval()

    train_records = [response_by_id[eid] for eid in train_examples if eid in response_by_id]
    if not train_records:
        raise RuntimeError("No training records found for full-distribution PCA.")
    full_embeddings = _collect_full_embeddings(
        train_records,
        tokenizer,
        model,
        layer_idx=args.layer,
        layer_mode=args.layer_mode,
    )
    max_components = min(full_embeddings.shape[0], full_embeddings.shape[1])
    full_take = min(args.proj_dim, max_components)
    if full_take <= 0:
        raise RuntimeError("Insufficient data for full-distribution PCA components.")
    pca_full = PCA(n_components=full_take, svd_solver="auto", random_state=args.seed)
    pca_full.fit(full_embeddings)
    full_pca_basis = pca_full.components_.astype(np.float32).T

    target_layer = intervene._resolve_layer_module(model, args.layer)
    model_dtype = next(model.parameters()).dtype
    node_basis_t = torch.tensor(node_pca_basis, dtype=torch.float32, device=model.device).to(dtype=model_dtype)
    full_basis_t = torch.tensor(full_pca_basis, dtype=torch.float32, device=model.device).to(dtype=model_dtype)

    per_example: List[Dict[str, Any]] = []
    baseline_exact = []
    baseline_partial = []
    node_exacts = []
    node_partials = []
    node_logit_diffs = []
    full_exacts = []
    full_partials = []
    full_logit_diffs = []

    total = len(selected)
    for idx, rec in enumerate(selected, start=1):
        prompt_text = intervene._render_prompt(tokenizer, rec.prompt)
        prefix_text_for_logits = ""
        if rec.parsed_text:
            start_idx = rec.raw_response.rfind(rec.parsed_text)
            if start_idx != -1:
                prefix_text_for_logits = rec.raw_response[:start_idx]
        answer_text = intervene._build_answer_text(rec.ground_truth_path)
        answer_mask = intervene._answer_token_mask(answer_text, tokenizer)
        node_token_ids = intervene._node_token_ids(rec.depth, tokenizer)

        base_logits = intervene._compute_answer_logits(
            prompt_text=prompt_text,
            prefix_text=prefix_text_for_logits,
            answer_text=answer_text,
            tokenizer=tokenizer,
            model=model,
            target_layer=None,
        )

        node_gen = intervene._run_generation(
            prompt_text,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            ablation_basis=node_basis_t,
            target_layer=target_layer,
            ground_truth_path=rec.ground_truth_path,
        )
        node_logits = intervene._compute_answer_logits(
            prompt_text=prompt_text,
            prefix_text=prefix_text_for_logits,
            answer_text=answer_text,
            tokenizer=tokenizer,
            model=model,
            ablation_basis=node_basis_t,
            target_layer=target_layer,
        )
        node_logit_gap = intervene._mean_abs_node_logit_diff(base_logits, node_logits, node_token_ids, answer_mask)

        full_gen = intervene._run_generation(
            prompt_text,
            tokenizer=tokenizer,
            model=model,
            max_new_tokens=args.max_new_tokens,
            ablation_basis=full_basis_t,
            target_layer=target_layer,
            ground_truth_path=rec.ground_truth_path,
        )
        full_logits = intervene._compute_answer_logits(
            prompt_text=prompt_text,
            prefix_text=prefix_text_for_logits,
            answer_text=answer_text,
            tokenizer=tokenizer,
            model=model,
            ablation_basis=full_basis_t,
            target_layer=target_layer,
        )
        full_logit_gap = intervene._mean_abs_node_logit_diff(base_logits, full_logits, node_token_ids, answer_mask)

        baseline_exact.append(float(rec.exact_match))
        baseline_partial.append(float(rec.partial_score))
        node_exacts.append(float(node_gen.exact_match))
        node_partials.append(float(node_gen.partial_score))
        node_logit_diffs.append(node_logit_gap)
        full_exacts.append(float(full_gen.exact_match))
        full_partials.append(float(full_gen.partial_score))
        full_logit_diffs.append(full_logit_gap)

        per_example.append(
            {
                "example_id": rec.example_id,
                "depth": rec.depth,
                "source": rec.source,
                "target": rec.target,
                "num_samples": rec.num_samples,
                "ground_truth_path": rec.ground_truth_path,
                "prompt": rec.prompt,
                "baseline": {
                    "response": rec.raw_response,
                    "parsed_path": rec.parsed_path,
                    "parsed_text": rec.parsed_text,
                    "format_ok": rec.format_ok,
                    "exact_match": rec.exact_match,
                    "partial_score": rec.partial_score,
                },
                "pca_ablation": {
                    "response": node_gen.response,
                    "parsed_path": node_gen.parsed_path,
                    "parsed_text": node_gen.parsed_text,
                    "format_ok": node_gen.format_ok,
                    "exact_match": node_gen.exact_match,
                    "partial_score": node_gen.partial_score,
                    "logit_diff": node_logit_gap,
                },
                "pca_full_ablation": {
                    "response": full_gen.response,
                    "parsed_path": full_gen.parsed_path,
                    "parsed_text": full_gen.parsed_text,
                    "format_ok": full_gen.format_ok,
                    "exact_match": full_gen.exact_match,
                    "partial_score": full_gen.partial_score,
                    "logit_diff": full_logit_gap,
                },
            }
        )

        if idx % 10 == 0 or idx == total:
            base_exact = float(np.mean(baseline_exact)) if baseline_exact else float("nan")
            node_exact = float(np.mean(node_exacts)) if node_exacts else float("nan")
            full_exact = float(np.mean(full_exacts)) if full_exacts else float("nan")
            node_logit = float(np.nanmean(node_logit_diffs)) if node_logit_diffs else float("nan")
            full_logit = float(np.nanmean(full_logit_diffs)) if full_logit_diffs else float("nan")
            print(
                f"Processed {idx}/{total} | "
                f"node Δ={node_exact - base_exact:+.4f} | "
                f"full Δ={full_exact - base_exact:+.4f} | "
                f"node logit Δ={node_logit:.4f} | full logit Δ={full_logit:.4f}"
            )

    base_exact = float(np.mean(baseline_exact)) if baseline_exact else float("nan")
    base_partial = float(np.mean(baseline_partial)) if baseline_partial else float("nan")
    node_exact = float(np.mean(node_exacts)) if node_exacts else float("nan")
    node_partial = float(np.mean(node_partials)) if node_partials else float("nan")
    node_logit = float(np.nanmean(node_logit_diffs)) if node_logit_diffs else float("nan")
    full_exact = float(np.mean(full_exacts)) if full_exacts else float("nan")
    full_partial = float(np.mean(full_partials)) if full_partials else float("nan")
    full_logit = float(np.nanmean(full_logit_diffs)) if full_logit_diffs else float("nan")

    aggregate = {
        "baseline_exact": base_exact,
        "baseline_partial": base_partial,
        "node_pca_exact": node_exact,
        "node_pca_partial": node_partial,
        "node_pca_accuracy_delta": node_exact - base_exact,
        "node_pca_partial_delta": node_partial - base_partial,
        "node_pca_logit_diff_mean": node_logit,
        "full_pca_exact": full_exact,
        "full_pca_partial": full_partial,
        "full_pca_accuracy_delta": full_exact - base_exact,
        "full_pca_partial_delta": full_partial - base_partial,
        "full_pca_logit_diff_mean": full_logit,
    }

    meta_out = {
        "probe_path": repo_relative(probe_fp),
        "responses_path": repo_relative(responses_fp),
        "model_id": model_id,
        "dataset_tag": dataset_tag,
        "layer": args.layer,
        "proj_dim": args.proj_dim,
        "pca_components": int(pca_info.get("n_components", -1)),
        "node_basis_rank": node_pca_basis.shape[1],
        "full_basis_rank": full_pca_basis.shape[1],
        "hidden_dim": node_pca_basis.shape[0],
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "test_example_count": len(selected),
        "bucket": args.bucket,
    }

    default_out = (
        path_utils.model_output_dir(dataset_tag, model_id)
        / "interventions"
        / f"pca_compare_proj{args.proj_dim}_pca{args.pca_components}_layer{args.layer}.npz"
    )
    output_path = args.output or default_out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        meta=np.array(meta_out, dtype=object),
        aggregate=np.array(aggregate, dtype=object),
        records=np.array(per_example, dtype=object),
    )

    print(f"Baseline exact {base_exact:.4f} | partial {base_partial:.4f}")
    print(f"Node PCA exact {node_exact:.4f} (Δ {node_exact - base_exact:+.4f}) | partial {node_partial:.4f} (Δ {node_partial - base_partial:+.4f})")
    print(f"Full PCA exact {full_exact:.4f} (Δ {full_exact - base_exact:+.4f}) | partial {full_partial:.4f} (Δ {full_partial - base_partial:+.4f})")
    print(f"Node PCA mean logit diff {node_logit:.4f} | Full PCA mean logit diff {full_logit:.4f}")
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
