#!/usr/bin/env python3
"""Regenerate tree traversal or GSM8K answers while ablating probe subspaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

# Ensure repository root is importable when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.tree.encoding import DEVICE, load_reasoning_model, set_global_seed
from cutter.utils.shared.chat import render_chat
from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.paths import intervention_path, probe_path, repo_relative, resolve_repo_path, responses_path
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_single_model_id
from cutter.utils.tree.prompting import SYSTEM_PROMPT as TREE_SYSTEM_PROMPT, extract_path, grade_path
from cutter.utils.math.prompting import (
    SYSTEM_PROMPT as MATH_SYSTEM_PROMPT,
    extract_gsm8k_final_answer_with_method,
    grade_gsm8k_answer,
)
from cutter.utils.math.responses import MathResponseRecord, load_math_responses
from cutter.utils.shared.basic import split_exact_only


# -----------------------------------------------------------------------------
# Data containers


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


@dataclass
class GenerationResult:
    response: str
    parsed_path: List[int]
    parsed_text: Optional[str]
    format_ok: bool
    exact_match: bool
    partial_score: float
    tokens: List[int]


@dataclass
class MathGenerationResult:
    response: str
    pred_answer: Optional[str]
    pred_method: str
    format_ok: bool
    exact_match: bool


# -----------------------------------------------------------------------------
# Utilities


def _load_probe(path: Path) -> Tuple[Dict[str, Any], Dict[int, Any], Dict[int, Any]]:
    payload = np.load(path, allow_pickle=True)
    meta = payload["meta"].item()
    encodings = payload["encodings"].item()
    results = payload["results"].item()
    return meta, encodings, results


def _orthonormal_basis(matrix: np.ndarray) -> np.ndarray:
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int((s > 1e-6).sum())
    rank = max(rank, 1)
    return u[:, :rank].astype(np.float32)


def _extract_depth_direction(layer_result: Dict[str, Any]) -> np.ndarray:
    depth_model = None
    if isinstance(layer_result, dict):
        depth_model = layer_result.get("depth_model")
        if depth_model is None and isinstance(layer_result.get("distance"), dict):
            depth_model = layer_result["distance"].get("depth_model")
    if depth_model is None:
        raise KeyError("Probe results missing 'depth_model' for depth-direction ablation.")

    named_steps = getattr(depth_model, "named_steps", None)
    scaler = named_steps.get("scale") if isinstance(named_steps, dict) else None
    reg = named_steps.get("reg") if isinstance(named_steps, dict) else None
    coef = None
    if reg is not None and hasattr(reg, "coef_"):
        coef = reg.coef_
    elif hasattr(depth_model, "coef_"):
        coef = depth_model.coef_
    if coef is None:
        raise KeyError("Depth probe model missing coefficients for direction extraction.")
    coef = np.asarray(coef, dtype=np.float32).reshape(-1)
    if scaler is not None and hasattr(scaler, "scale_"):
        scale = np.asarray(scaler.scale_, dtype=np.float32).reshape(-1)
        scale = np.where(scale == 0, 1.0, scale)
        coef = coef / scale
    return coef


def _resolve_distance_projection(layer_result: Dict[str, Any]) -> np.ndarray:
    if "projection_full" in layer_result:
        projection_raw = layer_result["projection_full"]
    elif "distance" in layer_result and isinstance(layer_result["distance"], dict) and "projection" in layer_result["distance"]:
        projection_raw = layer_result["distance"]["projection"]
    elif "projection" in layer_result:
        if "pca" in layer_result and isinstance(layer_result["pca"], dict):
            pca_info = layer_result["pca"]
            components = np.asarray(pca_info.get("components", []), dtype=np.float32)
            if components.size:
                projection_raw = components.T @ np.asarray(layer_result["projection"], dtype=np.float32)
            else:
                projection_raw = layer_result["projection"]
        else:
            projection_raw = layer_result["projection"]
    else:
        raise KeyError("Probe results missing 'projection' (checked both distance/projection and top-level projection).")
    return np.asarray(projection_raw, dtype=np.float32)


def _resolve_pca_basis(layer_result: Dict[str, Any], proj_dim: int) -> Tuple[Optional[np.ndarray], int, Optional[Dict[str, Any]]]:
    pca_basis = None
    pca_basis_rank = 0
    pca_info = layer_result.get("pca") if isinstance(layer_result.get("pca"), dict) else None
    if isinstance(pca_info, dict):
        components = np.asarray(pca_info.get("components", []), dtype=np.float32)
        if components.size:
            take = min(proj_dim, components.shape[0])
            if take > 0:
                pca_basis = components[:take].T.astype(np.float32)
                pca_basis_rank = pca_basis.shape[1]
    return pca_basis, pca_basis_rank, pca_info


def _resolve_depth_direction(layer_result: Dict[str, Any]) -> np.ndarray:
    depth_dir = _extract_depth_direction(layer_result)
    pca_info = layer_result.get("pca") if isinstance(layer_result.get("pca"), dict) else None
    if isinstance(pca_info, dict):
        components = np.asarray(pca_info.get("components", []), dtype=np.float32)
        if components.size:
            if depth_dir.shape[0] != components.shape[0]:
                raise ValueError(
                    f"Depth direction dim {depth_dir.shape[0]} does not match PCA components {components.shape[0]}."
                )
            depth_dir = components.T @ depth_dir
    return np.asarray(depth_dir, dtype=np.float32)


def _sample_random_basis(dim: int, rank: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    matrix = torch.randn(dim, rank, generator=generator, dtype=torch.float32)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def _resolve_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError("Unable to locate transformer layers on model.")


def _resolve_layer_module(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    layers = _resolve_layers(model)
    total_layers = len(layers)
    idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
    idx = max(0, min(idx, total_layers - 1))
    return layers[idx]


def _register_ablation_hook(layer: torch.nn.Module, basis: torch.Tensor):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        projected = hidden @ basis
        hidden_proj = projected @ basis.transpose(0, 1)
        updated = hidden - hidden_proj
        if isinstance(output, tuple):
            return (updated,) + output[1:]
        return updated
    return layer.register_forward_hook(hook)


def _register_zero_hook(layer: torch.nn.Module):
    def hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        updated = torch.zeros_like(hidden)
        if isinstance(output, tuple):
            return (updated,) + output[1:]
        return updated
    return layer.register_forward_hook(hook)


def _render_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": TREE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return render_chat(tokenizer, messages)


def _render_math_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": MATH_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return render_chat(tokenizer, messages)


def _select_hidden_layer(
    hidden_states: Optional[Tuple[torch.Tensor, ...]],
    layer_idx: int,
) -> torch.Tensor:
    if hidden_states is None or not hidden_states:
        raise ValueError("Model outputs did not include hidden states.")
    total_layers = len(hidden_states)
    idx = layer_idx if layer_idx >= 0 else total_layers + layer_idx
    idx = max(0, min(total_layers - 1, idx))
    return hidden_states[idx][0]


def _collect_full_embeddings(
    records: Iterable[Any],
    *,
    tokenizer,
    model,
    layer_idx: int,
    render_prompt: Callable[[Any], str],
    response_attr: str = "raw_response",
) -> np.ndarray:
    embeddings: List[np.ndarray] = []
    for rec in records:
        prompt_text = render_prompt(rec.prompt)
        response_text = getattr(rec, response_attr, "")
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
        hidden = _select_hidden_layer(outputs.hidden_states, layer_idx)
        embeddings.append(hidden.detach().float().cpu().numpy())
    if not embeddings:
        raise RuntimeError("No embeddings collected for full-CoT PCA.")
    return np.vstack(embeddings).astype(np.float32)


def _build_answer_text(path: Sequence[int]) -> str:
    return "PATH: " + " ".join(str(x) for x in path)


def _answer_token_mask(answer_text: str, tokenizer) -> torch.Tensor:
    enc = tokenizer(answer_text, return_offsets_mapping=True, add_special_tokens=False)
    mask = []
    for (start, end) in enc["offset_mapping"]:
        span = answer_text[start:end]
        mask.append(any(ch.isdigit() for ch in span))
    return torch.tensor(mask, dtype=torch.bool)


def _node_token_ids(tree_depth: int, tokenizer) -> List[int]:
    max_node = 2 ** (tree_depth + 1) - 1  # inclusive upper bound index
    # NOTE: For larger trees, some node strings become multi-token; we only keep the final token per number here.
    node_ids: List[int] = []
    seen = set()
    for nid in range(max_node + 1):
        toks = tokenizer.encode(str(nid), add_special_tokens=False)
        if not toks:
            continue
        tok = toks[-1]
        if tok not in seen:
            seen.add(tok)
            node_ids.append(tok)
    return sorted(node_ids)


def _output_tag(args: argparse.Namespace) -> Optional[str]:
    parts: List[str] = []
    if getattr(args, "answer_only", False):
        parts.append("answeronly")
    if getattr(args, "layer_sweep", False):
        parts.append("layersweep")
    return "_".join(parts) if parts else None


def _compute_answer_logits(
    *,
    prompt_text: str,
    prefix_text: str,
    answer_text: str,
    tokenizer,
    model,
    ablation_basis: Optional[torch.Tensor] = None,
    target_layer: Optional[torch.nn.Module],
    ablation_zero: bool = False,
) -> torch.Tensor:
    """Teacher-force gold answer; return logits for answer tokens only."""

    text = prompt_text + prefix_text
    if text and not text.endswith(" "):
        text += " "
    text += answer_text
    enc_full = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    answer_enc = tokenizer(answer_text, return_tensors="pt", add_special_tokens=False).to(model.device)
    answer_len = int(answer_enc["input_ids"].shape[1])

    handles = []
    if target_layer is not None:
        if ablation_zero:
            handles.append(_register_zero_hook(target_layer))
        elif ablation_basis is not None:
            handles.append(_register_ablation_hook(target_layer, ablation_basis))
    try:
        with torch.no_grad():
            outputs = model(
                input_ids=enc_full["input_ids"],
                attention_mask=enc_full["attention_mask"],
                output_hidden_states=False,
            )
        logits = outputs.logits
        return logits[0, -answer_len:, :].detach().float().cpu()
    finally:
        for h in handles:
            h.remove()


def _mean_abs_node_logit_diff(
    baseline_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    node_token_ids: Sequence[int],
    answer_mask: torch.Tensor,
) -> float:
    if baseline_logits.shape != variant_logits.shape:
        return float("nan")
    if not node_token_ids or not answer_mask.any():
        return float("nan")
    node_idx = torch.tensor(node_token_ids, dtype=torch.long, device=baseline_logits.device)
    diffs = (variant_logits[:, node_idx] - baseline_logits[:, node_idx]).abs().mean(dim=1)
    masked = diffs[answer_mask.to(diffs.device)]
    return float(masked.mean().item()) if masked.numel() > 0 else float("nan")


def _mean_abs_answer_logit_diff(
    baseline_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    answer_token_ids: Sequence[int],
) -> float:
    if baseline_logits.shape != variant_logits.shape:
        return float("nan")
    if not answer_token_ids:
        return float("nan")
    if baseline_logits.shape[0] != len(answer_token_ids):
        return float("nan")
    token_ids = torch.tensor(answer_token_ids, dtype=torch.long, device=baseline_logits.device)
    positions = torch.arange(len(answer_token_ids), device=baseline_logits.device)
    baseline_vals = baseline_logits[positions, token_ids]
    variant_vals = variant_logits[positions, token_ids]
    return float((variant_vals - baseline_vals).abs().mean().item())


def _math_prefix_for_logits(
    raw_response: str,
    pred_answer: Optional[str],
    gold_answer: str,
) -> tuple[str, bool]:
    if not raw_response:
        return "", False
    candidates = []
    if pred_answer:
        candidates.append(str(pred_answer))
    if gold_answer and gold_answer not in candidates:
        candidates.append(str(gold_answer))
    for cand in candidates:
        idx = raw_response.rfind(cand)
        if idx != -1:
            return raw_response[:idx], True
    return "", False


def _tree_ablation_record(
    gen: Optional[GenerationResult],
    logit_diff: float,
    *,
    answer_only: bool,
) -> Dict[str, Any]:
    if answer_only or gen is None:
        return {
            "response": "",
            "parsed_path": [],
            "parsed_text": None,
            "format_ok": None,
            "exact_match": None,
            "partial_score": float("nan"),
            "logit_diff": logit_diff,
        }
    return {
        "response": gen.response,
        "parsed_path": gen.parsed_path,
        "parsed_text": gen.parsed_text,
        "format_ok": gen.format_ok,
        "exact_match": gen.exact_match,
        "partial_score": gen.partial_score,
        "logit_diff": logit_diff,
    }


def _math_ablation_record(
    gen: Optional[MathGenerationResult],
    logit_diff: float,
    *,
    answer_only: bool,
) -> Dict[str, Any]:
    if answer_only or gen is None:
        return {
            "response": "",
            "pred_answer": None,
            "pred_method": "",
            "format_ok": None,
            "exact_match": None,
            "partial_score": float("nan"),
            "logit_diff": logit_diff,
        }
    return {
        "response": gen.response,
        "pred_answer": gen.pred_answer,
        "pred_method": gen.pred_method,
        "format_ok": gen.format_ok,
        "exact_match": gen.exact_match,
        "partial_score": 1.0 if gen.exact_match else 0.0,
        "logit_diff": logit_diff,
    }


def _run_generation(
    prompt_text: str,
    *,
    tokenizer,
    model,
    max_new_tokens: int,
    ablation_basis: Optional[torch.Tensor] = None,
    target_layer: Optional[torch.nn.Module] = None,
    ablation_zero: bool = False,
    ground_truth_path: Sequence[int],
) -> GenerationResult:
    handles = []
    if target_layer is not None:
        if ablation_zero:
            handles.append(_register_zero_hook(target_layer))
        elif ablation_basis is not None:
            handles.append(_register_ablation_hook(target_layer, ablation_basis))
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated.sequences[0, prompt_len:].tolist()
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parsed_path, parsed_text = extract_path(response_text)
        format_ok = bool(parsed_path)
        exact_match, _, _, partial_score = grade_path(parsed_path, ground_truth_path)
        return GenerationResult(
            response=response_text,
            parsed_path=parsed_path,
            parsed_text=parsed_text,
            format_ok=format_ok,
            exact_match=exact_match,
            partial_score=partial_score,
            tokens=new_tokens,
        )
    finally:
        for handle in handles:
            handle.remove()


def _run_math_generation(
    prompt_text: str,
    *,
    tokenizer,
    model,
    max_new_tokens: int,
    ablation_basis: Optional[torch.Tensor] = None,
    target_layer: Optional[torch.nn.Module] = None,
    ablation_zero: bool = False,
    gold_answer: str,
) -> MathGenerationResult:
    handles = []
    if target_layer is not None:
        if ablation_zero:
            handles.append(_register_zero_hook(target_layer))
        elif ablation_basis is not None:
            handles.append(_register_ablation_hook(target_layer, ablation_basis))
    try:
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                output_scores=False,
                return_dict_in_generate=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = generated.sequences[0, prompt_len:].tolist()
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred_answer, pred_method = extract_gsm8k_final_answer_with_method(response_text)
        format_ok = pred_answer is not None
        exact_match = grade_gsm8k_answer(pred_answer, gold_answer)
        return MathGenerationResult(
            response=response_text,
            pred_answer=pred_answer,
            pred_method=pred_method,
            format_ok=format_ok,
            exact_match=exact_match,
        )
    finally:
        for handle in handles:
            handle.remove()


def _load_responses(path: Path) -> List[ResponseRecord]:
    records: List[ResponseRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            rec = ResponseRecord.from_json(json.loads(line))
            if rec.example_id < 0:
                rec.example_id = idx
            records.append(rec)
    return records


# -----------------------------------------------------------------------------
# Main


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Intervene on probe subspace during generation (tree traversal or GSM8K).",
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--probe", type=Path, default=None, help="Optional explicit probe path. Defaults derived from dataset/model/proj dim.")
    parser.add_argument("--layer", type=int, default=10, help="Transformer layer to ablate.")
    parser.add_argument("--proj-dim", type=int, default=4, help="Projection dimension used for the probe file name.")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Number of PCA components used for the probe file name (-1 for full embeddings).",
    )
    parser.add_argument(
        "--basis",
        type=str,
        choices=("distance", "depth", "combined"),
        default="combined",
        help="Which probe basis to ablate: distance subspace, depth direction, or combined (distance+depth).",
    )
    parser.add_argument("--normalize-tree", dest="normalize_tree", action="store_true", help="Whether the probe was trained with tree-distance normalization (affects file name).")
    parser.add_argument("--normalize-depth", dest="normalize_tree", action="store_true", help=argparse.SUPPRESS)  # backward compatibility
    parser.add_argument("--responses", type=Path, default=None, help="Optional override for baseline responses JSONL.")
    parser.add_argument(
        "--bucket",
        type=str,
        default="exact",
        choices=("all", "exact", "partial", "zero"),
        help="Bucket of test examples to evaluate.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2000, help="Max new tokens to generate.")
    parser.add_argument("--num-random", type=int, default=1, help="Number of random subspace ablations per example.")
    parser.add_argument("--train-split", type=float, default=0.5, help="Train fraction (math setting only).")
    parser.add_argument("--gsm8k-split", type=str, default="all", help="GSM8K split tag for default dataset resolution.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of test examples (0 = all).")
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="Skip regeneration and accuracy metrics; compute logit diffs only.",
    )
    parser.add_argument(
        "--layer-sweep",
        action="store_true",
        help="Run the ablation for every layer and save a combined output file.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output npz under cutter/data/ for intervention results.")
    return parser.parse_args(argv)


def run_math_intervention(args: argparse.Namespace, model_id: str) -> None:
    dataset_tag = path_utils.resolve_dataset_tag(
        "math",
        args.dataset,
        num_samples=0,
        seed=args.seed,
        gsm8k_split=args.gsm8k_split,
    )
    if args.probe is None:
        raise ValueError("Math setting requires --probe pointing at the tree probe artifact.")
    probe_fp = args.probe

    meta, _, results = _load_probe(probe_fp)

    responses_fp = args.responses or responses_path(dataset_tag, model_id)
    all_responses = load_math_responses(responses_fp)
    train_records, test_records = split_exact_only(all_responses, args.train_split, args.seed)
    selected: List[MathResponseRecord] = list(test_records)
    if not selected:
        raise RuntimeError("No responses found for the requested test split.")
    if args.bucket != "all":
        filtered: List[MathResponseRecord] = []
        for rec in selected:
            if rec.exact_match:
                bucket = "exact"
            elif rec.format_ok:
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

    baseline_exact = np.mean([rec.exact_match for rec in selected])
    baseline_partial = np.mean([rec.partial_score for rec in selected])
    print(
        f"Loaded probe from {probe_fp} | test examples ({args.bucket}): {len(selected)} "
        f"(baseline exact {baseline_exact:.3f}, partial {baseline_partial:.3f})"
    )

    tokenizer, model = load_reasoning_model(model_id, device=DEVICE, use_half_precision=True)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    layer_modules = _resolve_layers(model)
    total_layers = len(layer_modules)
    layer_indices = list(range(total_layers)) if args.layer_sweep else [args.layer]
    missing_layers = [layer for layer in layer_indices if layer not in results]
    if missing_layers:
        raise ValueError(f"Layers missing in probe results: {missing_layers}")

    example_cache: List[Dict[str, Any]] = []
    for rec in selected:
        prompt_text = _render_math_prompt(tokenizer, rec.prompt)
        answer_text = str(rec.answer).strip()
        prefix_text_for_logits, prefix_found = _math_prefix_for_logits(
            rec.raw_response,
            rec.pred_answer,
            answer_text,
        )
        answer_token_ids = tokenizer.encode(answer_text, add_special_tokens=False)
        compute_logits = bool(rec.exact_match and prefix_found and answer_token_ids)
        base_answer_logits = None
        if compute_logits:
            base_answer_logits = _compute_answer_logits(
                prompt_text=prompt_text,
                prefix_text=prefix_text_for_logits,
                answer_text=answer_text,
                tokenizer=tokenizer,
                model=model,
                ablation_basis=None,
                target_layer=None,
            )
        example_cache.append(
            {
                "rec": rec,
                "prompt_text": prompt_text,
                "answer_text": answer_text,
                "prefix_text_for_logits": prefix_text_for_logits,
                "answer_token_ids": answer_token_ids,
                "compute_logits": compute_logits,
                "base_answer_logits": base_answer_logits,
            }
        )

    layer_results: List[Dict[str, Any]] = []
    for layer_idx in layer_indices:
        layer_result = results[layer_idx]
        projection = _resolve_distance_projection(layer_result)
        probe_basis_np = _orthonormal_basis(projection)
        distance_basis_rank = probe_basis_np.shape[1]
        hidden_dim = projection.shape[0]
        depth_direction = _resolve_depth_direction(layer_result)
        if depth_direction.shape[0] != hidden_dim:
            raise ValueError(
                f"Depth direction dim {depth_direction.shape[0]} does not match hidden dim {hidden_dim}."
            )
        depth_basis_np = _orthonormal_basis(depth_direction.reshape(-1, 1))
        depth_basis_rank = depth_basis_np.shape[1]
        combined_basis_np = _orthonormal_basis(np.column_stack([probe_basis_np, depth_direction]))
        combined_basis_rank = combined_basis_np.shape[1]
        if args.basis == "distance":
            selected_basis_np = probe_basis_np
        elif args.basis == "depth":
            selected_basis_np = depth_basis_np
        else:
            selected_basis_np = combined_basis_np
        basis_rank = selected_basis_np.shape[1]
        pca_basis_np, pca_basis_rank, pca_info = _resolve_pca_basis(layer_result, basis_rank)
        pca_basis: Optional[torch.Tensor] = None
        if pca_basis_np is None:
            print("Warning: PCA components not found in probe artifact; skipping PCA ablation baseline.")
        elif pca_basis_rank < basis_rank:
            print(f"Warning: PCA basis rank {pca_basis_rank} < basis rank {basis_rank}; using {pca_basis_rank} components.")
        if not train_records:
            raise RuntimeError("No training records found for full-CoT PCA basis.")

        target_layer = layer_modules[layer_idx]
        ablation_basis = torch.tensor(selected_basis_np, dtype=torch.float32, device=model.device).to(dtype=model_dtype)
        if pca_basis_np is not None:
            pca_basis = torch.tensor(pca_basis_np, dtype=torch.float32, device=model.device).to(dtype=model_dtype)
        full_pca_embeddings = _collect_full_embeddings(
            train_records,
            tokenizer=tokenizer,
            model=model,
            layer_idx=layer_idx,
            render_prompt=lambda prompt: _render_math_prompt(tokenizer, prompt),
        )
        full_pca_basis = None
        full_pca_basis_rank = 0
        max_components = min(full_pca_embeddings.shape[0], full_pca_embeddings.shape[1])
        full_take = min(basis_rank, max_components)
        if full_take <= 0:
            raise RuntimeError("Insufficient data for full-CoT PCA components.")
        pca_full = PCA(n_components=full_take, svd_solver="auto", random_state=args.seed)
        pca_full.fit(full_pca_embeddings)
        full_pca_basis_np = pca_full.components_.astype(np.float32).T
        full_pca_basis = torch.tensor(full_pca_basis_np, dtype=torch.float32, device=model.device).to(dtype=model_dtype)
        full_pca_basis_rank = full_pca_basis.shape[1]
        if full_pca_basis_rank < basis_rank:
            print(
                f"Warning: full-CoT PCA basis rank {full_pca_basis_rank} < basis rank {basis_rank}; using {full_pca_basis_rank} components."
            )

        per_example: List[Dict[str, Any]] = []
        baseline_exact_seen = 0.0
        baseline_partial_seen = 0.0
        probe_exacts: List[float] = []
        probe_partials: List[float] = []
        probe_logit_diffs: List[float] = []
        pca_exacts: List[float] = []
        pca_partials: List[float] = []
        pca_logit_diffs: List[float] = []
        full_pca_exacts: List[float] = []
        full_pca_partials: List[float] = []
        full_pca_logit_diffs: List[float] = []
        zero_exacts: List[float] = []
        zero_partials: List[float] = []
        zero_logit_diffs: List[float] = []

        random_exacts: List[List[float]] = [[] for _ in range(args.num_random)]
        random_partials: List[List[float]] = [[] for _ in range(args.num_random)]
        random_logit_diffs: List[List[float]] = [[] for _ in range(args.num_random)]

        total = len(example_cache)
        for idx, cache in enumerate(example_cache, start=1):
            rec = cache["rec"]
            prompt_text = cache["prompt_text"]
            answer_text = cache["answer_text"]
            prefix_text_for_logits = cache["prefix_text_for_logits"]
            answer_token_ids = cache["answer_token_ids"]
            compute_logits = cache["compute_logits"]
            base_answer_logits = cache["base_answer_logits"]

            probe_gen = None
            if not args.answer_only:
                probe_gen = _run_math_generation(
                    prompt_text,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    ablation_basis=ablation_basis,
                    target_layer=target_layer,
                    gold_answer=rec.answer,
                )
                probe_exacts.append(float(probe_gen.exact_match))
                probe_partials.append(1.0 if probe_gen.exact_match else 0.0)
            probe_logit_gap = float("nan")
            if compute_logits and base_answer_logits is not None:
                probe_answer_logits = _compute_answer_logits(
                    prompt_text=prompt_text,
                    prefix_text=prefix_text_for_logits,
                    answer_text=answer_text,
                    tokenizer=tokenizer,
                    model=model,
                    ablation_basis=ablation_basis,
                    target_layer=target_layer,
                )
                probe_logit_gap = _mean_abs_answer_logit_diff(
                    base_answer_logits,
                    probe_answer_logits,
                    answer_token_ids,
                )
            probe_logit_diffs.append(probe_logit_gap)

            pca_gen = None
            pca_logit_gap = float("nan")
            if pca_basis is not None:
                if not args.answer_only:
                    pca_gen = _run_math_generation(
                        prompt_text,
                        tokenizer=tokenizer,
                        model=model,
                        max_new_tokens=args.max_new_tokens,
                        ablation_basis=pca_basis,
                        target_layer=target_layer,
                        gold_answer=rec.answer,
                    )
                    pca_exacts.append(float(pca_gen.exact_match))
                    pca_partials.append(1.0 if pca_gen.exact_match else 0.0)
                if compute_logits and base_answer_logits is not None:
                    pca_answer_logits = _compute_answer_logits(
                        prompt_text=prompt_text,
                        prefix_text=prefix_text_for_logits,
                        answer_text=answer_text,
                        tokenizer=tokenizer,
                        model=model,
                        ablation_basis=pca_basis,
                        target_layer=target_layer,
                    )
                    pca_logit_gap = _mean_abs_answer_logit_diff(
                        base_answer_logits,
                        pca_answer_logits,
                        answer_token_ids,
                    )
            pca_logit_diffs.append(pca_logit_gap)

            full_pca_gen = None
            if not args.answer_only:
                full_pca_gen = _run_math_generation(
                    prompt_text,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    ablation_basis=full_pca_basis,
                    target_layer=target_layer,
                    gold_answer=rec.answer,
                )
                full_pca_exacts.append(float(full_pca_gen.exact_match))
                full_pca_partials.append(1.0 if full_pca_gen.exact_match else 0.0)
            full_pca_logit_gap = float("nan")
            if compute_logits and base_answer_logits is not None:
                full_pca_answer_logits = _compute_answer_logits(
                    prompt_text=prompt_text,
                    prefix_text=prefix_text_for_logits,
                    answer_text=answer_text,
                    tokenizer=tokenizer,
                    model=model,
                    ablation_basis=full_pca_basis,
                    target_layer=target_layer,
                )
                full_pca_logit_gap = _mean_abs_answer_logit_diff(
                    base_answer_logits,
                    full_pca_answer_logits,
                    answer_token_ids,
                )
            full_pca_logit_diffs.append(full_pca_logit_gap)

            zero_gen = None
            if not args.answer_only:
                zero_gen = _run_math_generation(
                    prompt_text,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    target_layer=target_layer,
                    ablation_zero=True,
                    gold_answer=rec.answer,
                )
                zero_exacts.append(float(zero_gen.exact_match))
                zero_partials.append(1.0 if zero_gen.exact_match else 0.0)
            zero_logit_gap = float("nan")
            if compute_logits and base_answer_logits is not None:
                zero_answer_logits = _compute_answer_logits(
                    prompt_text=prompt_text,
                    prefix_text=prefix_text_for_logits,
                    answer_text=answer_text,
                    tokenizer=tokenizer,
                    model=model,
                    target_layer=target_layer,
                    ablation_zero=True,
                )
                zero_logit_gap = _mean_abs_answer_logit_diff(
                    base_answer_logits,
                    zero_answer_logits,
                    answer_token_ids,
                )
            zero_logit_diffs.append(zero_logit_gap)

            random_runs: List[Dict[str, Any]] = []
            for ridx in range(args.num_random):
                seed = args.seed + 1 + ridx * 10_000 + rec.example_id
                rand_basis = _sample_random_basis(hidden_dim, basis_rank, seed).to(model.device, dtype=model_dtype)
                rand_gen = None
                if not args.answer_only:
                    rand_gen = _run_math_generation(
                        prompt_text,
                        tokenizer=tokenizer,
                        model=model,
                        max_new_tokens=args.max_new_tokens,
                        ablation_basis=rand_basis,
                        target_layer=target_layer,
                        gold_answer=rec.answer,
                    )
                    random_exacts[ridx].append(float(rand_gen.exact_match))
                    random_partials[ridx].append(1.0 if rand_gen.exact_match else 0.0)
                rand_logit_gap = float("nan")
                if compute_logits and base_answer_logits is not None:
                    rand_answer_logits = _compute_answer_logits(
                        prompt_text=prompt_text,
                        prefix_text=prefix_text_for_logits,
                        answer_text=answer_text,
                        tokenizer=tokenizer,
                        model=model,
                        ablation_basis=rand_basis,
                        target_layer=target_layer,
                    )
                    rand_logit_gap = _mean_abs_answer_logit_diff(
                        base_answer_logits,
                        rand_answer_logits,
                        answer_token_ids,
                    )
                random_logit_diffs[ridx].append(rand_logit_gap)
                random_runs.append(
                    {
                        "seed": seed,
                        "response": rand_gen.response if rand_gen is not None else "",
                        "pred_answer": rand_gen.pred_answer if rand_gen is not None else None,
                        "pred_method": rand_gen.pred_method if rand_gen is not None else "",
                        "format_ok": rand_gen.format_ok if rand_gen is not None else None,
                        "exact_match": rand_gen.exact_match if rand_gen is not None else None,
                        "partial_score": 1.0 if (rand_gen is not None and rand_gen.exact_match) else float("nan"),
                        "logit_diff": rand_logit_gap,
                    }
                )

            baseline_exact_seen += float(rec.exact_match)
            baseline_partial_seen += float(rec.partial_score)

            per_example.append(
                {
                    "example_id": rec.example_id,
                    "question": rec.question,
                    "answer": rec.answer,
                    "prompt": rec.prompt,
                    "baseline": {
                        "response": rec.raw_response,
                        "pred_answer": rec.pred_answer,
                        "pred_method": rec.pred_method,
                        "format_ok": rec.format_ok,
                        "exact_match": rec.exact_match,
                        "partial_score": rec.partial_score,
                    },
                    "probe_ablation": _math_ablation_record(
                        probe_gen,
                        probe_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "pca_ablation": _math_ablation_record(
                        pca_gen,
                        pca_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "full_pca_ablation": _math_ablation_record(
                        full_pca_gen,
                        full_pca_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "zero_ablation": _math_ablation_record(
                        zero_gen,
                        zero_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "random_ablation": random_runs,
                }
            )

            if idx % 10 == 0 or idx == total:
                prefix = f"Layer {layer_idx} | " if args.layer_sweep else ""
                if args.answer_only:
                    probe_logit_so_far = float(np.nanmean(probe_logit_diffs)) if probe_logit_diffs else float("nan")
                    pca_logit_so_far = float(np.nanmean(pca_logit_diffs)) if pca_logit_diffs else float("nan")
                    full_pca_logit_so_far = float(np.nanmean(full_pca_logit_diffs)) if full_pca_logit_diffs else float("nan")
                    zero_logit_so_far = float(np.nanmean(zero_logit_diffs)) if zero_logit_diffs else float("nan")
                    rand_logit_means = [float(np.nanmean(r[:idx])) if r else float("nan") for r in random_logit_diffs]
                    print(
                        f"{prefix}Processed {idx}/{total} | "
                        f"probe logit Δ={probe_logit_so_far:.4f} | "
                        f"pca logit Δ={pca_logit_so_far:.4f} | "
                        f"full pca logit Δ={full_pca_logit_so_far:.4f} | "
                        f"zero logit Δ={zero_logit_so_far:.4f} | "
                        f"rand logit Δ={rand_logit_means[0]:.4f}"
                    )
                else:
                    probe_exact_so_far = float(np.mean(probe_exacts)) if probe_exacts else float("nan")
                    probe_partial_so_far = float(np.mean(probe_partials)) if probe_partials else float("nan")
                    pca_exact_so_far = float(np.mean(pca_exacts)) if pca_exacts else float("nan")
                    pca_partial_so_far = float(np.mean(pca_partials)) if pca_partials else float("nan")
                    full_pca_exact_so_far = float(np.mean(full_pca_exacts)) if full_pca_exacts else float("nan")
                    full_pca_partial_so_far = float(np.mean(full_pca_partials)) if full_pca_partials else float("nan")
                    zero_exact_so_far = float(np.mean(zero_exacts)) if zero_exacts else float("nan")
                    zero_partial_so_far = float(np.mean(zero_partials)) if zero_partials else float("nan")
                    baseline_exact_so_far = baseline_exact_seen / idx
                    baseline_partial_so_far = baseline_partial_seen / idx
                    rand_exact_means = [float(np.mean(r[:idx])) if r else float("nan") for r in random_exacts]
                    rand_partial_means = [float(np.mean(r[:idx])) if r else float("nan") for r in random_partials]
                    print(
                        f"{prefix}Processed {idx}/{total} | "
                        f"exact Δ={probe_exact_so_far - baseline_exact_so_far:+.4f} "
                        f"(probe {probe_exact_so_far:.4f}, base {baseline_exact_so_far:.4f}) | "
                        f"partial Δ={probe_partial_so_far - baseline_partial_so_far:+.4f} "
                        f"(probe {probe_partial_so_far:.4f}, base {baseline_partial_so_far:.4f}) | "
                        f"pca exact {pca_exact_so_far:.4f} | pca partial {pca_partial_so_far:.4f} | "
                        f"full pca exact {full_pca_exact_so_far:.4f} | full pca partial {full_pca_partial_so_far:.4f} | "
                        f"zero exact {zero_exact_so_far:.4f} | zero partial {zero_partial_so_far:.4f} | "
                        f"rand exact {rand_exact_means[0]:.4f} | rand partial {rand_partial_means[0]:.4f}"
                    )

        if args.answer_only:
            probe_exact_rate = float("nan")
            probe_partial_mean = float("nan")
            pca_exact_rate = float("nan")
            pca_partial_mean = float("nan")
            full_pca_exact_rate = float("nan")
            full_pca_partial_mean = float("nan")
            zero_exact_rate = float("nan")
            zero_partial_mean = float("nan")
            random_exact_rates = [float("nan") for _ in range(args.num_random)]
            random_partial_means = [float("nan") for _ in range(args.num_random)]
        else:
            probe_exact_rate = float(np.mean(probe_exacts)) if probe_exacts else float("nan")
            probe_partial_mean = float(np.mean(probe_partials)) if probe_partials else float("nan")
            pca_exact_rate = float(np.mean(pca_exacts)) if pca_exacts else float("nan")
            pca_partial_mean = float(np.mean(pca_partials)) if pca_partials else float("nan")
            full_pca_exact_rate = float(np.mean(full_pca_exacts)) if full_pca_exacts else float("nan")
            full_pca_partial_mean = float(np.mean(full_pca_partials)) if full_pca_partials else float("nan")
            zero_exact_rate = float(np.mean(zero_exacts)) if zero_exacts else float("nan")
            zero_partial_mean = float(np.mean(zero_partials)) if zero_partials else float("nan")
            random_exact_rates = [float(np.mean(r)) if r else float("nan") for r in random_exacts]
            random_partial_means = [float(np.mean(r)) if r else float("nan") for r in random_partials]

        probe_logit_mean = float(np.nanmean(probe_logit_diffs)) if probe_logit_diffs else float("nan")
        pca_logit_mean = float(np.nanmean(pca_logit_diffs)) if pca_logit_diffs else float("nan")
        full_pca_logit_mean = float(np.nanmean(full_pca_logit_diffs)) if full_pca_logit_diffs else float("nan")
        zero_logit_mean = float(np.nanmean(zero_logit_diffs)) if zero_logit_diffs else float("nan")
        random_logit_means = [float(np.nanmean(r)) if r else float("nan") for r in random_logit_diffs]

        aggregate = {
            "baseline_exact": float(baseline_exact),
            "baseline_partial": float(baseline_partial),
            "probe_exact": probe_exact_rate,
            "probe_partial": probe_partial_mean,
            "probe_accuracy_delta": probe_exact_rate - baseline_exact,
            "probe_partial_delta": probe_partial_mean - baseline_partial,
            "probe_logit_diff_mean": probe_logit_mean,
            "pca_exact": pca_exact_rate,
            "pca_partial": pca_partial_mean,
            "pca_accuracy_delta": pca_exact_rate - baseline_exact,
            "pca_partial_delta": pca_partial_mean - baseline_partial,
            "pca_logit_diff_mean": pca_logit_mean,
            "full_pca_exact": full_pca_exact_rate,
            "full_pca_partial": full_pca_partial_mean,
            "full_pca_accuracy_delta": full_pca_exact_rate - baseline_exact,
            "full_pca_partial_delta": full_pca_partial_mean - baseline_partial,
            "full_pca_logit_diff_mean": full_pca_logit_mean,
            "zero_exact": zero_exact_rate,
            "zero_partial": zero_partial_mean,
            "zero_accuracy_delta": zero_exact_rate - baseline_exact,
            "zero_partial_delta": zero_partial_mean - baseline_partial,
            "zero_logit_diff_mean": zero_logit_mean,
            "random_exact": random_exact_rates,
            "random_partial": random_partial_means,
            "random_accuracy_delta": [r - baseline_exact for r in random_exact_rates],
            "random_partial_delta": [r - baseline_partial for r in random_partial_means],
            "random_logit_diff_mean": random_logit_means,
        }

        meta_out = {
            "probe_path": repo_relative(probe_fp),
            "responses_path": repo_relative(responses_fp),
            "model_id": model_id,
            "dataset_tag": dataset_tag,
            "layer": layer_idx,
            "basis_rank": basis_rank,
            "basis": args.basis,
            "distance_basis_rank": distance_basis_rank,
            "depth_basis_rank": depth_basis_rank,
            "combined_basis_rank": combined_basis_rank,
            "depth_direction_norm": float(np.linalg.norm(depth_direction)),
            "pca_basis_rank": pca_basis_rank,
            "full_pca_basis_rank": full_pca_basis_rank,
            "pca_components": int(pca_info.get("n_components", -1)) if isinstance(pca_info, dict) else -1,
            "hidden_dim": hidden_dim,
            "num_random": args.num_random,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "test_example_count": len(selected),
            "baseline_exact": float(baseline_exact),
            "baseline_partial": float(baseline_partial),
            "bucket": args.bucket,
            "train_split": args.train_split,
            "full_pca_train_examples": len(train_records),
            "answer_only": args.answer_only,
            "layer_sweep": args.layer_sweep,
        }

        layer_results.append(
            {
                "layer": layer_idx,
                "aggregate": aggregate,
                "records": per_example,
                "meta": meta_out,
            }
        )

        if args.answer_only:
            print(
                f"Layer {layer_idx} | {args.basis.capitalize()} ablation mean logit diff (answer tokens): "
                f"{aggregate['probe_logit_diff_mean']:.4f}"
            )
        else:
            print(
                f"Layer {layer_idx} | {args.basis.capitalize()} ablation accuracy delta (exact): "
                f"{aggregate['probe_accuracy_delta']:.4f} | partial delta: {aggregate['probe_partial_delta']:.4f}"
            )

    sample_layer = layer_indices[0] if layer_indices else args.layer
    sample_pca_info = results.get(sample_layer, {}).get("pca") if isinstance(results.get(sample_layer), dict) else None
    pca_components = int(sample_pca_info.get("n_components", -1)) if isinstance(sample_pca_info, dict) else args.pca_components
    layer_tag: int | str = "all" if args.layer_sweep else args.layer
    default_out = intervention_path(
        dataset_tag,
        model_id,
        args.proj_dim,
        pca_components,
        layer_tag,
        tag=_output_tag(args),
    )
    output_path = args.output or default_out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.layer_sweep:
        meta_out = {
            "probe_path": repo_relative(probe_fp),
            "responses_path": repo_relative(responses_fp),
            "model_id": model_id,
            "dataset_tag": dataset_tag,
            "layers": layer_indices,
            "num_layers": len(layer_indices),
            "num_random": args.num_random,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "test_example_count": len(selected),
            "baseline_exact": float(baseline_exact),
            "baseline_partial": float(baseline_partial),
            "bucket": args.bucket,
            "train_split": args.train_split,
            "answer_only": args.answer_only,
            "layer_sweep": True,
        }
        aggregate_out = {
            "per_layer": [{"layer": lr["layer"], "aggregate": lr["aggregate"]} for lr in layer_results]
        }
        records_out = layer_results
    else:
        aggregate_out = layer_results[0]["aggregate"]
        records_out = layer_results[0]["records"]
        meta_out = layer_results[0]["meta"]

    np.savez_compressed(
        output_path,
        meta=np.array(meta_out, dtype=object),
        aggregate=np.array(aggregate_out, dtype=object),
        records=np.array(records_out, dtype=object),
    )

    print(f"Saved intervention results to {output_path}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    set_global_seed(args.seed)
    model_id = resolve_single_model_id(args.reasoning_models, args.chat_models)
    if args.setting == "math":
        run_math_intervention(args, model_id)
        return

    dataset_tag = path_utils.resolve_dataset_tag(
        "tree",
        args.dataset,
        num_samples=0,
        seed=args.seed,
        gsm8k_split=args.gsm8k_split,
    )
    derived_probe = probe_path(dataset_tag, model_id, args.proj_dim, args.normalize_tree, args.pca_components)
    probe_fp = args.probe or derived_probe
    if args.probe is None and not probe_fp.exists():
        legacy_name = f"probe_proj{args.proj_dim}.npz"
        if args.normalize_tree:
            legacy_name = f"probe_normtree_proj{args.proj_dim}.npz"
        legacy_probe = probe_fp.parent / legacy_name
        if legacy_probe.exists():
            probe_fp = legacy_probe

    meta, encodings, results = _load_probe(probe_fp)

    # Responses default to the dataset/model folder, but allow overrides or probe meta fallback.
    default_responses_path = responses_path(dataset_tag, model_id)
    meta_responses = meta.get("responses_path")
    if args.responses:
        responses_fp = args.responses
    elif meta_responses:
        responses_fp = resolve_repo_path(Path(meta_responses))
    else:
        responses_fp = default_responses_path
    all_responses = _load_responses(responses_fp)
    response_by_id = {rec.example_id: rec for rec in all_responses}

    tokenizer, model = load_reasoning_model(model_id, device=DEVICE, use_half_precision=True)
    model.eval()
    model_dtype = next(model.parameters()).dtype
    layer_modules = _resolve_layers(model)
    total_layers = len(layer_modules)
    layer_indices = list(range(total_layers)) if args.layer_sweep else [args.layer]
    missing_layers = [layer for layer in layer_indices if layer not in results]
    if missing_layers:
        raise ValueError(f"Layers missing in probe results: {missing_layers}")

    layer_results: List[Dict[str, Any]] = []
    for layer_idx in layer_indices:
        enc_layer = encodings[layer_idx]
        train_idx = np.asarray(enc_layer["train_idx"], dtype=int)
        test_idx = np.asarray(enc_layer["test_idx"], dtype=int)
        example_ids = np.asarray(enc_layer["example_ids"], dtype=int)
        train_examples = sorted(set(example_ids[train_idx].tolist()))
        test_examples = sorted(set(example_ids[test_idx].tolist()))

        train_records = [response_by_id[eid] for eid in train_examples if eid in response_by_id]
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

        baseline_exact = np.mean([rec.exact_match for rec in selected])
        baseline_partial = np.mean([rec.partial_score for rec in selected])

        layer_result = results[layer_idx]
        projection = _resolve_distance_projection(layer_result)
        probe_basis_np = _orthonormal_basis(projection)
        distance_basis_rank = probe_basis_np.shape[1]
        hidden_dim = projection.shape[0]
        depth_direction = _resolve_depth_direction(layer_result)
        if depth_direction.shape[0] != hidden_dim:
            raise ValueError(
                f"Depth direction dim {depth_direction.shape[0]} does not match hidden dim {hidden_dim}."
            )
        depth_basis_np = _orthonormal_basis(depth_direction.reshape(-1, 1))
        depth_basis_rank = depth_basis_np.shape[1]
        combined_basis_np = _orthonormal_basis(np.column_stack([probe_basis_np, depth_direction]))
        combined_basis_rank = combined_basis_np.shape[1]
        if args.basis == "distance":
            selected_basis_np = probe_basis_np
        elif args.basis == "depth":
            selected_basis_np = depth_basis_np
        else:
            selected_basis_np = combined_basis_np
        basis_rank = selected_basis_np.shape[1]
        pca_basis_np, pca_basis_rank, pca_info = _resolve_pca_basis(layer_result, basis_rank)
        pca_basis: Optional[torch.Tensor] = None
        if pca_basis_np is None:
            print("Warning: PCA components not found in probe artifact; skipping PCA ablation baseline.")
        elif pca_basis_rank < basis_rank:
            print(f"Warning: PCA basis rank {pca_basis_rank} < basis rank {basis_rank}; using {pca_basis_rank} components.")
        if not train_records:
            raise RuntimeError("No training records found for full-CoT PCA basis.")

        print(
            f"Loaded probe from {probe_fp} | layer {layer_idx} | basis {args.basis} rank {basis_rank} | hidden dim {hidden_dim}"
        )
        print(
            f"Test examples ({args.bucket}): {len(selected)} (baseline exact {baseline_exact:.3f}, partial {baseline_partial:.3f})"
        )

        target_layer = layer_modules[layer_idx]
        ablation_basis = torch.tensor(selected_basis_np, dtype=torch.float32, device=model.device).to(dtype=model_dtype)
        if pca_basis_np is not None:
            pca_basis = torch.tensor(pca_basis_np, dtype=torch.float32, device=model.device).to(dtype=model_dtype)
        full_pca_embeddings = _collect_full_embeddings(
            train_records,
            tokenizer=tokenizer,
            model=model,
            layer_idx=layer_idx,
            render_prompt=lambda prompt: _render_prompt(tokenizer, prompt),
        )
        max_components = min(full_pca_embeddings.shape[0], full_pca_embeddings.shape[1])
        full_take = min(basis_rank, max_components)
        if full_take <= 0:
            raise RuntimeError("Insufficient data for full-CoT PCA components.")
        pca_full = PCA(n_components=full_take, svd_solver="auto", random_state=args.seed)
        pca_full.fit(full_pca_embeddings)
        full_pca_basis_np = pca_full.components_.astype(np.float32).T
        full_pca_basis = torch.tensor(full_pca_basis_np, dtype=torch.float32, device=model.device).to(dtype=model_dtype)
        full_pca_basis_rank = full_pca_basis.shape[1]
        if full_pca_basis_rank < basis_rank:
            print(
                f"Warning: full-CoT PCA basis rank {full_pca_basis_rank} < basis rank {basis_rank}; using {full_pca_basis_rank} components."
            )

        example_cache: List[Dict[str, Any]] = []
        for rec in selected:
            prompt_text = _render_prompt(tokenizer, rec.prompt)
            prefix_text_for_logits = ""
            if rec.parsed_text:
                start_idx = rec.raw_response.rfind(rec.parsed_text)
                if start_idx != -1:
                    prefix_text_for_logits = rec.raw_response[:start_idx]
            answer_text = _build_answer_text(rec.ground_truth_path)
            answer_mask = _answer_token_mask(answer_text, tokenizer)
            node_token_ids = _node_token_ids(rec.depth, tokenizer)
            base_answer_logits = _compute_answer_logits(
                prompt_text=prompt_text,
                prefix_text=prefix_text_for_logits,
                answer_text=answer_text,
                tokenizer=tokenizer,
                model=model,
                ablation_basis=None,
                target_layer=None,
            )
            example_cache.append(
                {
                    "rec": rec,
                    "prompt_text": prompt_text,
                    "prefix_text_for_logits": prefix_text_for_logits,
                    "answer_text": answer_text,
                    "answer_mask": answer_mask,
                    "node_token_ids": node_token_ids,
                    "base_answer_logits": base_answer_logits,
                }
            )

        per_example: List[Dict[str, Any]] = []
        baseline_exact_seen = 0.0
        baseline_partial_seen = 0.0
        probe_logit_diffs: List[float] = []
        probe_exacts: List[float] = []
        probe_partials: List[float] = []
        pca_logit_diffs: List[float] = []
        pca_exacts: List[float] = []
        pca_partials: List[float] = []
        full_pca_logit_diffs: List[float] = []
        full_pca_exacts: List[float] = []
        full_pca_partials: List[float] = []
        zero_logit_diffs: List[float] = []
        zero_exacts: List[float] = []
        zero_partials: List[float] = []

        random_logit_diffs: List[List[float]] = [[] for _ in range(args.num_random)]
        random_exacts: List[List[float]] = [[] for _ in range(args.num_random)]
        random_partials: List[List[float]] = [[] for _ in range(args.num_random)]

        total = len(example_cache)
        for idx, cache in enumerate(example_cache, start=1):
            rec = cache["rec"]
            prompt_text = cache["prompt_text"]
            prefix_text_for_logits = cache["prefix_text_for_logits"]
            answer_text = cache["answer_text"]
            answer_mask = cache["answer_mask"]
            node_token_ids = cache["node_token_ids"]
            base_answer_logits = cache["base_answer_logits"]

            probe_gen = None
            if not args.answer_only:
                probe_gen = _run_generation(
                    prompt_text,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    ablation_basis=ablation_basis,
                    target_layer=target_layer,
                    ground_truth_path=rec.ground_truth_path,
                )
                probe_exacts.append(float(probe_gen.exact_match))
                probe_partials.append(float(probe_gen.partial_score))
            probe_answer_logits = _compute_answer_logits(
                prompt_text=prompt_text,
                prefix_text=prefix_text_for_logits,
                answer_text=answer_text,
                tokenizer=tokenizer,
                model=model,
                ablation_basis=ablation_basis,
                target_layer=target_layer,
            )
            probe_logit_gap = _mean_abs_node_logit_diff(
                base_answer_logits,
                probe_answer_logits,
                node_token_ids,
                answer_mask,
            )
            probe_logit_diffs.append(probe_logit_gap)

            pca_gen = None
            pca_logit_gap = float("nan")
            if pca_basis is not None:
                if not args.answer_only:
                    pca_gen = _run_generation(
                        prompt_text,
                        tokenizer=tokenizer,
                        model=model,
                        max_new_tokens=args.max_new_tokens,
                        ablation_basis=pca_basis,
                        target_layer=target_layer,
                        ground_truth_path=rec.ground_truth_path,
                    )
                    pca_exacts.append(float(pca_gen.exact_match))
                    pca_partials.append(float(pca_gen.partial_score))
                pca_answer_logits = _compute_answer_logits(
                    prompt_text=prompt_text,
                    prefix_text=prefix_text_for_logits,
                    answer_text=answer_text,
                    tokenizer=tokenizer,
                    model=model,
                    ablation_basis=pca_basis,
                    target_layer=target_layer,
                )
                pca_logit_gap = _mean_abs_node_logit_diff(
                    base_answer_logits,
                    pca_answer_logits,
                    node_token_ids,
                    answer_mask,
                )
            pca_logit_diffs.append(pca_logit_gap)

            full_pca_gen = None
            if not args.answer_only:
                full_pca_gen = _run_generation(
                    prompt_text,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    ablation_basis=full_pca_basis,
                    target_layer=target_layer,
                    ground_truth_path=rec.ground_truth_path,
                )
                full_pca_exacts.append(float(full_pca_gen.exact_match))
                full_pca_partials.append(float(full_pca_gen.partial_score))
            full_pca_answer_logits = _compute_answer_logits(
                prompt_text=prompt_text,
                prefix_text=prefix_text_for_logits,
                answer_text=answer_text,
                tokenizer=tokenizer,
                model=model,
                ablation_basis=full_pca_basis,
                target_layer=target_layer,
            )
            full_pca_logit_gap = _mean_abs_node_logit_diff(
                base_answer_logits,
                full_pca_answer_logits,
                node_token_ids,
                answer_mask,
            )
            full_pca_logit_diffs.append(full_pca_logit_gap)

            zero_gen = None
            if not args.answer_only:
                zero_gen = _run_generation(
                    prompt_text,
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=args.max_new_tokens,
                    target_layer=target_layer,
                    ablation_zero=True,
                    ground_truth_path=rec.ground_truth_path,
                )
                zero_exacts.append(float(zero_gen.exact_match))
                zero_partials.append(float(zero_gen.partial_score))
            zero_answer_logits = _compute_answer_logits(
                prompt_text=prompt_text,
                prefix_text=prefix_text_for_logits,
                answer_text=answer_text,
                tokenizer=tokenizer,
                model=model,
                target_layer=target_layer,
                ablation_zero=True,
            )
            zero_logit_gap = _mean_abs_node_logit_diff(
                base_answer_logits,
                zero_answer_logits,
                node_token_ids,
                answer_mask,
            )
            zero_logit_diffs.append(zero_logit_gap)

            random_runs: List[Dict[str, Any]] = []
            for ridx in range(args.num_random):
                seed = args.seed + 1 + ridx * 10_000 + rec.example_id
                rand_basis = _sample_random_basis(hidden_dim, basis_rank, seed).to(model.device, dtype=model_dtype)
                rand_gen = None
                if not args.answer_only:
                    rand_gen = _run_generation(
                        prompt_text,
                        tokenizer=tokenizer,
                        model=model,
                        max_new_tokens=args.max_new_tokens,
                        ablation_basis=rand_basis,
                        target_layer=target_layer,
                        ground_truth_path=rec.ground_truth_path,
                    )
                    random_exacts[ridx].append(float(rand_gen.exact_match))
                    random_partials[ridx].append(float(rand_gen.partial_score))
                rand_answer_logits = _compute_answer_logits(
                    prompt_text=prompt_text,
                    prefix_text=prefix_text_for_logits,
                    answer_text=answer_text,
                    tokenizer=tokenizer,
                    model=model,
                    ablation_basis=rand_basis,
                    target_layer=target_layer,
                )
                rand_gap = _mean_abs_node_logit_diff(
                    base_answer_logits,
                    rand_answer_logits,
                    node_token_ids,
                    answer_mask,
                )
                random_logit_diffs[ridx].append(rand_gap)
                random_runs.append(
                    {
                        "seed": seed,
                        "response": rand_gen.response if rand_gen is not None else "",
                        "parsed_path": rand_gen.parsed_path if rand_gen is not None else [],
                        "parsed_text": rand_gen.parsed_text if rand_gen is not None else None,
                        "format_ok": rand_gen.format_ok if rand_gen is not None else None,
                        "exact_match": rand_gen.exact_match if rand_gen is not None else None,
                        "partial_score": rand_gen.partial_score if rand_gen is not None else float("nan"),
                        "logit_diff": rand_gap,
                    }
                )

            baseline_exact_seen += float(rec.exact_match)
            baseline_partial_seen += float(rec.partial_score)

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
                    "probe_ablation": _tree_ablation_record(
                        probe_gen,
                        probe_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "pca_ablation": _tree_ablation_record(
                        pca_gen,
                        pca_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "full_pca_ablation": _tree_ablation_record(
                        full_pca_gen,
                        full_pca_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "zero_ablation": _tree_ablation_record(
                        zero_gen,
                        zero_logit_gap,
                        answer_only=args.answer_only,
                    ),
                    "random_ablation": random_runs,
                }
            )

            if idx % 10 == 0 or idx == total:
                prefix = f"Layer {layer_idx} | " if args.layer_sweep else ""
                if args.answer_only:
                    probe_logit_so_far = float(np.nanmean(probe_logit_diffs)) if probe_logit_diffs else float("nan")
                    pca_logit_so_far = float(np.nanmean(pca_logit_diffs)) if pca_logit_diffs else float("nan")
                    full_pca_logit_so_far = float(np.nanmean(full_pca_logit_diffs)) if full_pca_logit_diffs else float("nan")
                    zero_logit_so_far = float(np.nanmean(zero_logit_diffs)) if zero_logit_diffs else float("nan")
                    rand_logit_means = [float(np.nanmean(r[:idx])) if r else float("nan") for r in random_logit_diffs]
                    print(
                        f"{prefix}Processed {idx}/{total} | "
                        f"probe logit Δ={probe_logit_so_far:.4f} | "
                        f"pca logit Δ={pca_logit_so_far:.4f} | "
                        f"full pca logit Δ={full_pca_logit_so_far:.4f} | "
                        f"zero logit Δ={zero_logit_so_far:.4f} | "
                        f"rand logit Δ={rand_logit_means[0]:.4f}"
                    )
                else:
                    probe_exact_so_far = float(np.mean(probe_exacts)) if probe_exacts else float("nan")
                    probe_partial_so_far = float(np.mean(probe_partials)) if probe_partials else float("nan")
                    probe_logit_so_far = float(np.nanmean(probe_logit_diffs)) if probe_logit_diffs else float("nan")
                    pca_exact_so_far = float(np.mean(pca_exacts)) if pca_exacts else float("nan")
                    pca_partial_so_far = float(np.mean(pca_partials)) if pca_partials else float("nan")
                    pca_logit_so_far = float(np.nanmean(pca_logit_diffs)) if pca_logit_diffs else float("nan")
                    full_pca_exact_so_far = float(np.mean(full_pca_exacts)) if full_pca_exacts else float("nan")
                    full_pca_partial_so_far = float(np.mean(full_pca_partials)) if full_pca_partials else float("nan")
                    full_pca_logit_so_far = float(np.nanmean(full_pca_logit_diffs)) if full_pca_logit_diffs else float("nan")
                    zero_exact_so_far = float(np.mean(zero_exacts)) if zero_exacts else float("nan")
                    zero_partial_so_far = float(np.mean(zero_partials)) if zero_partials else float("nan")
                    zero_logit_so_far = float(np.nanmean(zero_logit_diffs)) if zero_logit_diffs else float("nan")
                    baseline_exact_so_far = baseline_exact_seen / idx
                    baseline_partial_so_far = baseline_partial_seen / idx
                    rand_exact_means = [float(np.mean(r[:idx])) if r else float("nan") for r in random_exacts]
                    rand_partial_means = [float(np.mean(r[:idx])) if r else float("nan") for r in random_partials]
                    print(
                        f"{prefix}Processed {idx}/{total} | "
                        f"exact Δ={probe_exact_so_far - baseline_exact_so_far:+.4f} "
                        f"(probe {probe_exact_so_far:.4f}, base {baseline_exact_so_far:.4f}) | "
                        f"partial Δ={probe_partial_so_far - baseline_partial_so_far:+.4f} "
                        f"(probe {probe_partial_so_far:.4f}, base {baseline_partial_so_far:.4f}) | "
                        f"probe logit Δ={probe_logit_so_far:.4f} | "
                        f"pca exact {pca_exact_so_far:.4f} | pca partial {pca_partial_so_far:.4f} | "
                        f"pca logit Δ={pca_logit_so_far:.4f} | "
                        f"full pca exact {full_pca_exact_so_far:.4f} | full pca partial {full_pca_partial_so_far:.4f} | "
                        f"full pca logit Δ={full_pca_logit_so_far:.4f} | "
                        f"zero exact {zero_exact_so_far:.4f} | zero partial {zero_partial_so_far:.4f} | "
                        f"zero logit Δ={zero_logit_so_far:.4f} | "
                        f"rand exact {rand_exact_means[0]:.4f} | rand partial {rand_partial_means[0]:.4f}"
                    )

        if args.answer_only:
            probe_exact_rate = float("nan")
            probe_partial_mean = float("nan")
            pca_exact_rate = float("nan")
            pca_partial_mean = float("nan")
            full_pca_exact_rate = float("nan")
            full_pca_partial_mean = float("nan")
            zero_exact_rate = float("nan")
            zero_partial_mean = float("nan")
            random_exact_rates = [float("nan") for _ in range(args.num_random)]
            random_partial_means = [float("nan") for _ in range(args.num_random)]
        else:
            probe_exact_rate = float(np.mean(probe_exacts)) if probe_exacts else float("nan")
            probe_partial_mean = float(np.mean(probe_partials)) if probe_partials else float("nan")
            pca_exact_rate = float(np.mean(pca_exacts)) if pca_exacts else float("nan")
            pca_partial_mean = float(np.mean(pca_partials)) if pca_partials else float("nan")
            full_pca_exact_rate = float(np.mean(full_pca_exacts)) if full_pca_exacts else float("nan")
            full_pca_partial_mean = float(np.mean(full_pca_partials)) if full_pca_partials else float("nan")
            zero_exact_rate = float(np.mean(zero_exacts)) if zero_exacts else float("nan")
            zero_partial_mean = float(np.mean(zero_partials)) if zero_partials else float("nan")
            random_exact_rates = [float(np.mean(r)) if r else float("nan") for r in random_exacts]
            random_partial_means = [float(np.mean(r)) if r else float("nan") for r in random_partials]

        probe_logit_mean = float(np.nanmean(probe_logit_diffs)) if probe_logit_diffs else float("nan")
        pca_logit_mean = float(np.nanmean(pca_logit_diffs)) if pca_logit_diffs else float("nan")
        full_pca_logit_mean = float(np.nanmean(full_pca_logit_diffs)) if full_pca_logit_diffs else float("nan")
        zero_logit_mean = float(np.nanmean(zero_logit_diffs)) if zero_logit_diffs else float("nan")
        random_logit_means = [float(np.nanmean(r)) if r else float("nan") for r in random_logit_diffs]

        aggregate = {
            "baseline_exact": float(baseline_exact),
            "baseline_partial": float(baseline_partial),
            "probe_exact": probe_exact_rate,
            "probe_partial": probe_partial_mean,
            "probe_accuracy_delta": probe_exact_rate - baseline_exact,
            "probe_partial_delta": probe_partial_mean - baseline_partial,
            "probe_logit_diff_mean": probe_logit_mean,
            "pca_exact": pca_exact_rate,
            "pca_partial": pca_partial_mean,
            "pca_accuracy_delta": pca_exact_rate - baseline_exact,
            "pca_partial_delta": pca_partial_mean - baseline_partial,
            "pca_logit_diff_mean": pca_logit_mean,
            "full_pca_exact": full_pca_exact_rate,
            "full_pca_partial": full_pca_partial_mean,
            "full_pca_accuracy_delta": full_pca_exact_rate - baseline_exact,
            "full_pca_partial_delta": full_pca_partial_mean - baseline_partial,
            "full_pca_logit_diff_mean": full_pca_logit_mean,
            "zero_exact": zero_exact_rate,
            "zero_partial": zero_partial_mean,
            "zero_accuracy_delta": zero_exact_rate - baseline_exact,
            "zero_partial_delta": zero_partial_mean - baseline_partial,
            "zero_logit_diff_mean": zero_logit_mean,
            "random_exact": random_exact_rates,
            "random_partial": random_partial_means,
            "random_accuracy_delta": [r - baseline_exact for r in random_exact_rates],
            "random_partial_delta": [r - baseline_partial for r in random_partial_means],
            "random_logit_diff_mean": random_logit_means,
        }

        meta_out = {
            "probe_path": repo_relative(probe_fp),
            "responses_path": repo_relative(responses_fp),
            "model_id": model_id,
            "dataset_tag": dataset_tag,
            "layer": layer_idx,
            "basis_rank": basis_rank,
            "basis": args.basis,
            "distance_basis_rank": distance_basis_rank,
            "depth_basis_rank": depth_basis_rank,
            "combined_basis_rank": combined_basis_rank,
            "depth_direction_norm": float(np.linalg.norm(depth_direction)),
            "pca_basis_rank": pca_basis_rank,
            "full_pca_basis_rank": full_pca_basis_rank,
            "pca_components": int(pca_info.get("n_components", -1)) if isinstance(pca_info, dict) else -1,
            "hidden_dim": hidden_dim,
            "num_random": args.num_random,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "test_example_count": len(selected),
            "baseline_exact": float(baseline_exact),
            "baseline_partial": float(baseline_partial),
            "bucket": args.bucket,
            "full_pca_train_examples": len(train_records),
            "answer_only": args.answer_only,
            "layer_sweep": args.layer_sweep,
        }

        layer_results.append(
            {
                "layer": layer_idx,
                "aggregate": aggregate,
                "records": per_example,
                "meta": meta_out,
            }
        )

        if args.answer_only:
            print(
                f"Layer {layer_idx} | {args.basis.capitalize()} ablation mean logit diff (answer tokens): "
                f"{aggregate['probe_logit_diff_mean']:.4f}"
            )
        else:
            print(
                f"Layer {layer_idx} | {args.basis.capitalize()} ablation accuracy delta (exact): "
                f"{aggregate['probe_accuracy_delta']:.4f} | partial delta: {aggregate['probe_partial_delta']:.4f}"
            )

    sample_layer = layer_indices[0] if layer_indices else args.layer
    sample_pca_info = results.get(sample_layer, {}).get("pca") if isinstance(results.get(sample_layer), dict) else None
    pca_components = int(sample_pca_info.get("n_components", -1)) if isinstance(sample_pca_info, dict) else args.pca_components
    layer_tag: int | str = "all" if args.layer_sweep else args.layer
    default_out = intervention_path(
        dataset_tag,
        model_id,
        args.proj_dim,
        pca_components,
        layer_tag,
        tag=_output_tag(args),
    )
    output_path = args.output or default_out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.layer_sweep:
        meta_out = {
            "probe_path": repo_relative(probe_fp),
            "responses_path": repo_relative(responses_fp),
            "model_id": model_id,
            "dataset_tag": dataset_tag,
            "layers": layer_indices,
            "num_layers": len(layer_indices),
            "num_random": args.num_random,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "bucket": args.bucket,
            "answer_only": args.answer_only,
            "layer_sweep": True,
        }
        aggregate_out = {
            "per_layer": [{"layer": lr["layer"], "aggregate": lr["aggregate"]} for lr in layer_results]
        }
        records_out = layer_results
    else:
        aggregate_out = layer_results[0]["aggregate"]
        records_out = layer_results[0]["records"]
        meta_out = layer_results[0]["meta"]

    np.savez_compressed(
        output_path,
        meta=np.array(meta_out, dtype=object),
        aggregate=np.array(aggregate_out, dtype=object),
        records=np.array(records_out, dtype=object),
    )

    print(f"Saved intervention results to {output_path}")


if __name__ == "__main__":
    main()
