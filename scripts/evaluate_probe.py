#!/usr/bin/env python3
"""Train/evaluate distance and depth probes on model responses."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

# Ensure repository root is importable when running from scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.tree.encoding import DEVICE, set_global_seed
from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.paths import embeddings_path, embeddings_pca_path, probe_path, repo_relative, responses_path
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_single_model_id
from cutter.utils.shared.embeddings_cache import load_embedding_payload
from cutter.utils.tree.trees import DistanceProbeConfig, pairwise_tree_distances, tree_depth
from cutter.utils.tree.probing import evaluate_probes
from cutter.utils.shared.basic import split_balanced, split_exact_only


# -----------------------------------------------------------------------------
# Data containers


@dataclass
class EmbeddingRecord:
    """Back-compat container for unpickling old embedding caches."""

    example_id: int
    parsed_path: List[int]
    parsed_text: Optional[str]
    token_ids: List[int]
    token_offsets: List[Tuple[int, int]]
    layers: List[int]
    hidden_dim: int
    embeddings_by_layer: Dict[int, np.ndarray]
    prompt_tokens: int
    model_id: str
    dataset_path: str


@dataclass
class ResponseRecord:
    example_id: int
    depth: int
    source: int
    target: int
    num_samples: int
    prompt: str
    ground_truth_path: List[int]
    canonical_path: List[int]
    canonical_source: int
    canonical_target: int
    label_mapping: List[int]
    raw: str
    parsed_path: List[int]
    parsed_text: Optional[str]
    format_ok: bool
    exact_match: bool
    partial_score: float

    @classmethod
    def from_json(cls, row: Mapping[str, Any]) -> "ResponseRecord":
        return cls(
            example_id=int(row.get("example_id", -1)),
            depth=int(row["depth"]),
            source=int(row["source"]),
            target=int(row["target"]),
            num_samples=int(row.get("num_samples", row.get("sample_rate", -1))),
            prompt=str(row["prompt"]),
            ground_truth_path=[int(x) for x in row["ground_truth_path"]],
            canonical_path=[int(x) for x in row.get("canonical_path", row.get("ground_truth_path", []))],
            canonical_source=int(row.get("canonical_source", row.get("source", -1))),
            canonical_target=int(row.get("canonical_target", row.get("target", -1))),
            label_mapping=[int(x) for x in row.get("label_mapping", [])],
            raw=str(row.get("model_raw", "")),
            parsed_path=[int(x) for x in row.get("parsed_path", [])],
            parsed_text=row.get("parsed_text"),
            format_ok=bool(row.get("format_ok", False)),
            exact_match=bool(row.get("exact_match", False)),
            partial_score=float(row.get("partial_score", 0.0)),
        )


# -----------------------------------------------------------------------------
# Mapping utilities


def _invert_mapping(mapping: Sequence[int]) -> Dict[int, int]:
    """Return label->structural mapping for a provided structural->label permutation."""

    return {label: idx for idx, label in enumerate(mapping)}


# -----------------------------------------------------------------------------
# Probe helpers
# -----------------------------------------------------------------------------


def _evaluate_layer(data: Dict[str, Any], cfg: DistanceProbeConfig, normalize_tree: Optional[float], depth_alpha: float) -> Dict[str, Any]:
    """Run distance/depth probe evaluation for a single layer using shared utils."""

    encoded = {
        "layer": {
            "X": np.asarray(data["X"], dtype=np.float32),
            "D": np.asarray(data["dist"], dtype=np.float32),
            "depth": np.asarray(data["depth"], dtype=np.float32),
            "example_ids": np.asarray(data.get("example_ids", []), dtype=np.int64),
            "train_idx": np.asarray(data["train_idx"], dtype=int),
            "test_idx": np.asarray(data["test_idx"], dtype=int),
        }
    }
    exact_mask = np.asarray(data.get("example_is_exact", []), dtype=bool)
    if exact_mask.size:
        encoded["layer"]["example_is_exact"] = exact_mask
    results = evaluate_probes(encoded, cfg, device=DEVICE, normalize_tree=normalize_tree, depth_alpha=depth_alpha)
    return results["layer"]


# -----------------------------------------------------------------------------
# Core pipeline


def _resolve_layer_range(start: int, end: Optional[int], step: int, total_layers: int) -> List[int]:
    if step <= 0:
        raise ValueError("layer_step must be > 0")
    if end is None:
        end = total_layers
    if end <= start:
        raise ValueError("layer_end must be greater than layer_start")
    return list(range(start, end, step))


def load_responses(path: Path) -> List[ResponseRecord]:
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




def load_embedding_cache(path: Path) -> Dict[int, Any]:
    cache, _ = load_embedding_payload(path)
    return cache


def build_layer_from_cache(
    train_records: Sequence[ResponseRecord],
    test_records: Sequence[ResponseRecord],
    layer_idx: int,
    cache: Mapping[int, Any],
) -> Optional[Dict[str, Any]]:
    holder = {
        "X_list": [],
        "depth_list": [],
        "node_ids": [],
        "example_ids": [],
        "example_accuracy": [],
        "example_depths": [],
        "train_idx": [],
        "test_idx": [],
        "example_is_exact": [],
    }
    for split_name, records in (("train", train_records), ("test", test_records)):
        for rec in records:
            if not rec.parsed_path:
                continue
            emb_entry = cache.get(rec.example_id)
            if emb_entry is None:
                continue
            inv_map = _invert_mapping(rec.label_mapping) if rec.label_mapping else None

            def _to_canonical(node_id: int) -> int:
                if inv_map is None:
                    return node_id
                return inv_map.get(node_id, node_id)

            canonical_parsed = [_to_canonical(nid) for nid in rec.parsed_path]
            depth_vals_full = [tree_depth(nid) for nid in canonical_parsed]
            layers_dict = emb_entry["embeddings_by_layer"] if isinstance(emb_entry, dict) else emb_entry.embeddings_by_layer
            emb = layers_dict.get(layer_idx)
            if emb is None or emb.size == 0:
                continue
            count = emb.shape[0]
            effective_count = min(count, len(rec.parsed_path))
            if effective_count == 0:
                continue
            path_nodes = canonical_parsed[:effective_count]
            depth_vals = depth_vals_full[:effective_count]
            if effective_count != count:
                emb = np.asarray(emb, dtype=np.float32)[:effective_count]
                count = effective_count
            holder["X_list"].append(np.asarray(emb, dtype=np.float32))
            holder["depth_list"].extend(depth_vals)
            holder["node_ids"].extend(path_nodes)
            holder["example_ids"].extend([rec.example_id] * count)
            holder["example_accuracy"].extend([float(rec.partial_score)] * count)
            holder["example_depths"].extend([rec.depth] * count)
            holder["example_is_exact"].extend([bool(rec.exact_match)] * count)
            if split_name == "train":
                start = len(holder["train_idx"]) + len(holder["test_idx"])
                holder["train_idx"].extend(range(start, start + count))
            else:
                start = len(holder["train_idx"]) + len(holder["test_idx"])
                holder["test_idx"].extend(range(start, start + count))

    if not holder["X_list"]:
        return None
    X = np.vstack(holder["X_list"]).astype(np.float32)
    return {
        "X": X,
        "depth": np.array(holder["depth_list"], dtype=np.float32),
        "node_ids": np.array(holder["node_ids"], dtype=np.int64),
        "example_ids": np.array(holder["example_ids"], dtype=np.int64),
        "example_accuracy": np.array(holder["example_accuracy"], dtype=np.float32),
        "example_depths": np.array(holder["example_depths"], dtype=np.int64),
        "example_is_exact": np.array(holder["example_is_exact"], dtype=bool),
        "train_idx": np.array(holder["train_idx"], dtype=int),
        "test_idx": np.array(holder["test_idx"], dtype=int),
    }


def _save_payload(output_path: Path, meta: Dict[str, Any], encodings: Dict[int, Any], results: Dict[int, Any]) -> None:
    payload_meta = np.array(meta, dtype=object)
    payload_enc = np.array(encodings, dtype=object)
    payload_results = np.array(results, dtype=object)
    np.savez_compressed(output_path, meta=payload_meta, encodings=payload_enc, results=payload_results)


def _compact_encodings(encodings: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Drop large tensors (X/dist/batches) before saving; keep split/index metadata."""

    compact: Dict[int, Dict[str, Any]] = {}
    for layer, data in encodings.items():
        compact[layer] = {
            "example_ids": np.asarray(data.get("example_ids", []), dtype=np.int64),
            "example_depths": np.asarray(data.get("example_depths", []), dtype=np.int64),
            "example_accuracy": np.asarray(data.get("example_accuracy", []), dtype=np.float32),
            "example_is_exact": np.asarray(data.get("example_is_exact", []), dtype=bool),
            "node_ids": np.asarray(data.get("node_ids", []), dtype=np.int64),
            "depth": np.asarray(data.get("depth", []), dtype=np.float32),
            "train_idx": np.asarray(data.get("train_idx", []), dtype=np.int64),
            "test_idx": np.asarray(data.get("test_idx", []), dtype=np.int64),
        }
    return compact


def _compact_encoding(data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "example_ids": np.asarray(data.get("example_ids", []), dtype=np.int64),
        "example_depths": np.asarray(data.get("example_depths", []), dtype=np.int64),
        "example_accuracy": np.asarray(data.get("example_accuracy", []), dtype=np.float32),
        "example_is_exact": np.asarray(data.get("example_is_exact", []), dtype=bool),
        "node_ids": np.asarray(data.get("node_ids", []), dtype=np.int64),
        "depth": np.asarray(data.get("depth", []), dtype=np.float32),
        "train_idx": np.asarray(data.get("train_idx", []), dtype=np.int64),
        "test_idx": np.asarray(data.get("test_idx", []), dtype=np.int64),
    }


def _apply_pca_to_encodings(
    encodings: Dict[int, Dict[str, Any]],
    n_components: int,
    seed: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    if n_components < 0:
        return encodings, {}
    if n_components == 0:
        raise ValueError("pca_components must be -1 (full) or a positive integer.")
    projected: Dict[int, Dict[str, Any]] = {}
    pca_info: Dict[int, Dict[str, Any]] = {}
    for layer, data in encodings.items():
        X = np.asarray(data["X"], dtype=np.float32)
        train_idx = np.asarray(data["train_idx"], dtype=int)
        if train_idx.size == 0:
            raise RuntimeError(f"No training examples found for layer {layer}.")
        max_components = min(X.shape[0], X.shape[1])
        effective_components = min(n_components, max_components)
        if effective_components >= X.shape[1]:
            projected[layer] = data
            continue
        pca = PCA(n_components=effective_components, svd_solver="auto", random_state=seed)
        pca.fit(X[train_idx])
        projected_X = pca.transform(X).astype(np.float32)
        updated = dict(data)
        updated["X"] = projected_X
        projected[layer] = updated
        pca_info[layer] = {
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "n_components": int(effective_components),
            "n_features": int(X.shape[1]),
        }
    return projected, pca_info


def _fmt(val: Optional[float]) -> str:
    if val is None or np.isnan(val):
        return "   nan"
    return f"{val:6.3f}"


def _print_metrics_table(results: Dict[int, Any]) -> None:
    if not results:
        print("No results to display.")
        return
    # Columns: layer | dist MSE (train / test / test ex / test in) | depth MSE (train / test / test ex / test in)
    headers = [
        "Layer",
        "dist train",
        "dist test",
        "dist test ex",
        "dist test in",
        "depth train",
        "depth test",
        "depth test ex",
        "depth test in",
    ]
    widths = [7, 12, 12, 13, 13, 13, 13, 14, 14]

    def _row(fields, col_widths):
        return " ".join(f"{field:>{w}}" for field, w in zip(fields, col_widths))

    print("\nProbe metrics (MSE):")
    print(_row(headers, widths))

    for layer in sorted(results.keys()):
        dist_train = _fmt(results[layer].get("dist_mse_train"))
        dist_test = _fmt(results[layer].get("dist_mse_test"))
        dist_test_ex = _fmt(results[layer].get("dist_mse_test_exact"))
        dist_test_in = _fmt(results[layer].get("dist_mse_test_inexact"))
        depth = results[layer].get("depth", {}) or {}
        depth_train = _fmt(depth.get("train", {}).get("mse") if isinstance(depth, dict) else None)
        depth_test = _fmt(depth.get("test", {}).get("mse") if isinstance(depth, dict) else None)
        depth_test_ex = _fmt(depth.get("test_exact", {}).get("mse") if isinstance(depth, dict) else None)
        depth_test_in = _fmt(depth.get("test_inexact", {}).get("mse") if isinstance(depth, dict) else None)
        print(
            _row(
                [
                    str(layer),
                    dist_train,
                    dist_test,
                    dist_test_ex,
                    dist_test_in,
                    depth_train,
                    depth_test,
                    depth_test_ex,
                    depth_test_in,
                ],
                widths,
            )
        )


# -----------------------------------------------------------------------------
# CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical probes on model responses.",
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
        help="Reasoning (R1-distilled) parameter counts (e.g., 7B). Pass 'none' to skip reasoning models.",
    )
    parser.add_argument(
        "--chat-models",
        nargs="+",
        default=DEFAULT_CHAT_SIZES,
        help="Non-reasoning chat parameter counts (e.g., 7B). Pass 'none' to skip chat models.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-split", type=float, default=0.5, help="Train fraction for response-level split.")
    parser.add_argument(
        "--split-type",
        type=str,
        default="random",
        choices=("random", "exact-only"),
        help="Split strategy for train/test responses.",
    )
    parser.add_argument("--num-runs", type=int, default=1, help="Number of evaluation runs to aggregate.")
    parser.add_argument("--layer-start", type=int, default=0, help="Starting layer index (inclusive).")
    parser.add_argument("--layer-end", type=int, default=None, help="Ending layer index (exclusive). Defaults to all layers.")
    parser.add_argument("--layer-step", type=int, default=1, help="Layer stride.")
    parser.add_argument("--proj-dim", type=int, default=4, help="Projection dimension for distance probe.")
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Number of PCA components to project embeddings before probe fitting (-1 for full embeddings).",
    )
    parser.add_argument("--steps", type=int, default=1500, help="Training steps for distance probe.")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate for distance probe.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for distance probe.")
    parser.add_argument("--fit-geometry", type=str, default="euclidean", choices=("euclidean", "hyperbolic"), help="Probe geometry.")
    parser.add_argument(
        "--pair-weighting",
        type=str,
        default="inverse_freq",
        choices=("none", "inverse_freq"),
        help="Reweight distance pairs during training.",
    )
    parser.add_argument("--depth-alpha", type=float, default=1e-2, help="Regularization strength for the linear depth probe on activations.")
    parser.add_argument("--normalize-tree", dest="normalize_tree", action="store_true", help="Normalize tree distances by max tree depth when training/evaluating probes.")
    parser.add_argument("--normalize-depth", dest="normalize_tree", action="store_true", help=argparse.SUPPRESS)  # backward compatibility
    parser.add_argument("--output", type=Path, default=None, help="Optional override for probe output path.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.num_runs < 1:
        raise ValueError("num_runs must be >= 1")
    set_global_seed(args.seed)

    # Resolve cached responses/embeddings using the shared dataset/model folder layout.
    model_id = resolve_single_model_id(args.reasoning_models, args.chat_models)
    dataset_tag = args.dataset
    responses_fp = responses_path(dataset_tag, model_id)
    embeddings_fp = embeddings_path(dataset_tag, model_id)

    responses = load_responses(responses_fp)
    print(f"Loaded {len(responses)} responses from {responses_fp}")
    depths = [rec.depth for rec in responses]
    sample_counts = {rec.num_samples for rec in responses if rec.num_samples >= 0}
    depth_tag = f"depth{min(depths)}-{max(depths)}" if depths else "depthNA"
    sample_tag = f"n{sorted(sample_counts)[0]}" if sample_counts else "nall"

    # Determine embeddings sidecar path.
    emb_path = embeddings_fp
    cache: Dict[int, Any] = {}
    emb_meta: Dict[str, Any] = {}
    if args.pca_components >= 0:
        pca_path = embeddings_pca_path(dataset_tag, model_id, args.pca_components)
        if pca_path.exists():
            cache, emb_meta = load_embedding_payload(pca_path)
            cache_split_type = emb_meta.get("pca_split_type")
            if cache_split_type != args.split_type:
                print(
                    f"Warning: PCA cache split_type={cache_split_type} does not match requested "
                    f"split_type={args.split_type}; recomputing PCA on current split."
                )
                cache = {}
                emb_meta = {}
            else:
                emb_path = pca_path
        if not cache:
            emb_path = embeddings_fp
            cache, emb_meta = load_embedding_payload(embeddings_fp)
    else:
        cache, emb_meta = load_embedding_payload(embeddings_fp)
    if not cache:
        raise RuntimeError(f"No cached embeddings found in {emb_path}")
    # Infer available layers from cache (prefer keys of embeddings_by_layer)
    first_entry = next(iter(cache.values()))
    if isinstance(first_entry, dict):
        layer_keys = list(getattr(first_entry.get("embeddings_by_layer"), "keys", lambda: [])())
        if not layer_keys:
            layer_keys = list(first_entry.get("layers", []))
    else:
        layer_keys = list(getattr(getattr(first_entry, "embeddings_by_layer", {}), "keys", lambda: [])())
        if not layer_keys:
            layer_keys = list(getattr(first_entry, "layers", []))
    if not layer_keys:
        raise RuntimeError("No layer indices found in embedding cache.")
    max_layer = max(layer_keys)
    total_layers = max_layer + 1
    layer_indices = _resolve_layer_range(args.layer_start, args.layer_end, args.layer_step, total_layers)
    print(f"Using cached embeddings from {emb_path}; layers: {layer_indices}")

    max_tree_depth = max((rec.depth for rec in responses), default=1)
    normalize_tree = max_tree_depth if args.normalize_tree else None

    def _split_records(seed: int) -> Tuple[List[ResponseRecord], List[ResponseRecord]]:
        if args.split_type == "exact-only":
            train_records, test_records = split_exact_only(responses, args.train_split, seed, exact_attr="exact_match")
            print(f"Train split (exact only): {len(train_records)} | Test split: {len(test_records)}")
            if not train_records:
                raise RuntimeError("No exact-match responses available for training.")
            return train_records, test_records
        train_records, test_records = split_balanced(responses, args.train_split, seed, exact_attr="exact_match")
        print(f"Train split (balanced random): {len(train_records)} | Test split: {len(test_records)}")
        return train_records, test_records

    def _run_once(run_seed: int) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Any], Dict[int, Any], int, int]:
        set_global_seed(run_seed)
        train_records, test_records = _split_records(run_seed)
        cache_pca_components = int(emb_meta.get("pca_components", -1))
        cache_pca_info = emb_meta.get("pca") if isinstance(emb_meta, dict) else None
        preprojected = args.pca_components >= 0 and cache_pca_components >= 0 and isinstance(cache_pca_info, dict)
        if preprojected and cache_pca_components != args.pca_components:
            print(
                f"Warning: embeddings cache uses PCA={cache_pca_components} but requested {args.pca_components}; "
                "skipping additional projection."
            )
        if preprojected:
            print(f"Using pre-projected embeddings (PCA={cache_pca_components}) from {emb_path}")
        elif args.pca_components >= 0:
            print(f"Projecting embeddings to first {args.pca_components} PCA components (fit on train split).")

        cfg = DistanceProbeConfig(
            proj_dim=args.proj_dim,
            lr=args.lr,
            weight_decay=args.weight_decay,
            steps=args.steps,
            seed=run_seed,
            fit_geometry=args.fit_geometry,
            pair_weighting=args.pair_weighting,
        )

        results: Dict[int, Any] = {}
        encodings: Dict[int, Dict[str, Any]] = {}
        dist_cache: Optional[np.ndarray] = None
        dist_cache_nodes: Optional[np.ndarray] = None
        pca_info: Dict[int, Dict[str, Any]] = {}

        for layer in layer_indices:
            data = build_layer_from_cache(train_records, test_records, layer, cache)
            if data is None:
                continue
            node_ids = np.asarray(data.get("node_ids", []), dtype=np.int64)
            if dist_cache is not None and dist_cache_nodes is not None and np.array_equal(node_ids, dist_cache_nodes):
                data["dist"] = dist_cache
            else:
                data["dist"] = pairwise_tree_distances(node_ids)
                dist_cache = data["dist"]
                dist_cache_nodes = node_ids

            if not preprojected and args.pca_components >= 0:
                projected, layer_pca = _apply_pca_to_encodings({layer: data}, args.pca_components, run_seed)
                data = projected[layer]
                if layer in layer_pca:
                    pca_info[layer] = layer_pca[layer]
            elif preprojected:
                if isinstance(cache_pca_info, dict) and layer in cache_pca_info:
                    pca_info[layer] = cache_pca_info[layer]

            if args.pca_components >= 0:
                data["depth_X"] = data["X"]

            print(f"Training probes for layer {layer} ...")
            res = _evaluate_layer(data, cfg, normalize_tree, args.depth_alpha)
            if layer in pca_info:
                components = np.asarray(pca_info[layer]["components"], dtype=np.float32)
                projection = np.asarray(res.get("projection"), dtype=np.float32)
                res["projection_full"] = components.T @ projection
                res["pca"] = pca_info[layer]
            results[layer] = res
            encodings[layer] = data

        if not results:
            raise RuntimeError("No layer data collected from cached embeddings.")
        return encodings, results, pca_info, len(train_records), len(test_records)

    results: Dict[int, Any] = {}
    encodings_to_save: Dict[int, Dict[str, Any]] = {}
    run_seeds = [args.seed + run_idx for run_idx in range(args.num_runs)]
    train_counts: List[int] = []
    test_counts: List[int] = []

    if args.num_runs == 1:
        run_encodings, results, _, train_count, test_count = _run_once(args.seed)
        train_counts.append(train_count)
        test_counts.append(test_count)
        encodings_to_save = {layer: _compact_encoding(data) for layer, data in run_encodings.items()}
    else:
        metrics_acc: Dict[int, Dict[str, List[float]]] = {}
        depth_acc: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
        best_results: Dict[int, Any] = {}
        best_scores: Dict[int, float] = {}
        best_seeds: Dict[int, int] = {}
        best_encodings: Dict[int, Dict[str, Any]] = {}

        for run_seed in run_seeds:
            print(f"\n=== Run seed {run_seed} ===")
            encodings, run_results, _, train_count, test_count = _run_once(run_seed)
            train_counts.append(train_count)
            test_counts.append(test_count)
            for layer, res in run_results.items():
                # Accumulate distance metrics
                for key, val in res.items():
                    if not key.startswith("dist_"):
                        continue
                    if not isinstance(val, (float, int, np.floating)):
                        continue
                    if val is None or np.isnan(val):
                        continue
                    metrics_acc.setdefault(layer, {}).setdefault(key, []).append(float(val))

                # Accumulate depth metrics (mse/pearson only)
                depth_block = res.get("depth", {})
                if isinstance(depth_block, dict):
                    for split_name, split_data in depth_block.items():
                        if not isinstance(split_data, dict):
                            continue
                        for metric in ("mse", "pearson"):
                            val = split_data.get(metric)
                            if val is None or (isinstance(val, float) and np.isnan(val)):
                                continue
                            depth_acc.setdefault(layer, {}).setdefault(split_name, {}).setdefault(metric, []).append(float(val))

                # Track best-performing probe per layer
                curr = res.get("dist_mse_test")
                best = best_scores.get(layer)
                better = False
                if curr is not None and not (isinstance(curr, float) and np.isnan(curr)):
                    if best is None or (isinstance(best, float) and np.isnan(best)) or curr < best:
                        better = True
                elif best is None:
                    better = True
                if better:
                    best_scores[layer] = curr
                    best_results[layer] = res
                    best_encodings[layer] = _compact_encoding(encodings[layer])
                    best_seeds[layer] = run_seed

        for layer in sorted(best_results.keys()):
            res = dict(best_results[layer])
            stats: Dict[str, Any] = {"dist": {}, "depth": {}}
            for key, vals in metrics_acc.get(layer, {}).items():
                if not vals:
                    continue
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                stats["dist"][key] = {"mean": mean_val, "std": std_val}
            for split_name, metric_dict in depth_acc.get(layer, {}).items():
                stats["depth"][split_name] = {}
                for metric, vals in metric_dict.items():
                    if not vals:
                        continue
                    mean_val = float(np.mean(vals))
                    std_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
                    stats["depth"][split_name][metric] = {"mean": mean_val, "std": std_val}
            res["stats"] = stats
            res["best_run_seed"] = best_seeds.get(layer)
            results[layer] = res
        encodings_to_save = best_encodings

    meta = {
        "model": model_id,
        "train_split": args.train_split,
        "seed": args.seed,
        "layers": layer_indices,
        "proj_dim": args.proj_dim,
        "pca_components": args.pca_components,
        "fit_geometry": args.fit_geometry,
        "steps": args.steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "device": DEVICE,
        "depth_alpha": args.depth_alpha,
        "normalize_tree": bool(args.normalize_tree),
        "pair_weighting": args.pair_weighting,
        "split_type": args.split_type,
        "responses_path": repo_relative(responses_fp),
        "embeddings_path": repo_relative(emb_path),
        "train_count": int(train_counts[0]) if train_counts else 0,
        "test_count": int(test_counts[0]) if test_counts else 0,
        "train_counts": train_counts,
        "test_counts": test_counts,
        "num_runs": args.num_runs,
        "run_seeds": run_seeds,
        "best_run_metric": "dist_mse_test",
        "depth_tag": depth_tag,
        "sample_tag": sample_tag,
        "dataset_tag": dataset_tag,
    }

    output_path = args.output or probe_path(dataset_tag, model_id, args.proj_dim, args.normalize_tree, args.pca_components)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_payload(output_path, meta, encodings_to_save, results)
    _print_metrics_table(results)
    print(f"Saved probe results to {output_path}")


if __name__ == "__main__":
    main()
