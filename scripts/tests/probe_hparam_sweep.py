#!/usr/bin/env python3
"""Lightweight sweep of distance probe hyperparameters to check overfitting."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

# NOTE: this script lives under cutter/scripts/testing/, one level deeper than other scripts.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.tree.encoding import set_global_seed
from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.paths import embeddings_path, responses_path
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_single_model_id
from cutter.utils.tree.trees import DistanceProbeConfig
from cutter.scripts import evaluate_probe as eval_probe


def _parse_float_list(values: Sequence[str]) -> List[float]:
    return [float(v) for v in values]


def _parse_int_list(values: Sequence[str]) -> List[int]:
    return [int(v) for v in values]


def run_sweep(
    *,
    dataset_tag: str,
    model_id: str,
    layers: Iterable[int],
    lrs: Sequence[float],
    steps: Sequence[int],
    proj_dim: int,
    weight_decay: float,
    train_split: float,
    seed: int,
    normalize_tree: bool,
) -> None:
    set_global_seed(seed)

    dataset_path = path_utils.dataset_path_from_tag(dataset_tag)
    responses_fp = responses_path(dataset_tag, model_id)
    embeddings_fp = embeddings_path(dataset_tag, model_id)

    responses = eval_probe.load_responses(responses_fp)
    train_records, test_records = eval_probe.split_dataset(responses, train_split, seed)
    if not train_records:
        raise RuntimeError("No exact-match responses available for training.")

    cache = eval_probe.load_embedding_cache(embeddings_fp)
    selected_layers = list(layers)
    encodings = eval_probe.build_embeddings_from_cache(train_records, test_records, selected_layers, cache)
    max_tree_depth = max((rec.depth for rec in responses), default=1)
    norm_tree = max_tree_depth if normalize_tree else None

    layer_str = ",".join(str(l) for l in selected_layers)
    print(f"Dataset: {dataset_tag} ({dataset_path}) | Model: {model_id} | Layers: {layer_str}")
    print(f"Train split (exact only): {len(train_records)} | Test split: {len(test_records)}")
    print(f"LRs: {lrs} | Steps: {steps} | proj_dim: {proj_dim} | normalize_tree: {normalize_tree}")

    for layer_idx in selected_layers:
        data = encodings[layer_idx]
        for lr_val in lrs:
            for step_val in steps:
                cfg = DistanceProbeConfig(
                    proj_dim=proj_dim,
                    lr=lr_val,
                    weight_decay=weight_decay,
                    steps=step_val,
                    seed=seed,
                    fit_geometry="euclidean",
                )
                res = eval_probe._evaluate_layer(  # type: ignore[attr-defined]
                    data,
                    cfg,
                    norm_tree,
                    depth_alpha=1e-2,
                )
                dist_train = res.get("dist_corr_train", float("nan"))
                dist_test = res.get("dist_corr_test", float("nan"))
                depth = res.get("depth", {})
                depth_train = depth.get("train", {}).get("pearson") if isinstance(depth, dict) else None
                depth_test = depth.get("test", {}).get("pearson") if isinstance(depth, dict) else None
                depth_train_val = depth_train if depth_train is not None else float("nan")
                depth_test_val = depth_test if depth_test is not None else float("nan")
                print(
                    f"[layer {layer_idx:02d}] lr={lr_val:.3g} steps={step_val:4d} "
                    f"dist pearson train={dist_train:.3f} "
                    f"test={dist_test:.3f} | "
                    f"depth pearson train={depth_train_val:.3f} "
                    f"test={depth_test_val:.3f}"
                )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep distance probe hyperparameters on depth1-4_n500 / 7B.",
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
    parser.add_argument(
        "--layers",
        type=str,
        default="2",
        help="Comma-separated layer indices to sweep (e.g., '2,5,10').",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        default=["1e-3", "5e-3", "1e-2"],
        help="Learning rates to sweep.",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["500", "1000", "1500"],
        help="Training steps to sweep.",
    )
    parser.add_argument("--proj-dim", type=int, default=4, help="Projection dimension for the distance probe.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    parser.add_argument("--train-split", type=float, default=0.5, help="Train fraction (exact-match responses only).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--normalize-tree", dest="normalize_tree", action="store_true", help="Normalize distances/depths by max tree depth.")
    parser.add_argument("--normalize-depth", dest="normalize_tree", action="store_true", help=argparse.SUPPRESS)  # backward compatibility
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_id = resolve_single_model_id(args.reasoning_models, args.chat_models)
    layers = [int(tok) for tok in args.layers.split(",") if tok.strip()]
    run_sweep(
        dataset_tag=args.dataset,
        model_id=model_id,
        layers=layers,
        lrs=_parse_float_list(args.lrs),
        steps=_parse_int_list(args.steps),
        proj_dim=args.proj_dim,
        weight_decay=args.weight_decay,
        train_split=args.train_split,
        seed=args.seed,
        normalize_tree=bool(args.normalize_tree),
    )


if __name__ == "__main__":
    main()
