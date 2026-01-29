#!/usr/bin/env python3
"""Lightweight wrapper to run the traversal pipeline for multiple models.

Steps:
1) Generate the traversal dataset (shared across models).
2) For each model (reasoning + non-reasoning):
   a) Generate model responses + embeddings.
   b) Train/evaluate hierarchical probes.
   c) Optionally run interventions.

All progress is printed with timestamps for easy monitoring on a VM.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import cutter.utils.shared.paths as path_utils
from cutter.utils.shared.paths import parse_gsm8k_dataset_tag, parse_tree_dataset_tag, resolve_dataset_tag
from cutter.utils.shared.basic import parse_range_arg
from cutter.utils.shared.embeddings_cache import ensure_pca_cache
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_model_pairs

REPO_ROOT = PROJECT_ROOT


def timestamp(msg: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def run_command(cmd: List[str], desc: str) -> None:
    timestamp(desc)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    timestamp(f"Completed: {desc}\n")


def _force_flags(force_args: List[str]) -> set[str]:
    normalized = {tok.lower() for tok in force_args}
    if "all" in normalized:
        return {"dataset", "responses", "probes", "interventions"}
    return normalized


def _already_exists(path: Path, force_stage: str, force_flags: set[str]) -> bool:
    if path.exists() and force_stage not in force_flags:
        timestamp(f"Skip (exists): {path}")
        return True
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traversal dataset generation, model responses, probes, and optional interventions for multiple models.",
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--depth-range",
        type=int,
        nargs="+",
        default=[1, 2],
        help="One or two integers: depth or [min max] depth range (inclusive).",
    )
    parser.add_argument(
        "--steps-range",
        type=int,
        nargs="+",
        default=[1, 2],
        help="One or two integers: steps or [min max] steps range (inclusive).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of traversal examples to sample (0 = sample uniformly across steps).",
    )
    parser.add_argument(
        "--gsm8k-split",
        type=str,
        default="all",
        help="GSM8K split tag for math setting (train, test, all).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2000, help="Max new tokens for generation.")
    parser.add_argument("--generation-limit", type=int, default=0, help="Optional cap on examples passed to generate_model_responses (0 = all).")
    parser.add_argument("--train-split", type=float, default=0.5, help="Train fraction for probe evaluation.")
    parser.add_argument(
        "--proj-dims",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5],
        help="Projection dimensions for distance probes (one run per dim).",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        default=10,
        help="Number of PCA components to use when training probes (-1 for full embeddings).",
    )
    parser.add_argument("--probe-steps", type=int, default=1500, help="Training steps for distance probe.")
    parser.add_argument("--probe-layer-step", type=int, default=1, help="Layer stride for probe evaluation.")
    parser.add_argument(
        "--split-type",
        type=str,
        default="random",
        choices=("random", "exact-only"),
        help="Split strategy for train/test responses.",
    )
    parser.add_argument("--num-runs", type=int, default=1, help="Number of evaluation runs to aggregate.")
    parser.add_argument("--normalize-tree", dest="normalize_tree", action="store_true", help="Normalize tree distances when training/evaluating probes.")
    parser.add_argument("--normalize-depth", dest="normalize_tree", action="store_true", help=argparse.SUPPRESS)  # backward compatibility
    parser.add_argument(
        "--force",
        nargs="+",
        default=[],
        help="Force recompute for specific stages: choose from dataset, responses, probes, interventions, all.",
    )
    parser.add_argument("--run-interventions", action="store_true", help="Also run intervention script after probes.")
    parser.add_argument("--intervention-layer", type=int, default=10, help="Transformer layer index for intervention.")
    parser.add_argument("--intervention-num-random", type=int, default=1, help="Random subspace ablations per example.")
    parser.add_argument(
        "--intervention-basis",
        type=str,
        default="combined",
        choices=("distance", "depth", "combined"),
        help="Which probe basis to ablate in interventions.",
    )
    parser.add_argument("--probe-path", type=str, default=None, help="Explicit probe path for math interventions.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    force_set = _force_flags(args.force)
    model_pairs = resolve_model_pairs(args.reasoning_models, args.chat_models)
    if not model_pairs:
        raise ValueError("No models selected. Provide --reasoning-models or --chat-models.")
    proj_dims: List[int] = []
    for dim in args.proj_dims:
        if dim <= 0:
            raise ValueError("proj-dims must be positive integers")
        if dim not in proj_dims:
            proj_dims.append(dim)
    all_models: List[str] = [pair[2] for pair in model_pairs]
    if args.setting == "math":
        dataset_tag = resolve_dataset_tag(
            "math",
            args.dataset,
            num_samples=args.num_samples,
            seed=args.seed,
            gsm8k_split=args.gsm8k_split,
        )
        dataset_path = path_utils.gsm8k_dataset_path_from_tag(dataset_tag)
        if not _already_exists(dataset_path, "dataset", force_set):
            split, num_samples, dataset_seed = parse_gsm8k_dataset_tag(dataset_tag)
            cmd = [
                sys.executable,
                "cutter/scripts/create_dataset.py",
                "--setting",
                "math",
                f"--num-samples={num_samples}",
                f"--seed={dataset_seed}",
                f"--gsm8k-split={split}",
                "--dataset",
                dataset_tag,
            ]
            run_command(cmd, "Generate GSM8K dataset")

        timestamp(f"Processing {len(all_models)} models with GSM8K dataset {dataset_path} | proj dims {proj_dims}\n")

        for family, size_token, model_id in model_pairs:
            reasoning_arg = size_token if family == "reasoning" else "none"
            chat_arg = size_token if family == "chat" else "none"
            model_tag = path_utils.model_tag(model_id)
            timestamp(f"Model: {model_id} ({model_tag})")
            responses_fp = path_utils.responses_path(dataset_tag, model_id)

            if not _already_exists(responses_fp, "responses", force_set):
                gen_cmd = [
                    sys.executable,
                    "cutter/scripts/generate_responses.py",
                    "--setting",
                    "math",
                    "--dataset",
                    dataset_tag,
                    "--reasoning-models",
                    reasoning_arg,
                    "--chat-models",
                    chat_arg,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--seed",
                    str(args.seed),
                ]
                if args.generation_limit > 0:
                    gen_cmd.extend(["--limit", str(args.generation_limit)])
                run_command(gen_cmd, f"Generate GSM8K responses for {model_tag}")

            if args.run_interventions:
                if not args.probe_path:
                    raise ValueError("Math setting requires --probe-path to run interventions.")
                for proj_dim in proj_dims:
                    intervention_cmd = [
                        sys.executable,
                        "cutter/scripts/intervene.py",
                        "--setting",
                        "math",
                        "--dataset",
                        dataset_tag,
                        "--probe",
                        str(args.probe_path),
                        "--reasoning-models",
                        reasoning_arg,
                        "--chat-models",
                        chat_arg,
                        "--proj-dim",
                        str(proj_dim),
                        "--pca-components",
                        str(args.pca_components),
                        "--basis",
                        str(args.intervention_basis),
                        "--layer",
                        str(args.intervention_layer),
                        "--num-random",
                        str(args.intervention_num_random),
                        "--seed",
                        str(args.seed),
                        "--train-split",
                        str(args.train_split),
                    ]
                    intervention_fp = path_utils.intervention_path(dataset_tag, model_id, proj_dim, args.pca_components, args.intervention_layer)
                    if _already_exists(intervention_fp, "interventions", force_set):
                        timestamp(f"Skip interventions (exists): {intervention_fp}")
                    else:
                        run_command(intervention_cmd, f"Run GSM8K interventions for {model_tag} (proj_dim={proj_dim})")

        timestamp("Pipeline complete.")
        return

    if args.dataset:
        dataset_tag = resolve_dataset_tag(
            "tree",
            args.dataset,
            num_samples=args.num_samples,
            seed=args.seed,
            gsm8k_split=args.gsm8k_split,
        )
        min_depth, max_depth, num_samples, steps_range = parse_tree_dataset_tag(dataset_tag)
        min_steps, max_steps = steps_range
    else:
        min_depth, max_depth = parse_range_arg(args.depth_range, "depth-range", min_value=0)
        min_steps, max_steps = parse_range_arg(args.steps_range, "steps-range", min_value=1)
        steps_tag = (min_steps, max_steps) if min_steps != max_steps else min_steps
        dataset_tag = path_utils.dataset_tag(min_depth, max_depth, args.num_samples, steps_tag)
        num_samples = args.num_samples
    dataset_path = path_utils.dataset_path_from_tag(dataset_tag)

    if not _already_exists(dataset_path, "dataset", force_set):
        cmd = [
            sys.executable,
            "cutter/scripts/create_dataset.py",
            "--setting",
            "tree",
            "--depth-range",
            str(min_depth),
            str(max_depth),
            f"--num-samples={num_samples}",
            f"--seed={args.seed}",
            "--steps-range",
            str(min_steps),
            str(max_steps),
            "--dataset",
            dataset_tag,
        ]
        run_command(cmd, "Generate traversal dataset")

    timestamp(f"Processing {len(all_models)} models with dataset {dataset_path} | proj dims {proj_dims}\n")

    for family, size_token, model_id in model_pairs:
        reasoning_arg = size_token if family == "reasoning" else "none"
        chat_arg = size_token if family == "chat" else "none"
        model_tag = path_utils.model_tag(model_id)
        timestamp(f"Model: {model_id} ({model_tag})")
        # Derived locations keep outputs grouped by dataset/model pair.
        responses_fp = path_utils.responses_path(dataset_tag, model_id)
        if args.pca_components >= 0:
            embeddings_fp = path_utils.embeddings_pca_path(dataset_tag, model_id, args.pca_components)
        else:
            embeddings_fp = path_utils.embeddings_path(dataset_tag, model_id)

        responses_ready = _already_exists(responses_fp, "responses", force_set)
        embeddings_ready = _already_exists(embeddings_fp, "responses", force_set)
        if responses_ready and args.pca_components >= 0 and not embeddings_ready:
            ensured = ensure_pca_cache(
                dataset_tag,
                model_id,
                args.pca_components,
                args.train_split,
                args.seed,
                args.split_type,
            )
            if ensured is not None and ensured.exists():
                embeddings_ready = _already_exists(embeddings_fp, "responses", force_set)

        if not (responses_ready and embeddings_ready):
            gen_cmd = [
                sys.executable,
                "cutter/scripts/generate_responses.py",
                "--setting",
                "tree",
                "--dataset",
                dataset_tag,
                "--reasoning-models",
                reasoning_arg,
                "--chat-models",
                chat_arg,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--seed",
                str(args.seed),
                "--pca-components",
                str(args.pca_components),
                "--train-split",
                str(args.train_split),
                "--split-type",
                str(args.split_type),
            ]
            if args.generation_limit > 0:
                gen_cmd.extend(["--limit", str(args.generation_limit)])
            run_command(gen_cmd, f"Generate responses for {model_tag}")

        for proj_dim in proj_dims:
            probe_fp = path_utils.probe_path(dataset_tag, model_id, proj_dim, args.normalize_tree, args.pca_components)
            if _already_exists(probe_fp, "probes", force_set):
                timestamp(f"Skip probes (exists): {probe_fp}")
            else:
                probe_cmd = [
                    sys.executable,
                    "cutter/scripts/evaluate_probe.py",
                    "--dataset",
                    dataset_tag,
                    "--reasoning-models",
                    reasoning_arg,
                    "--chat-models",
                    chat_arg,
                    "--train-split",
                    str(args.train_split),
                    "--proj-dim",
                    str(proj_dim),
                    "--pca-components",
                    str(args.pca_components),
                    "--steps",
                    str(args.probe_steps),
                    "--layer-step",
                    str(args.probe_layer_step),
                    "--split-type",
                    str(args.split_type),
                    "--num-runs",
                    str(args.num_runs),
                ]
                if args.normalize_tree:
                    probe_cmd.append("--normalize-tree")
                run_command(probe_cmd, f"Train/evaluate probes for {model_tag} (proj_dim={proj_dim})")

            if args.run_interventions:
                intervention_cmd = [
                    sys.executable,
                    "cutter/scripts/intervene.py",
                    "--setting",
                    "tree",
                    "--dataset",
                    dataset_tag,
                    "--probe",
                    str(probe_fp),
                    "--reasoning-models",
                    reasoning_arg,
                    "--chat-models",
                    chat_arg,
                    "--proj-dim",
                    str(proj_dim),
                    "--pca-components",
                    str(args.pca_components),
                    "--basis",
                    str(args.intervention_basis),
                ]
                if args.normalize_tree:
                    intervention_cmd.append("--normalize-tree")
                intervention_cmd.extend(
                    [
                        "--layer",
                        str(args.intervention_layer),
                        "--num-random",
                        str(args.intervention_num_random),
                    ]
                )
                intervention_fp = path_utils.intervention_path(dataset_tag, model_id, proj_dim, args.pca_components, args.intervention_layer)
                if _already_exists(intervention_fp, "interventions", force_set):
                    timestamp(f"Skip interventions (exists): {intervention_fp}")
                else:
                    run_command(intervention_cmd, f"Run interventions for {model_tag} (proj_dim={proj_dim})")

    timestamp("Pipeline complete.")


if __name__ == "__main__":
    main()
