"""Lightweight unit tests for probe utilities.

These are quick synthetic checks to ensure:
- Distance probe training improves over a random baseline.
- Depth probe regression can recover simple depth features.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repository root importable when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.tree.probing import evaluate_probes, fit_depth_probe
from cutter.utils.tree.trees import DistanceProbeConfig, pairwise_tree_distances, tree_depth


def test_distance_probe_beats_random_baseline() -> None:
    rng = np.random.default_rng(0)
    node_ids = np.arange(6, dtype=int)
    X = rng.normal(size=(len(node_ids), 8)).astype(np.float32)
    dist = pairwise_tree_distances(node_ids)
    depth = np.array([tree_depth(nid) for nid in node_ids], dtype=np.float32)

    train_idx = np.array([0, 1, 2, 3], dtype=int)
    test_idx = np.array([4, 5], dtype=int)

    encoded = {
        "layer": {
            "X": X,
            "D": dist,
            "depth": depth,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "example_ids": np.zeros_like(depth, dtype=int),
            "example_is_exact": np.ones_like(depth, dtype=bool),
        }
    }

    cfg = DistanceProbeConfig(
        proj_dim=4,
        lr=5e-2,
        weight_decay=1e-4,
        steps=400,
        seed=0,
        fit_geometry="euclidean",
        pair_weighting="none",
    )

    results = evaluate_probes(encoded, cfg, normalize_tree=None, depth_alpha=1e-2)["layer"]
    assert results["dist_mse_train"] < results["dist_mse_rand_train"], "trained probe should beat random baseline (train MSE)"
    assert results["dist_corr_train"] is not None and results["dist_corr_train"] > 0.5, "trained probe should capture tree structure"


def test_depth_probe_recovers_depths() -> None:
    depth_vals = np.array([0, 1, 1, 2, 2, 3], dtype=np.float32)
    feats = np.stack([depth_vals, depth_vals ** 2], axis=1).astype(np.float32)
    rng = np.random.default_rng(1)
    feats += rng.normal(scale=0.01, size=feats.shape).astype(np.float32)

    train_idx = np.arange(4, dtype=int)
    test_idx = np.arange(4, len(depth_vals), dtype=int)
    exact_mask = np.ones_like(depth_vals, dtype=bool)

    metrics, _ = fit_depth_probe(
        features=feats,
        depths=depth_vals,
        train_idx=train_idx,
        test_idx=test_idx,
        exact_mask=exact_mask,
        alpha=1e-3,
        normalize_tree=None,
    )

    assert metrics["train"]["mse"] is not None and metrics["train"]["mse"] < 1e-3
    assert metrics["test"]["mse"] is not None and metrics["test"]["mse"] < 5e-2


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__]))
