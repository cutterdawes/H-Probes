#!/usr/bin/env python3
"""Generate paper figures from saved probe/intervention artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams, rcParamsDefault
from matplotlib.font_manager import font_scalings
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib import gridspec, patheffects
from matplotlib.patches import FancyBboxPatch, Rectangle
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from cutter.utils.shared import paths as path_utils
from cutter.utils.shared.models import DEFAULT_CHAT_SIZES, DEFAULT_REASONING_SIZES, resolve_model_pairs
from cutter.utils.shared.embeddings_cache import load_embedding_payload
from cutter.utils.tree.probing import pairwise_distance, transform_probe_space, evaluate_probes
from cutter.utils.tree.trees import DistanceProbeConfig, pairwise_tree_distances, tree_distance
from cutter.utils.tree.encoding import DEVICE, set_global_seed
from cutter.scripts.evaluate_probe import load_responses, build_layer_from_cache

sns.set_theme(style="whitegrid")


REASONING_LAYER_MAP = {
    "1.5B": 21,
    "7B": 21,
    "14B": 31,
}
CHAT_LAYER_MAP = {
    "1.8B": 17,
    "7B": 25,
    "14B": 25,
}

PROBE_COLORS = {
    "probe": "#1f77b4",
    "random": "#d62728",
    "pca": "#7c3aed",
    "full_pca": "#0ea5e9",
    "zero": "#111827",
    "baseline": "#4b5563",
}

ABLATION_COLORS = {
    "probe": "#1d4ed8",
    "random": "#d1d5db",
    "full_pca": "#bfc5cd",
    "pca": "#949aa7",
    "zero": "#5a6472",
    "baseline": "#949aa7",
}

SPLIT_PALETTE = {
    "train": "#1f77b4",
    "test": "#d62728",
    "test_exact": "#b91c1c",
    "test_inexact": "#fca5a5",
    "shuf_train": "#6b7280",
}

COLORCODE = "depth"  # options: depth, subtree, order
NODE_MARKER_SIZE = 280


def _warn(msg: str) -> None:
    print(f"[visualize] Warning: {msg}")


def _fig_path(dataset_tag: str, model_tag: str, fig_name: str) -> Path:
    filename = f"{fig_name}.png"
    return PROJECT_ROOT / "cutter" / "figures" / "paper" / dataset_tag / model_tag / filename


def _resolve_font_size(value: Any, base_size: float) -> float:
    if isinstance(value, (int, float, np.number)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            key = value.lower()
            if key in font_scalings:
                return float(base_size) * float(font_scalings[key])
    return float(base_size)


def _scaled_text_params(scale: float) -> dict:
    keys = [
        "font.size",
        "axes.titlesize",
        "axes.labelsize",
        "xtick.labelsize",
        "ytick.labelsize",
        "legend.fontsize",
        "figure.titlesize",
    ]
    default_base = float(rcParamsDefault.get("font.size", 10.0))
    base_raw = rcParams.get("font.size", default_base)
    base = _resolve_font_size(base_raw, default_base)
    scaled: Dict[str, float] = {}
    for key in keys:
        resolved = _resolve_font_size(rcParams.get(key, base), base)
        if key == "legend.fontsize":
            scaled[key] = resolved
        else:
            scaled[key] = resolved * scale
    return scaled


def _resolve_layer(family: str, size_token: str) -> int:
    if family == "reasoning":
        if size_token in REASONING_LAYER_MAP:
            return REASONING_LAYER_MAP[size_token]
        _warn(f"Unknown reasoning size '{size_token}', defaulting layer=31")
        return 31
    if family == "chat":
        if size_token in CHAT_LAYER_MAP:
            return CHAT_LAYER_MAP[size_token]
        _warn(f"Unknown chat size '{size_token}', defaulting layer=25")
        return 25
    _warn(f"Unknown family '{family}', defaulting layer=31")
    return 31


def _load_probe_artifact(dataset_tag: str, model_id: str, proj_dim: int, pca_components: int, normalize_tree: bool) -> Tuple[dict, dict, dict] | None:
    probe_fp = path_utils.probe_path(dataset_tag, model_id, proj_dim, normalize_tree, pca_components)
    if probe_fp.exists():
        payload = np.load(probe_fp, allow_pickle=True)
        return payload["meta"].item(), payload["encodings"].item(), payload["results"].item()
    # legacy fallback
    legacy_name = f"probe_normtree_proj{proj_dim}.npz" if normalize_tree else f"probe_proj{proj_dim}.npz"
    legacy_fp = path_utils.probe_output_dir(dataset_tag, model_id) / legacy_name
    if legacy_fp.exists():
        payload = np.load(legacy_fp, allow_pickle=True)
        return payload["meta"].item(), payload["encodings"].item(), payload["results"].item()
    _warn(f"Probe artifact missing: {probe_fp}")
    return None


def _resolve_meta_embeddings_path(meta: dict | None) -> Path | None:
    """Resolve a probe artifact's stored embeddings path, if present."""

    if not isinstance(meta, dict):
        return None
    raw_path = meta.get("embeddings_path")
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
        if candidate.exists():
            return candidate
    return None


def _load_embeddings_cache(
    dataset_tag: str,
    model_id: str,
    pca_components: int,
    meta: dict | None = None,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]] | None:
    # Prefer full embeddings so the probe's PCA basis (stored in results) is
    # applied consistently, matching the notebook behavior.
    meta_path = _resolve_meta_embeddings_path(meta)
    candidates: List[Path] = []
    if meta_path is not None:
        candidates.append(meta_path)
    candidates.append(path_utils.embeddings_path(dataset_tag, model_id))
    candidates.append(path_utils.embeddings_pca_path(dataset_tag, model_id, pca_components))

    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            return load_embedding_payload(path)
    _warn(f"Embeddings cache missing for {model_id} ({dataset_tag})")
    return None


def _resolve_probe_projection(res: dict) -> np.ndarray:
    if isinstance(res, dict) and res.get("projection_full") is not None:
        return res["projection_full"]
    if isinstance(res, dict) and isinstance(res.get("distance"), dict) and res["distance"].get("projection") is not None:
        return res["distance"]["projection"]
    if isinstance(res, dict) and res.get("projection") is not None:
        return res["projection"]
    raise ValueError("No projection found in probe results.")


def _pca_transform(features: np.ndarray, res: dict) -> np.ndarray:
    pca = res.get("pca") if isinstance(res, dict) else None
    if not pca:
        return features
    components = np.asarray(pca.get("components", []), dtype=np.float32)
    mean = np.asarray(pca.get("mean", 0.0), dtype=np.float32)
    if components.size == 0:
        return features
    if components.ndim != 2 or features.ndim != 2:
        return features
    n_features = features.shape[1]
    mean_len = mean.shape[0] if mean.ndim == 1 else None

    if components.shape[1] == n_features:
        mean_adj = mean if mean_len == n_features else np.zeros(n_features, dtype=np.float32)
        return (features - mean_adj) @ components.T
    if components.shape[0] == n_features:
        if mean_len == n_features:
            return (features - mean) @ components
        return features
    return features


def _align_features_to_projection(features: np.ndarray, res: dict, B: np.ndarray | None) -> np.ndarray:
    if B is None or features.ndim != 2:
        return _pca_transform(features, res)
    if B.shape[0] == features.shape[1]:
        return features
    projected = _pca_transform(features, res)
    if projected.shape[1] == B.shape[0]:
        return projected
    if features.shape[1] == B.shape[0]:
        return features
    return projected


def _project_embeddings(arr: np.ndarray, B: np.ndarray, geometry: str, info: dict, res: dict) -> np.ndarray:
    features = _align_features_to_projection(arr.astype(np.float32), res, B)
    proj = features @ B
    if geometry == "hyperbolic":
        proj = transform_probe_space(proj, info)
    return proj


def _project_to_2d(Z: np.ndarray) -> np.ndarray:
    if Z.ndim != 2:
        raise ValueError(f"Expected 2D array for projection, got shape {Z.shape}")
    if Z.shape[1] <= 2:
        return Z
    if Z.shape[0] == 0:
        return Z[:, :2]
    Z_centered = Z - Z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Z_centered, full_matrices=False)
    components = vt[:2]
    return Z_centered @ components.T


def _stat_or_value(res: dict, key: str) -> Tuple[float, float]:
    stats = res.get("stats", {}).get("dist", {})
    if key in stats:
        entry = stats.get(key, {})
        return float(entry.get("mean", float("nan"))), float(entry.get("std", float("nan")))
    val = res.get(key)
    return (float(val) if val is not None else float("nan")), float("nan")


def _depth_stat_or_value(res: dict, split: str, metric: str) -> Tuple[float, float]:
    stats = res.get("stats", {}).get("depth", {})
    if split in stats and metric in stats[split]:
        entry = stats[split][metric]
        return float(entry.get("mean", float("nan"))), float(entry.get("std", float("nan")))
    depth_block = res.get("depth", {}) if isinstance(res, dict) else {}
    split_data = depth_block.get(split, {}) if isinstance(depth_block, dict) else {}
    val = split_data.get(metric)
    return (float(val) if val is not None else float("nan")), float("nan")


def _build_metrics_df(results: Dict[int, Any]) -> pd.DataFrame:
    rows = []
    for layer, res in results.items():
        if not isinstance(res, dict):
            continue
        for split in ["train", "test", "test_exact", "test_inexact", "shuf_train", "shuf_test"]:
            mse_key = f"dist_mse_{split}"
            corr_key = f"dist_corr_{split}"
            mse_val, mse_std = _stat_or_value(res, mse_key)
            corr_val, corr_std = _stat_or_value(res, corr_key)
            rows.append(
                {
                    "layer": layer,
                    "metric": "distance",
                    "split": split,
                    "mse": mse_val,
                    "mse_std": mse_std,
                    "pearson": corr_val,
                    "pearson_std": corr_std,
                }
            )
        for split in ["train", "test", "test_exact", "test_inexact", "shuf_train", "shuf_test"]:
            mse_val, mse_std = _depth_stat_or_value(res, split, "mse")
            pearson_val, pearson_std = _depth_stat_or_value(res, split, "pearson")
            rows.append(
                {
                    "layer": layer,
                    "metric": "depth",
                    "split": split,
                    "mse": mse_val,
                    "mse_std": mse_std,
                    "pearson": pearson_val,
                    "pearson_std": pearson_std,
                }
            )
    return pd.DataFrame(rows)


def _plot_layerwise_statistics(results: Dict[int, Any]) -> plt.Figure:
    with plt.rc_context(_scaled_text_params(1.5)):
        df = _build_metrics_df(results)
        filtered = df[df.split.isin(["train", "test_exact", "test_inexact", "shuf_train"])]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

        hue_order = ["train", "test_exact", "test_inexact", "shuf_train"]
        label_mapping = {
            "train": "Train",
            "test_exact": "Test Exact",
            "test_inexact": "Test Inexact",
            "shuf_train": "Shuffled",
        }
        dashes_mapping = {
            "train": (),
            "test_exact": (),
            "test_inexact": (),
            "shuf_train": (1, 2),
        }

        rows = [("pearson", (0, 1)), ("mse", (0, None))]
        metrics = ["distance", "depth"]

        for row_idx, (score_col, ylim) in enumerate(rows):
            for col_idx, metric in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                for split in hue_order:
                    subset = filtered[(filtered.metric == metric) & (filtered.split == split)].sort_values("layer")
                    if subset.empty:
                        continue
                    x = subset["layer"].to_numpy()
                    y = subset[score_col].to_numpy()
                    std_col = f"{score_col}_std"
                    y_std = subset[std_col].to_numpy() if std_col in subset else np.full_like(y, np.nan, dtype=float)
                    y_std = np.where(np.isfinite(y_std), y_std, 0.0)
                    ax.plot(
                        x,
                        y,
                        color=SPLIT_PALETTE.get(split, "#1f77b4"),
                        label=label_mapping.get(split, split),
                        dashes=dashes_mapping.get(split, ()),
                    )
                    if np.any(y_std > 0):
                        ax.fill_between(
                            x,
                            y - y_std,
                            y + y_std,
                            color=SPLIT_PALETTE.get(split, "#1f77b4"),
                            alpha=0.2,
                        )
                ax.set_ylim(ylim)
                if row_idx == 1:
                    ax.set_xlabel("Layer")
                else:
                    ax.set_xlabel("")
                if col_idx == 0:
                    label = "MSE" if score_col == "mse" else score_col.title()
                    ax.set_ylabel(label)
                else:
                    ax.set_ylabel("")
        axes[0, 0].set_title("Distance")
        axes[0, 1].set_title("Depth")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[0, 1].legend(handles, labels, loc="upper right", frameon=False)
        fig.tight_layout()
        return fig


def _root_child_label(node_id: int) -> str:
    if node_id == 0:
        return "root"
    child = node_id
    parent = (child - 1) // 2
    while parent > 0:
        child = parent
        parent = (parent - 1) // 2
    return "left" if child == 1 else "right"


SUBTREE_CMAP = ListedColormap(["#1f77b4", "#d62728", "#7f7f7f"])
SUBTREE_BOUNDARIES = [-0.5, 0.5, 1.5, 2.5]
SUBTREE_TICKS = [0, 1, 2]
SUBTREE_LABELS = ["Left", "Right", "Root"]

_PATH_ORDER_CACHE: Dict[Path, Tuple[Dict[int, Dict[int, int]], int]] = {}


def _parse_path_tokens(text: Any) -> List[int]:
    tokens = str(text).replace(",", " ").split()
    parsed: List[int] = []
    for tok in tokens:
        try:
            parsed.append(int(tok))
        except Exception:
            continue
    return parsed


def _load_path_orders(responses_path: Path) -> Tuple[Dict[int, Dict[int, int]], int]:
    lookup: Dict[int, Dict[int, int]] = {}
    max_len = 0
    if responses_path and responses_path.exists():
        with responses_path.open() as f:
            for line in f:
                rec = json.loads(line)
                ex_id = rec.get("example_id")
                if ex_id is None:
                    continue
                seq = rec.get("parsed_path") or rec.get("path")
                if isinstance(seq, str):
                    seq = _parse_path_tokens(seq)
                elif not seq:
                    txt = rec.get("parsed_text")
                    seq = _parse_path_tokens(txt) if txt else []
                if not seq:
                    txt = rec.get("parsed_text")
                    if txt:
                        seq = _parse_path_tokens(txt)
                if not seq:
                    continue
                orders = {int(node_id): idx for idx, node_id in enumerate(seq)}
                lookup[int(ex_id)] = orders
                max_len = max(max_len, len(seq))
    return lookup, max_len


def _get_path_orders(responses_path: Path | None) -> Tuple[Dict[int, Dict[int, int]], int]:
    if responses_path is None:
        return {}, 0
    cached = _PATH_ORDER_CACHE.get(responses_path)
    if cached is None:
        cached = _load_path_orders(responses_path)
        _PATH_ORDER_CACHE[responses_path] = cached
    return cached


def _order_cmap(n_orders: int) -> ListedColormap:
    if n_orders <= 0:
        return ListedColormap(np.array([[0.7, 0.7, 0.7, 1.0]]))
    grad = plt.get_cmap("plasma")(np.linspace(0.15, 0.85, n_orders))
    colors = np.vstack(([0.7, 0.7, 0.7, 1.0], grad))
    return ListedColormap(colors)


def _prepare_colorcode(
    colorcode: str,
    node_ids: np.ndarray,
    depths: np.ndarray,
    example_ids: np.ndarray,
    responses_path: Path | None = None,
) -> Tuple[np.ndarray, Any, Any, str, list[int] | None, list[str] | None]:
    code = (colorcode or "depth").lower()
    if code == "subtree":
        labels = np.array([_root_child_label(int(nid)) for nid in node_ids])
        mapping = {"left": 0, "right": 1, "root": 2}
        values = np.array([mapping[label] for label in labels], dtype=float)
        cmap = SUBTREE_CMAP
        norm = BoundaryNorm(SUBTREE_BOUNDARIES, cmap.N)
        cbar_label = "Subtree"
        tick_locs = SUBTREE_TICKS
        tick_labels = SUBTREE_LABELS
    elif code == "order":
        example_ids = np.asarray(example_ids)
        node_ids_arr = np.asarray(node_ids)
        if example_ids.size != node_ids_arr.size:
            raise ValueError("example_ids and node_ids must align for order color coding")
        lookup, max_len = _get_path_orders(responses_path)
        values = np.array(
            [lookup.get(int(ex_id), {}).get(int(nid), -1) for ex_id, nid in zip(example_ids, node_ids_arr)],
            dtype=float,
        )
        n_orders = max(max_len, int(values[values >= 0].max()) + 1 if np.any(values >= 0) else 0)
        cmap = _order_cmap(n_orders)
        boundaries = np.arange(-1.5, n_orders + 0.5, 1.0)
        norm = BoundaryNorm(boundaries, cmap.N)
        cbar_label = "Path order"
        tick_locs = [-1] + list(range(n_orders))
        tick_labels = ["Not in path"] + [str(i) for i in range(n_orders)]
    else:
        values = np.asarray(depths)
        cbar_label = "Depth"
        if values.size:
            dmin = int(values.min())
            dmax = int(values.max())
            boundaries = np.arange(dmin - 0.5, dmax + 1.5, 1.0)
            cmap = plt.get_cmap("Blues", dmax - dmin + 1)
            norm = BoundaryNorm(boundaries, cmap.N)
            tick_locs = list(range(dmin, dmax + 1))
            tick_labels = [str(d) for d in tick_locs]
        else:
            cmap = "Blues"
            norm = None
            tick_locs = tick_labels = None
    return values, cmap, norm, cbar_label, tick_locs, tick_labels


def _collect_test_exact_ids(encodings: dict, responses_path: Path, embeddings_cache: Dict[int, Any], layer: int, depth_filter: int, min_path_len: int) -> List[int]:
    data = encodings[layer]
    example_ids = np.asarray(data.get("example_ids", []))
    if example_ids.size == 0:
        raise ValueError("Encodings missing example_ids; cannot map to embeddings")
    train_idx = set(data.get("train_idx", []))
    splits = np.array(["train" if i in train_idx else "test" for i in range(len(example_ids))])
    per_example_split: Dict[int, str] = {}
    for idx, ex_id in enumerate(example_ids):
        per_example_split.setdefault(int(ex_id), splits[idx])
    test_ids = {ex for ex, sp in per_example_split.items() if sp == "test"}

    exact_ids = set()
    depth_ok = set()
    path_len_ok = set()
    with responses_path.open() as f:
        for line in f:
            rec = json.loads(line)
            ex_id = rec.get("example_id")
            if rec.get("depth") == depth_filter:
                depth_ok.add(ex_id)
            if rec.get("exact_match"):
                exact_ids.add(ex_id)
            parsed_path = rec.get("parsed_path") or rec.get("path")
            if parsed_path and len(parsed_path) >= min_path_len:
                path_len_ok.add(ex_id)

    available_ids = set(embeddings_cache.keys())
    candidates = sorted(test_ids & exact_ids & available_ids & depth_ok & path_len_ok)
    return candidates


def _example_distance_mse(
    encodings: dict,
    results: dict,
    example_id: int,
    layer: int,
    normalize_tree: bool,
    normalization_factor: float | None,
    meta: dict | None = None,
) -> float:
    data = encodings[layer]
    res = results[layer]
    X = data["X"]
    example_ids = np.asarray(data.get("example_ids", []))
    node_ids = np.asarray(data.get("node_ids", []))
    mask_example = example_ids == example_id
    X_example = np.asarray(X[mask_example])
    node_ids_example = node_ids[mask_example]

    B = _resolve_probe_projection(res)
    fit_geometry = meta.get("fit_geometry", "euclidean") if isinstance(meta, dict) else "euclidean"
    info = {
        "geometry": res.get("geometry", fit_geometry),
        "center": res.get("center"),
        "curvature": res.get("curvature"),
        "normalized_tree": res.get("normalized_tree", meta.get("normalize_tree", False) if isinstance(meta, dict) else False),
    }
    geometry = info.get("geometry", fit_geometry)

    X_example = _align_features_to_projection(X_example, res, B)
    Z_example = X_example @ B
    pred_dist = pairwise_distance(Z_example, geometry, center=info.get("center"), curvature=info.get("curvature"))

    n = len(node_ids_example)
    true_dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            true_dist[i, j] = true_dist[j, i] = float(tree_distance(int(node_ids_example[i]), int(node_ids_example[j])))

    if normalize_tree and normalization_factor:
        true_dist = true_dist / normalization_factor
        pred_dist = pred_dist / normalization_factor

    tri = np.triu(np.ones_like(true_dist, dtype=bool), k=1)
    diff = true_dist[tri] - pred_dist[tri]
    return float(np.mean(diff ** 2))


def _plot_example_projection(
    ax,
    encodings: dict,
    results: dict,
    example_id: int,
    label_mapping: list[int] | None,
    parsed_path: list[int] | None,
    responses_path: Path | None = None,
    meta: dict | None = None,
    colorcode: str = COLORCODE,
    *,
    add_labels: bool = False,
    label_size: int = 9,
    node_size: int = NODE_MARKER_SIZE,
    pad_scale: float = 0.15,
):
    data = encodings
    res = results
    X = data["X"]
    depths = data["depth"]
    example_ids = np.asarray(data.get("example_ids", []))
    node_ids = np.asarray(data.get("node_ids", []))
    label_mapping = label_mapping or []
    inverse_label = {int(label): idx for idx, label in enumerate(label_mapping)} if label_mapping else {}
    color_values, cmap, norm, _, _, _ = _prepare_colorcode(colorcode, node_ids, depths, example_ids, responses_path)
    B = _resolve_probe_projection(res)
    fit_geometry = meta.get("fit_geometry", "euclidean") if isinstance(meta, dict) else "euclidean"
    info = {
        "geometry": res.get("geometry", fit_geometry),
        "center": res.get("center"),
        "curvature": res.get("curvature"),
        "normalized_tree": res.get("normalized_tree", meta.get("normalize_tree", False) if isinstance(meta, dict) else False),
    }
    geometry = info.get("geometry", fit_geometry)

    Z = _project_embeddings(X, B, geometry, info, res)
    Z = _project_to_2d(Z)

    mask_example = example_ids == example_id
    Z_example = Z[mask_example]
    depths_example = depths[mask_example]
    colors_example = color_values[mask_example]
    node_ids_example = node_ids[mask_example]

    path_node_ids = []
    if parsed_path:
        for tok in parsed_path:
            try:
                label_val = int(tok)
                canonical_id = inverse_label.get(label_val, label_val)
                path_node_ids.append(int(canonical_id))
            except Exception:
                continue

    path_indices = []
    if path_node_ids:
        path_node_ids = [int(nid) for nid in path_node_ids]
        node_ids_list = [int(nid) for nid in node_ids_example]
        if node_ids_list[: len(path_node_ids)] == path_node_ids:
            path_indices = list(range(len(path_node_ids)))
        else:
            ptr = 0
            for idx, nid in enumerate(node_ids_list):
                if ptr < len(path_node_ids) and nid == path_node_ids[ptr]:
                    path_indices.append(idx)
                    ptr += 1

    # Rotate embeddings to make depth levels as horizontally level as possible
    overall_center = Z_example.mean(axis=0)
    centered = Z_example - overall_center
    unique_depths = np.unique(depths_example)

    def _leveling_cost(theta: float) -> float:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = centered @ R.T
        cost = 0.0
        for d in unique_depths:
            mask = depths_example == d
            if mask.any():
                y = rotated[mask, 1]
                y_mean = float(y.mean())
                diff = y - y_mean
                cost += float((diff * diff).sum())
        return cost

    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 361)
    costs = np.array([_leveling_cost(theta) for theta in angles])
    best_theta = float(angles[int(costs.argmin())])

    cos_a, sin_a = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    Z_example = centered @ R.T + overall_center

    # Invert y-axis so lowest-depth nodes end up near the top
    Z_example[:, 1] = 2 * overall_center[1] - Z_example[:, 1]

    for node_id in np.unique(node_ids_example):
        idxs = np.where(node_ids_example == node_id)[0]
        if idxs.size > 1:
            coords = Z_example[idxs]
            line_color = cmap(norm(colors_example[idxs[0]]))
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                linestyle="-",
                color="black",
                alpha=0.9,
                lw=4.2,
                zorder=0,
            )
            ax.plot(
                coords[:, 0],
                coords[:, 1],
                linestyle="-",
                color=line_color,
                alpha=0.85,
                lw=3.2,
                zorder=1,
            )

    if path_node_ids and len(path_indices) > 1:
        x_range = Z_example[:, 0].max() - Z_example[:, 0].min()
        y_range = Z_example[:, 1].max() - Z_example[:, 1].min()
        avg_range = (x_range + y_range) / 2
        node_radius_estimate = avg_range * 0.02
        arrow_color = "#dc2626"

        for i in range(len(path_indices) - 1):
            idx1, idx2 = path_indices[i], path_indices[i + 1]
            x1, y1 = Z_example[idx1, 0], Z_example[idx1, 1]
            x2, y2 = Z_example[idx2, 0], Z_example[idx2, 1]
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                shrink_factor = 1.0 - (node_radius_estimate * 0.7) / dist
                x2_adj = x1 + dx * shrink_factor
                y2_adj = y1 + dy * shrink_factor
            else:
                x2_adj, y2_adj = x2, y2
            ax.annotate(
                "",
                xy=(x2_adj, y2_adj),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5, alpha=0.75, mutation_scale=28),
                zorder=0,
            )
            label_x = (x1 + x2_adj) / 2
            label_y = (y1 + y2_adj) / 2
            step_label = "step 1" if i == 0 else f"{i + 1}"
            ax.text(
                label_x,
                label_y,
                step_label,
                color="black",
                fontsize=9,
                ha="center",
                va="center",
                zorder=2,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
            )

    ax.scatter(
        Z_example[:, 0],
        Z_example[:, 1],
        c=colors_example,
        cmap=cmap,
        norm=norm,
        edgecolors="k",
        s=node_size,
        marker="o",
        zorder=1,
    )

    if add_labels:
        for (x, y), nid, depth in zip(Z_example, node_ids_example, depths_example):
            text_color = "white" if int(depth) == 2 else "black"
            ax.text(
                x,
                y,
                str(int(nid)),
                fontsize=label_size,
                weight="bold",
                color=text_color,
                ha="center",
                va="center",
                zorder=2,
            )

    x_min, x_max = Z_example[:, 0].min(), Z_example[:, 0].max()
    y_min, y_max = Z_example[:, 1].min(), Z_example[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    padding = max_range * pad_scale
    ax.set_xlim(x_center - max_range / 2 - padding, x_center + max_range / 2 + padding)
    ax.set_ylim(y_center - max_range / 2 - padding, y_center + max_range / 2 + padding)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")


def _wrap_lines(lines: Sequence[str], width: int = 65) -> List[str]:
    import textwrap

    wrapped: List[str] = []
    for line in lines:
        if line.strip() and all(ch.isdigit() or ch.isspace() for ch in line):
            wrapped.append(line)
            continue
        wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return wrapped


def _abridge_text_blocks(
    prompt: str,
    model_raw: str,
    max_prompt_lines: int = 6,
    max_model_lines: int = 5,
    width: int = 65,
    model_width: int = 60,
):
    """Return wrapped prompt/model snippets. PATH handled separately."""
    prompt_lines = prompt.splitlines()
    if "Guidelines:" in prompt_lines:
        prompt_lines = prompt_lines[: prompt_lines.index("Guidelines:")]

    tree_graph_idx = prompt_lines.index("Tree Graph:") if "Tree Graph:" in prompt_lines else None
    task_idx = None
    if tree_graph_idx is not None:
        depth_idxs = [i for i, ln in enumerate(prompt_lines[:tree_graph_idx]) if ln.strip().startswith("Tree depth:")]
        if depth_idxs:
            depth_idx = depth_idxs[0]
            prompt_lines = prompt_lines[:depth_idx] + prompt_lines[tree_graph_idx:]
    task_idxs = [i for i, ln in enumerate(prompt_lines) if ln.strip().startswith("Task:")]
    if task_idxs:
        task_idx = task_idxs[0]

    prompt_snip = []
    preamble = prompt_lines[:3]
    prompt_snip.extend(preamble)

    seen = set()
    deduped = []
    for ln in prompt_snip:
        if ln not in seen:
            deduped.append(ln)
            seen.add(ln)
    prompt_snip = deduped[: max_prompt_lines + 2]

    task_lines = []
    if task_idx is not None:
        task_lines.append(prompt_lines[task_idx])
        if task_idx + 1 < len(prompt_lines):
            task_lines.append(prompt_lines[task_idx + 1])
    prompt_snip = [ln for ln in prompt_snip if ln.strip() != "Tree Graph:" and not ln.strip().isdigit()]
    prompt_snip = _wrap_lines(prompt_snip, width=width)
    task_lines = _wrap_lines(task_lines, width=width)

    model_lines = model_raw.splitlines()
    wrapped_model = _wrap_lines(model_lines, width=model_width)
    model_snip = wrapped_model[:max_model_lines]
    if wrapped_model and len(wrapped_model) > max_model_lines:
        model_snip[-1] = model_snip[-1] + "..."
    elif model_snip:
        model_snip[-1] = model_snip[-1] + "..."
    else:
        model_snip = []

    return prompt_snip, task_lines, model_snip


def _parse_tree_ascii(tree_ascii: str | None):
    if not tree_ascii:
        return [], []
    lines = [ln.rstrip("\n") for ln in tree_ascii.splitlines() if ln.strip()]
    nodes = []
    for row_idx, line in enumerate(lines):
        for match in re.finditer(r"\d+", line):
            label = int(match.group())
            col = (match.start() + match.end() - 1) / 2.0
            nodes.append({"label": label, "row": row_idx, "col": col})
    if not nodes:
        return [], []

    cols = [n["col"] for n in nodes]
    min_col, max_col = min(cols), max(cols)
    max_row = max(n["row"] for n in nodes)

    for n in nodes:
        n["x"] = 0.5 if max_col == min_col else (n["col"] - min_col) / (max_col - min_col)
        n["y"] = 1.0 if max_row == 0 else 1.0 - (n["row"] / max_row)

    nodes_by_row = {}
    for n in nodes:
        nodes_by_row.setdefault(n["row"], []).append(n)

    edges = []
    for row_idx in range(1, max_row + 1):
        parents = nodes_by_row.get(row_idx - 1, [])
        if not parents:
            continue
        for n in nodes_by_row.get(row_idx, []):
            parent = min(parents, key=lambda p: abs(p["col"] - n["col"]))
            edges.append((parent, n))

    return nodes, edges


def _plot_test_with_example(
    encodings: dict,
    results: dict,
    example_id: int,
    prompt: str,
    model_raw: str,
    path_line: str,
    tree_ascii: str | None,
    label_mapping: list[int] | None,
    parsed_path: list[int] | None,
    responses_path: Path | None = None,
    meta: dict | None = None,
    colorcode: str = COLORCODE,
) -> plt.Figure:
    data = encodings
    res = results
    X = data["X"]
    depths = data["depth"]
    example_ids = np.asarray(data.get("example_ids", []))
    node_ids = np.asarray(data.get("node_ids", []))
    label_mapping = label_mapping or []
    label_lookup = {idx: int(label) for idx, label in enumerate(label_mapping)} if label_mapping else {}
    inverse_label = {int(label): idx for idx, label in enumerate(label_mapping)} if label_mapping else {}
    labels_example = np.array([label_lookup.get(int(nid), int(nid)) for nid in node_ids[example_ids == example_id]])

    train_idx = set(data.get("train_idx", []))
    splits = np.array(["train" if i in train_idx else "test" for i in range(len(depths))])

    color_values, cmap, norm, cbar_label, tick_locs, tick_labels = _prepare_colorcode(
        colorcode, node_ids, depths, example_ids, responses_path
    )
    B = _resolve_probe_projection(res)
    fit_geometry = meta.get("fit_geometry", "euclidean") if isinstance(meta, dict) else "euclidean"
    info = {
        "geometry": res.get("geometry", fit_geometry),
        "center": res.get("center"),
        "curvature": res.get("curvature"),
        "normalized_tree": res.get("normalized_tree", meta.get("normalize_tree", False) if isinstance(meta, dict) else False),
    }
    geometry = info.get("geometry", fit_geometry)

    Z = _project_embeddings(np.asarray(X, dtype=np.float32), B, geometry, info, res)
    Z = _project_to_2d(Z)

    mask_example = example_ids == example_id
    Z_example = Z[mask_example]
    depths_example = depths[mask_example]
    colors_example = color_values[mask_example]
    node_ids_example = node_ids[mask_example]

    path_node_ids = []
    path_tokens = []
    if parsed_path:
        path_tokens = [str(tok) for tok in parsed_path]
    elif path_line and path_line.upper().startswith("PATH:"):
        path_tokens = path_line.split()[1:]
    for tok in path_tokens:
        try:
            label_val = int(tok)
            canonical_id = inverse_label.get(label_val, label_val)
            path_node_ids.append(int(canonical_id))
        except Exception:
            continue

    path_indices = []
    if path_node_ids:
        node_ids_list = [int(nid) for nid in node_ids_example]
        if node_ids_list[: len(path_node_ids)] == path_node_ids:
            path_indices = list(range(len(path_node_ids)))
        else:
            ptr = 0
            for idx, nid in enumerate(node_ids_list):
                if ptr < len(path_node_ids) and nid == path_node_ids[ptr]:
                    path_indices.append(idx)
                    ptr += 1

    overall_center = Z_example.mean(axis=0)
    centered = Z_example - overall_center
    unique_depths = np.unique(depths_example)

    def _leveling_cost(theta: float) -> float:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = centered @ R.T
        cost = 0.0
        for d in unique_depths:
            mask = depths_example == d
            if mask.any():
                y = rotated[mask, 1]
                y_mean = float(y.mean())
                diff = y - y_mean
                cost += float((diff * diff).sum())
        return cost

    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 361)
    costs = np.array([_leveling_cost(theta) for theta in angles])
    best_theta = float(angles[int(costs.argmin())])

    cos_a, sin_a = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    Z_example = centered @ R.T + overall_center

    Z_example[:, 1] = 2 * overall_center[1] - Z_example[:, 1]

    fig, axes = plt.subplots(1, 2, figsize=(17, 9), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(wspace=0.01)

    for spine in axes[0].spines.values():
        spine.set_visible(False)
    axes[0].set_facecolor("none")
    axes[0].patch.set_alpha(0.0)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_aspect("auto", adjustable="box")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    for spine in axes[1].spines.values():
        spine.set_visible(False)

    prompt_snip, task_lines, model_snip = _abridge_text_blocks(prompt, model_raw, model_width=60)
    y = 0.98
    line_h = 0.062
    user_texts = []
    model_texts = []
    user_text_x = 0.08
    model_text_x = 0.02
    depth_cmap = plt.cm.Blues

    t = axes[0].text(user_text_x, y, "User:", fontsize=13, fontweight="bold", family="monospace", va="top", ha="left")
    user_texts.append(t)
    y -= line_h
    for ln in prompt_snip:
        t = axes[0].text(user_text_x, y, ln, fontsize=13, family="monospace", va="top", ha="left")
        user_texts.append(t)
        y -= line_h

    tree_ax = None
    if tree_ascii:
        tree_nodes, tree_edges = _parse_tree_ascii(tree_ascii)
        if tree_nodes:
            tree_rows = max(n["row"] for n in tree_nodes) + 1
            tree_h = min(0.34, line_h * max(4, tree_rows) * 1.0)
            tree_w = 0.55
            y -= line_h * 0.15
            tree_y1 = y
            tree_y0 = tree_y1 - tree_h
            tree_x0 = min(0.2, user_text_x + 0.015)
            tree_ax = axes[0].inset_axes([tree_x0, tree_y0, tree_w, tree_h], transform=axes[0].transAxes)
            tree_ax.set_axis_off()

            all_xs = [n["x"] for n in tree_nodes]
            target_center = (min(all_xs) + max(all_xs)) / 2
            shifted_x = [n["x"] - target_center + 0.5 for n in tree_nodes]
            min_x, max_x = min(shifted_x), max(shifted_x)
            if max_x - min_x > 0:
                scale = 1.0 / (max_x - min_x)
                for n, sx in zip(tree_nodes, shifted_x):
                    n["x"] = (sx - min_x) * scale
            else:
                for n in tree_nodes:
                    n["x"] = 0.5

            root_nodes = [n for n in tree_nodes if n["row"] == 0]
            child_nodes = [n for n in tree_nodes if n["row"] > 0]
            if root_nodes and child_nodes:
                child_xs = [n["x"] for n in child_nodes]
                center_x = (min(child_xs) + max(child_xs)) / 2
                for n in root_nodes:
                    n["x"] = center_x

            for parent, child in tree_edges:
                tree_ax.plot([parent["x"], child["x"]], [parent["y"], child["y"]], color="#999999", lw=1.2, zorder=0)

            canonical_labels = [inverse_label.get(int(n["label"]), int(n["label"])) for n in tree_nodes]
            depth_vals = [int(np.floor(np.log2(lbl + 1))) for lbl in canonical_labels]
            min_depth = min(depth_vals)
            max_depth = max(depth_vals)
            if max_depth == min_depth:
                max_depth = min_depth + 1
            depth_norm_tree = Normalize(vmin=min_depth, vmax=max_depth)

            for n, d in zip(tree_nodes, depth_vals):
                color = depth_cmap(depth_norm_tree(d))
                tree_ax.scatter(n["x"], n["y"], s=180, c=[color], edgecolors="k", linewidths=0.8, zorder=2)
                text_color = "white" if int(d) == 2 else "black"
                tree_ax.text(
                    n["x"],
                    n["y"],
                    str(n["label"]),
                    fontsize=8,
                    weight="bold",
                    color=text_color,
                    ha="center",
                    va="center",
                    zorder=3,
                )

            tree_ax.set_xlim(-0.08, 1.08)
            tree_ax.set_ylim(-0.08, 1.08)
            y = tree_y0 - line_h * 0.3

    for ln in task_lines:
        t = axes[0].text(user_text_x, y, ln, fontsize=13, family="monospace", va="top", ha="left")
        user_texts.append(t)
        y -= line_h

    y -= line_h * 0.5
    t = axes[0].text(model_text_x, y, "Model:", fontsize=13, fontweight="bold", family="monospace", va="top", ha="left")
    model_texts.append(t)
    y -= line_h
    for ln in model_snip:
        t = axes[0].text(model_text_x, y, ln, fontsize=13, family="monospace", va="top", ha="left")
        model_texts.append(t)
        y -= line_h

    depth_norm = Normalize(vmin=depths_example.min(), vmax=depths_example.max())
    if path_tokens:
        y -= line_h * 0.2
        t = axes[0].text(model_text_x, y, "PATH:", fontsize=13, family="monospace", va="top", ha="left")
        model_texts.append(t)
        x_offset = model_text_x + 0.11
        for tok in path_tokens:
            try:
                label_val = int(tok)
                canonical_id = inverse_label.get(label_val, label_val)
                node_depth = int(np.floor(np.log2(canonical_id + 1)))
            except Exception:
                node_depth = depths_example.min()
            color = depth_cmap(depth_norm(node_depth))
            path_effect = [patheffects.withStroke(linewidth=1.0, foreground="black")]
            t = axes[0].text(
                x_offset,
                y,
                tok,
                fontsize=13,
                weight="light",
                family="monospace",
                va="top",
                ha="left",
                color=color,
                path_effects=path_effect,
            )
            model_texts.append(t)
            x_offset += 0.06 * max(len(tok), 1) + 0.02
            if x_offset > 0.96:
                y -= line_h
                x_offset = 0.11
        y -= line_h * 0.6

    for node_id in np.unique(node_ids_example):
        idxs = np.where(node_ids_example == node_id)[0]
        if idxs.size > 1:
            coords = Z_example[idxs]
            line_color = cmap(norm(colors_example[idxs[0]]))
            axes[1].plot(coords[:, 0], coords[:, 1], linestyle="-", color="black", alpha=0.9, lw=4.2, zorder=0)
            axes[1].plot(
                coords[:, 0],
                coords[:, 1],
                linestyle="-",
                color=line_color,
                alpha=0.85,
                lw=3.2,
                zorder=1,
            )

    if path_node_ids and len(path_indices) > 1:
        x_range = Z_example[:, 0].max() - Z_example[:, 0].min()
        y_range = Z_example[:, 1].max() - Z_example[:, 1].min()
        avg_range = (x_range + y_range) / 2
        node_radius_estimate = avg_range * 0.02
        arrow_color = "#dc2626"
        for i in range(len(path_indices) - 1):
            idx1, idx2 = path_indices[i], path_indices[i + 1]
            x1, y1 = Z_example[idx1, 0], Z_example[idx1, 1]
            x2, y2 = Z_example[idx2, 0], Z_example[idx2, 1]
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                shrink_factor = 1.0 - (node_radius_estimate * 0.7) / dist
                x2_adj = x1 + dx * shrink_factor
                y2_adj = y1 + dy * shrink_factor
            else:
                x2_adj, y2_adj = x2, y2
            axes[1].annotate(
                "",
                xy=(x2_adj, y2_adj),
                xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.5, alpha=0.75, mutation_scale=28),
                zorder=0,
            )
            label_x = (x1 + x2_adj) / 2
            label_y = (y1 + y2_adj) / 2
            step_label = "step 1" if i == 0 else f"{i + 1}"
            axes[1].text(
                label_x,
                label_y,
                step_label,
                color="black",
                fontsize=9,
                ha="center",
                va="center",
                zorder=2,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0),
            )

    right_node_size = NODE_MARKER_SIZE * 1.25
    scatter = axes[1].scatter(
        Z_example[:, 0],
        Z_example[:, 1],
        c=colors_example,
        cmap=cmap,
        norm=norm,
        edgecolors="k",
        s=right_node_size,
        marker="o",
        zorder=1,
    )

    for coords, label, depth in zip(Z_example, labels_example, depths_example):
        x, y = coords[:2]
        text_color = "white" if int(depth) == 2 else "black"
        axes[1].text(
            x,
            y,
            str(label),
            fontsize=9,
            weight="bold",
            color=text_color,
            ha="center",
            va="center",
        )

    x_min, x_max = Z_example[:, 0].min(), Z_example[:, 0].max()
    y_min, y_max = Z_example[:, 1].min(), Z_example[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    padding = max_range * 0.15

    axes[1].set_xlim(x_center - max_range / 2 - padding, x_center + max_range / 2 + padding)
    axes[1].set_ylim(y_center - max_range / 2 - padding, y_center + max_range / 2 + padding)

    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[1].set_aspect("equal", adjustable="box")

    bottom, top = 0.07, 0.93
    left_margin, right_margin = 0.04, 0.03
    inter_panel = 0.02
    cb_pad = 0.012
    cb_w = 0.018
    H = top - bottom

    fig_w, fig_h = fig.get_size_inches()
    right_w = H * (fig_h / fig_w)

    cax_x0 = 1.0 - right_margin - cb_w
    right_x1 = cax_x0 - cb_pad
    right_x0 = right_x1 - right_w

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_px_w = fig.bbox.width
    req_left_px = 0.0
    for t in axes[0].texts:
        try:
            bb = t.get_window_extent(renderer=renderer)
        except Exception:
            continue
        x_start = float(t.get_position()[0])
        denom = max(1e-6, 1.0 - x_start)
        req_left_px = max(req_left_px, bb.width / denom)
    req_left_w = (req_left_px + 30.0) / fig_px_w
    max_left_w = max(0.1, right_x0 - inter_panel - left_margin)
    left_w = min(req_left_w, max_left_w)

    axes[0].set_position([left_margin, bottom, left_w, H])
    axes[1].set_position([right_x0, bottom, right_w, H])

    cb_h = H * 0.6
    cb_y0 = bottom + (H - cb_h) / 2
    cax = fig.add_axes([cax_x0, cb_y0, cb_w, cb_h])
    cbar = fig.colorbar(scatter, cax=cax)
    cbar.set_label(cbar_label, rotation=270, labelpad=15)
    if tick_locs is not None:
        cbar.set_ticks(tick_locs)
        cbar.set_ticklabels(tick_labels)
    if hasattr(cbar, "solids"):
        cbar.solids.set_alpha(1)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    def _bubble_from_artists(artists, pad_px: float = 8.0, extra_bottom_px: float = 0.0, x_offset: float = 0.0):
        bbs = []
        for artist in artists:
            try:
                bbs.append(artist.get_window_extent(renderer=renderer))
            except Exception:
                continue
        if not bbs:
            return
        x0 = min(bb.x0 for bb in bbs) - pad_px
        y0 = min(bb.y0 for bb in bbs) - pad_px - extra_bottom_px
        x1 = max(bb.x1 for bb in bbs) + pad_px
        y1 = max(bb.y1 for bb in bbs) + pad_px
        (ax_x0, ax_y0) = axes[0].transAxes.inverted().transform((x0, y0))
        (ax_x1, ax_y1) = axes[0].transAxes.inverted().transform((x1, y1))
        ax_x0 += x_offset
        ax_x1 += x_offset
        bubble = FancyBboxPatch(
            (ax_x0, ax_y0),
            ax_x1 - ax_x0,
            ax_y1 - ax_y0,
            boxstyle="round,pad=0.0,rounding_size=0.02",
            linewidth=0.8,
            edgecolor="#cccccc",
            facecolor="white",
            alpha=1.0,
            transform=axes[0].transAxes,
            clip_on=False,
            zorder=0.5,
        )
        axes[0].add_patch(bubble)

    user_artists = user_texts + ([tree_ax] if tree_ax is not None else [])
    _bubble_from_artists(user_artists, pad_px=16.0, extra_bottom_px=8.0)
    _bubble_from_artists(model_texts, pad_px=16.0, extra_bottom_px=8.0)

    pad_frac = 0.02

    def _panel_from_ax(ax, *, fill: bool):
        pos = ax.get_position().frozen()
        x0 = pos.x0 - pad_frac * pos.width
        y0 = pos.y0 - pad_frac * pos.height
        w = pos.width * (1 + 2 * pad_frac)
        h = pos.height * (1 + 2 * pad_frac)
        if fill:
            bg = Rectangle(
                (x0, y0),
                w,
                h,
                linewidth=0.0,
                edgecolor="none",
                facecolor="#f5f5f5",
                alpha=0.6,
                transform=fig.transFigure,
                clip_on=False,
                zorder=-5,
            )
            fig.add_artist(bg)
        outline = Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=0.8,
            edgecolor="#cccccc",
            facecolor="none",
            alpha=1.0,
            transform=fig.transFigure,
            clip_on=False,
            zorder=10,
        )
        fig.add_artist(outline)

    _panel_from_ax(axes[0], fill=True)
    _panel_from_ax(axes[1], fill=False)
    return fig


def _extract_tree_and_path(responses_path: Path, example_id: int):
    model_path_line = None
    tree_ascii = None
    prompt = ""
    model_raw = ""
    label_mapping = []
    parsed_path = []
    with responses_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("example_id") == example_id:
                model_raw = rec.get("model_raw", "")
                for ln in model_raw.splitlines()[::-1]:
                    stripped = ln.strip()
                    if stripped.upper().startswith("PATH:"):
                        model_path_line = stripped
                        break
                parsed_path = rec.get("parsed_path") or []
                if model_path_line is None and parsed_path:
                    model_path_line = "PATH: " + " ".join(str(x) for x in parsed_path)
                elif model_path_line is None and rec.get("parsed_text"):
                    model_path_line = f"PATH: {rec['parsed_text']}"
                prompt = rec.get("prompt", "")
                label_mapping = rec.get("label_mapping") or []
                if prompt:
                    lines = prompt.splitlines()
                    if "Tree Graph:" in lines:
                        start = lines.index("Tree Graph:") + 1
                        collected = []
                        for ln in lines[start:]:
                            if ln.strip() == "" and collected:
                                break
                            collected.append(ln)
                        tree_ascii = "\n".join(collected).strip()
                break
    return prompt, model_raw, tree_ascii, model_path_line, label_mapping, parsed_path


def _plot_probe_geometry_grid(
    encodings: dict,
    results: dict,
    chosen: Sequence[int],
    responses_path: Path,
    meta: dict | None = None,
    colorcode: str = COLORCODE,
) -> plt.Figure:
    grid_rows, grid_cols = 2, 4
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(22, 8))
    axes = np.ravel(axes)
    for ax, ex_id in zip(axes, chosen):
        label_mapping, parsed_path = _load_path_info(responses_path, ex_id)
        _plot_example_projection(
            ax,
            encodings,
            results,
            ex_id,
            label_mapping,
            parsed_path,
            responses_path=responses_path,
            meta=meta,
            colorcode=colorcode,
        )
    for ax in axes[len(chosen):]:
        ax.axis("off")
    fig.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.2)
    return fig


def _plot_true_tree(ax, tree_ascii: str | None, label_mapping: list[int] | None, parsed_path: list[int] | None):
    label_mapping = label_mapping or []
    parsed_path = parsed_path or []
    inverse_label = {int(label): idx for idx, label in enumerate(label_mapping)} if label_mapping else {}

    nodes, edges = _parse_tree_ascii(tree_ascii)
    if not nodes:
        ax.axis("off")
        return None, None

    all_xs = [n["x"] for n in nodes]
    target_center = (min(all_xs) + max(all_xs)) / 2
    shifted_x = [n["x"] - target_center + 0.5 for n in nodes]
    min_x, max_x = min(shifted_x), max(shifted_x)
    if max_x - min_x > 0:
        scale = 1.0 / (max_x - min_x)
        for n, sx in zip(nodes, shifted_x):
            n["x"] = (sx - min_x) * scale
    else:
        for n in nodes:
            n["x"] = 0.5

    root_nodes = [n for n in nodes if n["row"] == 0]
    child_nodes = [n for n in nodes if n["row"] > 0]
    if root_nodes and child_nodes:
        child_xs = [n["x"] for n in child_nodes]
        center_x = (min(child_xs) + max(child_xs)) / 2
        for n in root_nodes:
            n["x"] = center_x

    for n in nodes:
        lbl = int(n["label"])
        n["canonical"] = int(inverse_label.get(lbl, lbl))

    y_scale = 0.8
    for n in nodes:
        n["y"] = 0.5 + (n["y"] - 0.5) * y_scale

    canonical_ids = [n["canonical"] for n in nodes]
    max_depth = max(int(np.floor(np.log2(cid + 1))) for cid in canonical_ids)
    depth_cmap = plt.get_cmap("Blues", max_depth + 1)
    depth_norm = Normalize(vmin=0, vmax=max_depth)

    for parent, child in edges:
        ax.plot([parent["x"], child["x"]], [parent["y"], child["y"]], color="#aaaaaa", lw=0.9, zorder=0, clip_on=False)

    path_labels = []
    for tok in parsed_path:
        try:
            path_labels.append(int(tok))
        except Exception:
            continue

    pos_by_label = {n["label"]: (n["x"], n["y"]) for n in nodes}
    edge_dir_counts = {}
    for a, b in zip(path_labels, path_labels[1:]):
        edge_dir_counts[(a, b)] = edge_dir_counts.get((a, b), 0) + 1

    if len(path_labels) > 1:
        arrow_color = "#dc2626"
        edge_dir_seen = {}
        for i, (a, b) in enumerate(zip(path_labels, path_labels[1:])):
            if a not in pos_by_label or b not in pos_by_label:
                continue
            x0, y0 = pos_by_label[a]
            x1, y1 = pos_by_label[b]
            dir_dx, dir_dy = x1 - x0, y1 - y0
            dir_dist = np.hypot(dir_dx, dir_dy) or 1.0

            if edge_dir_counts.get((a, b), 0) and edge_dir_counts.get((b, a), 0):
                u, v = sorted((a, b))
                ux, uy = pos_by_label[u]
                vx, vy = pos_by_label[v]
                base_dx, base_dy = vx - ux, vy - uy
                base_dist = np.hypot(base_dx, base_dy) or 1.0
                sign = 1.0 if (a, b) == (u, v) else -1.0
                base_offset = 0.02
                idx = edge_dir_seen.get((a, b), 0)
                offset = base_offset * (idx + 1) * sign
                edge_dir_seen[(a, b)] = idx + 1
                ox = -base_dy / base_dist * offset
                oy = base_dx / base_dist * offset
            else:
                ox = 0.0
                oy = 0.0

            shrink = 0.05
            x0_adj = x0 + dir_dx * shrink + ox
            y0_adj = y0 + dir_dy * shrink + oy
            x1_adj = x1 - dir_dx * shrink + ox
            y1_adj = y1 - dir_dy * shrink + oy

            ax.annotate(
                "",
                xy=(x1_adj, y1_adj),
                xytext=(x0_adj, y0_adj),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.7, alpha=0.85, mutation_scale=22, shrinkA=0, shrinkB=0),
                annotation_clip=False,
                zorder=1,
            )

            label_x = (x0_adj + x1_adj) / 2
            label_y = (y0_adj + y1_adj) / 2
            step_label = "step 1" if i == 0 else f"{i + 1}"
            ax.text(
                label_x,
                label_y,
                step_label,
                color="#c2410c",
                fontsize=9,
                ha="center",
                va="center",
                zorder=2,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.35),
                clip_on=False,
            )

    for n in nodes:
        cid = n["canonical"]
        d = int(np.floor(np.log2(cid + 1)))
        color = depth_cmap(depth_norm(d))
        ax.scatter(n["x"], n["y"], s=320, c=[color], edgecolors="k", linewidths=0.6, zorder=2, clip_on=False)
        text_color = "white" if d >= max_depth - 1 else "black"
        ax.text(n["x"], n["y"], str(cid), fontsize=8, weight="bold", color=text_color, ha="center", va="center", zorder=3, clip_on=False)

    ax.relim()
    ax.autoscale_view()
    ax.margins(0.1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")

    return depth_cmap, depth_norm


def _draw_path_text(ax, parsed_path: list[int] | None, label_mapping: list[int] | None, depth_cmap, depth_norm):
    inverse_label = {int(label): idx for idx, label in enumerate(label_mapping or [])} if label_mapping else {}
    tokens = [str(tok) for tok in (parsed_path or [])]
    if not tokens:
        return
    start_x = 0.34
    y = 0.2
    ax.text(start_x, y, "PATH:", transform=ax.transAxes, ha="right", va="center", fontsize=12, color="black", clip_on=False)
    x = start_x + 0.02
    text_outline = [patheffects.withStroke(linewidth=2.6, foreground="black")]
    for tok in tokens:
        try:
            label_val = int(tok)
            canonical_id = inverse_label.get(label_val, label_val)
            d = int(np.floor(np.log2(int(canonical_id) + 1)))
        except Exception:
            d = 0
        color = depth_cmap(depth_norm(d))
        ax.text(
            x,
            y,
            tok,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=12,
            color=color,
            clip_on=False,
            path_effects=text_outline,
        )
        x += 0.045 * max(1, len(tok)) + 0.01


def _layout_projection_combo(
    encodings: dict,
    results: dict,
    example_id: int,
    responses_path: Path | None,
    meta: dict | None,
    colorcode: str,
):
    data = encodings
    res = results
    X = data["X"]
    depths = data["depth"]
    example_ids = np.asarray(data.get("example_ids", []))
    node_ids = np.asarray(data.get("node_ids", []))
    color_values, cmap, norm, _, _, _ = _prepare_colorcode(colorcode, node_ids, depths, example_ids, responses_path)

    B = _resolve_probe_projection(res)
    fit_geometry = meta.get("fit_geometry", "euclidean") if isinstance(meta, dict) else "euclidean"
    info = {
        "geometry": res.get("geometry", fit_geometry),
        "center": res.get("center"),
        "curvature": res.get("curvature"),
        "normalized_tree": res.get("normalized_tree", meta.get("normalize_tree", False) if isinstance(meta, dict) else False),
    }
    geometry = info.get("geometry", fit_geometry)

    Z = _project_embeddings(X, B, geometry, info, res)
    Z = _project_to_2d(Z)

    mask_example = example_ids == example_id
    Z_example = Z[mask_example]
    depths_example = depths[mask_example]
    colors_example = color_values[mask_example]
    node_ids_example = node_ids[mask_example]

    overall_center = Z_example.mean(axis=0)
    centered = Z_example - overall_center
    unique_depths = np.unique(depths_example)

    def _leveling_cost(theta: float) -> float:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        rotated = centered @ R.T
        cost = 0.0
        for d in unique_depths:
            mask = depths_example == d
            if mask.any():
                y = rotated[mask, 1]
                y_mean = float(y.mean())
                diff = y - y_mean
                cost += float((diff * diff).sum())
        return cost

    angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 361)
    costs = np.array([_leveling_cost(theta) for theta in angles])
    best_theta = float(angles[int(costs.argmin())])

    cos_a, sin_a = np.cos(best_theta), np.sin(best_theta)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    Z_example = centered @ R.T + overall_center

    Z_example[:, 1] = 2 * overall_center[1] - Z_example[:, 1]

    return Z_example, depths_example, colors_example, node_ids_example, cmap, norm


def _path_indices_combo(node_ids_example, parsed_path, label_mapping):
    inverse_label = {int(label): idx for idx, label in enumerate(label_mapping)} if label_mapping else {}
    path_node_ids = []
    for tok in parsed_path:
        try:
            label_val = int(tok)
            canonical_id = inverse_label.get(label_val, label_val)
            path_node_ids.append(int(canonical_id))
        except Exception:
            continue

    path_indices = []
    if path_node_ids:
        node_ids_list = [int(nid) for nid in node_ids_example]
        if node_ids_list[: len(path_node_ids)] == path_node_ids:
            path_indices = list(range(len(path_node_ids)))
        else:
            ptr = 0
            for idx, nid in enumerate(node_ids_list):
                if ptr < len(path_node_ids) and nid == path_node_ids[ptr]:
                    path_indices.append(idx)
                    ptr += 1
    return path_node_ids, path_indices


def _plot_example_projection_combo(
    ax,
    encodings: dict,
    results: dict,
    example_id: int,
    label_mapping=None,
    parsed_path=None,
    *,
    responses_path: Path | None = None,
    meta: dict | None = None,
    colorcode: str = COLORCODE,
    add_labels: bool = False,
    label_size: int = 10,
    node_size: int = 280,
    pad_scale: float = 0.15,
    zoom: float = 1.0,
):
    _ = zoom
    label_mapping = label_mapping or []
    parsed_path = parsed_path or []
    Z_example, depths_example, colors_example, node_ids_example, cmap, norm = _layout_projection_combo(
        encodings, results, example_id, responses_path, meta, colorcode
    )
    path_node_ids, path_indices = _path_indices_combo(node_ids_example, parsed_path, label_mapping)

    for node_id in np.unique(node_ids_example):
        idxs = np.where(node_ids_example == node_id)[0]
        if idxs.size > 1:
            coords = Z_example[idxs]
            line_color = cmap(norm(colors_example[idxs[0]]))
            ax.plot(coords[:, 0], coords[:, 1], linestyle="-", color="black", alpha=0.9, lw=4.2, zorder=0, clip_on=False)
            ax.plot(coords[:, 0], coords[:, 1], linestyle="-", color=line_color, alpha=0.85, lw=3.2, zorder=1, clip_on=False)

    if path_node_ids and len(path_indices) > 1:
        x_range = Z_example[:, 0].max() - Z_example[:, 0].min()
        y_range = Z_example[:, 1].max() - Z_example[:, 1].min()
        avg_range = (x_range + y_range) / 2
        node_radius_estimate = avg_range * 0.022
        arrow_color = "#dc2626"

        for i in range(len(path_indices) - 1):
            idx1, idx2 = path_indices[i], path_indices[i + 1]
            x1, y1 = Z_example[idx1, 0], Z_example[idx1, 1]
            x2, y2 = Z_example[idx2, 0], Z_example[idx2, 1]

            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0:
                shrink = (node_radius_estimate * 1.1)
                shrink_factor = max(0.0, 1.0 - shrink / dist)
                x2_adj = x1 + dx * shrink_factor
                y2_adj = y1 + dy * shrink_factor
                x1_adj = x1 + dx * (shrink / dist) * 0.2
                y1_adj = y1 + dy * (shrink / dist) * 0.2
            else:
                x1_adj, y1_adj = x1, y1
                x2_adj, y2_adj = x2, y2

            ax.annotate(
                "",
                xy=(x2_adj, y2_adj),
                xytext=(x1_adj, y1_adj),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=1.9, alpha=0.85, mutation_scale=38, shrinkA=0, shrinkB=0),
                annotation_clip=False,
                zorder=0,
            )

            label_x = (x1_adj + x2_adj) / 2
            label_y = (y1_adj + y2_adj) / 2
            step_label = "step 1" if i == 0 else f"{i + 1}"
            ax.text(
                label_x,
                label_y,
                step_label,
                color="black",
                fontsize=10,
                ha="center",
                va="center",
                zorder=2,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5),
                clip_on=False,
            )

    ax.scatter(
        Z_example[:, 0],
        Z_example[:, 1],
        c=colors_example,
        cmap=cmap,
        norm=norm,
        edgecolors="k",
        s=node_size,
        marker="o",
        zorder=1,
        clip_on=False,
    )

    if add_labels:
        for (x, y), nid, depth in zip(Z_example, node_ids_example, depths_example):
            text_color = "white" if int(depth) == 2 else "black"
            ax.text(x, y, str(int(nid)), fontsize=label_size, weight="bold", color=text_color, ha="center", va="center", clip_on=False)

    ax.relim()
    ax.autoscale_view()
    ax.margins(pad_scale)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="datalim")


def _plot_probe_geometry_combo(
    encodings: dict,
    results: dict,
    chosen: Sequence[int],
    responses_path: Path,
    meta: dict | None = None,
    colorcode: str = COLORCODE,
) -> plt.Figure:
    if not chosen:
        raise ValueError("No examples available for probe-geometry combo plot.")
    first_example = chosen[0]
    _, _, tree_ascii, _, label_mapping, parsed_path = _extract_tree_and_path(responses_path, first_example)
    fig = plt.figure(figsize=(18, 12.5))
    outer = gridspec.GridSpec(2, 1, height_ratios=[1.45, 1.0], hspace=0.0)

    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[3.0, 1.0], wspace=0.0)
    ax_proj = fig.add_subplot(inner[0, 0])
    ax_tree = fig.add_subplot(inner[0, 1])

    _plot_example_projection_combo(
        ax_proj,
        encodings,
        results,
        first_example,
        label_mapping,
        parsed_path,
        responses_path=responses_path,
        meta=meta,
        colorcode=colorcode,
        add_labels=True,
        label_size=12,
        node_size=1200,
        pad_scale=0.7,
    )

    depth_cmap, depth_norm = _plot_true_tree(ax_tree, tree_ascii, label_mapping, parsed_path)

    ax_proj.text(
        0.55,
        0.7,
        "Model Activations Projected to Hierarchical Subspace",
        transform=ax_proj.transAxes,
        ha="center",
        va="center",
        fontsize=15,
    )
    ax_tree.text(
        0.5,
        0.8,
        "Ground-Truth Tree Traversal",
        transform=ax_tree.transAxes,
        ha="center",
        va="center",
        fontsize=15,
    )

    if depth_cmap is not None:
        _draw_path_text(ax_tree, parsed_path, label_mapping, depth_cmap, depth_norm)

    bottom = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], wspace=0.0, hspace=0.0)
    axes = [fig.add_subplot(bottom[r, c]) for r in range(2) for c in range(3)]
    for ax, ex_id in zip(axes, chosen[1:7]):
        lm, pp = _load_path_info(responses_path, ex_id)
        _plot_example_projection_combo(
            ax,
            encodings,
            results,
            ex_id,
            lm,
            pp,
            responses_path=responses_path,
            meta=meta,
            colorcode=colorcode,
            add_labels=False,
            node_size=480,
            pad_scale=0.9,
        )
    for ax in axes[len(chosen[1:7]) :]:
        ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax_tree_pos = ax_tree.get_position().frozen()
    ax_tree.set_position([ax_tree_pos.x0 - 0.045, ax_tree_pos.y0, ax_tree_pos.width, ax_tree_pos.height])
    return fig

def _load_path_info(responses_path: Path, example_id: int):
    label_mapping = []
    parsed_path = []
    with responses_path.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("example_id") == example_id:
                label_mapping = rec.get("label_mapping") or []
                parsed_path = rec.get("parsed_path") or []
                break
    return label_mapping, parsed_path


def _max_tree_depth(responses_path: Path) -> int:
    max_depth = 1
    with responses_path.open() as f:
        for line in f:
            rec = json.loads(line)
            depth = rec.get("depth")
            if isinstance(depth, (int, float)):
                max_depth = max(max_depth, int(depth))
    return max_depth


def _ensure_layer_encoding(encodings: dict, layer: int, dataset_tag: str, model_id: str, pca_components: int) -> dict:
    data = encodings.get(layer, {})
    if isinstance(data, dict) and "X" in data and "dist" in data:
        return data
    cache_payload = _load_embeddings_cache(dataset_tag, model_id, pca_components)
    if cache_payload is None:
        raise RuntimeError("Embeddings cache missing; cannot rebuild encodings.")
    cache, _ = cache_payload
    example_ids = np.asarray(data.get("example_ids", []), dtype=int)
    node_ids = np.asarray(data.get("node_ids", []), dtype=int)
    depths = np.asarray(data.get("depth", []), dtype=float)
    blocks = []
    idx = 0
    while idx < len(example_ids):
        ex_id = int(example_ids[idx])
        count = 1
        while idx + count < len(example_ids) and example_ids[idx + count] == ex_id:
            count += 1
        rec = cache.get(ex_id)
        if rec is None:
            raise KeyError(f"Missing embedding cache entry for example {ex_id}")
        emb_layer = rec.get("embeddings_by_layer", {}).get(layer)
        if emb_layer is None:
            raise KeyError(f"No embeddings for layer {layer} in example {ex_id}")
        block = np.asarray(emb_layer, dtype=np.float32)[:count]
        if block.shape[0] != count:
            raise ValueError(f"Embedding rows {block.shape[0]} do not match expected count {count} for example {ex_id}")
        blocks.append(block)
        idx += count
    X = np.vstack(blocks) if blocks else np.empty((0, 0), dtype=np.float32)
    n = len(node_ids)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = float(tree_distance(int(node_ids[i]), int(node_ids[j])))
    return {**data, "X": X, "dist": dist, "depth": depths, "node_ids": node_ids}


def _plot_ablation_stats(intervention_path: Path, bucket_exact: bool = True) -> plt.Figure | None:
    if not intervention_path.exists():
        _warn(f"Intervention file missing: {intervention_path}")
        return None
    payload = np.load(intervention_path, allow_pickle=True)
    records = payload["records"].tolist()
    rows = []
    for rec in records:
        base = rec["baseline"]
        probe = rec["probe_ablation"]
        pca = rec.get("pca_ablation", {})
        full_pca = rec.get("full_pca_ablation", {})
        zero = rec.get("zero_ablation", {})
        rand = rec.get("random_ablation", [{}])[0] if rec.get("random_ablation") else {}
        rows.append(
            {
                "baseline_exact": bool(base.get("exact_match", False)),
                "baseline_partial": float(base.get("partial_score", 0.0)),
                "probe_exact": bool(probe.get("exact_match", False)),
                "probe_partial": float(probe.get("partial_score", np.nan)),
                "probe_logit_diff": float(probe.get("logit_diff", np.nan)),
                "pca_exact": bool(pca.get("exact_match", False)),
                "pca_partial": float(pca.get("partial_score", np.nan)),
                "pca_logit_diff": float(pca.get("logit_diff", np.nan)),
                "full_pca_exact": bool(full_pca.get("exact_match", False)),
                "full_pca_partial": float(full_pca.get("partial_score", np.nan)),
                "full_pca_logit_diff": float(full_pca.get("logit_diff", np.nan)),
                "zero_exact": bool(zero.get("exact_match", False)),
                "zero_partial": float(zero.get("partial_score", np.nan)),
                "zero_logit_diff": float(zero.get("logit_diff", np.nan)),
                "rand_exact": bool(rand.get("exact_match", False)),
                "rand_partial": float(rand.get("partial_score", np.nan)),
                "rand_logit_diff": float(rand.get("logit_diff", np.nan)),
            }
        )
    df = pd.DataFrame(rows)
    if bucket_exact:
        df = df[df["baseline_exact"]].copy()
        if df.empty:
            _warn("No exact examples for ablation stats.")
            return None

    baseline_exact_rate = df["baseline_exact"].mean()
    baseline_partial_mean = df["baseline_partial"].mean()

    stats = {
        "probe_partial_delta": df["probe_partial"].mean() - baseline_partial_mean,
        "pca_partial_delta": df["pca_partial"].mean() - baseline_partial_mean,
        "full_pca_partial_delta": df["full_pca_partial"].mean() - baseline_partial_mean,
        "zero_partial_delta": df["zero_partial"].mean() - baseline_partial_mean,
        "rand_partial_delta": df["rand_partial"].mean() - baseline_partial_mean,
    }

    base_font = _resolve_font_size(rcParams.get("font.size", rcParamsDefault.get("font.size", 10.0)),
                                   float(rcParamsDefault.get("font.size", 10.0)))
    with plt.rc_context(_scaled_text_params(1.5)):
        fig, (ax, ax_hist) = plt.subplots(
            1,
            2,
            figsize=(12, 4),
            gridspec_kw={"width_ratios": [1, 3]},
        )
        bars = [
            (stats["probe_partial_delta"], "Probe", ABLATION_COLORS["probe"]),
            (stats["rand_partial_delta"], "Random", ABLATION_COLORS["random"]),
            (stats["full_pca_partial_delta"], "CoT PCs", ABLATION_COLORS["full_pca"]),
            (stats["pca_partial_delta"], "Tree PCs", ABLATION_COLORS["pca"]),
            (stats["zero_partial_delta"], "Full Space", ABLATION_COLORS["zero"]),
        ]
        bars = [b for b in bars if not np.isnan(b[0])]
        x = np.arange(1)
        width = 0.18
        offsets = np.linspace(-width * (len(bars) - 1) / 2, width * (len(bars) - 1) / 2, len(bars))
        for offset, (val, label, color) in zip(offsets, bars):
            ax.bar(x + offset, val, width, label=label, color=color, alpha=0.6)
        ax.axhline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xticks([])
        ax.set_ylabel("Accuracy Delta")
        ax.legend(loc="lower left", fontsize=base_font)
        # Logit distributions
        probe_vals = df["probe_logit_diff"].dropna()
        pca_vals = df["pca_logit_diff"].dropna()
        full_pca_vals = df["full_pca_logit_diff"].dropna()
        zero_vals = df["zero_logit_diff"].dropna()
        rand_vals = df["rand_logit_diff"].dropna()
        series = [probe_vals, pca_vals, zero_vals, rand_vals]
        if not full_pca_vals.empty:
            series.append(full_pca_vals)
        combined = pd.concat(series)
        if combined.empty:
            _warn("No logit differences available for ablation histogram.")
            fig.tight_layout()
            return fig
        bins = np.histogram_bin_edges(combined, bins=40)
        ax_hist.hist(probe_vals, bins=bins, alpha=0.45, label="Probe", color=ABLATION_COLORS["probe"])
        ax_hist.hist(
            rand_vals,
            bins=bins,
            histtype="step",
            linewidth=2,
            label="Random",
            color=ABLATION_COLORS["random"],
        )
        if not full_pca_vals.empty:
            ax_hist.hist(
                full_pca_vals,
                bins=bins,
                histtype="step",
                linewidth=2,
                label="CoT PCs",
                color=ABLATION_COLORS["full_pca"],
            )
        ax_hist.hist(
            pca_vals,
            bins=bins,
            histtype="step",
            linewidth=2,
            label="Tree PCs",
            color=ABLATION_COLORS["pca"],
        )
        ax_hist.hist(
            zero_vals,
            bins=bins,
            histtype="step",
            linewidth=2,
            label="Full Space",
            color=ABLATION_COLORS["zero"],
        )
        ax_hist.set_xlabel("Logit Difference")
        ax_hist.set_ylabel("Count")

        y_max = ax_hist.get_ylim()[1]
        label_specs = [
            ("Probe", probe_vals, ABLATION_COLORS["probe"]),
            ("Random", rand_vals, ABLATION_COLORS["random"]),
            ("CoT PCs", full_pca_vals, ABLATION_COLORS["full_pca"]),
            ("Tree PCs", pca_vals, ABLATION_COLORS["pca"]),
            ("Full Space", zero_vals, ABLATION_COLORS["zero"]),
        ]
        label_specs = [(lbl, vals, color) for lbl, vals, color in label_specs if len(vals) > 0]
        for idx, (label, vals, color) in enumerate(label_specs):
            median_val = float(np.median(vals))
            ax_hist.vlines(median_val, 0, y_max, colors=color, linestyles=":", linewidth=2, alpha=0.9)
            y_text = y_max * (0.98 - 0.06 * idx)
            ax_hist.text(
                median_val,
                y_text,
                label,
                color=color,
                ha="center",
                va="bottom",
                fontsize=base_font,
            )

        fig.tight_layout()
        return fig


def _plot_layersweep_logit_shifts(layer_sweep_path: Path) -> plt.Figure | None:
    if not layer_sweep_path.exists():
        _warn(f"Layer-sweep file missing: {layer_sweep_path}")
        return None
    npz_layer = np.load(layer_sweep_path, allow_pickle=True)
    records = npz_layer["records"].tolist()
    layers = []
    probe_means = []
    pca_means = []
    full_pca_means = []
    zero_means = []
    random_means = []

    for entry in records:
        layer = int(entry.get("layer"))
        layer_records = entry.get("records", [])
        rows = []
        for rec in layer_records:
            base = rec["baseline"]
            if not base.get("exact_match", False):
                continue
            probe = rec["probe_ablation"]
            pca = rec.get("pca_ablation", {})
            full_pca = rec.get("full_pca_ablation", {})
            zero = rec.get("zero_ablation", {})
            random_runs = rec.get("random_ablation", [])
            if random_runs:
                rand_logit_diffs = [float(r.get("logit_diff", np.nan)) for r in random_runs]
                rand_logit_mean = np.nanmean(rand_logit_diffs) if rand_logit_diffs else np.nan
            else:
                rand_logit_mean = np.nan
            rows.append(
                {
                    "probe_logit_diff": float(probe.get("logit_diff", np.nan)),
                    "pca_logit_diff": float(pca.get("logit_diff", np.nan)),
                    "full_pca_logit_diff": float(full_pca.get("logit_diff", np.nan)),
                    "zero_logit_diff": float(zero.get("logit_diff", np.nan)),
                    "rand_logit_diff": rand_logit_mean,
                }
            )
        if not rows:
            continue
        df_layer = pd.DataFrame(rows)
        layers.append(layer)
        probe_means.append(float(np.nanmean(df_layer["probe_logit_diff"])))
        pca_means.append(float(np.nanmean(df_layer["pca_logit_diff"])))
        full_pca_means.append(float(np.nanmean(df_layer["full_pca_logit_diff"])))
        zero_means.append(float(np.nanmean(df_layer["zero_logit_diff"])))
        random_means.append(float(np.nanmean(df_layer["rand_logit_diff"])))

    if not layers:
        _warn("No layer-sweep logit data found.")
        return None
    order = np.argsort(layers)
    layers = list(np.array(layers)[order])
    probe_means = list(np.array(probe_means)[order])
    pca_means = list(np.array(pca_means)[order])
    full_pca_means = list(np.array(full_pca_means)[order])
    zero_means = list(np.array(zero_means)[order])
    random_means = list(np.array(random_means)[order])

    with plt.rc_context(_scaled_text_params(1.5)):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(layers, probe_means, label="Probe ablation", color=ABLATION_COLORS["probe"], linewidth=2)
        ax.plot(layers, random_means, label="Random ablation", color=ABLATION_COLORS["random"], linewidth=2)
        if any(not np.isnan(x) for x in full_pca_means):
            ax.plot(layers, full_pca_means, label="CoT PCs ablation", color=ABLATION_COLORS["full_pca"], linewidth=2)
        if any(not np.isnan(x) for x in pca_means):
            ax.plot(layers, pca_means, label="Tree PCs ablation", color=ABLATION_COLORS["pca"], linewidth=2)
        if any(not np.isnan(x) for x in zero_means):
            ax.plot(layers, zero_means, label="Full Space ablation", color=ABLATION_COLORS["zero"], linewidth=2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Logit Difference")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig


def _principal_angle_cosines(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    Qa, _ = np.linalg.qr(basis_a)
    Qb, _ = np.linalg.qr(basis_b)
    singular_vals = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(singular_vals, -1.0, 1.0)


def _mean_principal_angle_cosine(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    cosines = _principal_angle_cosines(basis_a, basis_b)
    return float(np.mean(cosines))


def _resolve_depth_probe_weight(res: dict) -> np.ndarray:
    from sklearn.pipeline import Pipeline

    depth_model = res.get("depth_model")
    if depth_model is None:
        depth_block = res.get("depth", {})
        if isinstance(depth_block, dict):
            depth_model = depth_block.get("model")
    if depth_model is None or not isinstance(depth_model, Pipeline):
        raise ValueError("No depth probe model available.")
    ridge = depth_model.named_steps.get("reg")
    coef = np.asarray(ridge.coef_, dtype=np.float32)
    scaler = depth_model.named_steps.get("scale")
    if scaler is not None:
        std = np.asarray(scaler.scale_, dtype=np.float32)
        std = np.where(std > 1e-10, std, 1.0)
        coef = coef / std
    pca_info = res.get("pca")
    if pca_info is not None and pca_info.get("components") is not None:
        components = np.asarray(pca_info["components"], dtype=np.float32)
        coef = components.T @ coef
    norm = np.linalg.norm(coef)
    if norm > 0:
        coef = coef / norm
    return coef.reshape(-1, 1)


def _plot_probe_similarities(
    responses: Sequence[Any],
    cache: Dict[int, Any],
    layer: int,
    proj_dim: int,
    pca_components: int,
    lr: float,
    weight_decay: float,
    steps: int,
    fit_geometry: str,
    depth_alpha: float,
    seed: int,
    num_splits: int,
) -> plt.Figure | None:
    if not responses:
        _warn("No responses available for probe similarities.")
        return None

    def make_balanced_subsets(records: Sequence[Any], num_splits: int, seed: int) -> List[List[Any]]:
        rng = np.random.default_rng(seed)
        exact_records = [rec for rec in records if bool(getattr(rec, "exact_match", False))]
        inexact_records = [rec for rec in records if not bool(getattr(rec, "exact_match", False))]
        rng.shuffle(exact_records)
        rng.shuffle(inexact_records)
        exact_splits = np.array_split(exact_records, num_splits)
        inexact_splits = np.array_split(inexact_records, num_splits)
        subsets: List[List[Any]] = []
        for idx in range(num_splits):
            subset = list(exact_splits[idx]) + list(inexact_splits[idx])
            rng.shuffle(subset)
            subsets.append(subset)
        return subsets

    def build_embeddings_from_cache(train_records: Sequence[Any]) -> Dict[int, Dict[str, Any]]:
        encodings: Dict[int, Dict[str, Any]] = {}
        dist_cache = None
        dist_cache_nodes = None
        data = build_layer_from_cache(train_records, [], layer, cache)
        if data is None:
            return encodings
        node_ids = np.asarray(data.get("node_ids", []), dtype=np.int64)
        if dist_cache is not None and dist_cache_nodes is not None and np.array_equal(node_ids, dist_cache_nodes):
            data["dist"] = dist_cache
        else:
            data["dist"] = pairwise_tree_distances(node_ids)
            dist_cache = data["dist"]
            dist_cache_nodes = node_ids
        data["D"] = data["dist"]
        encodings[layer] = data
        return encodings

    def apply_pca(encodings: Dict[int, Dict[str, Any]], n_components: int, seed: int):
        if n_components < 0:
            return encodings, {}
        if n_components == 0:
            raise ValueError("pca_components must be -1 or a positive integer.")
        projected = {}
        pca_info = {}
        for lyr, data in encodings.items():
            X = np.asarray(data["X"], dtype=np.float32)
            train_idx = np.asarray(data["train_idx"], dtype=int)
            if train_idx.size == 0:
                continue
            max_components = min(X.shape[0], X.shape[1])
            effective = min(n_components, max_components)
            if effective >= X.shape[1]:
                projected[lyr] = data
                continue
            pca = PCA(n_components=effective, svd_solver="auto", random_state=seed)
            pca.fit(X[train_idx])
            X_proj = pca.transform(X).astype(np.float32)
            updated = dict(data)
            updated["X"] = X_proj
            projected[lyr] = updated
            pca_info[lyr] = {
                "components": pca.components_.astype(np.float32),
                "mean": pca.mean_.astype(np.float32),
                "n_components": int(effective),
                "n_features": int(X.shape[1]),
            }
        return projected, pca_info

    def train_probe_for_subset(split_idx: int, subset: Sequence[Any]):
        if not subset:
            return None
        encodings = build_embeddings_from_cache(subset)
        if layer not in encodings:
            return None
        cfg = DistanceProbeConfig(
            proj_dim=proj_dim,
            lr=lr,
            weight_decay=weight_decay,
            steps=steps,
            seed=int(seed + split_idx),
            fit_geometry=fit_geometry,
        )
        encodings, pca_info = apply_pca(encodings, pca_components, int(seed + split_idx))
        results: Dict[int, Dict[str, Any]] = {}
        for lyr, data in encodings.items():
            res_layer = evaluate_probes({"layer": data}, cfg, device=DEVICE, normalize_tree=None, depth_alpha=depth_alpha)["layer"]
            packaged = {
                "distance": {
                    "train": {"pearson": res_layer.get("dist_corr_train"), "mse": res_layer.get("dist_mse_train")},
                    "test": {"pearson": res_layer.get("dist_corr_test"), "mse": res_layer.get("dist_mse_test")},
                },
                "depth": res_layer.get("depth", {}),
                "projection": res_layer.get("projection"),
                "geometry": res_layer.get("geometry"),
                "center": res_layer.get("center"),
                "curvature": res_layer.get("curvature"),
                "normalized_tree": res_layer.get("normalized_tree"),
                "depth_model": res_layer.get("depth_model"),
            }
            if lyr in pca_info:
                packaged["pca"] = pca_info[lyr]
                if res_layer.get("projection") is not None:
                    packaged["projection_full"] = pca_info[lyr]["components"].T @ res_layer["projection"]
            results[lyr] = packaged
        return results

    subsets = make_balanced_subsets(responses, num_splits, seed)
    depth_runs: Dict[int, Dict[str, Any]] = {}
    for split_idx, subset in enumerate(subsets):
        results = train_probe_for_subset(split_idx, subset)
        if results is None:
            continue
        depth_runs[split_idx] = {"results": results}

    keys = [k for k in sorted(depth_runs.keys()) if layer in depth_runs[k]["results"]]
    if len(keys) < 2:
        _warn("Not enough splits for probe similarities.")
        return None

    dist_matrix = np.eye(len(keys), dtype=np.float32)
    depth_matrix = np.eye(len(keys), dtype=np.float32)
    for i, key_i in enumerate(keys):
        res_i = depth_runs[key_i]["results"][layer]
        B_i = _resolve_probe_projection(res_i)
        try:
            weight_i = _resolve_depth_probe_weight(res_i)
        except Exception:
            weight_i = None
        for j, key_j in enumerate(keys):
            if j <= i:
                if j < i:
                    dist_matrix[i, j] = dist_matrix[j, i]
                    depth_matrix[i, j] = depth_matrix[j, i]
                continue
            res_j = depth_runs[key_j]["results"][layer]
            B_j = _resolve_probe_projection(res_j)
            dist_matrix[i, j] = dist_matrix[j, i] = _mean_principal_angle_cosine(B_i, B_j)
            if weight_i is not None:
                try:
                    weight_j = _resolve_depth_probe_weight(res_j)
                    depth_matrix[i, j] = depth_matrix[j, i] = float(np.dot(weight_i.flatten(), weight_j.flatten()))
                except Exception:
                    depth_matrix[i, j] = depth_matrix[j, i] = np.nan

    with plt.rc_context(_scaled_text_params(1.5)):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axis_label = "Split"
        axes[0].grid(False)
        axes[1].grid(False)
        im_dist = axes[0].imshow(dist_matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        axes[0].set_xticks(range(len(keys)))
        axes[0].set_xticklabels(keys)
        axes[0].set_yticks(range(len(keys)))
        axes[0].set_yticklabels(keys)
        axes[0].set_xlabel(axis_label)
        axes[0].set_ylabel(axis_label)
        axes[0].set_title("Distance")
        cbar_dist = fig.colorbar(im_dist, ax=axes[0], fraction=0.046, pad=0.04)
        cbar_dist.set_label("Mean cosine")

        im_depth = axes[1].imshow(depth_matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        axes[1].set_xticks(range(len(keys)))
        axes[1].set_xticklabels(keys)
        axes[1].set_yticks(range(len(keys)))
        axes[1].set_yticklabels(keys)
        axes[1].set_xlabel(axis_label)
        axes[1].set_ylabel(axis_label)
        axes[1].set_title("Depth")
        cbar_depth = fig.colorbar(im_depth, ax=axes[1], fraction=0.046, pad=0.04)
        cbar_depth.set_label("Cosine")
        fig.tight_layout()
        return fig


def _find_layer_sweep_path(dataset_tag: str, model_id: str, proj_dim: int, pca_components: int) -> Path | None:
    base_dir = path_utils.intervention_output_dir(dataset_tag, model_id)
    patterns = [
        f"interventions_proj{proj_dim}_pca{pca_components}_layerall_answeronly.npz",
        f"interventions_proj{proj_dim}_pca{pca_components}_layerall*.npz",
    ]
    for pattern in patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate visualization figures from probe/intervention artifacts.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=path_utils.DEFAULT_TREE_DATASET_TAG,
        help="Dataset tag to load artifacts from.",
    )
    parser.add_argument(
        "--reasoning-models",
        nargs="+",
        default=["7B"],
        help="Reasoning model sizes (e.g., 7B). Use 'none' to skip.",
    )
    parser.add_argument(
        "--chat-models",
        nargs="+",
        default=["7B"],
        help="Chat model sizes (e.g., 7B). Use 'none' to skip.",
    )
    parser.add_argument(
        "--figs",
        nargs="+",
        default=["all"],
        help="Figure groups to generate: geometry, ablations, performance, or all.",
    )
    parser.add_argument("--proj-dim", type=int, default=5, help="Projection dimension used in probe filenames.")
    parser.add_argument("--pca-components", type=int, default=10, help="PCA components used in probe filenames.")
    parser.add_argument("--normalize-tree", action="store_true", help="Use normalized tree distance probe filenames.")
    parser.add_argument("--depth-filter", type=int, default=2, help="Depth filter for probe-geometry examples.")
    parser.add_argument("--min-path-len", type=int, default=8, help="Minimum path length for probe-geometry examples.")
    parser.add_argument("--num-examples", type=int, default=8, help="Number of examples to select for probe-geometry.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for similarity splits.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    fig_requests = [fig.lower() for fig in args.figs]
    if "all" in set(fig_requests):
        fig_requests = ["performance", "geometry", "ablations"]

    category_map = {
        "geometry": ["probe-geometry"] + [f"example_{i}" for i in range(max(0, args.num_examples))],
        "ablations": ["ablation-statistics", "ablation-layersweep"],
        "performance": ["layerwise-statistics", "probe-similarities"],
    }
    known_figs = {name for names in category_map.values() for name in names}

    expanded: List[str] = []
    for fig in fig_requests:
        if fig in category_map:
            expanded.extend(category_map[fig])
        elif fig in known_figs or fig.startswith("example_"):
            expanded.append(fig)
        else:
            _warn(f"Unknown fig group '{fig}'")
    # Preserve order but drop duplicates.
    seen = set()
    ordered_figs: List[str] = []
    for fig in expanded:
        key = fig.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered_figs.append(fig)

    model_pairs = resolve_model_pairs(args.reasoning_models, args.chat_models)
    if not model_pairs:
        raise ValueError("No models selected. Provide --reasoning-models or --chat-models.")

    for family, size_token, model_id in model_pairs:
        model_tag = path_utils.model_tag(model_id)
        layer = _resolve_layer(family, size_token)
        fig_base_dir = PROJECT_ROOT / "cutter" / "figures" / "paper" / args.dataset / model_tag
        fig_base_dir.mkdir(parents=True, exist_ok=True)

        needs_probe = any(fig in ordered_figs for fig in ["layerwise-statistics", "probe-geometry"]) or any(
            fig.startswith("example_") for fig in ordered_figs
        )
        probe_artifact = None
        if needs_probe:
            probe_artifact = _load_probe_artifact(
                args.dataset, model_id, args.proj_dim, args.pca_components, args.normalize_tree
            )
        if probe_artifact is None and needs_probe:
            _warn(f"Skipping probe-based figures for {model_tag} (probe artifact missing).")

        if probe_artifact:
            meta, encodings, results = probe_artifact

        # Precompute probe-geometry example selection once per model so example_{i} matches notebook ordering.
        chosen_examples: List[int] | None = None
        responses_path = path_utils.responses_path(args.dataset, model_id)
        if probe_artifact and (
            "probe-geometry" in ordered_figs or any(fig.startswith("example_") for fig in ordered_figs)
        ):
            if not responses_path.exists():
                _warn(f"Responses missing: {responses_path}")
            else:
                try:
                    enc_layer = _ensure_layer_encoding(encodings, layer, args.dataset, model_id, args.pca_components)
                    encodings[layer] = enc_layer
                    embedding_payload = _load_embeddings_cache(args.dataset, model_id, args.pca_components, meta)
                    if embedding_payload is not None:
                        embeddings_cache, _ = embedding_payload
                        candidates = _collect_test_exact_ids(
                            encodings,
                            responses_path,
                            embeddings_cache,
                            layer,
                            args.depth_filter,
                            args.min_path_len,
                        )
                        if candidates:
                            normalization_factor = _max_tree_depth(responses_path) if meta.get("normalize_tree") else None
                            ranked = sorted(
                                candidates,
                                key=lambda ex_id: _example_distance_mse(
                                    encodings,
                                    results,
                                    ex_id,
                                    layer,
                                    bool(meta.get("normalize_tree")),
                                    normalization_factor,
                                    meta,
                                ),
                            )
                            chosen_examples = ranked[: min(args.num_examples, len(ranked))]
                        else:
                            _warn("No candidates found for probe-geometry/example plots.")
                except Exception as exc:
                    _warn(f"Failed to prepare probe-geometry/example selection: {exc}")

        for fig in ordered_figs:
            try:
                if fig == "layerwise-statistics":
                    if not probe_artifact:
                        continue
                    fig_obj = _plot_layerwise_statistics(results)
                    out_path = _fig_path(args.dataset, model_tag, fig)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_obj.savefig(out_path, dpi=300)
                    plt.close(fig_obj)
                    print(f"[visualize] Saved {out_path}")
                elif fig == "probe-geometry":
                    if not probe_artifact:
                        continue
                    if chosen_examples is None:
                        _warn("Probe-geometry selection unavailable; skipping.")
                        continue
                    fig_obj = _plot_probe_geometry_combo(
                        encodings[layer],
                        results[layer],
                        chosen_examples,
                        responses_path,
                        meta=meta,
                        colorcode=COLORCODE,
                    )
                    out_path = _fig_path(args.dataset, model_tag, fig)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_obj.savefig(out_path, dpi=300)
                    plt.close(fig_obj)
                    print(f"[visualize] Saved {out_path}")
                elif fig.startswith("example_"):
                    if not probe_artifact:
                        continue
                    if chosen_examples is None:
                        _warn("Example selection unavailable; skipping.")
                        continue
                    try:
                        idx = int(fig.split("_", 1)[1])
                    except Exception:
                        _warn(f"Invalid example index in fig name '{fig}'")
                        continue
                    if idx >= len(chosen_examples):
                        _warn(f"Example index {idx} out of range (only {len(chosen_examples)} available).")
                        continue
                    ex_id = chosen_examples[idx]
                    prompt, model_raw, tree_ascii, path_line, label_mapping, parsed_path = _extract_tree_and_path(
                        responses_path, ex_id
                    )
                    fig_obj = _plot_test_with_example(
                        encodings[layer],
                        results[layer],
                        ex_id,
                        prompt,
                        model_raw,
                        path_line,
                        tree_ascii,
                        label_mapping,
                        parsed_path,
                        responses_path=responses_path,
                        meta=meta,
                        colorcode=COLORCODE,
                    )
                    out_path = _fig_path(args.dataset, model_tag, fig)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_obj.savefig(out_path, dpi=300)
                    plt.close(fig_obj)
                    print(f"[visualize] Saved {out_path}")
                elif fig == "probe-similarities":
                    responses_path = path_utils.responses_path(args.dataset, model_id)
                    if not responses_path.exists():
                        _warn(f"Responses missing: {responses_path}")
                        continue
                    responses = load_responses(responses_path)
                    embedding_payload = _load_embeddings_cache(
                        args.dataset,
                        model_id,
                        args.pca_components,
                        meta if probe_artifact else None,
                    )
                    if embedding_payload is None:
                        continue
                    cache, _ = embedding_payload
                    # Try to inherit optimizer params from probe artifact if present
                    lr = 1e-2
                    weight_decay = 1e-4
                    steps = 1500
                    fit_geometry = "euclidean"
                    depth_alpha = 1e-2
                    if probe_artifact:
                        meta = probe_artifact[0]
                        lr = float(meta.get("lr", lr))
                        weight_decay = float(meta.get("weight_decay", weight_decay))
                        steps = int(meta.get("steps", steps))
                        fit_geometry = str(meta.get("fit_geometry", fit_geometry))
                        depth_alpha = float(meta.get("depth_alpha", depth_alpha))
                        pca_components = int(meta.get("pca_components", args.pca_components))
                    else:
                        pca_components = args.pca_components
                    fig_obj = _plot_probe_similarities(
                        responses,
                        cache,
                        layer,
                        args.proj_dim,
                        pca_components,
                        lr,
                        weight_decay,
                        steps,
                        fit_geometry,
                        depth_alpha,
                        args.seed,
                        num_splits=5,
                    )
                    if fig_obj is None:
                        continue
                    out_path = _fig_path(args.dataset, model_tag, fig)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig_obj.savefig(out_path, dpi=300)
                    plt.close(fig_obj)
                    print(f"[visualize] Saved {out_path}")
                elif fig == "ablation-statistics":
                    intervention_fp = path_utils.intervention_path(args.dataset, model_id, args.proj_dim, args.pca_components, layer)
                    fig_obj = _plot_ablation_stats(intervention_fp, bucket_exact=True)
                    if fig_obj is None:
                        continue
                    out_path = _fig_path(args.dataset, model_tag, fig)
                    fig_obj.savefig(out_path, dpi=300)
                    plt.close(fig_obj)
                    print(f"[visualize] Saved {out_path}")
                elif fig == "ablation-layersweep":
                    layer_sweep_path = _find_layer_sweep_path(args.dataset, model_id, args.proj_dim, args.pca_components)
                    if layer_sweep_path is None:
                        _warn("Layer sweep file missing; skipping.")
                        continue
                    fig_obj = _plot_layersweep_logit_shifts(layer_sweep_path)
                    if fig_obj is None:
                        continue
                    out_path = _fig_path(args.dataset, model_tag, fig)
                    fig_obj.savefig(out_path, dpi=300)
                    plt.close(fig_obj)
                    print(f"[visualize] Saved {out_path}")
                else:
                    _warn(f"Unknown fig option '{fig}'")
            except Exception as exc:
                _warn(f"Failed to generate {fig} for {model_tag}: {exc}")
                continue


if __name__ == "__main__":
    main()
