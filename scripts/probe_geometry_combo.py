import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib import gridspec, patheffects as pe
import re

# Parameters (mirrors cutter/notebooks/probe-geometry.ipynb)
NORMALIZE = False
DEPTH_RANGE = [1, 2]
STEPS_RANGE = [1, 2]
NUM_SAMPLES = 1000


def _parse_range(values, name: str, min_value: int = 1) -> tuple[int, int]:
    if isinstance(values, (int, float)):
        values = [int(values)]
    if len(values) == 1:
        range_min = range_max = int(values[0])
    elif len(values) == 2:
        range_min, range_max = [int(v) for v in values]
    else:
        raise ValueError(f"{name} must have 1 or 2 integers")
    if range_min < min_value or range_max < min_value or range_max < range_min:
        raise ValueError(f"{name} must be >= {min_value} and max >= min")
    return range_min, range_max


def _format_steps_tag(num_steps) -> str:
    if isinstance(num_steps, (list, tuple)):
        if len(num_steps) != 2:
            raise ValueError("num_steps must have 1 or 2 integers")
        min_steps, max_steps = num_steps
    else:
        min_steps = max_steps = int(num_steps)
    if min_steps == max_steps == 1:
        return ""
    if min_steps == max_steps:
        return f"_steps{min_steps}"
    return f"_steps{min_steps}-{max_steps}"


def _dataset_tag(min_depth: int, max_depth: int, num_samples: int, num_steps) -> str:
    suffix = f"n{num_samples}" if num_samples > 0 else "nall"
    steps = _format_steps_tag(num_steps)
    return f"depth{min_depth}-{max_depth}_{suffix}{steps}"


MIN_DEPTH, MAX_DEPTH = _parse_range(DEPTH_RANGE, "depth-range", min_value=1)
MIN_STEPS, MAX_STEPS = _parse_range(STEPS_RANGE, "steps-range", min_value=1)
STEPS_TAG = (MIN_STEPS, MAX_STEPS) if MIN_STEPS != MAX_STEPS else MIN_STEPS
DATASET_TAG = _dataset_tag(MIN_DEPTH, MAX_DEPTH, NUM_SAMPLES, STEPS_TAG)
MODEL_TAG = "DeepSeek-R1-Distill-Qwen-14B"
PROJ_DIM = 5
PCA_COMPONENTS = 10
PCA_TRAIN_SPLIT = 0.5
PCA_SEED = 0
COLORCODE = "depth"
LAYER = 31
DEPTH_FILTER = 2
MIN_PATH_LEN = 8

NOTEBOOK_DIR = Path.cwd().resolve()
for candidate in [NOTEBOOK_DIR, *NOTEBOOK_DIR.parents]:
    if (candidate / "cutter").exists():
        PROJECT_ROOT = candidate
        break
else:
    raise RuntimeError("Could not locate repository root containing cutter/")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

norm_suffix = "_normtree" if NORMALIZE else ""
RESULTS_PATH = PROJECT_ROOT / f"cutter/data/{DATASET_TAG}/models/{MODEL_TAG}/probes/probe{norm_suffix}_proj{PROJ_DIM}_pca{PCA_COMPONENTS}.npz"

from cutter.utils.tree.probing import pairwise_distance, transform_probe_space
from cutter.scripts.evaluate_probe import load_embedding_cache
from cutter.utils.tree.trees import tree_distance
from cutter.utils.shared.embeddings_cache import ensure_pca_cache
from cutter.utils.shared.paths import embeddings_path as default_embeddings_path

if not RESULTS_PATH.exists():
    raise FileNotFoundError(f"Probe results not found: {RESULTS_PATH}")
artifact = np.load(RESULTS_PATH, allow_pickle=True)
meta = artifact["meta"].item()
encodings = artifact["encodings"].item()
results = artifact["results"].item()


def _pca_transform(features: np.ndarray, res: dict) -> np.ndarray:
    pca = res.get("pca") if isinstance(res, dict) else None
    if not pca:
        return features
    components = np.asarray(pca.get("components", []), dtype=np.float32)
    mean = np.asarray(pca.get("mean", 0.0), dtype=np.float32)
    if components.size == 0:
        return features
    if features.shape[1] == components.shape[0]:
        return features
    return (features - mean) @ components.T


def _resolve_probe_projection(res: dict) -> np.ndarray:
    if isinstance(res, dict) and isinstance(res.get("distance"), dict) and res["distance"].get("projection") is not None:
        return res["distance"]["projection"]
    if isinstance(res, dict) and res.get("projection") is not None:
        return res["projection"]
    raise ValueError("No projection found in results for the requested layer.")


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


def _locate_embeddings(meta: dict, results_path: Path, project_root: Path) -> Path:
    meta_path = meta.get("embeddings_path")
    if meta_path:
        p = Path(meta_path)
        if p.exists():
            return p
        if not p.is_absolute():
            p = project_root / p
            if p.exists():
                return p

    dataset_tag = meta.get("dataset_tag")
    model_id = meta.get("model")
    if dataset_tag and model_id and PCA_COMPONENTS >= 0:
        try:
            pca_components = int(meta.get("pca_components", PCA_COMPONENTS))
            train_split = float(meta.get("pca_train_split", PCA_TRAIN_SPLIT))
            seed = int(meta.get("pca_seed", PCA_SEED))
            split_type = str(meta.get("pca_split_type", "exact-only"))
            pca_path = ensure_pca_cache(
                dataset_tag,
                model_id,
                pca_components,
                train_split,
                seed,
                split_type,
            )
            if pca_path is not None and Path(pca_path).exists():
                return Path(pca_path)
        except Exception:
            pass

    candidates = [results_path.parent.parent / "embeddings.npz"]
    if dataset_tag and model_id:
        try:
            candidates.append(default_embeddings_path(dataset_tag, model_id))
        except Exception:
            pass
    for cand in candidates:
        if cand and Path(cand).exists():
            return Path(cand)
    return Path()


def _rebuild_layer_encoding(layer: int, data: dict, meta: dict, results_path: Path, project_root: Path) -> dict:
    if isinstance(data, dict) and "X" in data and "dist" in data:
        return data

    emb_path = _locate_embeddings(meta, results_path, project_root)
    cache = load_embedding_cache(emb_path) if emb_path.exists() else None
    if cache is None:
        raise RuntimeError(f"Cannot rebuild encodings without embeddings cache. Tried: {emb_path}")

    example_ids = np.asarray(data.get("example_ids", []), dtype=int)
    node_ids = np.asarray(data.get("node_ids", []), dtype=int)
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
        emb_layer = rec.get("embeddings_by_layer", {}).get(layer) if isinstance(rec, dict) else getattr(rec, "embeddings_by_layer", {}).get(layer)
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
    return {**data, "X": X, "dist": dist}


layer_data = encodings.get(LAYER, {})
needs_rebuild = not (isinstance(layer_data, dict) and "X" in layer_data and "dist" in layer_data)
if needs_rebuild:
    encodings[LAYER] = _rebuild_layer_encoding(LAYER, layer_data, meta, RESULTS_PATH, PROJECT_ROOT)

RESPONSES_PATH = PROJECT_ROOT / f"cutter/data/{DATASET_TAG}/models/{MODEL_TAG}/responses.jsonl"

SUBTREE_CMAP = ListedColormap(["#1f77b4", "#d62728", "#7f7f7f"])
SUBTREE_BOUNDARIES = [-0.5, 0.5, 1.5, 2.5]
SUBTREE_TICKS = [0, 1, 2]
SUBTREE_LABELS = ["Left", "Right", "Root"]


def _root_child_label(node_id: int) -> str:
    if node_id == 0:
        return "root"
    child = node_id
    parent = (child - 1) // 2
    while parent > 0:
        child = parent
        parent = (parent - 1) // 2
    return "left" if child == 1 else "right"


def prepare_colorcode(colorcode: str, node_ids: np.ndarray, depths: np.ndarray, example_ids: np.ndarray):
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


def project_embeddings(arr: np.ndarray, B: np.ndarray, geometry: str, info: Dict[str, Any], res: dict) -> np.ndarray:
    features = _pca_transform(arr.astype(np.float32), res)
    proj = features @ B
    if geometry == "hyperbolic":
        proj = transform_probe_space(proj, info)
    return proj


def collect_test_exact_ids(layer: int, depth_filter: int = DEPTH_FILTER, min_path_len: int = MIN_PATH_LEN):
    data = encodings[layer]
    example_ids = np.asarray(data.get("example_ids", []))
    if example_ids.size == 0:
        raise ValueError("Encodings missing example_ids; cannot map to embeddings")
    train_idx = set(data.get("train_idx", []))
    splits = np.array(["train" if i in train_idx else "test" for i in range(len(example_ids))])
    per_example_split = {}
    for idx, ex_id in enumerate(example_ids):
        per_example_split.setdefault(ex_id, splits[idx])
    test_ids = {ex for ex, sp in per_example_split.items() if sp == "test"}

    exact_ids = set()
    depth_ok = set()
    path_len_ok = set()
    with open(RESPONSES_PATH) as f:
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

    embedding_records = np.load(_locate_embeddings(meta, RESULTS_PATH, PROJECT_ROOT), allow_pickle=True)["embeddings"]
    available_ids = {rec["example_id"] for rec in embedding_records}
    candidates = sorted(test_ids & exact_ids & available_ids & depth_ok & path_len_ok)
    if not candidates:
        raise ValueError("No overlapping test_exact examples between encodings, responses, and embeddings.npz after filters")
    return candidates


def _max_tree_depth(responses_path: Path) -> int:
    max_depth = 1
    if responses_path and responses_path.exists():
        with open(responses_path) as f:
            for line in f:
                rec = json.loads(line)
                depth = rec.get("depth")
                if isinstance(depth, (int, float)):
                    max_depth = max(max_depth, int(depth))
    return max_depth


def _example_distance_mse(example_id: int, layer: int, normalize_tree: bool, normalization_factor: float | None) -> float:
    data = encodings[layer]
    res = results[layer]
    X = data["X"]
    example_ids = np.asarray(data.get("example_ids", []))
    node_ids = np.asarray(data.get("node_ids", []))
    mask_example = example_ids == example_id
    X_example = _pca_transform(np.asarray(X[mask_example]), res)
    node_ids_example = node_ids[mask_example]

    B = _resolve_probe_projection(res)
    info = {
        "geometry": res.get("geometry", meta.get("fit_geometry", "euclidean")),
        "center": res.get("center"),
        "curvature": res.get("curvature"),
        "normalized_tree": res.get("normalized_tree", meta.get("normalize_tree", False)),
    }
    geometry = info.get("geometry", meta.get("fit_geometry", "euclidean"))

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


def _load_path_info(example_id: int):
    label_mapping = []
    parsed_path = []
    tree_ascii = None
    with open(RESPONSES_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("example_id") == example_id:
                label_mapping = rec.get("label_mapping") or []
                parsed_path = rec.get("parsed_path") or []
                prompt = rec.get("prompt", "")
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
    return label_mapping, parsed_path, tree_ascii


# Select examples as in notebook
candidates = collect_test_exact_ids(LAYER, depth_filter=DEPTH_FILTER, min_path_len=MIN_PATH_LEN)
normalization_factor = _max_tree_depth(RESPONSES_PATH) if meta.get("normalize_tree") else None
ranked = sorted(
    candidates,
    key=lambda ex_id: _example_distance_mse(ex_id, LAYER, bool(meta.get("normalize_tree")), normalization_factor),
)
chosen = ranked[:8]
first_example = chosen[0]


# Plotting helpers

def _layout_projection(layer: int, example_id: int):
    data = encodings[layer]
    res = results[layer]
    X = data["X"]
    depths = data["depth"]
    example_ids = np.asarray(data.get("example_ids", []))
    node_ids = np.asarray(data.get("node_ids", []))
    color_values, cmap, norm, _, _, _ = prepare_colorcode(COLORCODE, node_ids, depths, example_ids)

    B = _resolve_probe_projection(res)
    info = {
        "geometry": res.get("geometry", meta.get("fit_geometry", "euclidean")),
        "center": res.get("center"),
        "curvature": res.get("curvature"),
        "normalized_tree": res.get("normalized_tree", meta.get("normalize_tree", False)),
    }
    geometry = info.get("geometry", meta.get("fit_geometry", "euclidean"))

    Z = project_embeddings(X, B, geometry, info, res)
    Z = _project_to_2d(Z)

    mask_example = example_ids == example_id
    Z_example = Z[mask_example]
    depths_example = depths[mask_example]
    colors_example = color_values[mask_example]
    node_ids_example = node_ids[mask_example]

    # Rotate embeddings to make depth levels horizontally level as possible
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

    return Z_example, depths_example, colors_example, node_ids_example, cmap, norm


def _path_indices(node_ids_example, parsed_path, label_mapping):
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


def plot_example_projection(ax, layer: int, example_id: int, label_mapping=None, parsed_path=None, *, add_labels: bool = False, label_size: int = 10, node_size: int = 280, pad_scale: float = 0.15, zoom: float = 1.0):
    label_mapping = label_mapping or []
    parsed_path = parsed_path or []
    Z_example, depths_example, colors_example, node_ids_example, cmap, norm = _layout_projection(layer, example_id)
    path_node_ids, path_indices = _path_indices(node_ids_example, parsed_path, label_mapping)

    # Connect multiple embeddings of same node
    for node_id in np.unique(node_ids_example):
        idxs = np.where(node_ids_example == node_id)[0]
        if idxs.size > 1:
            coords = Z_example[idxs]
            line_color = cmap(norm(colors_example[idxs[0]]))
            ax.plot(coords[:, 0], coords[:, 1], linestyle="-", color="black", alpha=0.9, lw=4.2, zorder=0, clip_on=False)
            ax.plot(coords[:, 0], coords[:, 1], linestyle="-", color=line_color, alpha=0.85, lw=3.2, zorder=1, clip_on=False)

    # Path arrows
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

            # Step labels
            label_x = (x1_adj + x2_adj) / 2
            label_y = (y1_adj + y2_adj) / 2
            step_label = "step 1" if i == 0 else f"{i + 1}"
            ax.text(label_x, label_y, step_label, color="black", fontsize=10, ha="center", va="center", zorder=2,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.5), clip_on=False)

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


# Tree parsing borrowed from notebook logic

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


def plot_true_tree(ax, tree_ascii, label_mapping, parsed_path):
    label_mapping = label_mapping or []
    parsed_path = parsed_path or []
    inverse_label = {int(label): idx for idx, label in enumerate(label_mapping)} if label_mapping else {}

    nodes, edges = _parse_tree_ascii(tree_ascii)
    if not nodes:
        ax.axis("off")
        return

    # Recenter and scale x positions to center root over children
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

    # Slightly shorten the tree vertically (compress around center)
    y_scale = 0.8
    for n in nodes:
        n["y"] = 0.5 + (n["y"] - 0.5) * y_scale

    canonical_ids = [n["canonical"] for n in nodes]
    max_depth = max(int(np.floor(np.log2(cid + 1))) for cid in canonical_ids)
    depth_cmap = plt.get_cmap("Blues", max_depth + 1)
    depth_norm = Normalize(vmin=0, vmax=max_depth)

    # Edges
    for parent, child in edges:
        ax.plot([parent["x"], child["x"]], [parent["y"], child["y"]], color="#aaaaaa", lw=0.9, zorder=0, clip_on=False)

    # Traversal arrows (offset bidirectional pairs symmetrically)
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

            offset = 0.0
            if edge_dir_counts.get((a, b), 0) and edge_dir_counts.get((b, a), 0):
                # split to opposite sides of the edge using a canonical direction
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

            # step labels
            label_x = (x0_adj + x1_adj) / 2
            label_y = (y0_adj + y1_adj) / 2
            step_label = "step 1" if i == 0 else f"{i + 1}"
            ax.text(label_x, label_y, step_label, color="#c2410c", fontsize=9, ha="center", va="center", zorder=2,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.35), clip_on=False)

    # Nodes
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


def draw_path_text(ax, parsed_path, label_mapping, depth_cmap, depth_norm):
    inverse_label = {int(label): idx for idx, label in enumerate(label_mapping)} if label_mapping else {}
    tokens = [str(tok) for tok in parsed_path]
    if not tokens:
        return
    # Centered path text below tree, closer up
    start_x = 0.34
    y = 0.2
    ax.text(start_x, y, "PATH:", transform=ax.transAxes, ha="right", va="center", fontsize=12, color="black", clip_on=False)
    x = start_x + 0.02
    text_outline = [pe.withStroke(linewidth=2.6, foreground="black")]
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


label_mapping, parsed_path, tree_ascii = _load_path_info(first_example)

# Build figure
fig = plt.figure(figsize=(18, 12.5))
outer = gridspec.GridSpec(2, 1, height_ratios=[1.45, 1.0], hspace=0.0)

# Top: large projection + tree
inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[3.0, 1.0], wspace=0.0)
ax_proj = fig.add_subplot(inner[0, 0])
ax_tree = fig.add_subplot(inner[0, 1])

plot_example_projection(ax_proj, LAYER, first_example, label_mapping, parsed_path, add_labels=True, label_size=12, node_size=1200, pad_scale=0.7, zoom=0.95)

depth_cmap, depth_norm = plot_true_tree(ax_tree, tree_ascii, label_mapping, parsed_path)

# Titles pulled inside axes bounds (lowered)
ax_proj.text(0.55, 0.7, "Model Activations Projected to Hierarchical Subspace", transform=ax_proj.transAxes, ha="center", va="center", fontsize=15)
ax_tree.text(0.5, 0.8, "Ground-Truth Tree Traversal", transform=ax_tree.transAxes, ha="center", va="center", fontsize=15)

# Path text below tree
if depth_cmap is not None:
    draw_path_text(ax_tree, parsed_path, label_mapping, depth_cmap, depth_norm)

# Bottom: 2x3 grid using 2nd-7th examples
bottom = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1], wspace=0.0, hspace=0.0)
axes = [fig.add_subplot(bottom[r, c]) for r in range(2) for c in range(3)]
for ax, ex_id in zip(axes, chosen[1:7]):
    lm, pp, _ = _load_path_info(ex_id)
    plot_example_projection(ax, LAYER, ex_id, lm, pp, add_labels=False, node_size=520, pad_scale=0.9, zoom=0.55)
for ax in axes[len(chosen[1:7]):]:
    ax.axis("off")

# Adjust layout to reduce whitespace
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
ax_tree_pos = ax_tree.get_position().frozen()
ax_tree.set_position([ax_tree_pos.x0 - 0.045, ax_tree_pos.y0, ax_tree_pos.width, ax_tree_pos.height])

out_path = PROJECT_ROOT / "cutter/figures/misc/probe-geometry_combo-v15.png"
fig.savefig(out_path, dpi=300)
print(out_path)
