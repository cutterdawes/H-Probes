"""Helpers for loading, saving, and reducing embedding caches."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from cutter.utils.shared.basic import split_balanced, split_exact_only
from cutter.utils.shared.paths import embeddings_path, embeddings_pca_path, model_output_dir, responses_path

_PCA_CACHE_RE = re.compile(r"^embeddings_pca(\d+)\.npz$")


@dataclass
class _ResponseStub:
    example_id: int
    exact_match: bool


@dataclass
class _LegacyEmbeddingRecord:
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


def _ensure_legacy_embedding_record() -> None:
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "EmbeddingRecord"):
        setattr(main_mod, "EmbeddingRecord", _LegacyEmbeddingRecord)


def _normalize_meta(raw_meta: Any) -> Dict[str, Any]:
    if raw_meta is None:
        return {}
    if isinstance(raw_meta, np.ndarray) and raw_meta.shape == ():
        return dict(raw_meta.item())
    if isinstance(raw_meta, dict):
        return dict(raw_meta)
    return dict(raw_meta)


def list_pca_caches(output_dir: Path) -> Dict[int, Path]:
    caches: Dict[int, Path] = {}
    if not output_dir.exists():
        return caches
    for path in output_dir.glob("embeddings_pca*.npz"):
        match = _PCA_CACHE_RE.match(path.name)
        if not match:
            continue
        caches[int(match.group(1))] = path
    return caches


def load_embedding_meta(path: Path) -> Dict[str, Any]:
    payload = np.load(path, allow_pickle=True)
    return _normalize_meta(payload["meta"] if "meta" in payload else None)


def load_embedding_payload(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    _ensure_legacy_embedding_record()
    payload = np.load(path, allow_pickle=True)
    embeddings = payload["embeddings"] if "embeddings" in payload else []
    meta = _normalize_meta(payload["meta"] if "meta" in payload else None)
    cache: Dict[int, Dict[str, Any]] = {}
    for entry in embeddings:
        if isinstance(entry, dict):
            example_id = int(entry.get("example_id", -1))
            rec = entry
        else:
            example_id = int(getattr(entry, "example_id", -1))
            rec = {
                "example_id": example_id,
                "parsed_path": getattr(entry, "parsed_path", []),
                "parsed_text": getattr(entry, "parsed_text", None),
                "token_ids": getattr(entry, "token_ids", []),
                "token_offsets": getattr(entry, "token_offsets", []),
                "layers": getattr(entry, "layers", []),
                "hidden_dim": getattr(entry, "hidden_dim", 0),
                "embeddings_by_layer": getattr(entry, "embeddings_by_layer", {}),
                "prompt_tokens": getattr(entry, "prompt_tokens", 0),
                "model_id": getattr(entry, "model_id", ""),
                "dataset_path": getattr(entry, "dataset_path", ""),
            }
        if example_id >= 0:
            cache[example_id] = rec
    return cache, meta


def save_embedding_cache(path: Path, embeddings: Mapping[int, Dict[str, Any]], meta: Mapping[str, Any]) -> None:
    entries = [embeddings[key] for key in sorted(embeddings.keys())]
    emb_payload = np.array(entries, dtype=object)
    meta_payload = np.array(dict(meta), dtype=object)
    np.savez_compressed(path, embeddings=emb_payload, meta=meta_payload)


def _infer_layers(entries: Mapping[int, Dict[str, Any]]) -> List[int]:
    if not entries:
        return []
    first_entry = next(iter(entries.values()))
    layers_dict = first_entry.get("embeddings_by_layer", {})
    if isinstance(layers_dict, dict) and layers_dict:
        return sorted(int(layer) for layer in layers_dict.keys())
    return [int(layer) for layer in first_entry.get("layers", [])]


def _collect_train_example_ids(
    responses_path: Path,
    train_split: float,
    seed: int,
    split_type: str,
) -> List[int]:
    records: List[_ResponseStub] = []
    with responses_path.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = int(row.get("example_id", idx))
            exact_match = bool(row.get("exact_match", False))
            records.append(_ResponseStub(example_id=example_id, exact_match=exact_match))
    if split_type == "exact-only":
        train_records, _ = split_exact_only(records, train_split, seed, exact_attr="exact_match")
    else:
        train_records, _ = split_balanced(records, train_split, seed, exact_attr="exact_match")
    return [rec.example_id for rec in train_records]


def build_pca_cache_from_entries(
    entries: Mapping[int, Dict[str, Any]],
    train_example_ids: Iterable[int],
    pca_components: int,
    seed: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], int]:
    if pca_components < 0:
        raise ValueError("pca_components must be >= 0 to build a PCA cache.")
    if pca_components == 0:
        raise ValueError("pca_components must be -1 or a positive integer.")
    layers = _infer_layers(entries)
    if not layers:
        raise RuntimeError("No layers found in embedding entries.")
    train_ids = set(train_example_ids)
    pca_models: Dict[int, PCA] = {}
    pca_info: Dict[int, Dict[str, Any]] = {}
    for layer in layers:
        parts: List[np.ndarray] = []
        for example_id in train_ids:
            entry = entries.get(example_id)
            if entry is None:
                continue
            emb = entry.get("embeddings_by_layer", {}).get(layer)
            if emb is None or np.size(emb) == 0:
                continue
            parts.append(np.asarray(emb, dtype=np.float32))
        if not parts:
            raise RuntimeError(f"No training embeddings found for layer {layer}.")
        X_train = np.vstack(parts).astype(np.float32)
        max_components = min(X_train.shape[0], X_train.shape[1])
        effective = min(pca_components, max_components)
        if effective <= 0:
            raise RuntimeError(f"Insufficient data for PCA components at layer {layer}.")
        pca = PCA(n_components=effective, svd_solver="auto", random_state=seed)
        pca.fit(X_train)
        pca_models[layer] = pca
        pca_info[layer] = {
            "components": pca.components_.astype(np.float32),
            "mean": pca.mean_.astype(np.float32),
            "n_components": int(effective),
            "n_features": int(X_train.shape[1]),
        }
    hidden_dim = min(info["n_components"] for info in pca_info.values()) if pca_info else 0
    reduced_entries: Dict[int, Dict[str, Any]] = {}
    for example_id, entry in entries.items():
        emb_by_layer = entry.get("embeddings_by_layer", {})
        new_by_layer: Dict[int, np.ndarray] = {}
        for layer, emb in emb_by_layer.items():
            if emb is None or np.size(emb) == 0:
                continue
            pca = pca_models.get(int(layer))
            if pca is None:
                continue
            new_by_layer[int(layer)] = pca.transform(np.asarray(emb, dtype=np.float32)).astype(np.float32)
        new_entry = dict(entry)
        new_entry["embeddings_by_layer"] = new_by_layer
        new_entry["hidden_dim"] = hidden_dim
        reduced_entries[example_id] = new_entry
    return reduced_entries, pca_info, hidden_dim


def derive_pca_cache_from_existing(
    source_path: Path,
    target_components: int,
    output_path: Path,
) -> bool:
    cache, meta = load_embedding_payload(source_path)
    source_components = int(meta.get("pca_components", -1))
    if source_components < 0 or target_components >= source_components:
        return False
    reduced_entries: Dict[int, Dict[str, Any]] = {}
    for example_id, entry in cache.items():
        emb_by_layer = entry.get("embeddings_by_layer", {})
        new_by_layer: Dict[int, np.ndarray] = {}
        for layer, emb in emb_by_layer.items():
            if emb is None or np.size(emb) == 0:
                continue
            reduced = np.asarray(emb, dtype=np.float32)[:, :target_components]
            new_by_layer[int(layer)] = reduced
        new_entry = dict(entry)
        new_entry["embeddings_by_layer"] = new_by_layer
        new_entry["hidden_dim"] = target_components
        reduced_entries[example_id] = new_entry
    pca_info = meta.get("pca", {})
    new_pca_info: Dict[int, Dict[str, Any]] = {}
    if isinstance(pca_info, dict):
        for layer, info in pca_info.items():
            components = np.asarray(info.get("components", []), dtype=np.float32)
            if components.size == 0:
                continue
            take = min(target_components, components.shape[0])
            new_pca_info[int(layer)] = {
                "components": components[:take],
                "mean": np.asarray(info.get("mean", []), dtype=np.float32),
                "n_components": int(take),
                "n_features": int(info.get("n_features", components.shape[1] if components.ndim > 1 else 0)),
            }
    new_meta = dict(meta)
    new_meta["pca_components"] = int(target_components)
    new_meta["hidden_dim"] = int(target_components)
    if new_pca_info:
        new_meta["pca"] = new_pca_info
    if "pca_split_type" in meta:
        new_meta["pca_split_type"] = meta["pca_split_type"]
    save_embedding_cache(output_path, reduced_entries, new_meta)
    return True


def build_pca_cache_from_full(
    dataset_tag: str,
    model_id: str,
    pca_components: int,
    train_split: float,
    seed: int,
    split_type: str,
) -> Optional[Path]:
    full_path = embeddings_path(dataset_tag, model_id)
    if not full_path.exists():
        return None
    responses_fp = responses_path(dataset_tag, model_id)
    if not responses_fp.exists():
        return None
    entries, meta = load_embedding_payload(full_path)
    train_ids = _collect_train_example_ids(responses_fp, train_split, seed, split_type)
    reduced_entries, pca_info, hidden_dim = build_pca_cache_from_entries(entries, train_ids, pca_components, seed)
    output_path = embeddings_pca_path(dataset_tag, model_id, pca_components)
    new_meta = dict(meta)
    new_meta.update(
        {
            "pca_components": int(pca_components),
            "hidden_dim": int(hidden_dim),
            "pca": pca_info,
            "pca_train_split": float(train_split),
            "pca_seed": int(seed),
            "pca_split_type": str(split_type),
        }
    )
    save_embedding_cache(output_path, reduced_entries, new_meta)
    return output_path


def ensure_pca_cache(
    dataset_tag: str,
    model_id: str,
    pca_components: int,
    train_split: float,
    seed: int,
    split_type: str = "exact-only",
) -> Optional[Path]:
    if pca_components < 0:
        return embeddings_path(dataset_tag, model_id)
    output_path = embeddings_pca_path(dataset_tag, model_id, pca_components)
    if output_path.exists():
        meta = load_embedding_meta(output_path)
        cache_split_type = meta.get("pca_split_type")
        if cache_split_type == split_type:
            return output_path
    output_dir = model_output_dir(dataset_tag, model_id)
    caches = list_pca_caches(output_dir)
    larger = []
    for comp, path in caches.items():
        if comp <= pca_components:
            continue
        meta = load_embedding_meta(path)
        if meta.get("pca_split_type") == split_type:
            larger.append(comp)
    larger = sorted(larger)
    if larger:
        source_path = caches[larger[0]]
        if derive_pca_cache_from_existing(source_path, pca_components, output_path):
            return output_path
    built = build_pca_cache_from_full(dataset_tag, model_id, pca_components, train_split, seed, split_type)
    if built is not None and built.exists():
        return built
    return None
