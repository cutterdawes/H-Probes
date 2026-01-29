"""Model loading and sentence encoding helpers."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
# Lazy-import transformers inside functions to avoid heavy imports during module load

from .trees import Node, TreeExample, annotate_examples

# =============================================================================
# Module configuration
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
    "DEVICE",
    "set_global_seed",
    "load_reasoning_model",
    "get_sentence_embedding",
    "collect_hidden_states_for_spans",
    "encode_example",
    "encode_examples",
]

_CACHE_VERSION = 1
_CACHE_DIR_DEFAULT = Path(__file__).resolve().parents[2] / "data"


# =============================================================================
# Device & reproducibility utilities
# =============================================================================

def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Model loading
# =============================================================================

def load_reasoning_model(
    model_id: str,
    *,
    device: str = DEVICE,
    use_half_precision: bool = True,
) -> Tuple["AutoTokenizer", "AutoModelForCausalLM"]:
    """Load tokenizer and language model with the desired precision."""

    from transformers import AutoModelForCausalLM, AutoTokenizer  # local import

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    dtype = torch.float16 if (device == "cuda" and use_half_precision) else torch.float32
    import inspect

    model_kwargs: Dict[str, Any] = {"device_map": "auto" if device == "cuda" else None}
    signature = inspect.signature(AutoModelForCausalLM.from_pretrained)
    if "dtype" in signature.parameters:
        model_kwargs["dtype"] = dtype
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()
    return tokenizer, model


# =============================================================================
# Embedding helpers
# =============================================================================

def _resolve_model_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(DEVICE)


def _select_hidden_layer(
    hidden_states: Optional[Tuple[torch.Tensor, ...]],
    last_hidden_state: Optional[torch.Tensor],
    layer_idx: int,
    layer_mode: str,
) -> torch.Tensor:
    """Retrieve a token-level hidden-state matrix according to layer selection rules."""

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


def get_sentence_embedding(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    layer_idx: int,
    layer_mode: str = "layer",
    context: Optional[str] = None,
) -> np.ndarray:
    """Generate a sentence embedding with optional left-context conditioning.

    The model receives the preceding sentences as context, while the returned
    embedding continues to mean-pool only over the tokens of ``text`` itself.
    """

    model_device = _resolve_model_device(model)
    with torch.no_grad():
        context_ids: List[int] = []
        if context:
            context_ids = tokenizer.encode(context, add_special_tokens=False)
        target_ids = tokenizer.encode(text, add_special_tokens=False)

        # Use a simple newline separator so context and current sentence remain distinct.
        sep_ids: List[int] = tokenizer.encode("\n\n", add_special_tokens=False) if context_ids else []

        combined_ids: List[int] = []
        target_start = 0

        bos_id = getattr(tokenizer, "bos_token_id", None)
        if bos_id is not None and getattr(tokenizer, "add_bos_token", False):
            combined_ids.append(bos_id)
            target_start += 1

        combined_ids.extend(context_ids)
        target_start += len(context_ids)
        combined_ids.extend(sep_ids)
        target_start += len(sep_ids)
        combined_ids.extend(target_ids)
        target_len = len(target_ids)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None and getattr(tokenizer, "add_eos_token", False):
            combined_ids.append(eos_id)

        if not combined_ids:
            # Fallback for empty inputs; use tokenizer's encoding of a single space.
            combined_ids = tokenizer.encode(" ", add_special_tokens=False)
            target_start = 0
            target_len = len(combined_ids)

        input_ids = torch.tensor([combined_ids], dtype=torch.long, device=model_device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        tokens = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        outputs = model(**tokens, output_hidden_states=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        last_hidden = _select_hidden_layer(hidden_states, last_hidden_state, layer_idx, layer_mode)

        total_tokens = last_hidden.shape[0]
        target_start = max(0, min(target_start, total_tokens))
        target_end = max(target_start, min(target_start + target_len, total_tokens))

        target_mask = torch.zeros(total_tokens, dtype=torch.bool, device=last_hidden.device)
        if target_end > target_start:
            target_mask[target_start:target_end] = True

        input_ids_row = tokens["input_ids"][0]
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        if special_ids and target_mask.any():
            special_mask = torch.zeros_like(target_mask)
            for special in special_ids:
                special_mask |= input_ids_row == special
            target_mask &= ~special_mask

        if not target_mask.any():
            vector = last_hidden.mean(dim=0)
        else:
            vector = last_hidden[target_mask].mean(dim=0)
        return vector.detach().float().cpu().numpy()


def _random_sentence_embedding(text: str, dim: int, seed: int) -> np.ndarray:
    digest = hashlib.sha256(f"{seed}:{text}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(rng_seed)
    return rng.standard_normal(dim).astype(np.float32)


def _infer_model_identifier(model: Optional[AutoModelForCausalLM]) -> str:
    if model is None:
        return "synthetic"
    name = getattr(model, "name_or_path", None)
    if name:
        return str(name)
    config = getattr(model, "config", None)
    for attr in ("name_or_path", "_name_or_path"):
        value = getattr(config, attr, None) if config is not None else None
        if value:
            return str(value)
    return model.__class__.__name__


def _sanitize_for_filename(value: str) -> str:
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    return "".join(safe_chars).strip("_") or "cached"


def _examples_signature(examples: Sequence[TreeExample]) -> str:
    payload = []
    for example in examples:
        payload.append(
            {
                "name": example.name,
                "nodes": [(node.nid, node.parent, node.text) for node in example.nodes],
            }
        )
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _auto_cache_stem(
    *,
    examples: Sequence[TreeExample],
    model_id: str,
    layer_idx: int,
    layer_mode: str,
    train_test_split: float,
    seed: int,
    random_dim: int,
    use_model_embeddings: bool,
) -> str:
    split_marker = int(round(train_test_split * 1_000))
    parts = [
        "encodings",
        _sanitize_for_filename(model_id),
        f"layer{layer_idx}",
        _sanitize_for_filename(layer_mode.lower()),
        f"split{split_marker}",
        f"seed{seed}",
    ]
    if not use_model_embeddings:
        parts.append(f"rand{random_dim}")
    signature = _examples_signature(examples)[:16]
    parts.append(signature)
    return "_".join(parts)


def _meta_matches(stored: Dict[str, Any], current: Dict[str, Any]) -> bool:
    if stored.get("version") != current.get("version"):
        return False
    float_keys = {"train_test_split"}
    for key, expected in current.items():
        if key in float_keys:
            stored_val = stored.get(key)
            if stored_val is None or abs(float(stored_val) - float(expected)) > 1e-9:
                return False
        else:
            if stored.get(key) != expected:
                return False
    return True


def _encoded_dataset_size(data: Dict[str, Any]) -> int:
    X = data.get("X")
    if isinstance(X, np.ndarray):
        return int(X.shape[0])
    for key in ("depth", "texts", "ids"):
        value = data.get(key)
        if value is not None:
            try:
                return len(value)
            except TypeError:
                continue
    train = data.get("train_idx")
    test = data.get("test_idx")
    if train is not None and test is not None:
        try:
            return len(train) + len(test)
        except TypeError:
            return 0
    return 0


def _encoded_splits_valid(encoded_payload: Dict[str, Dict[str, Any]], train_test_split: float) -> bool:
    if not isinstance(encoded_payload, dict):
        return False
    enforce_test = train_test_split > 0
    for name, data in encoded_payload.items():
        if not isinstance(data, dict):
            return False
        total = _encoded_dataset_size(data)
        if total <= 1:
            continue
        train_idx = np.array(data.get("train_idx", []), dtype=int)
        test_idx = np.array(data.get("test_idx", []), dtype=int)
        if enforce_test and test_idx.size == 0:
            return False
        if np.intersect1d(train_idx, test_idx).size > 0:
            return False
        if np.union1d(train_idx, test_idx).size != total:
            return False
    return True


# =============================================================================
# Encoding and caching
# =============================================================================

def encode_example(
    example: TreeExample,
    tokenizer: Optional[AutoTokenizer],
    model: Optional[AutoModelForCausalLM],
    *,
    layer_idx: int,
    layer_mode: str,
    random_dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], np.ndarray]:
    """Encode a tree example into embeddings, distances, texts, and ids."""

    texts = [node.text for node in example.nodes]
    node_ids = [node.nid for node in example.nodes]
    embeddings_list = []
    context_fragments: List[str] = []
    for offset, text_item in enumerate(texts):
        if tokenizer is None or model is None:
            embeddings_list.append(_random_sentence_embedding(text_item, random_dim, seed + offset))
        else:
            context_text = "\n\n".join(context_fragments) if context_fragments else None
            embeddings_list.append(
                get_sentence_embedding(
                    text_item,
                    tokenizer,
                    model,
                    layer_idx=layer_idx,
                    layer_mode=layer_mode,
                    context=context_text,
                )
            )
        context_fragments.append(text_item)
    embeddings = np.stack(embeddings_list, axis=0)
    if example.depth is None or example.dist_mat is None:
        raise ValueError("Example missing depth or distance annotations. Run annotate_examples() first.")
    depths = np.array([example.depth[nid] for nid in node_ids], dtype=np.float32)
    return embeddings, example.dist_mat.astype(np.float32), texts, node_ids, depths


def collect_hidden_states_for_spans(
    text: str,
    spans: Sequence[Tuple[int, int]],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    *,
    layer_idx: int,
    layer_mode: str = "layer",
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """Collect hidden states for specific character spans within ``text``.

    Args:
        text: Full context to encode (already includes any prompt content).
        spans: Sequence of (start, end) character offsets within ``text``.
        tokenizer/model: Hugging Face components used for encoding.
        layer_idx/layer_mode: Layer selection parameters.

    Returns:
        (embeddings, token_indices, debug_info) where embeddings is a list of
        NumPy vectors ordered like ``spans``, token_indices stores the token
        position used for each span, and debug_info exposes raw offsets and IDs
        for downstream validation.
    """

    if not spans:
        return [], [], {"offsets": [], "token_ids": []}
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("collect_hidden_states_for_spans requires a fast tokenizer to access offsets.")

    model_device = _resolve_model_device(model)
    with torch.no_grad():
        encoding = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].to(model_device)
        attention_mask = encoding["attention_mask"].to(model_device)
        offsets = encoding["offset_mapping"][0].tolist()
        token_ids = encoding["input_ids"][0].tolist()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        selected_hidden = _select_hidden_layer(hidden_states, last_hidden_state, layer_idx, layer_mode)

        embeddings: List[np.ndarray] = []
        token_indices: List[int] = []
        for start, end in spans:
            token_range = [idx for idx, (s, e) in enumerate(offsets) if (e > start and s < end)]
            if not token_range:
                raise ValueError(f"Could not align character span {(start, end)} to tokenizer offsets.")
            token_idx = token_range[-1]
            token_indices.append(token_idx)
            embeddings.append(selected_hidden[token_idx].detach().float().cpu().numpy())

        debug_info: Dict[str, Any] = {
            "offsets": offsets,
            "token_ids": token_ids,
        }
        return embeddings, token_indices, debug_info


def encode_examples(
    examples: Sequence[TreeExample],
    tokenizer: Optional[AutoTokenizer],
    model: Optional[AutoModelForCausalLM],
    *,
    layer_idx: int,
    layer_mode: str,
    train_test_split: float,
    seed: int,
    random_dim: int = 128,
    cache: bool = False,
    cache_namespace: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    force_recompute: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """Encode all examples and generate reproducible train/test splits.

    When ``cache`` is enabled (or ``cache_namespace`` is provided), encodings are
    persisted under ``data/`` to avoid repeated model inference. Subsequent calls
    with the same configuration will load the cached artifacts unless
    ``force_recompute`` is True.
    """

    annotate_examples(examples)
    use_model_embeddings = tokenizer is not None and model is not None
    model_id = _infer_model_identifier(model if use_model_embeddings else None)
    examples_sig = _examples_signature(examples)
    cache_meta: Dict[str, Any] = {
        "version": _CACHE_VERSION,
        "model_id": model_id,
        "layer_idx": int(layer_idx),
        "layer_mode": layer_mode,
        "train_test_split": float(train_test_split),
        "seed": int(seed),
        "random_dim": int(random_dim) if not use_model_embeddings else None,
        "use_model_embeddings": bool(use_model_embeddings),
        "examples_signature": examples_sig,
    }

    cache_path: Optional[Path] = None
    use_cache = cache or cache_namespace is not None
    if use_cache:
        cache_root = Path(cache_dir) if cache_dir is not None else _CACHE_DIR_DEFAULT
        cache_root.mkdir(parents=True, exist_ok=True)
        if cache_namespace:
            stem = _sanitize_for_filename(cache_namespace)
        else:
            stem = _auto_cache_stem(
                examples=examples,
                model_id=model_id,
                layer_idx=layer_idx,
                layer_mode=layer_mode,
                train_test_split=train_test_split,
                seed=seed,
                random_dim=random_dim,
                use_model_embeddings=use_model_embeddings,
            )
        cache_path = cache_root / f"{stem}.npz"
        if cache_path.exists() and not force_recompute:
            try:
                with np.load(cache_path, allow_pickle=True) as cached:
                    stored_meta = cached["meta"].item()
                    if _meta_matches(stored_meta, cache_meta):
                        encoded_cached = cached["encoded"].item()
                        if isinstance(encoded_cached, dict):
                            if _encoded_splits_valid(encoded_cached, train_test_split):
                                return encoded_cached
                            print(
                                f"Cached encodings in {cache_path.name} have empty test splits; recomputing."
                            )
            except Exception:
                # Fall back to recomputing if cache is unreadable.
                pass

    encoded: Dict[str, Dict[str, Any]] = {}
    rng = np.random.default_rng(seed)
    for example in examples:
        embeddings, distances, texts, ids, depths = encode_example(
            example,
            tokenizer,
            model,
            layer_idx=layer_idx,
            layer_mode=layer_mode,
            random_dim=random_dim,
            seed=seed,
        )
        total = embeddings.shape[0]
        indices = np.arange(total)
        test_idx = np.array([], dtype=int)
        if total > 1 and train_test_split > 0:
            desired = int(np.round(train_test_split * total))
            test_count = max(1, min(total - 1, desired))
            if test_count > 0:
                test_idx = np.sort(rng.choice(indices, size=test_count, replace=False))
        train_idx = np.setdiff1d(indices, test_idx, assume_unique=True)
        encoded[example.name] = {
            "X": embeddings,
            "D": distances,
            "texts": texts,
            "ids": ids,
            "depth": depths,
            "train_idx": train_idx.astype(int).tolist(),
            "test_idx": test_idx.astype(int).tolist(),
        }

    if cache_path is not None:
        payload_encoded = np.array(encoded, dtype=object)
        payload_meta = np.array(cache_meta, dtype=object)
        np.savez_compressed(cache_path, encoded=payload_encoded, meta=payload_meta)
    return encoded
