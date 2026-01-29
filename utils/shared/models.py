"""Model size resolution helpers."""

from __future__ import annotations

from typing import Iterable, List, Tuple

# Map short parameter counts to full model ids for CLI ergonomics.
REASONING_MODEL_MAP = {
    "1.5B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "7B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "14B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "32B": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}
CHAT_MODEL_MAP = {
    "1.8B": "Qwen/Qwen1.5-1.8B-Chat",
    "4B": "Qwen/Qwen1.5-4B-Chat",
    "7B": "Qwen/Qwen1.5-7B-Chat",
    "14B": "Qwen/Qwen1.5-14B-Chat",
    "32B": "Qwen/Qwen1.5-32B-Chat",
}

DEFAULT_REASONING_SIZES: List[str] = ["7B"]
DEFAULT_CHAT_SIZES: List[str] = ["none"]


def _normalize_size_token(token: str) -> str:
    return str(token).upper().replace(" ", "")


def _resolve_size_tokens(size_tokens: Iterable[str], mapping: dict[str, str], label: str) -> List[str]:
    tokens = list(size_tokens)
    if len(tokens) == 1 and _normalize_size_token(tokens[0]) == "NONE":
        return []
    models: List[str] = []
    for tok in tokens:
        key = _normalize_size_token(tok)
        model_id = mapping.get(key)
        if model_id is None:
            supported = ", ".join(sorted(mapping.keys()))
            raise ValueError(f"Unknown {label} size '{tok}'. Supported: {supported}, or 'none'.")
        models.append(model_id)
    return models


def resolve_model_ids(
    reasoning_sizes: Iterable[str],
    chat_sizes: Iterable[str],
) -> Tuple[List[str], List[str]]:
    reasoning_models = _resolve_size_tokens(reasoning_sizes, REASONING_MODEL_MAP, "reasoning")
    chat_models = _resolve_size_tokens(chat_sizes, CHAT_MODEL_MAP, "chat")
    return reasoning_models, chat_models


def resolve_model_pairs(
    reasoning_sizes: Iterable[str],
    chat_sizes: Iterable[str],
) -> List[Tuple[str, str, str]]:
    """Return (family, size_token, model_id) tuples for selected sizes."""

    pairs: List[Tuple[str, str, str]] = []
    for tok in reasoning_sizes:
        key = _normalize_size_token(tok)
        if key == "NONE":
            continue
        model_id = REASONING_MODEL_MAP.get(key)
        if model_id is None:
            supported = ", ".join(sorted(REASONING_MODEL_MAP.keys()))
            raise ValueError(f"Unknown reasoning size '{tok}'. Supported: {supported}, or 'none'.")
        pairs.append(("reasoning", key, model_id))
    for tok in chat_sizes:
        key = _normalize_size_token(tok)
        if key == "NONE":
            continue
        model_id = CHAT_MODEL_MAP.get(key)
        if model_id is None:
            supported = ", ".join(sorted(CHAT_MODEL_MAP.keys()))
            raise ValueError(f"Unknown chat size '{tok}'. Supported: {supported}, or 'none'.")
        pairs.append(("chat", key, model_id))
    return pairs


def resolve_single_model_id(
    reasoning_sizes: Iterable[str],
    chat_sizes: Iterable[str],
    *,
    label: str = "model",
) -> str:
    reasoning_models, chat_models = resolve_model_ids(reasoning_sizes, chat_sizes)
    all_models = reasoning_models + chat_models
    if not all_models:
        raise ValueError(f"No {label} specified. Provide --reasoning-models or --chat-models.")
    if len(all_models) > 1:
        raise ValueError(f"Multiple {label}s specified. Provide exactly one size.")
    return all_models[0]
