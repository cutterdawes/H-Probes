"""Shared chat rendering helpers."""

from __future__ import annotations

from typing import Dict, List


def render_chat(tokenizer, messages: List[Dict[str, str]]) -> str:
    """Render messages using the tokenizer's chat template when available."""

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    joined = []
    for msg in messages:
        prefix = msg.get("role", "user").upper()
        joined.append(f"{prefix}: {msg.get('content', '')}")
    joined.append("ASSISTANT:")
    return "\n".join(joined)
