"""Ключ кеша учитывает текст, параметры модели и системный промпт."""

from __future__ import annotations

import hashlib
import json


def build_cache_key(
    *,
    message: str,
    model: str,
    temperature: float,
    max_tokens: int,
    system_prompt: str,
) -> str:
    payload = {
        "message": message,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "system_prompt": system_prompt,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
