"""Проверка структуры ответа модели и очистка текста."""

from __future__ import annotations

import json
import re

from errors import EMPTY_MODEL_RESPONSE_MESSAGE, LLMResponseError, PARSE_ERROR_MESSAGE

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.IGNORECASE | re.DOTALL)


def postprocess_summary(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise LLMResponseError(
            EMPTY_MODEL_RESPONSE_MESSAGE,
            log_message="empty_model_response",
        )

    text = _strip_fence(raw.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            PARSE_ERROR_MESSAGE,
            log_message=f"invalid_json: {exc.msg}",
        ) from exc

    if not isinstance(data, dict):
        raise LLMResponseError(PARSE_ERROR_MESSAGE, log_message="json_is_not_object")

    summary = data.get("summary")
    if not isinstance(summary, str):
        raise LLMResponseError(
            PARSE_ERROR_MESSAGE,
            log_message="summary_missing_or_not_string",
        )

    cleaned = re.sub(r"\s+", " ", summary).strip()
    if not cleaned:
        raise LLMResponseError(EMPTY_MODEL_RESPONSE_MESSAGE, log_message="empty_summary")
    return cleaned


def _strip_fence(text: str) -> str:
    match = _FENCE.fullmatch(text)
    if match:
        return match.group(1).strip()
    return text
