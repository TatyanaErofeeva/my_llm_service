"""Очистка и проверка JSON-ответа модели."""

from __future__ import annotations

import pytest

from errors import EMPTY_MODEL_RESPONSE_MESSAGE, LLMResponseError, PARSE_ERROR_MESSAGE
from llm.postprocess import postprocess_summary


def test_plain_json_and_extra_fields():
    assert postprocess_summary('{"summary": "Готово", "note": "ignore"}') == "Готово"


def test_fence_and_whitespace():
    raw = '```JSON\n{"summary": "  Раз\\nдва  "}\n```'
    assert postprocess_summary(raw) == "Раз два"


def test_invalid_json():
    with pytest.raises(LLMResponseError) as exc:
        postprocess_summary("просто текст")
    assert exc.value.message == PARSE_ERROR_MESSAGE
    assert exc.value.status_code == 502


def test_prose_around_fence_is_rejected():
    raw = 'Вот ответ:\n```json\n{"summary": "x"}\n```'
    with pytest.raises(LLMResponseError) as exc:
        postprocess_summary(raw)
    assert exc.value.message == PARSE_ERROR_MESSAGE


def test_summary_must_be_non_empty_string():
    with pytest.raises(LLMResponseError) as missing:
        postprocess_summary('{"summary": 1}')
    assert missing.value.message == PARSE_ERROR_MESSAGE

    with pytest.raises(LLMResponseError) as blank:
        postprocess_summary('{"summary": "   "}')
    assert blank.value.message == EMPTY_MODEL_RESPONSE_MESSAGE

    with pytest.raises(LLMResponseError):
        postprocess_summary("[]")
