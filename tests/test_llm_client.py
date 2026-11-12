"""Таймаут, ретраи только для временных ошибок и разбор конверта провайдера."""

from __future__ import annotations

import json
from dataclasses import replace

import requests

from errors import (
    EMPTY_MODEL_RESPONSE_MESSAGE,
    FALLBACK_MESSAGE,
    LLMResponseError,
    LLMUnavailableError,
    PARSE_ERROR_MESSAGE,
    REJECTED_REQUEST_MESSAGE,
)
from llm.client import LLMClient, extract_content


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _client(settings, **overrides):
    sleeps: list[float] = []
    client = LLMClient(
        replace(settings, api_key="test-key", **overrides),
        sleep_fn=sleeps.append,
    )
    return client, sleeps


def test_timeout_then_success_uses_exponential_delay(settings):
    client, sleeps = _client(settings)
    attempts = {"n": 0}

    def post(*_args, **_kwargs):
        attempts["n"] += 1
        if attempts["n"] < settings.max_attempts:
            raise requests.Timeout("timed out")
        return FakeResponse(200, {"choices": [{"message": {"content": "raw-text"}}]})

    client._session.post = post
    assert client.complete(system_prompt="s", user_prompt="u", request_id="r1") == "raw-text"
    base = settings.retry_base_delay_seconds
    assert attempts["n"] == 3
    assert sleeps == [base, base * 2]


def test_all_timeouts_become_fallback(settings):
    client, sleeps = _client(settings)
    attempts = {"n": 0}

    def post(*_args, **_kwargs):
        attempts["n"] += 1
        raise requests.ConnectionError("down")

    client._session.post = post
    try:
        client.complete(system_prompt="s", user_prompt="u", request_id="r1")
    except LLMUnavailableError as exc:
        assert exc.message == FALLBACK_MESSAGE
        assert exc.status_code == 503
    else:
        raise AssertionError("ожидался fallback")
    assert attempts["n"] == settings.max_attempts
    assert len(sleeps) == settings.max_attempts - 1


def test_http_429_and_503_are_retried(settings):
    client, sleeps = _client(settings)
    attempts = {"n": 0}

    def post(*_args, **_kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResponse(429, {"error": "rate"})
        if attempts["n"] == 2:
            return FakeResponse(503, {"error": "down"})
        return FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})

    client._session.post = post
    assert client.complete(system_prompt="s", user_prompt="u", request_id="r1") == "ok"
    assert attempts["n"] == 3
    assert len(sleeps) == 2


def test_http_400_is_not_retried(settings):
    client, sleeps = _client(settings)
    attempts = {"n": 0}

    def post(*_args, **_kwargs):
        attempts["n"] += 1
        return FakeResponse(400, {"error": "bad"})

    client._session.post = post
    try:
        client.complete(system_prompt="s", user_prompt="u", request_id="r1")
    except LLMResponseError as exc:
        assert exc.message == REJECTED_REQUEST_MESSAGE
        assert exc.status_code == 502
    else:
        raise AssertionError("ожидался отказ провайдера")
    assert attempts["n"] == 1
    assert sleeps == []


def test_single_attempt_does_not_sleep(settings):
    client, sleeps = _client(settings, max_attempts=1)

    def post(*_args, **_kwargs):
        raise requests.Timeout("timed out")

    client._session.post = post
    try:
        client.complete(system_prompt="s", user_prompt="u", request_id="r1")
    except LLMUnavailableError:
        pass
    else:
        raise AssertionError("ожидался fallback")
    assert sleeps == []


def test_request_uses_timeout_model_and_prompts(settings):
    client, _sleeps = _client(settings, base_url="http://example.test/v1/")
    seen = {}

    def post(*args, **kwargs):
        seen["url"] = args[0]
        seen["timeout"] = kwargs["timeout"]
        seen["json"] = kwargs["json"]
        seen["headers"] = kwargs["headers"]
        return FakeResponse(200, {"choices": [{"message": {"content": "body"}}]})

    client._session.post = post
    assert client.complete(system_prompt="system", user_prompt="user", request_id="r1") == "body"
    assert seen["url"] == "http://example.test/v1/chat/completions"
    assert seen["timeout"] == settings.timeout_seconds
    assert seen["json"]["model"] == settings.model
    assert seen["json"]["temperature"] == settings.temperature
    assert seen["json"]["messages"][0] == {"role": "system", "content": "system"}
    assert seen["json"]["messages"][1] == {"role": "user", "content": "user"}
    assert seen["headers"]["Authorization"] == "Bearer test-key"


def test_missing_api_key_does_not_call_network(settings):
    client = LLMClient(replace(settings, api_key=""))

    def post(*_args, **_kwargs):
        raise AssertionError("сеть не должна вызываться")

    client._session.post = post
    try:
        client.complete(system_prompt="s", user_prompt="u", request_id="r1")
    except LLMUnavailableError as exc:
        assert exc.message == FALLBACK_MESSAGE
    else:
        raise AssertionError("ожидался fallback")


def test_invalid_provider_body_is_not_retried(settings):
    client, sleeps = _client(settings)
    attempts = {"n": 0}

    def post(*_args, **_kwargs):
        attempts["n"] += 1
        return FakeResponse(200, text="<html>")

    client._session.post = post
    try:
        client.complete(system_prompt="s", user_prompt="u", request_id="r1")
    except LLMResponseError as exc:
        assert exc.message == PARSE_ERROR_MESSAGE
    else:
        raise AssertionError("ожидалась ошибка разбора")
    assert attempts["n"] == 1
    assert sleeps == []


def test_empty_choices_are_response_error(settings):
    client, _sleeps = _client(settings)

    def post(*_args, **_kwargs):
        return FakeResponse(200, {"choices": []})

    client._session.post = post
    try:
        client.complete(system_prompt="s", user_prompt="u", request_id="r1")
    except LLMResponseError as exc:
        assert exc.status_code == 502
    else:
        raise AssertionError("ожидалась ошибка конверта")


def test_extract_content_from_parts():
    raw = extract_content(
        {"choices": [{"message": {"content": [{"type": "text", "text": "{\"summary\": \"A\"}"}]}}]}
    )
    assert raw == '{"summary": "A"}'


def test_extract_content_rejects_empty():
    try:
        extract_content({"choices": [{"message": {"content": "  "}}]})
    except LLMResponseError as exc:
        assert exc.message == EMPTY_MODEL_RESPONSE_MESSAGE
    else:
        raise AssertionError("пустой ответ должен быть ошибкой")
