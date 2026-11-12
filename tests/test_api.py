"""Сценарии HTTP-слоя: успех, валидация, кеш, fallback и разбор ответа."""

from __future__ import annotations

import logging
from dataclasses import replace

import requests

from config.logging_setup import LOGGER_NAME
from errors import (
    FALLBACK_MESSAGE,
    INTERNAL_ERROR_MESSAGE,
    LLMUnavailableError,
    PARSE_ERROR_MESSAGE,
)
from llm.client import LLMClient
from llm.prompt_builder import SYSTEM_PROMPT
from tests.fakes import FakeLLM


def _events(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == LOGGER_NAME]


def test_success_and_exact_repeat_comes_from_cache(app_factory, caplog):
    app, llm = app_factory()
    client = app.test_client()
    payload = {"message": "Текст для проверки."}

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        first = client.post("/chat", json=payload)
        second = client.post("/chat", json=payload)

    assert first.status_code == 200
    assert first.get_json() == {
        "scenario": "summarize",
        "summary": "Краткий итог.",
        "cached": False,
    }
    assert second.status_code == 200
    assert second.get_json()["cached"] is True
    assert second.get_json()["summary"] == "Краткий итог."
    assert len(llm.calls) == 1
    assert llm.calls[0]["system_prompt"] == SYSTEM_PROMPT
    assert llm.calls[0]["user_prompt"] == "Суммаризируй текст:\n\nТекст для проверки."

    events = _events(caplog)
    assert "request_received" in events
    assert "validation_ok" in events
    assert "scenario_selected" in events
    assert "prompt_built" in events
    assert "cache_miss" in events
    assert "cache_hit" in events
    assert events.count("model_response") == 1
    assert "postprocess_ok" in events
    assert "response_ready" in events


def test_surrounding_whitespace_uses_same_cache_entry(app_factory):
    app, llm = app_factory()
    client = app.test_client()
    assert client.post("/chat", json={"message": "  Текст для проверки.  "}).get_json()["cached"] is False
    assert client.post("/chat", json={"message": "Текст для проверки."}).get_json()["cached"] is True
    assert len(llm.calls) == 1


def test_changed_system_prompt_is_cache_miss(app_factory, monkeypatch):
    app, llm = app_factory()
    client = app.test_client()
    assert client.post("/chat", json={"message": "текст"}).get_json()["cached"] is False
    monkeypatch.setattr("llm.prompt_builder.SYSTEM_PROMPT", "Другая инструкция.")
    second = client.post("/chat", json={"message": "текст"})
    assert second.get_json()["cached"] is False
    assert len(llm.calls) == 2
    assert llm.calls[1]["system_prompt"] == "Другая инструкция."


def test_missing_message(app_factory):
    client = app_factory()[0].test_client()
    response = client.post("/chat", json={"text": "привет"})
    assert response.status_code == 400
    assert response.get_json() == {"error": "Поле message обязательно."}


def test_empty_message(app_factory):
    client = app_factory()[0].test_client()
    response = client.post("/chat", json={"message": " \n\t "})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Поле message не должно быть пустым."


def test_message_must_be_string(app_factory):
    client = app_factory()[0].test_client()
    response = client.post("/chat", json={"message": 12})
    assert response.status_code == 400
    assert response.get_json()["error"] == "Поле message должно быть строкой."


def test_message_length_limit(app_factory, settings):
    client = app_factory()[0].test_client()
    limit = settings.max_message_length
    ok = client.post("/chat", json={"message": "а" * limit})
    too_long = client.post("/chat", json={"message": "а" * (limit + 1)})
    assert ok.status_code == 200
    assert too_long.status_code == 400
    assert too_long.get_json()["error"] == f"Поле message не должно превышать {limit} символов."


def test_non_object_and_non_json(app_factory, caplog):
    client = app_factory()[0].test_client()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        as_list = client.post("/chat", json=["привет"])
        as_text = client.post("/chat", data="hello", content_type="text/plain")
    assert as_list.status_code == 400
    assert as_text.status_code == 400
    assert as_list.get_json()["error"] == "Тело запроса должно быть JSON-объектом."
    assert "validation_failed" in _events(caplog)


def test_invalid_model_json_is_not_cached(app_factory):
    app, llm = app_factory(FakeLLM(result="не json"))
    client = app.test_client()
    first = client.post("/chat", json={"message": "привет"})
    second = client.post("/chat", json={"message": "привет"})
    assert first.status_code == 502
    assert first.get_json()["error"] == PARSE_ERROR_MESSAGE
    assert second.status_code == 502
    assert len(llm.calls) == 2


def test_fenced_json_is_cleaned(app_factory):
    raw = '```json\n{"summary": "  Раз,   два.  "}\n```'
    app, _llm = app_factory(FakeLLM(result=raw))
    response = app.test_client().post("/chat", json={"message": "длинный текст"})
    assert response.status_code == 200
    assert response.get_json()["summary"] == "Раз, два."


def test_unavailable_model_returns_fallback_and_does_not_cache(app_factory):
    app, llm = app_factory(FakeLLM(error=LLMUnavailableError()))
    client = app.test_client()
    first = client.post("/chat", json={"message": "привет"})
    second = client.post("/chat", json={"message": "привет"})
    assert first.status_code == 503
    assert first.get_json() == {"error": FALLBACK_MESSAGE}
    assert second.status_code == 503
    assert len(llm.calls) == 2


def test_connection_error_retries_then_fallback(settings, monkeypatch):
    llm = LLMClient(
        replace(settings, api_key="test-key", retry_base_delay_seconds=0),
        sleep_fn=lambda _delay: None,
    )
    calls = {"n": 0}

    def boom(*_args, **_kwargs):
        calls["n"] += 1
        raise requests.ConnectionError("network down")

    monkeypatch.setattr(llm._session, "post", boom)
    client = create_client(settings, llm)
    payload = {"message": "сеть недоступна"}
    first = client.post("/chat", json=payload)
    second = client.post("/chat", json=payload)
    assert first.status_code == 503
    assert second.status_code == 503
    assert first.get_json()["error"] == FALLBACK_MESSAGE
    assert "network down" not in first.get_data(as_text=True)
    assert calls["n"] == settings.max_attempts * 2


def test_provider_rejection_is_not_retried(settings, monkeypatch):
    llm = LLMClient(replace(settings, api_key="test-key"), sleep_fn=lambda _delay: None)
    calls = {"n": 0}

    def reject(*_args, **_kwargs):
        calls["n"] += 1
        return _Response(400, {"error": "bad"})

    monkeypatch.setattr(llm._session, "post", reject)
    response = create_client(settings, llm).post("/chat", json={"message": "привет"})
    assert response.status_code == 502
    assert calls["n"] == 1
    assert "bad" not in response.get_data(as_text=True)


def test_unexpected_error_hides_internals(app_factory):
    class ExplodingLLM:
        def complete(self, *, system_prompt: str, user_prompt: str, request_id: str) -> str:
            raise RuntimeError("secret-bug")

    app, _llm = app_factory(ExplodingLLM())
    response = app.test_client().post("/chat", json={"message": "привет"})
    assert response.status_code == 500
    assert response.get_json()["error"] == INTERNAL_ERROR_MESSAGE
    assert "secret-bug" not in response.get_data(as_text=True)


def test_health_not_found_and_method(app_factory):
    client = app_factory()[0].test_client()
    assert client.get("/health").get_json() == {"status": "ok"}
    assert client.get("/missing").status_code == 404
    assert client.get("/chat").status_code == 405


def test_body_too_large(app_factory):
    client = app_factory()[0].test_client()
    payload = b'{"message": "' + (b"a" * 20000) + b'"}'
    response = client.post("/chat", data=payload, content_type="application/json")
    assert response.status_code == 413
    assert response.get_json()["error"] == "Тело запроса слишком большое."


def create_client(settings, llm):
    from api.app import create_app

    app = create_app(settings, llm_client=llm)
    app.testing = True
    return app.test_client()


class _Response:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload
