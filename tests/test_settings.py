"""Значения по умолчанию из задания и переопределение через окружение."""

from __future__ import annotations

from dataclasses import replace

import pytest

from config.settings import load_settings, validate_settings


def test_assignment_defaults(settings):
    assert settings.max_attempts == 3
    assert settings.retry_base_delay_seconds == 0.5
    assert settings.cache_ttl_seconds == 600
    assert settings.max_message_length == 1000
    assert settings.timeout_seconds == 15
    assert settings.temperature == 0.2
    assert settings.model == "gpt-4o-mini"
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.api_key == ""


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_MODEL", "demo-model")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.7")
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9999/v1")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "60")
    monkeypatch.setenv("LLM_MAX_ATTEMPTS", "2")
    loaded = load_settings()
    assert loaded.api_key == "secret"
    assert loaded.model == "demo-model"
    assert loaded.temperature == 0.7
    assert loaded.base_url == "http://127.0.0.1:9999/v1"
    assert loaded.cache_ttl_seconds == 60
    assert loaded.max_attempts == 2


def test_validate_rejects_too_many_attempts(settings):
    with pytest.raises(ValueError):
        validate_settings(replace(settings, max_attempts=5))
