"""Общие фикстуры. Секреты и переопределения из окружения в тесты не попадают."""

from __future__ import annotations

import logging

import pytest

from api.app import create_app
from cache.memory_cache import TTLCache
from config.settings import load_settings
from tests.fakes import FakeLLM

_ENV_KEYS = (
    "HOST",
    "PORT",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT_SECONDS",
    "LLM_MAX_ATTEMPTS",
    "CACHE_TTL_SECONDS",
    "LOG_LEVEL",
    "LOG_FILE",
)


@pytest.fixture(autouse=True)
def _isolate_logger():
    logger = logging.getLogger("my_llm_service")
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    yield


@pytest.fixture
def settings(monkeypatch):
    for name in _ENV_KEYS:
        monkeypatch.delenv(name, raising=False)
    return load_settings()


@pytest.fixture
def app_factory(settings):
    def factory(llm=None, ttl_seconds: float | None = None):
        llm_client = FakeLLM() if llm is None else llm
        cache = TTLCache(
            ttl_seconds=settings.cache_ttl_seconds if ttl_seconds is None else ttl_seconds
        )
        application = create_app(settings, llm_client=llm_client, cache=cache)
        application.testing = True
        return application, llm_client

    return factory
