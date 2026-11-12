"""Загрузка настроек из YAML и переменных окружения."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_seconds: float
    max_attempts: int
    retry_base_delay_seconds: float
    cache_ttl_seconds: int
    max_message_length: int
    log_file: Path
    log_level: str


def load_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    raw = _read_yaml(ROOT / "config" / "settings.yaml")
    server = _section(raw, "server")
    llm = _section(raw, "llm")
    cache = _section(raw, "cache")
    validation = _section(raw, "validation")
    logging_cfg = _section(raw, "logging")

    log_file_value = os.environ.get("LOG_FILE", logging_cfg.get("file", "logs/service.log"))
    log_file = Path(log_file_value)
    if not log_file.is_absolute():
        log_file = ROOT / log_file

    settings = Settings(
        host=os.environ.get("HOST", str(server.get("host", "127.0.0.1"))),
        port=_as_int("port", os.environ.get("PORT", server.get("port", 8000))),
        base_url=os.environ.get(
            "LLM_BASE_URL", str(llm.get("base_url", "https://api.openai.com/v1"))
        ).strip(),
        api_key=os.environ.get("LLM_API_KEY", "").strip(),
        model=os.environ.get("LLM_MODEL", str(llm.get("model", "gpt-4o-mini"))).strip(),
        temperature=_as_float(
            "temperature", os.environ.get("LLM_TEMPERATURE", llm.get("temperature", 0.2))
        ),
        max_tokens=_as_int("max_tokens", llm.get("max_tokens", 300)),
        timeout_seconds=_as_float(
            "timeout_seconds",
            os.environ.get("LLM_TIMEOUT_SECONDS", llm.get("timeout_seconds", 15)),
        ),
        max_attempts=_as_int(
            "max_attempts", os.environ.get("LLM_MAX_ATTEMPTS", llm.get("max_attempts", 3))
        ),
        retry_base_delay_seconds=_as_float(
            "retry_base_delay_seconds", llm.get("retry_base_delay_seconds", 0.5)
        ),
        cache_ttl_seconds=_as_int(
            "cache_ttl_seconds",
            os.environ.get("CACHE_TTL_SECONDS", cache.get("ttl_seconds", 600)),
        ),
        max_message_length=_as_int(
            "max_message_length", validation.get("max_message_length", 1000)
        ),
        log_file=log_file,
        log_level=os.environ.get("LOG_LEVEL", str(logging_cfg.get("level", "INFO"))),
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    if not settings.host.strip():
        raise ValueError("host не задан")
    if not 1 <= settings.port <= 65535:
        raise ValueError("port должен быть от 1 до 65535")
    if not settings.base_url.startswith(("http://", "https://")):
        raise ValueError("LLM base_url должен начинаться с http:// или https://")
    if not settings.model:
        raise ValueError("model не задана")
    if not 0 <= settings.temperature <= 2:
        raise ValueError("temperature должен быть от 0 до 2")
    if settings.max_tokens < 1:
        raise ValueError("max_tokens должен быть положительным")
    if settings.timeout_seconds <= 0:
        raise ValueError("timeout_seconds должен быть положительным")
    if not 1 <= settings.max_attempts <= 3:
        raise ValueError("max_attempts должен быть от 1 до 3")
    if settings.retry_base_delay_seconds < 0:
        raise ValueError("retry_base_delay_seconds не может быть отрицательным")
    if settings.cache_ttl_seconds <= 0:
        raise ValueError("cache_ttl_seconds должен быть положительным")
    if settings.max_message_length < 1:
        raise ValueError("max_message_length должен быть положительным")
    if settings.log_level.upper() not in _LOG_LEVELS:
        raise ValueError("Неизвестный уровень логирования")


def _read_yaml(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Не удалось прочитать {path.name}: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Некорректный YAML в {path.name}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} должен содержать объект")
    return raw


def _section(raw: dict, name: str) -> dict:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Секция {name} должна быть объектом")
    return value


def _as_int(name: str, value: object) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Некорректное целое значение {name}") from exc


def _as_float(name: str, value: object) -> float:
    try:
        if isinstance(value, bool):
            raise TypeError
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Некорректное числовое значение {name}") from exc
