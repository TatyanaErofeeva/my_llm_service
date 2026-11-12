"""Структурированные JSON-логи в консоль и в файл."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

LOGGER_NAME = "my_llm_service"
_RESERVED_FIELDS = {"timestamp", "level", "logger", "event", "exception"}


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class JsonFormatter(logging.Formatter):
    """Одна запись лога — один JSON-объект."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if key not in _RESERVED_FIELDS:
                    payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger


def log_event(level: int, event: str, **fields: object) -> None:
    get_logger().log(level, event, extra={"fields": fields})


def log_exception(event: str, **fields: object) -> None:
    get_logger().exception(event, extra={"fields": fields})


def setup_logging(log_file: Path, level: str = "INFO") -> None:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper()))
    logger.propagate = False
    if logger.handlers:
        return

    formatter = JsonFormatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
