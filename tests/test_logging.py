"""JSON-формат логов."""

from __future__ import annotations

import json
import logging
import sys

from config.logging_setup import JsonFormatter, LOGGER_NAME, setup_logging


def test_formatter_includes_fields_and_traceback():
    formatter = JsonFormatter()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = logging.LogRecord(
            name=LOGGER_NAME,
            level=logging.ERROR,
            pathname=__file__,
            lineno=1,
            msg="request_failed",
            args=(),
            exc_info=sys.exc_info(),
        )
    record.fields = {"request_id": "abc", "event": "must-not-overwrite"}
    payload = json.loads(formatter.format(record))
    assert payload["event"] == "request_failed"
    assert payload["level"] == "ERROR"
    assert payload["logger"] == LOGGER_NAME
    assert payload["request_id"] == "abc"
    assert "RuntimeError" in payload["exception"]
    assert "timestamp" in payload


def test_setup_writes_json_line(tmp_path):
    logger = logging.getLogger(LOGGER_NAME)
    original_handlers = list(logger.handlers)
    original_propagate = logger.propagate
    original_level = logger.level
    log_file = tmp_path / "service.log"
    try:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        setup_logging(log_file, "INFO")
        logger.info("cache_hit", extra={"fields": {"cache_key": "abc", "elapsed_ms": 1.5}})
        payload = json.loads(log_file.read_text(encoding="utf-8").splitlines()[-1])
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
        for handler in original_handlers:
            logger.addHandler(handler)
        logger.propagate = original_propagate
        logger.setLevel(original_level)

    assert payload["event"] == "cache_hit"
    assert payload["cache_key"] == "abc"
    assert payload["elapsed_ms"] == 1.5
