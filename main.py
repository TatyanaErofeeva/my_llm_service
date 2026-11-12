"""Точка входа: python main.py"""

from __future__ import annotations

import logging
import sys

from api.app import create_app
from config.logging_setup import log_event, setup_logging
from config.settings import load_settings


def main() -> None:
    try:
        settings = load_settings()
    except (OSError, ValueError) as exc:
        print(f"Не удалось загрузить конфигурацию: {exc}", file=sys.stderr)
        sys.exit(1)

    setup_logging(settings.log_file, settings.log_level)
    log_event(
        logging.INFO,
        "service_started",
        host=settings.host,
        port=settings.port,
        model=settings.model,
        temperature=settings.temperature,
        timeout_seconds=settings.timeout_seconds,
        max_attempts=settings.max_attempts,
        cache_ttl_seconds=settings.cache_ttl_seconds,
    )
    print(
        f"Сервис слушает http://{settings.host}:{settings.port}  (POST /chat)",
        flush=True,
    )
    app = create_app(settings)
    app.run(host=settings.host, port=settings.port, debug=False, use_reloader=False, threaded=True)


if __name__ == "__main__":
    main()
