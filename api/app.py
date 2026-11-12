"""HTTP-эндпоинт /chat и сборка зависимостей."""

from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from api.validation import extract_message
from cache.memory_cache import TTLCache
from config.logging_setup import elapsed_ms, log_event, log_exception
from config.settings import Settings, load_settings
from errors import INTERNAL_ERROR_MESSAGE, ServiceError, ValidationError
from llm.client import LLMClient
from services.chat_service import ChatService

MAX_BODY_BYTES = 16 * 1024
BODY_TOO_LARGE_MESSAGE = "Тело запроса слишком большое."


def create_app(
    settings: Settings | None = None,
    llm_client: LLMClient | None = None,
    cache: TTLCache | None = None,
) -> Flask:
    settings = settings or load_settings()
    if llm_client is None:
        llm_client = LLMClient(settings)
    if cache is None:
        cache = TTLCache(ttl_seconds=settings.cache_ttl_seconds)
    service = ChatService(settings, llm_client, cache)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.json.ensure_ascii = False

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    @app.post("/chat")
    def chat():
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        log_event(
            logging.INFO,
            "request_received",
            request_id=request_id,
            path=request.path,
            method=request.method,
            remote_addr=request.remote_addr,
            content_length=request.content_length,
        )
        if request.content_length is not None and request.content_length > MAX_BODY_BYTES:
            log_event(
                logging.WARNING,
                "validation_failed",
                request_id=request_id,
                status_code=413,
                error=BODY_TOO_LARGE_MESSAGE,
                elapsed_ms=elapsed_ms(started),
            )
            return jsonify({"error": BODY_TOO_LARGE_MESSAGE}), 413
        try:
            message = extract_message(request.get_json(silent=True), service.settings.max_message_length)
            log_event(
                logging.INFO,
                "validation_ok",
                request_id=request_id,
                elapsed_ms=elapsed_ms(started),
                message_chars=len(message),
                user_message=message,
            )
            result = service.handle(message, request_id=request_id)
            log_event(
                logging.INFO,
                "response_ready",
                request_id=request_id,
                status_code=200,
                cached=result["cached"],
                elapsed_ms=elapsed_ms(started),
            )
            return jsonify(result), 200
        except ValidationError as exc:
            log_event(
                logging.WARNING,
                "validation_failed",
                request_id=request_id,
                status_code=exc.status_code,
                error=exc.message,
                elapsed_ms=elapsed_ms(started),
            )
            return jsonify({"error": exc.message}), exc.status_code
        except ServiceError as exc:
            log_event(
                logging.ERROR if exc.status_code >= 500 else logging.WARNING,
                "request_failed",
                request_id=request_id,
                status_code=exc.status_code,
                error=exc.message,
                reason=exc.log_message,
                error_type=type(exc).__name__,
                elapsed_ms=elapsed_ms(started),
            )
            return jsonify({"error": exc.message}), exc.status_code
        except RequestEntityTooLarge:
            raise
        except Exception:
            log_exception(
                "request_failed",
                request_id=request_id,
                status_code=500,
                elapsed_ms=elapsed_ms(started),
            )
            return jsonify({"error": INTERNAL_ERROR_MESSAGE}), 500

    @app.errorhandler(413)
    def request_too_large(_exc):
        return jsonify({"error": BODY_TOO_LARGE_MESSAGE}), 413

    @app.errorhandler(404)
    def not_found(_exc):
        return jsonify({"error": "Ресурс не найден."}), 404

    @app.errorhandler(405)
    def method_not_allowed(_exc):
        return jsonify({"error": "Метод не поддерживается."}), 405

    return app
