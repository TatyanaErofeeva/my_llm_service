"""Вызов OpenAI-совместимого API с таймаутом и ограниченными ретраями."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import requests

from config.logging_setup import elapsed_ms, log_event
from config.settings import Settings
from errors import (
    EMPTY_MODEL_RESPONSE_MESSAGE,
    LLMResponseError,
    LLMUnavailableError,
    PARSE_ERROR_MESSAGE,
    REJECTED_REQUEST_MESSAGE,
)


class _TransientLLMError(Exception):
    """Временный сбой: таймаут, обрыв сети, 408, 429 или 5xx."""


class LLMClient:
    def __init__(self, settings: Settings, sleep_fn: Callable[[float], None] | None = None) -> None:
        self._settings = settings
        self._sleep = sleep_fn or time.sleep
        self._session = requests.Session()

    def complete(self, *, system_prompt: str, user_prompt: str, request_id: str) -> str:
        if not self._settings.api_key:
            log_event(
                logging.ERROR,
                "llm_failed",
                request_id=request_id,
                reason="missing_api_key",
                attempts=0,
            )
            raise LLMUnavailableError()

        url = self._settings.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self._settings.model,
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, self._settings.max_attempts + 1):
            started = time.perf_counter()
            try:
                content = self._request(url, payload, headers, request_id, attempt)
            except _TransientLLMError as exc:
                will_retry = attempt < self._settings.max_attempts
                log_event(
                    logging.WARNING,
                    "llm_attempt_failed",
                    request_id=request_id,
                    attempt=attempt,
                    max_attempts=self._settings.max_attempts,
                    elapsed_ms=elapsed_ms(started),
                    reason=str(exc),
                    will_retry=will_retry,
                )
                if not will_retry:
                    log_event(
                        logging.ERROR,
                        "llm_failed",
                        request_id=request_id,
                        attempts=attempt,
                        reason=str(exc),
                    )
                    raise LLMUnavailableError() from exc
                delay = self._settings.retry_base_delay_seconds * (2 ** (attempt - 1))
                log_event(
                    logging.INFO,
                    "llm_retry",
                    request_id=request_id,
                    attempt=attempt,
                    delay_seconds=delay,
                )
                self._sleep(delay)
                continue

            log_event(
                logging.INFO,
                "llm_success",
                request_id=request_id,
                attempt=attempt,
                elapsed_ms=elapsed_ms(started),
                response_chars=len(content),
            )
            return content

        raise LLMUnavailableError()

    def _request(
        self,
        url: str,
        payload: dict,
        headers: dict,
        request_id: str,
        attempt: int,
    ) -> str:
        log_event(
            logging.INFO,
            "llm_request",
            request_id=request_id,
            attempt=attempt,
            url=url,
            model=self._settings.model,
            temperature=self._settings.temperature,
            timeout_seconds=self._settings.timeout_seconds,
        )
        try:
            response = self._session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._settings.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise _TransientLLMError(f"{type(exc).__name__}: {exc}") from exc

        if response.status_code in (408, 429) or response.status_code >= 500:
            raise _TransientLLMError(f"http_{response.status_code}")

        if response.status_code >= 400:
            log_event(
                logging.ERROR,
                "llm_rejected",
                request_id=request_id,
                status_code=response.status_code,
                body_preview=response.text[:500],
            )
            raise LLMResponseError(REJECTED_REQUEST_MESSAGE, log_message="provider_rejected")

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMResponseError(
                PARSE_ERROR_MESSAGE,
                log_message="invalid_provider_json",
            ) from exc
        return extract_content(data)


def extract_content(data: object) -> str:
    if not isinstance(data, dict):
        raise LLMResponseError(PARSE_ERROR_MESSAGE, log_message="provider_payload_not_object")
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMResponseError(PARSE_ERROR_MESSAGE, log_message="provider_choices_missing")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LLMResponseError(PARSE_ERROR_MESSAGE, log_message="provider_message_missing")

    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        content = "".join(parts)

    if not isinstance(content, str) or not content.strip():
        raise LLMResponseError(EMPTY_MODEL_RESPONSE_MESSAGE, log_message="empty_content")
    return content
