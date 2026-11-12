"""Сценарий суммаризации: промпт, кеш, модель, очистка ответа."""

from __future__ import annotations

import logging
import time

from cache.keys import build_cache_key
from cache.memory_cache import TTLCache
from config.logging_setup import elapsed_ms, log_event
from config.settings import Settings
from errors import LLMResponseError
from llm.client import LLMClient
from llm.postprocess import postprocess_summary
from llm.prompt_builder import build_summarize_prompt


class ChatService:
    def __init__(self, settings: Settings, llm_client: LLMClient, cache: TTLCache) -> None:
        self.settings = settings
        self._llm = llm_client
        self._cache = cache

    def handle(self, message: str, request_id: str) -> dict:
        scenario_started = time.perf_counter()
        scenario = "summarize"
        log_event(
            logging.INFO,
            "scenario_selected",
            request_id=request_id,
            scenario=scenario,
            elapsed_ms=elapsed_ms(scenario_started),
        )

        prompt_started = time.perf_counter()
        system_prompt, user_prompt = build_summarize_prompt(message)
        log_event(
            logging.INFO,
            "prompt_built",
            request_id=request_id,
            elapsed_ms=elapsed_ms(prompt_started),
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        cache_started = time.perf_counter()
        cache_key = build_cache_key(
            message=message,
            model=self.settings.model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            system_prompt=system_prompt,
        )
        cached = self._cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("summary"), str):
            log_event(
                logging.INFO,
                "cache_hit",
                request_id=request_id,
                cache_key=cache_key,
                elapsed_ms=elapsed_ms(cache_started),
                ttl_seconds=self.settings.cache_ttl_seconds,
            )
            return {
                "scenario": scenario,
                "summary": cached["summary"],
                "cached": True,
            }

        log_event(
            logging.INFO,
            "cache_miss",
            request_id=request_id,
            cache_key=cache_key,
            elapsed_ms=elapsed_ms(cache_started),
            ttl_seconds=self.settings.cache_ttl_seconds,
        )

        raw = self._llm.complete(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            request_id=request_id,
        )
        preview = raw if len(raw) <= 4000 else raw[:4000] + "…"
        log_event(
            logging.INFO,
            "model_response",
            request_id=request_id,
            model_response=preview,
            model_response_chars=len(raw),
        )

        post_started = time.perf_counter()
        try:
            summary = postprocess_summary(raw)
        except LLMResponseError as exc:
            log_event(
                logging.ERROR,
                "postprocess_failed",
                request_id=request_id,
                elapsed_ms=elapsed_ms(post_started),
                reason=exc.log_message,
            )
            raise

        self._cache.set(cache_key, {"summary": summary})
        log_event(
            logging.INFO,
            "postprocess_ok",
            request_id=request_id,
            elapsed_ms=elapsed_ms(post_started),
            summary=summary,
        )
        return {"scenario": scenario, "summary": summary, "cached": False}
