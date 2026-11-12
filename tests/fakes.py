"""Тестовый двойник модели. Не используется сервисом."""

from __future__ import annotations


class FakeLLM:
    def __init__(self, result: str = '{"summary": "Краткий итог."}', error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls: list[dict] = []

    def complete(self, *, system_prompt: str, user_prompt: str, request_id: str) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "request_id": request_id,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result
