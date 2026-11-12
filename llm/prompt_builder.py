"""Сборка системной инструкции и пользовательского запроса."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Суммаризируй текст пользователя кратко, "
    "на русском языке, в одном-трёх предложениях. Не добавляй фактов, "
    'которых нет в тексте. Верни только JSON-объект вида {"summary": "краткий пересказ"} '
    "без markdown и без пояснений."
)


def build_summarize_prompt(message: str) -> tuple[str, str]:
    user_prompt = f"Суммаризируй текст:\n\n{message}"
    return SYSTEM_PROMPT, user_prompt
