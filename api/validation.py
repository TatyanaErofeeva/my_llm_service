"""Проверка входного JSON перед бизнес-логикой."""

from __future__ import annotations

from errors import ValidationError


def extract_message(payload: object, max_length: int) -> str:
    if not isinstance(payload, dict):
        raise ValidationError("Тело запроса должно быть JSON-объектом.")
    if "message" not in payload:
        raise ValidationError("Поле message обязательно.")
    message = payload["message"]
    if not isinstance(message, str):
        raise ValidationError("Поле message должно быть строкой.")
    if len(message) > max_length:
        raise ValidationError(f"Поле message не должно превышать {max_length} символов.")
    cleaned = message.strip()
    if not cleaned:
        raise ValidationError("Поле message не должно быть пустым.")
    return cleaned
