"""Ошибки слоёв сервиса и тексты, которые видит пользователь."""

from __future__ import annotations


FALLBACK_MESSAGE = "Сервис временно недоступен, попробуйте позже"
PARSE_ERROR_MESSAGE = "Не удалось обработать ответ модели. Попробуйте ещё раз."
EMPTY_MODEL_RESPONSE_MESSAGE = "Модель вернула пустой ответ."
REJECTED_REQUEST_MESSAGE = (
    "Модель отклонила запрос. Проверьте текст и повторите попытку."
)
INTERNAL_ERROR_MESSAGE = "Внутренняя ошибка сервиса."


class ServiceError(Exception):
    """Ожидаемая ошибка с HTTP-статусом и безопасным текстом для клиента."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        log_message: str | None = None,
    ) -> None:
        super().__init__(log_message or message)
        self.message = message
        self.status_code = status_code
        self.log_message = log_message or message


class ValidationError(ServiceError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)


class LLMUnavailableError(ServiceError):
    def __init__(self, message: str = FALLBACK_MESSAGE) -> None:
        super().__init__(message, status_code=503)


class LLMResponseError(ServiceError):
    def __init__(
        self,
        message: str = PARSE_ERROR_MESSAGE,
        *,
        log_message: str | None = None,
    ) -> None:
        super().__init__(message, status_code=502, log_message=log_message)
