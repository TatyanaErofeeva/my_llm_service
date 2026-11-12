"""Потокобезопасный словарь с TTL."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable


class TTLCache:
    def __init__(self, ttl_seconds: float, clock: Callable[[], float] | None = None) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds должен быть положительным")
        self._ttl = float(ttl_seconds)
        self._clock = clock or time.monotonic
        self._items: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        now = self._clock()
        with self._lock:
            self._purge(now)
            item = self._items.get(key)
            if item is None:
                return None
            return copy.deepcopy(item[1])

    def set(self, key: str, value: dict) -> None:
        if not isinstance(value, dict):
            raise TypeError("В кеш можно положить только словарь")
        now = self._clock()
        with self._lock:
            self._purge(now)
            self._items[key] = (now + self._ttl, copy.deepcopy(value))

    def _purge(self, now: float) -> None:
        expired = [key for key, (expires_at, _) in self._items.items() if expires_at <= now]
        for key in expired:
            del self._items[key]
