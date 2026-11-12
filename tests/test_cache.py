"""Ключ кеша и TTL."""

from __future__ import annotations

from cache.keys import build_cache_key
from cache.memory_cache import TTLCache


def _key(**overrides) -> str:
    payload = {
        "message": "текст",
        "model": "demo",
        "temperature": 0.2,
        "max_tokens": 300,
        "system_prompt": "инструкция",
    }
    payload.update(overrides)
    return build_cache_key(**payload)


def test_key_changes_when_request_model_or_prompt_changes():
    base = _key()
    variants = [
        _key(message="другой"),
        _key(model="other"),
        _key(temperature=0.3),
        _key(max_tokens=100),
        _key(system_prompt="другая инструкция"),
    ]
    assert len(base) == 64
    int(base, 16)
    assert base == _key()
    assert len({base, *variants}) == 6


def test_ttl_expiry_and_refresh():
    now = {"t": 100.0}
    cache = TTLCache(10, clock=lambda: now["t"])
    cache.set("a", {"summary": "x"})
    now["t"] = 109.999
    assert cache.get("a") == {"summary": "x"}
    now["t"] = 110.0
    assert cache.get("a") is None

    now["t"] = 200.0
    cache.set("a", {"summary": "x"})
    now["t"] = 205.0
    cache.set("a", {"summary": "y"})
    now["t"] = 214.0
    assert cache.get("a") == {"summary": "y"}
    now["t"] = 215.0
    assert cache.get("a") is None


def test_stored_value_is_isolated_from_caller():
    cache = TTLCache(10, clock=lambda: 0.0)
    payload = {"summary": "x"}
    cache.set("a", payload)
    payload["summary"] = "changed"
    loaded = cache.get("a")
    assert loaded == {"summary": "x"}
    loaded["summary"] = "mutated"
    assert cache.get("a") == {"summary": "x"}


def test_ttl_must_be_positive():
    try:
        TTLCache(0)
    except ValueError:
        return
    raise AssertionError("нулевой TTL должен быть отклонён")
