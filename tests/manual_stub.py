"""Локальная OpenAI-совместимая заглушка для ручной проверки.

Первые STUB_FAIL_FIRST запросов отвечают HTTP 500, чтобы был виден retry.
Если в пользовательском тексте есть маркер __bad_json__, заглушка возвращает
не-JSON и сервис должен ответить 502. Это поведение заглушки, не сервиса.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = "127.0.0.1"
PORT = 9999
FAIL_FIRST = int(os.environ.get("STUB_FAIL_FIRST", "0"))


class Handler(BaseHTTPRequestHandler):
    fail_remaining = FAIL_FIRST

    def do_GET(self) -> None:
        self._send(200, {"status": "ok"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        user_text = _user_text(raw)

        if "__bad_json__" in user_text:
            self._send(200, {"choices": [{"message": {"content": "это не json"}}]})
            return

        if Handler.fail_remaining > 0:
            Handler.fail_remaining -= 1
            self._send(500, {"error": "temporary"})
            return

        compact = " ".join(_source_text(user_text).split())
        summary = "Краткий пересказ: " + compact[:120]
        content = json.dumps({"summary": summary}, ensure_ascii=False)
        self._send(200, {"choices": [{"message": {"content": content}}]})

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def _user_text(raw: bytes) -> str:
    try:
        payload = json.loads(raw.decode("utf-8"))
        return str(payload["messages"][-1]["content"])
    except (UnicodeError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


def _source_text(user_text: str) -> str:
    marker = "\n\n"
    if marker in user_text:
        return user_text.split(marker, 1)[1]
    return user_text


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(
        f"Заглушка слушает http://{HOST}:{PORT}/v1/chat/completions "
        f"(первые ошибки: {FAIL_FIRST})",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
