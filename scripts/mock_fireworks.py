#!/usr/bin/env python3
"""Tiny OpenAI-compatible mock server for Track 1 container tests.

It lets you verify the Docker contract without a real Fireworks API key:
- the app reads FIREWORKS_* env vars
- the OpenAI client sends requests through FIREWORKS_BASE_URL
- the container reads /input/tasks.json
- the container writes /output/results.json

This is only for local/CI compliance testing. It does not test real answer accuracy.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re


def _answer_for(prompt: str) -> str:
    low = prompt.lower()
    if "1,200" in prompt or "1200" in prompt:
        return "1269.6"
    if "export button fails" in low or "crashes every time" in low:
        return "Mixed — it includes positive feedback, but a serious failure creates a negative user experience."
    if "july 4, 2026" in low and "maria santos" in low:
        return "Maria Santos: Person; AMD: Organization; Austin, Texas: Location; July 4, 2026: Date."
    if "len(num)" in prompt:
        return "The bug is len(num), because num is undefined. Use len(nums):\n\ndef average(nums):\n    total = 0\n    for n in nums:\n        total += n\n    return total / len(nums)"
    if "alice" in low and "bob" in low and "carla" in low and "bird" in low:
        return "Carla owns the bird, Alice owns the cat, and Bob owns the dog."
    if "is_palindrome" in low:
        return "import re\n\ndef is_palindrome(text):\n    cleaned = re.sub(r'[^a-z0-9]', '', text.lower())\n    return cleaned == cleaned[::-1]"
    if "kubernetes" in low:
        return "Kubernetes is an open-source platform for running and managing containerized applications. It automates deployment, scaling, networking, and recovery for application workloads. It helps teams operate distributed systems consistently across servers and clouds."
    if "summarise" in low or "summarize" in low:
        return "Cloud computing lets companies rent scalable infrastructure over the internet, lowering upfront costs and speeding deployment."
    return "Mock answer: the Track 1 Docker pipeline is working."


class Handler(BaseHTTPRequestHandler):
    server_version = "MockFireworks/1.0"

    def do_GET(self) -> None:
        if self.path in {"/", "/health", "/v1/health"}:
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length)
        prompt = ""
        model = "mock-model"
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
            model = str(body.get("model") or model)
            messages = body.get("messages") or []
            for msg in reversed(messages):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    prompt = str(msg.get("content") or "")
                    break
        except Exception:
            pass

        response = {
            "id": "mock-chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": _answer_for(prompt)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        self._send_json(response)

    def log_message(self, fmt: str, *args: object) -> None:
        if os.getenv("MOCK_FIREWORKS_VERBOSE"):
            super().log_message(fmt, *args)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    host = os.getenv("MOCK_FIREWORKS_HOST", "0.0.0.0")
    port = int(os.getenv("MOCK_FIREWORKS_PORT", "8000"))
    print(f"Mock Fireworks server listening on http://{host}:{port}/v1", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
