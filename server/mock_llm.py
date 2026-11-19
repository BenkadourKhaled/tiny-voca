"""Development-only fake OpenAI-compatible LLM for testing the voice service.

It answers any question with a short Arabic sentence. It is NOT a real model:
it exists so the full pipeline (question -> answer -> tashkeel -> audio) can be
tested without an API key or an Ollama installation.

Run:  python server/mock_llm.py 8011
Then point the service at it:
  TINY_VOCA_LLM_URL=http://127.0.0.1:8011/v1/chat/completions
  TINY_VOCA_LLM_KEY=local
  TINY_VOCA_LLM_MODEL=mock-ar
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

ANSWER = "مرحبا بك، يسعدني التحدث معك اليوم."


def payload(question: str) -> dict:
    """OpenAI-shaped response body for a chat completion request."""
    return {
        "id": "mock-1",
        "object": "chat.completion",
        "model": "mock-ar",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": ANSWER},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": len(question.split()), "completion_tokens": 7, "total_tokens": 7},
    }


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 - http.server API
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        question = request.get("messages", [{}])[-1].get("content", "")
        body = json.dumps(payload(question), ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 - some clients probe /v1/models
        body = json.dumps({"object": "list", "data": [{"id": "mock-ar"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass                                    # keep the test output quiet


def main(port: int = 8011) -> None:
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8011)
