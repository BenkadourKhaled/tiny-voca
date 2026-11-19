"""Answer generation through an OpenAI-compatible chat endpoint.

The endpoint is configurable so the same code works with a hosted API or a
local server (Ollama, vLLM, llama.cpp...). No key is ever returned to the
client. The prompt asks for short, unvocalised Modern Standard Arabic: the
diacritics are added afterwards by ``server.diacritize``.
"""

from __future__ import annotations

import os
import re

import httpx

SYSTEM_PROMPT = (
    "Answer in Modern Standard Arabic. Use one or two short sentences, at most "
    "40 words. Plain text only: no diacritics, no Latin letters, no emoji, no "
    "Markdown, no quotation marks. Do not mention that you are an AI."
)

# Arabic letters and marks (including diacritics), Arabic digits, Arabic and
# common punctuation, spaces. Everything else (Latin text, emoji, URLs) is
# replaced by a space so the G2P never sees a symbol it cannot pronounce.
_ALLOWED = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF"
    r"0-9\u0660-\u0669\s.,!؟،؛:\-]"
)
_SPACES = re.compile(r"\s+")


def sanitize(text: str) -> str:
    """Remove characters the Arabic front end cannot pronounce."""
    clean = "".join(char if _ALLOWED.fullmatch(char) else " " for char in text)
    return _SPACES.sub(" ", clean).strip()


def cut_at_sentence(text: str, max_chars: int = 300) -> str:
    """Trim a reply to ``max_chars``, preferably after a sentence mark."""
    text = _SPACES.sub(" ", text).strip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    for mark in ("؟", ".", "!", "؛"):
        position = window.rfind(mark)
        if position >= max_chars // 2:
            return window[: position + 1].strip()
    position = window.rfind(" ")
    return (window[:position] if position > 0 else window).strip()


def generate_answer(question: str, *, client=None, history=None) -> str:
    """Generate a short Arabic reply, optionally aware of the conversation.

    ``history`` is a list of ``{"role": "user"|"assistant", "content": str}``
    turns. Only the last six turns are kept, so the prompt stays small.
    ``client`` is an object with a ``post(url, json=, headers=, timeout=)``
    method; it is only used by the tests. In production the module uses httpx.
    """
    url = os.getenv("TINY_VOCA_LLM_URL", "https://api.openai.com/v1/chat/completions")
    key = os.getenv("TINY_VOCA_LLM_KEY")
    model = os.getenv("TINY_VOCA_LLM_MODEL", "gpt-4o-mini")
    timeout = float(os.getenv("TINY_VOCA_LLM_TIMEOUT", "20"))
    if not key:
        raise RuntimeError(
            "TINY_VOCA_LLM_KEY is not set; for a local server set any value "
            "in it and point TINY_VOCA_LLM_URL at its /v1/chat/completions endpoint"
        )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = str(turn.get("content", "")).strip()
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 160,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if client is None:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    else:
        response = client.post(url, json=payload, headers=headers, timeout=timeout)
    data = response.json()
    return str(data["choices"][0]["message"]["content"]).strip()
