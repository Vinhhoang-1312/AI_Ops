# Group 1, MLOps (DDM501.22), FSB
# Nguyen Sy Hung (25MSA33055)
"""Optional hosted-LLM backend for response generation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib import error, request


DEFAULT_RESPONSE_BACKEND = "auto"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-20b:free"
DEFAULT_OPENROUTER_APP_TITLE = "agentic-rag-based-it-support"


class LlmServiceError(RuntimeError):
    """Raised when the configured LLM backend cannot satisfy a request."""


@dataclass(frozen=True)
class LlmGenerationResult:
    """Generated text plus provider token usage, when returned by the backend."""

    text: str
    usage_details: dict[str, int] | None = None
    provider_usage: dict[str, Any] | None = None


def _response_backend() -> str:
    return (os.getenv("IT_SUPPORT_RESPONSE_BACKEND") or DEFAULT_RESPONSE_BACKEND).strip().lower()


def _openrouter_api_key() -> str:
    return (os.getenv("OPENROUTER_API_KEY") or "").strip()


def configured_openrouter_model() -> str:
    """Return the configured OpenRouter model name."""
    return (os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_MODEL).strip()


def _openrouter_timeout() -> int:
    raw = (os.getenv("OPENROUTER_TIMEOUT_SECONDS") or "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def _extract_text_from_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()
    return ""


def _extract_openrouter_usage(body: dict[str, Any]) -> tuple[dict[str, int] | None, dict[str, Any] | None]:
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None, None

    usage_details: dict[str, int] = {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    if isinstance(prompt_tokens, int):
        usage_details["input"] = prompt_tokens
    if isinstance(completion_tokens, int):
        usage_details["output"] = completion_tokens
    if isinstance(total_tokens, int):
        usage_details["total"] = total_tokens
    elif "input" in usage_details or "output" in usage_details:
        usage_details["total"] = usage_details.get("input", 0) + usage_details.get("output", 0)

    return usage_details or None, usage


def call_openrouter_result(prompt: str) -> LlmGenerationResult:
    api_key = _openrouter_api_key()
    if not api_key:
        raise LlmServiceError("OPENROUTER_API_KEY is not set")

    base_url = (os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).rstrip("/")
    model = configured_openrouter_model()
    app_title = (os.getenv("OPENROUTER_APP_TITLE") or DEFAULT_OPENROUTER_APP_TITLE).strip()
    http_referer = (os.getenv("OPENROUTER_HTTP_REFERER") or "").strip()

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "X-Title": app_title,
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer

    req = request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers=headers,
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=_openrouter_timeout()) as response:
            raw_body = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise LlmServiceError(f"OpenRouter request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise LlmServiceError(f"OpenRouter request failed: {exc.reason}") from exc

    try:
        body = json.loads(raw_body)
    except json.JSONDecodeError as exc:
        raise LlmServiceError("OpenRouter returned a non-JSON response") from exc

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LlmServiceError(f"OpenRouter response did not include choices: {body!r}")

    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LlmServiceError(f"OpenRouter response did not include a message payload: {body!r}")

    text = _extract_text_from_message(message)
    if not text:
        raise LlmServiceError("OpenRouter response message content was empty")
    usage_details, provider_usage = _extract_openrouter_usage(body)
    return LlmGenerationResult(text=text, usage_details=usage_details, provider_usage=provider_usage)


def call_openrouter(prompt: str) -> str:
    """Return only OpenRouter text for backward-compatible callers."""
    return call_openrouter_result(prompt).text


def maybe_generate_text_result(prompt: str) -> Optional[LlmGenerationResult]:
    """Return an LLM response plus usage when configured, else ``None`` for local fallback."""

    backend = _response_backend()
    if backend == "template":
        return None

    api_key = _openrouter_api_key()
    if not api_key:
        if backend == "auto":
            return None
        raise LlmServiceError("IT_SUPPORT_RESPONSE_BACKEND=openrouter requires OPENROUTER_API_KEY")

    try:
        return call_openrouter_result(prompt)
    except LlmServiceError:
        if backend == "auto":
            return None
        raise


def maybe_generate_text(prompt: str) -> Optional[str]:
    """Return an LLM response when configured, else ``None`` for local fallback."""
    result = maybe_generate_text_result(prompt)
    return result.text if result is not None else None
