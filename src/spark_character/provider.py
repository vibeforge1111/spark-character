"""OpenAI-compatible direct provider.

Works with anything that exposes an OpenAI-compatible /chat/completions
endpoint: Z.AI, MiniMax, OpenAI itself, Ollama in OpenAI-compat mode,
Together, Groq, etc.

We deliberately do not depend on any vendor SDK. One httpx call, one
response shape.
"""

from __future__ import annotations

import json as _json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

# Maximum response body size in bytes (50 MB). Prevents OOM from
# misconfigured providers returning multi-GB payloads.
MAX_RESPONSE_BODY_BYTES: int = 50 * 1024 * 1024


class ResponseBodyTooLarge(RuntimeError):
    """Raised when a provider response exceeds the allowed body size."""


def _check_content_length(resp: httpx.Response) -> None:
    """Reject responses whose Content-Length header exceeds MAX_RESPONSE_BODY_BYTES."""
    content_length = resp.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError:
            return  # malformed; we'll catch size issues during read
        if length > MAX_RESPONSE_BODY_BYTES:
            raise ResponseBodyTooLarge(
                f"Provider response Content-Length {length} bytes "
                f"exceeds limit of {MAX_RESPONSE_BODY_BYTES} bytes."
            )


def _read_stream_body(resp: httpx.Response) -> bytes:
    """Read a streaming response body, enforcing MAX_RESPONSE_BODY_BYTES.

    Must be called while the response stream is still open (inside a
    ``client.stream(...)`` context).
    """
    _check_content_length(resp)
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_bytes(chunk_size=65536):
        total += len(chunk)
        if total > MAX_RESPONSE_BODY_BYTES:
            raise ResponseBodyTooLarge(
                f"Provider response body exceeded "
                f"{MAX_RESPONSE_BODY_BYTES} bytes while streaming."
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_stream_body_async(resp: httpx.Response) -> bytes:
    """Async variant of _read_stream_body."""
    _check_content_length(resp)
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.aiter_bytes(chunk_size=65536):
        total += len(chunk)
        if total > MAX_RESPONSE_BODY_BYTES:
            raise ResponseBodyTooLarge(
                f"Provider response body exceeded "
                f"{MAX_RESPONSE_BODY_BYTES} bytes while streaming."
            )
        chunks.append(chunk)
    return b"".join(chunks)


ALLOWED_PROVIDER_HOSTS = frozenset(
    {
        "api.z.ai",
        "api.minimax.io",
        "api.openai.com",
        "api.anthropic.com",
        "api.groq.com",
        "api.together.xyz",
        "localhost",
        "127.0.0.1",
        "::1",
    }
)


@dataclass(frozen=True)
class ProviderSpec:
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 60.0

    @classmethod
    def from_env(
        cls,
        *,
        api_key_env: str = "ZAI_API_KEY",
        base_url_env: str = "ZAI_BASE_URL",
        model_env: str = "ZAI_MODEL",
        default_base_url: str = "https://api.z.ai/api/coding/paas/v4/",
        default_model: str = "glm-5.1",
    ) -> "ProviderSpec":
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key: env var {api_key_env} is not set."
            )
        return cls(
            base_url=validate_provider_base_url(os.environ.get(base_url_env, default_base_url)),
            model=os.environ.get(model_env, default_model),
            api_key=api_key,
        )


def validate_provider_base_url(base_url: str) -> str:
    parsed = urlparse(str(base_url).strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" and host not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("Provider base URL must use HTTPS.")
    if not host or host not in ALLOWED_PROVIDER_HOSTS:
        allowed = ", ".join(sorted(ALLOWED_PROVIDER_HOSTS))
        raise RuntimeError(f"Provider base URL host is not allowed: {host or '<missing>'}. Allowed hosts: {allowed}.")
    return str(base_url).strip()


def _join_url(base_url: str, path_name: str) -> str:
    safe_base_url = validate_provider_base_url(base_url)
    return f"{safe_base_url.rstrip('/')}/{path_name.lstrip('/')}"


def _parse_provider_response_json(resp: httpx.Response, raw_body: bytes) -> dict[str, Any]:
    """Parse provider JSON without leaking raw provider bodies in errors.

    *raw_body* is the pre-read, size-checked response bytes so that
    ``resp.json()`` (which would re-read without a size cap) is never
    called directly.
    """
    content_type = (resp.headers.get("content-type") or "").lower()
    if content_type and "json" not in content_type:
        raise RuntimeError(
            f"Provider returned non-JSON content-type (status {resp.status_code}): {content_type}"
        )
    try:
        body = _json.loads(raw_body)
    except ValueError as exc:
        raise RuntimeError(f"Provider returned invalid JSON (status {resp.status_code}).") from exc
    if not isinstance(body, dict):
        raise RuntimeError("Provider JSON body must be an object.")
    return body


def call_provider(
    *,
    provider: ProviderSpec,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.7,
    extra_messages: list[dict[str, str]] | None = None,
    disable_thinking: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    """Synchronous chat-completions call. Returns the assistant text only.

    Pass disable_thinking=True for short structured outputs (judge scores,
    classifiers) when the backend is a reasoning model whose thinking
    phase can exhaust the token budget before any visible output lands.

    Pass tools=[{...}] to attach native tools the provider supports (e.g.
    Z.AI's `web_search`). The provider chooses when to call them; the
    final assistant text is returned to the caller.
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    messages.append({"role": "user", "content": user_prompt})
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    if tools:
        payload["tools"] = tools
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    url = _join_url(provider.base_url, "chat/completions")
    with httpx.Client(timeout=provider.timeout_seconds) as client:
        with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            raw_body = _read_stream_body(resp)
            body = _parse_provider_response_json(resp, raw_body)
    return _extract_text(body)


async def call_provider_async(
    *,
    provider: ProviderSpec,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.7,
    extra_messages: list[dict[str, str]] | None = None,
    disable_thinking: bool = False,
    tools: list[dict[str, Any]] | None = None,
) -> str:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    messages.append({"role": "user", "content": user_prompt})
    payload: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if disable_thinking:
        payload["thinking"] = {"type": "disabled"}
    if tools:
        payload["tools"] = tools
    headers = {
        "Authorization": f"Bearer {provider.api_key}",
        "Content-Type": "application/json",
    }
    url = _join_url(provider.base_url, "chat/completions")
    async with httpx.AsyncClient(timeout=provider.timeout_seconds) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            raw_body = await _read_stream_body_async(resp)
            body = _parse_provider_response_json(resp, raw_body)
    return _extract_text(body)


_THINK_BLOCK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN_ONLY = re.compile(r"<think\b[^>]*>.*?(?=<\w|$)", re.IGNORECASE | re.DOTALL)


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks that some providers
    (notably MiniMax) emit inline as literal text inside content."""
    if not text or "<think" not in text.lower():
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    if "<think" in cleaned.lower():
        cleaned = _THINK_OPEN_ONLY.sub("", cleaned)
    return cleaned.strip()


def _extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return _strip_think_blocks(str(content)).strip()
