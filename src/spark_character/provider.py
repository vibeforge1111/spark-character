"""OpenAI-compatible direct provider.

Works with anything that exposes an OpenAI-compatible /chat/completions
endpoint: Z.AI, MiniMax, OpenAI itself, Ollama in OpenAI-compat mode,
Together, Groq, etc.

We deliberately do not depend on any vendor SDK. One httpx call, one
response shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx


MAX_RESPONSE_BODY_BYTES = 50 * 1024 * 1024
MAX_PROVIDER_ATTEMPTS = 4
RETRYABLE_PROVIDER_STATUS_CODES = frozenset({429})
RETRYABLE_PROVIDER_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout)


class ResponseBodyTooLarge(RuntimeError):
    """Raised when a provider response exceeds the bounded body size."""


def _check_content_length(resp: httpx.Response) -> None:
    content_length = resp.headers.get("content-length")
    if content_length is None:
        return
    try:
        length = int(content_length)
    except ValueError:
        return
    if length > MAX_RESPONSE_BODY_BYTES:
        raise ResponseBodyTooLarge(
            f"Provider response Content-Length {length} bytes exceeds "
            f"the {MAX_RESPONSE_BODY_BYTES}-byte limit."
        )


def _read_stream_body(resp: httpx.Response) -> bytes:
    """Read a provider stream while enforcing its decoded body-size limit."""
    _check_content_length(resp)
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_bytes(chunk_size=64 * 1024):
        total += len(chunk)
        if total > MAX_RESPONSE_BODY_BYTES:
            raise ResponseBodyTooLarge(
                f"Provider response body exceeded the {MAX_RESPONSE_BODY_BYTES}-byte limit."
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _read_stream_body_async(resp: httpx.Response) -> bytes:
    """Async variant of :func:`_read_stream_body`."""
    _check_content_length(resp)
    chunks: list[bytes] = []
    total = 0
    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
        total += len(chunk)
        if total > MAX_RESPONSE_BODY_BYTES:
            raise ResponseBodyTooLarge(
                f"Provider response body exceeded the {MAX_RESPONSE_BODY_BYTES}-byte limit."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _retry_delay_seconds(attempt: int, resp: httpx.Response | None = None) -> float:
    """Return bounded backoff, honoring a numeric Retry-After when present."""
    if resp is not None:
        retry_after = resp.headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(max(float(retry_after), 0.0), 60.0)
            except ValueError:
                pass
    return float(min(2**attempt, 8))


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

    @staticmethod
    def _env_or_default(env_name: str, default: str) -> str:
        value = (os.environ.get(env_name) or "").strip()
        return value or default

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
        api_key = (os.environ.get(api_key_env) or "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key: env var {api_key_env} is not set."
            )
        return cls(
            base_url=validate_provider_base_url(cls._env_or_default(base_url_env, default_base_url)),
            model=cls._env_or_default(model_env, default_model),
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


def _parse_provider_response_json(
    resp: httpx.Response,
    raw_body: bytes | None = None,
) -> dict[str, Any]:
    """Parse provider JSON without leaking raw provider bodies in errors."""
    content_type = (resp.headers.get("content-type") or "").lower()
    if content_type and "json" not in content_type:
        raise RuntimeError(
            f"Provider returned non-JSON content-type (status {resp.status_code}): {content_type}"
        )
    try:
        if raw_body is None:
            raw_body = resp.content
            if len(raw_body) > MAX_RESPONSE_BODY_BYTES:
                raise ResponseBodyTooLarge(
                    f"Provider response body exceeded the {MAX_RESPONSE_BODY_BYTES}-byte limit."
                )
        body = json.loads(raw_body)
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
        for attempt in range(MAX_PROVIDER_ATTEMPTS):
            retry_delay: float | None = None
            try:
                with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if (
                        resp.status_code in RETRYABLE_PROVIDER_STATUS_CODES
                        and attempt + 1 < MAX_PROVIDER_ATTEMPTS
                    ):
                        retry_delay = _retry_delay_seconds(attempt, resp)
                    else:
                        resp.raise_for_status()
                        body = _parse_provider_response_json(resp, _read_stream_body(resp))
                        return _extract_text(body)
            except RETRYABLE_PROVIDER_EXCEPTIONS:
                if attempt + 1 >= MAX_PROVIDER_ATTEMPTS:
                    raise
                retry_delay = _retry_delay_seconds(attempt)
            if retry_delay is not None:
                time.sleep(retry_delay)
    raise RuntimeError("Provider retry loop ended without a response.")


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
        for attempt in range(MAX_PROVIDER_ATTEMPTS):
            retry_delay: float | None = None
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if (
                        resp.status_code in RETRYABLE_PROVIDER_STATUS_CODES
                        and attempt + 1 < MAX_PROVIDER_ATTEMPTS
                    ):
                        retry_delay = _retry_delay_seconds(attempt, resp)
                    else:
                        resp.raise_for_status()
                        raw_body = await _read_stream_body_async(resp)
                        body = _parse_provider_response_json(resp, raw_body)
                        return _extract_text(body)
            except RETRYABLE_PROVIDER_EXCEPTIONS:
                if attempt + 1 >= MAX_PROVIDER_ATTEMPTS:
                    raise
                retry_delay = _retry_delay_seconds(attempt)
            if retry_delay is not None:
                await asyncio.sleep(retry_delay)
    raise RuntimeError("Provider retry loop ended without a response.")


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
