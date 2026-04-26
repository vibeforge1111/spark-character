"""OpenAI-compatible direct provider.

Works with anything that exposes an OpenAI-compatible /chat/completions
endpoint: Z.AI, MiniMax, OpenAI itself, Ollama in OpenAI-compat mode,
Together, Groq, etc.

We deliberately do not depend on any vendor SDK. One httpx call, one
response shape.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx


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
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
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
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        body = resp.json()
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
