"""OpenAI-compatible direct provider.

Works with anything that exposes an OpenAI-compatible /chat/completions
endpoint: Z.AI, MiniMax, OpenAI itself, Ollama in OpenAI-compat mode,
Together, Groq, etc.

We deliberately do not depend on any vendor SDK. One httpx call, one
response shape.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import httpx


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
            base_url=os.environ.get(base_url_env, default_base_url),
            model=os.environ.get(model_env, default_model),
            api_key=api_key,
        )


def _join_url(base_url: str, path_name: str) -> str:
    return f"{base_url.rstrip('/')}/{path_name.lstrip('/')}"


def call_provider(
    *,
    provider: ProviderSpec,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 600,
    temperature: float = 0.7,
    extra_messages: list[dict[str, str]] | None = None,
    disable_thinking: bool = False,
) -> str:
    """Synchronous chat-completions call. Returns the assistant text only.

    Pass disable_thinking=True for short structured outputs (judge scores,
    classifiers) when the backend is a reasoning model whose thinking
    phase can exhaust the token budget before any visible output lands.
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


def _extract_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    msg = choices[0].get("message") or {}
    content = msg.get("content") or msg.get("reasoning_content") or ""
    return str(content).strip()
