"""Provider response extraction tests."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager

import httpx
import pytest

import spark_character.provider as provider_module
from spark_character.provider import (
    ProviderSpec,
    ResponseBodyTooLarge,
    _read_stream_body,
    _extract_text,
    _join_url,
    _parse_provider_response_json,
    _strip_think_blocks,
    validate_provider_base_url,
)


def _response(status_code: int, *, content: bytes = b"", content_type: str = "application/json") -> httpx.Response:
    return httpx.Response(
        status_code,
        content=content,
        headers={"content-type": content_type},
        request=httpx.Request("POST", "https://api.z.ai/v1/chat/completions"),
    )


class _SyncClient:
    def __init__(self, outcomes: list[httpx.Response | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def __enter__(self) -> "_SyncClient":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @contextmanager
    def stream(self, *_args: object, **_kwargs: object):
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        yield outcome


class _AsyncStream:
    def __init__(self, outcome: httpx.Response | Exception) -> None:
        self.outcome = outcome

    async def __aenter__(self) -> httpx.Response:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    async def __aexit__(self, *_args: object) -> None:
        return None


class _AsyncClient:
    def __init__(self, outcomes: list[httpx.Response | Exception]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    async def __aenter__(self) -> "_AsyncClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def stream(self, *_args: object, **_kwargs: object) -> _AsyncStream:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        return _AsyncStream(outcome)


class _ChunkedStream(httpx.SyncByteStream):
    def __iter__(self):
        yield b"12"
        yield b"345"


def test_strip_complete_think_block() -> None:
    text = "<think>The user is asking about TVL</think>TVL is Total Value Locked."
    assert _strip_think_blocks(text) == "TVL is Total Value Locked."


def test_strip_multiline_think_block() -> None:
    text = "<think>\nLet me reason through this.\nStep 1: ...\n</think>\n\nShip it."
    assert _strip_think_blocks(text) == "Ship it."


def test_strip_open_only_think_when_no_close() -> None:
    text = "<think>I'm reasoning here without closing.\n\nThe answer is yes."
    cleaned = _strip_think_blocks(text)
    assert "<think" not in cleaned.lower()


def test_passthrough_when_no_think() -> None:
    text = "TVL is Total Value Locked."
    assert _strip_think_blocks(text) == text


def test_extract_text_strips_think_in_content() -> None:
    body = {
        "choices": [
            {"message": {"content": "<think>reasoning</think>Final answer here.", "role": "assistant"}}
        ]
    }
    assert _extract_text(body) == "Final answer here."


def test_extract_text_falls_back_to_reasoning_content() -> None:
    body = {
        "choices": [
            {"message": {"content": "", "reasoning_content": "Here is the reply.", "role": "assistant"}}
        ]
    }
    assert _extract_text(body) == "Here is the reply."


def test_extract_text_handles_missing_choices() -> None:
    assert _extract_text({}) == ""
    assert _extract_text({"choices": []}) == ""


def test_provider_base_url_requires_allowed_https_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "secret")
    monkeypatch.setenv("ZAI_BASE_URL", "https://evil.example/v1")
    with pytest.raises(RuntimeError, match="not allowed"):
        ProviderSpec.from_env()


def test_provider_env_blank_overrides_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", " secret ")
    monkeypatch.setenv("ZAI_BASE_URL", "   ")
    monkeypatch.setenv("ZAI_MODEL", "")

    spec = ProviderSpec.from_env(default_base_url="https://api.z.ai/v1/", default_model="default-model")

    assert spec == ProviderSpec(
        base_url="https://api.z.ai/v1/",
        model="default-model",
        api_key="secret",
    )


def test_provider_env_rejects_whitespace_only_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "   ")

    with pytest.raises(RuntimeError, match="Missing API key"):
        ProviderSpec.from_env()


def test_provider_base_url_rejects_plain_http_for_remote_host() -> None:
    with pytest.raises(RuntimeError, match="HTTPS"):
        validate_provider_base_url("http://api.z.ai/api/coding/paas/v4/")


def test_provider_base_url_allows_known_https_provider() -> None:
    assert validate_provider_base_url("https://api.z.ai/api/coding/paas/v4/") == "https://api.z.ai/api/coding/paas/v4/"


def test_provider_base_url_allows_localhost_for_local_openai_compatible_servers() -> None:
    assert validate_provider_base_url("http://127.0.0.1:11434/v1") == "http://127.0.0.1:11434/v1"


def test_join_url_validates_direct_provider_spec_base_url() -> None:
    with pytest.raises(RuntimeError, match="not allowed"):
        _join_url("https://evil.example/v1", "chat/completions")


def test_parse_provider_response_json_rejects_html_content_type_without_body_preview() -> None:
    resp = httpx.Response(200, text="<html>sensitive-marker</html>", headers={"content-type": "text/html"})

    with pytest.raises(RuntimeError, match="non-JSON content-type") as exc_info:
        _parse_provider_response_json(resp)

    assert "sensitive-marker" not in str(exc_info.value)


def test_parse_provider_response_json_accepts_object() -> None:
    resp = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "ok"}}]},
        headers={"content-type": "application/json"},
    )

    body = _parse_provider_response_json(resp)

    assert body["choices"][0]["message"]["content"] == "ok"


def test_parse_provider_response_json_rejects_invalid_json_without_body_preview() -> None:
    resp = httpx.Response(200, content=b"sensitive-marker", headers={"content-type": "application/json"})

    with pytest.raises(RuntimeError, match="invalid JSON") as exc_info:
        _parse_provider_response_json(resp)

    assert "sensitive-marker" not in str(exc_info.value)


def test_parse_provider_response_json_rejects_non_object_json() -> None:
    resp = httpx.Response(200, content=b"[]", headers={"content-type": "application/json"})

    with pytest.raises(RuntimeError, match="must be an object"):
        _parse_provider_response_json(resp)


def test_stream_body_rejects_oversized_content_length(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "MAX_RESPONSE_BODY_BYTES", 4)
    response = httpx.Response(200, headers={"content-length": "5"})

    with pytest.raises(ResponseBodyTooLarge, match="exceeds"):
        _read_stream_body(response)


def test_stream_body_rejects_oversized_chunked_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provider_module, "MAX_RESPONSE_BODY_BYTES", 4)
    response = httpx.Response(200, stream=_ChunkedStream())

    with pytest.raises(ResponseBodyTooLarge, match="exceeded"):
        _read_stream_body(response)


def test_call_provider_retries_rate_limit_then_returns(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _SyncClient(
        [
            _response(429),
            _response(200, content=b'{"choices":[{"message":{"content":"ok"}}]}'),
        ]
    )
    sleeps: list[float] = []
    monkeypatch.setattr(provider_module.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(provider_module.time, "sleep", sleeps.append)

    result = provider_module.call_provider(
        provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
        system_prompt="system",
        user_prompt="user",
    )

    assert result == "ok"
    assert client.calls == 2
    assert sleeps == [1.0]


def test_call_provider_does_not_replay_ambiguous_read_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    timeout = httpx.ReadTimeout("timed out", request=httpx.Request("POST", "https://api.z.ai/v1/chat/completions"))
    client = _SyncClient([timeout, _response(200)])
    monkeypatch.setattr(provider_module.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _delay: None)

    with pytest.raises(httpx.ReadTimeout):
        provider_module.call_provider(
            provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
            system_prompt="system",
            user_prompt="user",
        )

    assert client.calls == 1


def test_call_provider_does_not_replay_gateway_error(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _SyncClient([_response(503), _response(200)])
    monkeypatch.setattr(provider_module.httpx, "Client", lambda **_kwargs: client)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _delay: None)

    with pytest.raises(httpx.HTTPStatusError):
        provider_module.call_provider(
            provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
            system_prompt="system",
            user_prompt="user",
        )

    assert client.calls == 1


def test_call_provider_bounds_response_before_json_parse(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _SyncClient([_response(200, content=b"12345")])
    monkeypatch.setattr(provider_module, "MAX_RESPONSE_BODY_BYTES", 4)
    monkeypatch.setattr(provider_module.httpx, "Client", lambda **_kwargs: client)

    with pytest.raises(ResponseBodyTooLarge):
        provider_module.call_provider(
            provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
            system_prompt="system",
            user_prompt="user",
        )


def test_call_provider_async_retries_connect_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = httpx.ConnectError("refused", request=httpx.Request("POST", "https://api.z.ai/v1/chat/completions"))
    client = _AsyncClient(
        [failure, _response(200, content=b'{"choices":[{"message":{"content":"async ok"}}]}')]
    )
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(provider_module.httpx, "AsyncClient", lambda **_kwargs: client)
    monkeypatch.setattr(provider_module.asyncio, "sleep", fake_sleep)

    result = asyncio.run(
        provider_module.call_provider_async(
            provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
            system_prompt="system",
            user_prompt="user",
        )
    )

    assert result == "async ok"
    assert client.calls == 2
    assert sleeps == [1.0]
