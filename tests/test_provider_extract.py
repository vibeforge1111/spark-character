"""Provider response extraction tests."""

from __future__ import annotations

import pytest

from spark_character.provider import ProviderSpec, _extract_text, _join_url, _strip_think_blocks, validate_provider_base_url


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
