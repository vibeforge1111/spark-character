"""Pipeline tests with the network call patched out."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from spark_character import (
    ProviderSpec,
    generate,
    generate_with_critique,
    generate_with_critique_async,
    load_critic,
    load_persona,
)
from spark_character.critic import CritiqueResult, _interpret
from spark_character.pipeline import _accept_rewrite_or_keep


PROVIDER = ProviderSpec(
    base_url="https://example.invalid/v1",
    model="stub-model",
    api_key="stub-key",
)


def test_generate_returns_draft_only() -> None:
    persona = load_persona("v1")
    with patch(
        "spark_character.pipeline.call_provider",
        return_value="My read: ship the redesign now.",
    ):
        result = generate("Should I ship?", provider=PROVIDER, persona=persona)
    assert result.final == "My read: ship the redesign now."
    assert result.draft == "My read: ship the redesign now."
    assert not result.rewritten
    assert result.persona_version == "v1"
    assert result.critic_version is None


def test_local_gate_pass_keeps_draft_without_claiming_critic_version() -> None:
    persona = load_persona("v1")
    critic = load_critic("v1")
    with patch(
        "spark_character.pipeline.call_provider",
        return_value="Ship the redesign now. Three reasons follow.",
    ), patch(
        "spark_character.pipeline.score_persona",
        return_value=SimpleNamespace(passed=True),
    ), patch("spark_character.pipeline.critique") as critic_call:
        result = generate_with_critique(
            "Should I ship?",
            provider=PROVIDER,
            persona=persona,
            critic=critic,
        )
    assert result.final == "Ship the redesign now. Three reasons follow."
    assert not result.rewritten
    assert result.critic_version is None
    critic_call.assert_not_called()


def test_async_local_gate_pass_keeps_draft_without_claiming_critic_version() -> None:
    persona = load_persona("v1")
    critic = load_critic("v1")
    provider_call = AsyncMock(return_value="Ship the redesign now. Three reasons follow.")
    critic_call = AsyncMock()
    with patch("spark_character.pipeline.call_provider_async", provider_call), patch(
        "spark_character.pipeline.score_persona",
        return_value=SimpleNamespace(passed=True),
    ), patch("spark_character.pipeline.critique_async", critic_call):
        result = asyncio.run(
            generate_with_critique_async(
                "Should I ship?",
                provider=PROVIDER,
                persona=persona,
                critic=critic,
            )
        )

    assert result.final == "Ship the redesign now. Three reasons follow."
    assert not result.rewritten
    assert result.critic_version is None
    critic_call.assert_not_awaited()


def test_critic_pass_token_requires_exact_match() -> None:
    draft = "Ship the redesign now."
    result = _interpret(draft, " pass ")

    assert result.final == draft
    assert result.rewritten is False


def test_critic_does_not_treat_pass_prefixed_words_as_pass_token() -> None:
    draft = "Ship the redesign now."
    for response in ("PASSING", "PASSAGE", "PASSION"):
        result = _interpret(draft, response)
        assert result.final == response
        assert result.rewritten is True


def test_critique_rewrites_when_persona_violation_present() -> None:
    persona = load_persona("v1")
    critic = load_critic("v1")
    bad_draft = "Great question \u2014 how can I help you today?"
    rewritten = "Ship now. Three reasons follow."
    with patch(
        "spark_character.pipeline.call_provider",
        return_value=bad_draft,
    ), patch(
        "spark_character.critic.call_provider",
        return_value=rewritten,
    ):
        result = generate_with_critique(
            "Should I ship?",
            provider=PROVIDER,
            persona=persona,
            critic=critic,
        )
    assert result.final == rewritten
    assert result.draft == bad_draft
    assert result.rewritten is True


def test_accept_rewrite_requires_strict_score_improvement() -> None:
    draft = "Ship it."
    rewrite = "Ship it now."
    result = CritiqueResult(final=rewrite, rewritten=True, draft=draft)

    with patch(
        "spark_character.pipeline.score_persona",
        side_effect=[SimpleNamespace(mean=0.8), SimpleNamespace(mean=0.8)],
    ):
        assert _accept_rewrite_or_keep(draft, result) == draft
