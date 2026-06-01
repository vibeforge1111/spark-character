"""Pipeline tests with the network call patched out."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from spark_character import (
    ProviderSpec,
    generate,
    generate_with_critique,
    load_critic,
    load_persona,
)
from spark_character.critic import CritiqueResult
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


def test_critique_pass_keeps_draft() -> None:
    persona = load_persona("v1")
    critic = load_critic("v1")
    with patch(
        "spark_character.pipeline.call_provider",
        return_value="Ship the redesign now. Three reasons follow.",
    ), patch(
        "spark_character.critic.call_provider",
        return_value="PASS",
    ):
        result = generate_with_critique(
            "Should I ship?",
            provider=PROVIDER,
            persona=persona,
            critic=critic,
        )
    assert result.final == "Ship the redesign now. Three reasons follow."
    assert not result.rewritten
    assert result.critic_version == "v1"


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
