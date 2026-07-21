"""Harness adapter identity and critic loading boundaries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import spark_character.harness_adapter as adapter
from spark_character.critic import CriticSpec
from spark_character.persona import PersonaSpec


def _result(persona: PersonaSpec, critic: CriticSpec | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        final="final",
        draft="draft",
        rewritten=critic is not None,
        persona_version=persona.version,
        critic_version=critic.version if critic is not None else None,
    )


def test_implicit_persona_reloads_for_each_harness_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    personas = iter([PersonaSpec("v1", "one"), PersonaSpec("v2", "two")])
    loaded_kinds: list[str | None] = []

    def load_persona(*, provider_kind: str | None) -> PersonaSpec:
        loaded_kinds.append(provider_kind)
        return next(personas)

    async def generate(_prompt: str, *, persona: PersonaSpec, **_kwargs) -> SimpleNamespace:
        return _result(persona)

    monkeypatch.setattr(adapter, "detect_provider_kind", lambda _provider: "openai")
    monkeypatch.setattr(adapter, "load_persona", load_persona)
    monkeypatch.setattr(adapter, "generate_async", generate)

    run = adapter.build_run_fn(provider=SimpleNamespace())

    assert asyncio.run(run("first")).persona_version == "v1"
    assert asyncio.run(run("second")).persona_version == "v2"
    assert loaded_kinds == ["openai", "openai"]


def test_explicit_persona_remains_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = PersonaSpec("v7", "pinned")

    async def generate(_prompt: str, *, persona: PersonaSpec, **_kwargs) -> SimpleNamespace:
        return _result(persona)

    monkeypatch.setattr(
        adapter,
        "load_persona",
        lambda **_kwargs: pytest.fail("explicit persona must not be replaced"),
    )
    monkeypatch.setattr(adapter, "generate_async", generate)

    run = adapter.build_run_fn(provider=SimpleNamespace(), persona=explicit)

    assert asyncio.run(run("first")).persona_version == "v7"
    assert asyncio.run(run("second")).persona_version == "v7"


def test_implicit_critic_reloads_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    critics = iter([CriticSpec("v1", "one"), CriticSpec("v2", "two")])

    async def generate(
        _prompt: str,
        *,
        persona: PersonaSpec,
        critic: CriticSpec,
        **_kwargs,
    ) -> SimpleNamespace:
        return _result(persona, critic)

    monkeypatch.setattr(adapter, "load_persona", lambda **_kwargs: PersonaSpec("v1", "persona"))
    monkeypatch.setattr(adapter, "load_critic", lambda: next(critics))
    monkeypatch.setattr(adapter, "generate_with_critique_async", generate)

    run = adapter.build_run_fn(provider=SimpleNamespace(), use_critic=True)

    assert asyncio.run(run("first")).critic_version == "v1"
    assert asyncio.run(run("second")).critic_version == "v2"


def test_disabled_critic_is_never_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def generate(_prompt: str, *, persona: PersonaSpec, **_kwargs) -> SimpleNamespace:
        return _result(persona)

    monkeypatch.setattr(adapter, "load_persona", lambda **_kwargs: PersonaSpec("v1", "persona"))
    monkeypatch.setattr(
        adapter,
        "load_critic",
        lambda: pytest.fail("disabled critic must not load"),
    )
    monkeypatch.setattr(adapter, "generate_async", generate)

    run = adapter.build_run_fn(provider=SimpleNamespace(), use_critic=False)

    assert asyncio.run(run("prompt")).critic_version is None
