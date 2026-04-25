"""High-level generate-and-critique pipeline.

Two public entry points:

- generate(): one provider call with the persona system prompt. Cheap.
- generate_with_critique(): generate, then run the critic-rewriter
  pass. Roughly 2x tokens, much higher persona fidelity.

The harness wraps either of these as a run_fn to score voice during
evolution cycles.
"""

from __future__ import annotations

from dataclasses import dataclass

from .critic import (
    CriticSpec,
    CritiqueResult,
    critique,
    critique_async,
    load_critic,
)
from .persona import PersonaSpec, detect_provider_kind, load_persona
from .provider import ProviderSpec, call_provider, call_provider_async
from .scoring import score_persona


@dataclass(frozen=True)
class GenerationResult:
    final: str
    draft: str
    rewritten: bool
    persona_version: str
    critic_version: str | None


def generate(
    user_message: str,
    *,
    provider: ProviderSpec,
    persona: PersonaSpec | None = None,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 600,
    temperature: float = 0.7,
    disable_thinking: bool = True,
    tools: list[dict] | None = None,
) -> GenerationResult:
    """Generate a Spark reply. disable_thinking defaults to True so the
    reasoning phase of reasoning models (GLM 5.1, o1-style) does not
    leak structured "1. Analyze the Request" prefixes into the visible
    output when the token budget is tight. Pass False if you want the
    model to think aloud (only meaningful for some backends).

    Pass tools=[{...}] to attach native tools the backend supports for
    this turn (e.g. [{"type": "web_search", "web_search": {"enable": True}}]
    on Z.AI). The model decides when to call them; the final reply text
    is returned.

    When persona is None, the active version is loaded with the
    matching provider overlay automatically (Z.AI, MiniMax, etc.).
    Pass an explicit persona to override that behavior."""
    p = persona or load_persona(provider_kind=detect_provider_kind(provider))
    draft = call_provider(
        provider=provider,
        system_prompt=p.system_prompt,
        user_prompt=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_messages=history,
        disable_thinking=disable_thinking,
        tools=tools,
    )
    return GenerationResult(
        final=draft,
        draft=draft,
        rewritten=False,
        persona_version=p.version,
        critic_version=None,
    )


def generate_with_critique(
    user_message: str,
    *,
    provider: ProviderSpec,
    persona: PersonaSpec | None = None,
    critic: CriticSpec | None = None,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 600,
    temperature: float = 0.7,
    always_critique: bool = False,
    disable_thinking: bool = True,
) -> GenerationResult:
    """Generate, then run the critic only if the local scorers flag a
    persona violation in the draft. Set always_critique=True to bypass
    the gate and run the critic on every reply."""
    p = persona or load_persona(provider_kind=detect_provider_kind(provider))
    c = critic or load_critic()
    draft = call_provider(
        provider=provider,
        system_prompt=p.system_prompt,
        user_prompt=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_messages=history,
        disable_thinking=disable_thinking,
    )
    if not always_critique and score_persona(draft).passed:
        return GenerationResult(
            final=draft,
            draft=draft,
            rewritten=False,
            persona_version=p.version,
            critic_version=c.version,
        )
    result: CritiqueResult = critique(
        provider=provider, persona=p, critic=c, draft=draft, max_tokens=max_tokens
    )
    final = _accept_rewrite_or_keep(draft, result)
    return GenerationResult(
        final=final,
        draft=draft,
        rewritten=final != draft,
        persona_version=p.version,
        critic_version=c.version,
    )


async def generate_async(
    user_message: str,
    *,
    provider: ProviderSpec,
    persona: PersonaSpec | None = None,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 600,
    temperature: float = 0.7,
    disable_thinking: bool = True,
) -> GenerationResult:
    p = persona or load_persona(provider_kind=detect_provider_kind(provider))
    draft = await call_provider_async(
        provider=provider,
        system_prompt=p.system_prompt,
        user_prompt=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_messages=history,
        disable_thinking=disable_thinking,
    )
    return GenerationResult(
        final=draft,
        draft=draft,
        rewritten=False,
        persona_version=p.version,
        critic_version=None,
    )


async def generate_with_critique_async(
    user_message: str,
    *,
    provider: ProviderSpec,
    persona: PersonaSpec | None = None,
    critic: CriticSpec | None = None,
    history: list[dict[str, str]] | None = None,
    max_tokens: int = 600,
    temperature: float = 0.7,
    always_critique: bool = False,
    disable_thinking: bool = True,
) -> GenerationResult:
    p = persona or load_persona(provider_kind=detect_provider_kind(provider))
    c = critic or load_critic()
    draft = await call_provider_async(
        provider=provider,
        system_prompt=p.system_prompt,
        user_prompt=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_messages=history,
        disable_thinking=disable_thinking,
    )
    if not always_critique and score_persona(draft).passed:
        return GenerationResult(
            final=draft,
            draft=draft,
            rewritten=False,
            persona_version=p.version,
            critic_version=c.version,
        )
    result: CritiqueResult = await critique_async(
        provider=provider, persona=p, critic=c, draft=draft, max_tokens=max_tokens
    )
    final = _accept_rewrite_or_keep(draft, result)
    return GenerationResult(
        final=final,
        draft=draft,
        rewritten=final != draft,
        persona_version=p.version,
        critic_version=c.version,
    )


def _accept_rewrite_or_keep(draft: str, result: CritiqueResult) -> str:
    """Only accept a rewrite if it actually improves the persona score.

    Defends against critic outputs that leak meta-commentary like
    "Let me check the draft against the rules:" by comparing scores
    and falling back to the draft when the rewrite scores worse.
    """
    if not result.rewritten:
        return draft
    draft_score = score_persona(draft)
    rewrite_score = score_persona(result.final)
    if rewrite_score.mean >= draft_score.mean:
        return result.final
    return draft
