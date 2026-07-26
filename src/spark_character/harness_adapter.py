"""Adapter to plug spark-character into the spark-self-evolving-harness.

The harness expects an async `run_fn(prompt: str) -> Result` where Result
has a `final_response: str` attribute. This module provides exactly that.

Usage from the harness side:

    from spark_character.harness_adapter import build_run_fn

    run_fn = build_run_fn(provider=ProviderSpec.from_env())

    score = await L5HygieneEvaluator(...).evaluate(run_fn=run_fn)

Or with the critic-rewrite pass enabled:

    run_fn = build_run_fn(provider=..., use_critic=True)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Awaitable, Callable

from .critic import CriticSpec, load_critic
from .persona import PersonaSpec, detect_provider_kind, load_persona
from .pipeline import generate_async, generate_with_critique_async
from .provider import ProviderSpec


@dataclass
class HarnessResult:
    final_response: str
    draft: str
    rewritten: bool
    persona_version: str
    critic_version: str | None


RunFn = Callable[[str], Awaitable[HarnessResult]]


def build_run_fn(
    *,
    provider: ProviderSpec,
    persona: PersonaSpec | None = None,
    critic: CriticSpec | None = None,
    use_critic: bool = False,
    max_tokens: int = 600,
    temperature: float = 0.7,
) -> RunFn:
    provider_kind = detect_provider_kind(provider) if persona is None else None

    async def _run(prompt: str) -> HarnessResult:
        p = persona if persona is not None else load_persona(provider_kind=provider_kind)
        c = critic if critic is not None else (load_critic() if use_critic else None)
        if use_critic and c is not None:
            result = await generate_with_critique_async(
                prompt,
                provider=provider,
                persona=p,
                critic=c,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            result = await generate_async(
                prompt,
                provider=provider,
                persona=p,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        return HarnessResult(
            final_response=result.final,
            draft=result.draft,
            rewritten=result.rewritten,
            persona_version=result.persona_version,
            critic_version=result.critic_version,
        )

    return _run
