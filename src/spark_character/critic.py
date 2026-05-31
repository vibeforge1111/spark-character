"""Persona critic-rewriter.

Takes a draft reply, runs it through a critic LLM with the persona spec
as the rule set, returns either the original (if it passes) or a
rewritten version.

Cheap, provider-agnostic, evolvable. The critic prompt itself lives in
artifacts/critic.{version}.md so the harness can mutate it the same way
it mutates the persona.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .persona import ARTIFACTS_DIR, PersonaSpec
from .prompt_guard import sanitize_prompt_text
from .provider import ProviderSpec, call_provider, call_provider_async

DEFAULT_CRITIC_VERSION = "v1"
PASS_TOKEN = "PASS"


@dataclass(frozen=True)
class CriticSpec:
    version: str
    text: str

    @property
    def system_prompt(self) -> str:
        return self.text.strip()


@dataclass(frozen=True)
class CritiqueResult:
    final: str
    rewritten: bool
    draft: str


def load_critic(version: str = DEFAULT_CRITIC_VERSION) -> CriticSpec:
    path = ARTIFACTS_DIR / f"critic.{version}.md"
    if not path.exists():
        raise FileNotFoundError("Critic artifact not found")
    return CriticSpec(version=version, text=sanitize_prompt_text(path.read_text(encoding="utf-8")))