"""Persona scoring (P1-P5).

Pure-function scorers that mirror the harness evaluators. Importable from
both the harness (as a run_fn target) and from this package's pipeline
for self-checks. No external dependencies.

The five axes:

- P1 em_dash:    hard fail if any em dash is present
- P2 plumbing:   penalty per internal subsystem leak
- P3 reset:      hard fail on canned check-in greetings
- P4 lead:       hard fail if first sentence is a hedge opener
- P5 voice_fit:  heuristic on warmth, directness, low formality
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .output_sanitizer import EM_DASH_FAMILY

EM_DASH = "\u2014"

PLUMBING_TERMS = (
    "spark researcher",
    "researcher bridge",
    "researcher runtime",
    "bridge error",
    "bridge mode",
    "memory bridge",
    "raw episode",
    "structured evidence",
    "belief packet",
    "domain chip",
    "active chip",
    "router",
    "routing decision",
    "guardrail",
    "trace ref",
    "trace_ref",
    "gateway",
    "provider fallback",
    "provider execution",
    "internal advisory",
)

PLUMBING_PATTERN = re.compile(
    "(?:" + "|".join(re.escape(t) for t in PLUMBING_TERMS) + ")",
    re.IGNORECASE,
)

RESET_PHRASES = (
    r"how can i help (?:you )?(?:today|now)\??",
    r"what(?:'s| is) on your mind\??",
    r"what are you working on\??",
    r"what(?:'s| is) up\??",
    r"how can i assist (?:you )?(?:today|now)?\??",
    r"how may i help (?:you )?(?:today|now)?\??",
    r"is there anything (?:else )?i can help (?:you )?with\??",
    r"what (?:would|do) you (?:like|want) to (?:talk about|focus on|work on)\??",
)
RESET_PATTERN = re.compile("|".join(RESET_PHRASES), re.IGNORECASE)

HEDGE_OPENERS = (
    r"^(?:great|good|interesting|excellent|that(?:'s| is) a (?:great|good|interesting)) (?:question|point)",
    r"^(?:certainly|of course|absolutely|sure thing|sure[!,.])",
    r"^(?:i think|i believe|i feel|in my opinion|it depends|it really depends)",
    r"^(?:well[,!.]|so[,!.]|hmm[,!.]|let me|let's see)",
    r"^(?:before (?:i|we) (?:answer|dive|get into))",
    r"^(?:to answer your question)",
    r"^(?:that(?:'s| is) a (?:tough|hard|tricky))",
)
HEDGE_PATTERN = re.compile("|".join(HEDGE_OPENERS), re.IGNORECASE)

ROBOTIC_TICS = (
    r"\bi am an? (?:ai|assistant|language model|llm)\b",
    r"\bas an? (?:ai|assistant|language model|llm)\b",
    r"\bi (?:do not|don't) have (?:personal )?(?:feelings|opinions|preferences)\b",
    r"\bi cannot (?:browse|access real-time|provide medical|provide legal)\b",
    r"\bplease (?:note|be aware|keep in mind) that\b",
    r"\bi hope this helps[!.]?\b",
    r"\bfeel free to ask\b",
    r"\bdon't hesitate to\b",
)
ROBOTIC_PATTERN = re.compile("|".join(ROBOTIC_TICS), re.IGNORECASE)

WARM_MARKERS = (
    r"\b(?:got it|fair|honestly|tbh|yeah|nah|noted|locked in|on it)\b",
    r"\bhere's (?:what|the|my)\b",
    r"\b(?:my call|my read|my take|the call) (?:is|here is)\b",
)
WARM_PATTERN = re.compile("|".join(WARM_MARKERS), re.IGNORECASE)


@dataclass(frozen=True)
class PersonaScore:
    p1_em_dash: float
    p2_plumbing: float
    p2_hits: tuple[str, ...]
    p3_reset: float
    p4_lead: float
    p5_voice: float
    p5_reason: str

    @property
    def mean(self) -> float:
        return round(
            (self.p1_em_dash + self.p2_plumbing + self.p3_reset + self.p4_lead + self.p5_voice) / 5,
            3,
        )

    @property
    def passed(self) -> bool:
        return all(
            v >= 0.7 for v in (self.p1_em_dash, self.p2_plumbing, self.p3_reset, self.p4_lead, self.p5_voice)
        )


def score_persona(text: str) -> PersonaScore:
    p1 = 0.0 if any(ch in text for ch in EM_DASH_FAMILY) else 1.0
    plumbing_hits = tuple(sorted({m.lower() for m in PLUMBING_PATTERN.findall(text)}))
    p2 = max(0.0, 1.0 - 0.25 * len(plumbing_hits)) if plumbing_hits else 1.0
    p3 = 0.0 if RESET_PATTERN.search(text) else 1.0
    first = _first_sentence(text)
    p4 = 0.0 if HEDGE_PATTERN.search(first) else 1.0
    p5_score, p5_reason = _voice_score(text)
    return PersonaScore(
        p1_em_dash=p1,
        p2_plumbing=p2,
        p2_hits=plumbing_hits,
        p3_reset=p3,
        p4_lead=p4,
        p5_voice=round(p5_score, 3),
        p5_reason=p5_reason,
    )


def _first_sentence(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", stripped, maxsplit=1)
    return parts[0] if parts else stripped


def _voice_score(text: str) -> tuple[float, str]:
    if not text.strip():
        return 0.0, "empty"
    robotic_hits = ROBOTIC_PATTERN.findall(text)
    warm_hits = WARM_PATTERN.findall(text)
    word_count = len(text.split())
    too_long_penalty = 0.0
    if word_count > 200:
        too_long_penalty = min(0.4, (word_count - 200) / 400)
    robotic_penalty = min(0.6, 0.2 * len(robotic_hits))
    warmth_bonus = min(0.2, 0.1 * len(warm_hits))
    raw = 1.0 - robotic_penalty - too_long_penalty + warmth_bonus
    score = max(0.0, min(1.0, raw))
    parts = []
    if robotic_hits:
        parts.append(f"robotic={len(robotic_hits)}")
    if too_long_penalty:
        parts.append(f"verbose={word_count}w")
    if warm_hits:
        parts.append(f"warm={len(warm_hits)}")
    return score, " ".join(parts) or "neutral"
