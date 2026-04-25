"""Numeric trait mutator for personality chips.

The prose mutator in evolve_persona.py rewrites the system prompt
markdown but cannot reason about OCEAN trait values. To actually evolve
a personality chip in the chip lab schema, we also need to mutate the
numbers: openness, conscientiousness, extraversion, agreeableness,
neuroticism, plus emotional_range entries (curiosity, satisfaction,
frustration, excitement, concern, humor).

This module:

1. Asks an LLM to propose bounded deltas based on diagnosed weaknesses.
2. Clamps the deltas to safe ranges (max ±0.10 per trait per cycle).
3. Returns a new PersonalityChip with mutated values and reasoning so
   the evolution loop can log why a particular trait moved.

Pair with the prose mutator: trait phase first (numbers), then prose
phase fed the new numbers as context so the spec stays consistent.

Closes ROADMAP gap #3.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, replace
from typing import Any

from .chip_loader import PersonalityChip
from .provider import ProviderSpec, call_provider


MAX_DELTA_PER_TRAIT = 0.10
MAX_DELTA_PER_EMOTIONAL_RANGE = 0.10
TRAIT_FIELDS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
EMOTIONAL_PROFILE_FIELDS = (
    "self_awareness",
    "self_regulation",
    "social_awareness",
)
EMOTIONAL_RANGE_KEYS = (
    "curiosity",
    "satisfaction",
    "frustration",
    "excitement",
    "concern",
    "humor",
)


TRAIT_MUTATOR_SYSTEM = (
    "You are a personality trait evolution engine for the Spark agent. "
    "You will see a baseline personality chip and a list of observed "
    "weaknesses on real model output. Your job is to propose small "
    "numerical deltas to the chip's OCEAN traits, emotional intelligence "
    "scores, and emotional range entries that would help the agent score "
    "better on those weaknesses next cycle.\n\n"
    "Hard constraints:\n"
    "- Each delta must be in the range [-0.10, +0.10]. Anything outside "
    "is clamped to that range.\n"
    "- All resulting values must remain in [0.0, 1.0] after clamping.\n"
    "- Do not propose deltas that would create internal contradictions "
    "(e.g. high agreeableness AND high disagreement frequency).\n"
    "- If a trait does not need to move based on the weaknesses, leave "
    "it out of the deltas object. Smaller deltas are better than larger.\n\n"
    "Output format: a single JSON object, no preamble, no code fences.\n"
    "{\n"
    "  \"reasoning\": \"short explanation of which weaknesses pulled which traits\",\n"
    "  \"deltas\": {\n"
    "    \"openness\": 0.03,\n"
    "    \"agreeableness\": -0.04\n"
    "  },\n"
    "  \"emotional_profile_deltas\": {\n"
    "    \"self_regulation\": 0.02\n"
    "  },\n"
    "  \"emotional_range_deltas\": {\n"
    "    \"frustration\": -0.02,\n"
    "    \"curiosity\": 0.03\n"
    "  }\n"
    "}\n"
    "Output ONLY this JSON object. No code fences, no commentary."
)


@dataclass(frozen=True)
class TraitMutationResult:
    chip: PersonalityChip
    deltas: dict[str, float]
    emotional_profile_deltas: dict[str, float]
    emotional_range_deltas: dict[str, float]
    reasoning: str
    raw_response: str


def mutate_trait_values(
    *,
    provider: ProviderSpec,
    chip: PersonalityChip,
    weaknesses: list[str],
    max_delta: float = MAX_DELTA_PER_TRAIT,
    temperature: float = 0.6,
) -> TraitMutationResult:
    """Propose and apply bounded trait deltas to a personality chip.

    Returns a TraitMutationResult with the mutated chip plus the deltas
    that were actually applied (after clamping). When no valid deltas
    can be parsed, returns the chip unchanged with empty delta dicts.
    """
    user_prompt = _build_user_prompt(chip, weaknesses)
    raw = call_provider(
        provider=provider,
        system_prompt=TRAIT_MUTATOR_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=600,
        temperature=temperature,
        disable_thinking=True,
    )
    parsed = _parse_trait_response(raw)
    deltas = _clamp_dict(parsed.get("deltas", {}), TRAIT_FIELDS, max_delta)
    profile_deltas = _clamp_dict(
        parsed.get("emotional_profile_deltas", {}),
        EMOTIONAL_PROFILE_FIELDS,
        max_delta,
    )
    range_deltas = _clamp_dict(
        parsed.get("emotional_range_deltas", {}),
        EMOTIONAL_RANGE_KEYS,
        max_delta,
    )
    new_chip = _apply_deltas(chip, deltas, profile_deltas, range_deltas)
    return TraitMutationResult(
        chip=new_chip,
        deltas=deltas,
        emotional_profile_deltas=profile_deltas,
        emotional_range_deltas=range_deltas,
        reasoning=str(parsed.get("reasoning") or ""),
        raw_response=raw,
    )


def _build_user_prompt(chip: PersonalityChip, weaknesses: list[str]) -> str:
    return (
        "[Baseline personality chip]\n"
        f"id: {chip.id}\n"
        f"name: {chip.name}\n"
        f"archetype: {chip.archetype}\n"
        f"voice_signature: {chip.voice_signature}\n"
        f"OCEAN: openness={chip.openness:.2f}, conscientiousness={chip.conscientiousness:.2f}, "
        f"extraversion={chip.extraversion:.2f}, agreeableness={chip.agreeableness:.2f}, "
        f"neuroticism={chip.neuroticism:.2f}\n"
        f"emotional_profile: self_awareness={chip.self_awareness:.2f}, "
        f"self_regulation={chip.self_regulation:.2f}, social_awareness={chip.social_awareness:.2f}, "
        f"empathy_style={chip.empathy_style}\n"
        f"emotional_range: {chip.emotional_range}\n"
        f"anti_patterns: {chip.anti_patterns[:5]}\n\n"
        "[Observed weaknesses on real model output]\n"
        + ("\n".join(f"- {w}" for w in weaknesses) if weaknesses else "- (none specifically diagnosed)")
        + "\n\n[Task]\nPropose bounded numerical deltas. Output the JSON object only."
    )


def _parse_trait_response(text: str) -> dict[str, Any]:
    """Extract the JSON object from the mutator response."""
    if not text:
        return {}
    raw = text.strip()
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
    open_match = re.search(r"\{", raw)
    if not open_match:
        return {}
    try:
        return json.loads(raw[open_match.start():])
    except json.JSONDecodeError:
        depth = 0
        start = open_match.start()
        for i in range(start, len(raw)):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        return {}
        return {}


def _clamp_dict(
    deltas_in: Any,
    allowed_keys: tuple[str, ...],
    max_delta: float,
) -> dict[str, float]:
    """Filter to allowed keys, coerce to float, clamp to ±max_delta."""
    out: dict[str, float] = {}
    if not isinstance(deltas_in, dict):
        return out
    for key, raw_val in deltas_in.items():
        if key not in allowed_keys:
            continue
        try:
            v = float(raw_val)
        except (TypeError, ValueError):
            continue
        v = max(-max_delta, min(max_delta, v))
        if v != 0.0:
            out[key] = round(v, 3)
    return out


def _apply_deltas(
    chip: PersonalityChip,
    trait_deltas: dict[str, float],
    profile_deltas: dict[str, float],
    range_deltas: dict[str, float],
) -> PersonalityChip:
    """Apply deltas to chip values, clamped to [0.0, 1.0]."""
    new_traits = {f: getattr(chip, f) for f in TRAIT_FIELDS}
    for k, d in trait_deltas.items():
        new_traits[k] = round(_clamp01(new_traits[k] + d), 3)
    new_profile = {f: getattr(chip, f) for f in EMOTIONAL_PROFILE_FIELDS}
    for k, d in profile_deltas.items():
        new_profile[k] = round(_clamp01(new_profile[k] + d), 3)
    new_range = dict(chip.emotional_range or {})
    for k, d in range_deltas.items():
        cur = float(new_range.get(k, 0.5))
        new_range[k] = round(_clamp01(cur + d), 3)
    return replace(
        chip,
        emotional_range=new_range,
        **new_traits,
        **new_profile,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def chip_to_yaml_dict(chip: PersonalityChip) -> dict[str, Any]:
    """Serialize a PersonalityChip back to a dict in the chip lab schema
    so it can be written as a .personality.yaml. Preserves _raw fields
    (vulnerabilities, strengths, preferences, anti_patterns, adaptive,
    safety) by merging the base raw spec with the mutated trait values."""
    base = copy.deepcopy(chip._raw or {})
    base.setdefault("schema", "spark-personality-chip.v1")
    identity = dict(base.get("identity", {}))
    identity["id"] = chip.id
    identity["name"] = chip.name
    if chip.archetype:
        identity["archetype"] = chip.archetype
    if chip.voice_signature:
        identity["voice_signature"] = chip.voice_signature
    if chip.tagline:
        identity["tagline"] = chip.tagline
    base["identity"] = identity
    base["traits"] = {
        "openness": chip.openness,
        "conscientiousness": chip.conscientiousness,
        "extraversion": chip.extraversion,
        "agreeableness": chip.agreeableness,
        "neuroticism": chip.neuroticism,
    }
    emo = dict(base.get("emotional_profile", {}))
    emo["self_awareness"] = chip.self_awareness
    emo["self_regulation"] = chip.self_regulation
    emo["social_awareness"] = chip.social_awareness
    if chip.empathy_style:
        emo["empathy_style"] = chip.empathy_style
    if chip.emotional_range:
        emo["emotional_range"] = dict(chip.emotional_range)
    if chip.emotional_triggers:
        emo["triggers"] = dict(chip.emotional_triggers)
    base["emotional_profile"] = emo
    return base
