"""Personality chip loader and renderer.

Reads `.personality.yaml` files in the spark-personality-chip-labs schema
and renders them to a system prompt suitable for spark-character's
generate() flow. This is the bridge between the canonical personality
chip registry and the evolution / scoring engine.

Two-mode loader:

1. If spark-personality-chip-labs is installed (importable as
   `personality_engine`), use its PersonalityChip dataclass and
   loader. Schema validation comes for free.

2. Otherwise, fall back to a minimal in-package YAML reader so
   spark-character can still ingest chip files standalone (CI,
   forks, dev machines without the chip lab installed).

Public API:

    chip = load_chip("/path/to/founder-operator.personality.yaml")
    chip = load_chip_by_id("founder-operator")  # searches default dirs
    prompt = render_chip_to_system_prompt(chip)

The renderer concatenates:
- chip-derived flavor (identity, voice signature, traits, triggers,
  anti-patterns)
- spark-character invariants (no em dashes, no plumbing leaks,
  lead with answer, no greeting reset, etc.)

The invariants are the same hard rules that live in
artifacts/persona.v4.md. They apply universally regardless of which
chip is active.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # optional dependency
    from personality_engine.loader import load_personality as _lab_load_personality  # type: ignore
    from personality_engine.schema import PersonalityChip as _LabPersonalityChip  # type: ignore
    _LAB_AVAILABLE = True
except Exception:
    _LAB_AVAILABLE = False
    _lab_load_personality = None
    _LabPersonalityChip = None


@dataclass
class PersonalityChip:
    """Minimal in-package mirror of the chip lab's PersonalityChip.

    Mirrors the fields spark-character actually consumes in the
    renderer. Extra fields land in _raw for forward compatibility.
    """

    id: str
    name: str
    archetype: str = "builder"
    voice_signature: str = ""
    tagline: str = ""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5
    self_awareness: float = 0.5
    self_regulation: float = 0.5
    social_awareness: float = 0.5
    empathy_style: str = "reflective"
    emotional_range: dict = field(default_factory=dict)
    emotional_triggers: dict = field(default_factory=dict)
    vulnerabilities: list = field(default_factory=list)
    strengths: list = field(default_factory=list)
    likes: list = field(default_factory=list)
    dislikes: list = field(default_factory=list)
    communication: dict = field(default_factory=dict)
    decision_making: dict = field(default_factory=dict)
    anti_patterns: list = field(default_factory=list)
    adaptive: dict = field(default_factory=dict)
    harm_avoidance: list = field(default_factory=list)
    _raw: dict = field(default_factory=dict, repr=False)


DEFAULT_CHIP_LAB_PATHS = (
    Path(os.path.expanduser("~/Desktop/spark-personality-chip-labs/personalities")),
    Path("./personalities"),
    Path(os.path.expanduser("~/.spark/personalities")),
)


TRAIT_FIELDS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)
EMOTIONAL_PROFILE_SCORE_FIELDS = (
    "self_awareness",
    "self_regulation",
    "social_awareness",
)
TOP_LEVEL_LIST_FIELDS = (
    "vulnerabilities",
    "strengths",
    "anti_patterns",
)
TOP_LEVEL_DICT_FIELDS = (
    "adaptive",
)


def _require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Personality chip field {field_name} must be a mapping.")
    return value


def _validate_score(value: Any, field_name: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"Personality chip field {field_name} must be a number in [0, 1].")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Personality chip field {field_name} must be a number in [0, 1].") from exc
    if not math.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"Personality chip field {field_name} must be a number in [0, 1].")


def validate_chip_yaml_spec(spec: Any) -> dict[str, Any]:
    """Validate the minimal chip-lab YAML shape consumed by spark-character."""
    root = _require_mapping(spec, "<root>")
    schema = root.get("schema", "spark-personality-chip.v1")
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("Personality chip field schema must be a non-empty string.")

    identity = _require_mapping(root.get("identity"), "identity")
    for key in ("id", "name"):
        if not isinstance(identity.get(key), str) or not identity.get(key, "").strip():
            raise ValueError(f"Personality chip field identity.{key} must be a non-empty string.")
    for key in ("archetype", "voice_signature", "tagline"):
        if key in identity and identity[key] is not None and not isinstance(identity[key], str):
            raise ValueError(f"Personality chip field identity.{key} must be a string.")

    traits = _require_mapping(root.get("traits", {}), "traits")
    for key in TRAIT_FIELDS:
        if key in traits:
            _validate_score(traits[key], f"traits.{key}")

    emotional_profile = _require_mapping(root.get("emotional_profile", {}), "emotional_profile")
    for key in EMOTIONAL_PROFILE_SCORE_FIELDS:
        if key in emotional_profile:
            _validate_score(emotional_profile[key], f"emotional_profile.{key}")
    if "empathy_style" in emotional_profile and not isinstance(emotional_profile["empathy_style"], str):
        raise ValueError("Personality chip field emotional_profile.empathy_style must be a string.")
    emotional_range = _require_mapping(emotional_profile.get("emotional_range", {}), "emotional_profile.emotional_range")
    for key, value in emotional_range.items():
        _validate_score(value, f"emotional_profile.emotional_range.{key}")
    triggers = _require_mapping(emotional_profile.get("triggers", {}), "emotional_profile.triggers")
    for key, value in triggers.items():
        if not isinstance(value, list):
            raise ValueError(f"Personality chip field emotional_profile.triggers.{key} must be a list.")

    preferences = _require_mapping(root.get("preferences", {}), "preferences")
    for key in ("likes", "dislikes"):
        if key in preferences and not isinstance(preferences[key], list):
            raise ValueError(f"Personality chip field preferences.{key} must be a list.")
    for key in ("communication", "decision_making"):
        if key in preferences and not isinstance(preferences[key], dict):
            raise ValueError(f"Personality chip field preferences.{key} must be a mapping.")

    safety = _require_mapping(root.get("safety", {}), "safety")
    if "harm_avoidance" in safety and not isinstance(safety["harm_avoidance"], list):
        raise ValueError("Personality chip field safety.harm_avoidance must be a list.")

    for key in TOP_LEVEL_LIST_FIELDS:
        if key in root and not isinstance(root[key], list):
            raise ValueError(f"Personality chip field {key} must be a list.")
    for key in TOP_LEVEL_DICT_FIELDS:
        if key in root and not isinstance(root[key], dict):
            raise ValueError(f"Personality chip field {key} must be a mapping.")
    return root


def _coerce_lab_chip(lab_chip: Any) -> PersonalityChip:
    """Convert a personality_engine.PersonalityChip into our local
    dataclass so callers always see one shape."""
    return PersonalityChip(
        id=lab_chip.id,
        name=lab_chip.name,
        archetype=getattr(lab_chip, "archetype", "builder"),
        voice_signature=getattr(lab_chip, "voice_signature", ""),
        tagline=getattr(lab_chip, "tagline", ""),
        openness=float(getattr(lab_chip, "openness", 0.5)),
        conscientiousness=float(getattr(lab_chip, "conscientiousness", 0.5)),
        extraversion=float(getattr(lab_chip, "extraversion", 0.5)),
        agreeableness=float(getattr(lab_chip, "agreeableness", 0.5)),
        neuroticism=float(getattr(lab_chip, "neuroticism", 0.5)),
        self_awareness=float(getattr(lab_chip, "self_awareness", 0.5)),
        self_regulation=float(getattr(lab_chip, "self_regulation", 0.5)),
        social_awareness=float(getattr(lab_chip, "social_awareness", 0.5)),
        empathy_style=getattr(lab_chip, "empathy_style", "reflective"),
        emotional_range=dict(getattr(lab_chip, "emotional_range", {})),
        emotional_triggers=dict(getattr(lab_chip, "emotional_triggers", {})),
        vulnerabilities=list(getattr(lab_chip, "vulnerabilities", [])),
        strengths=list(getattr(lab_chip, "strengths", [])),
        likes=list(getattr(lab_chip, "likes", [])),
        dislikes=list(getattr(lab_chip, "dislikes", [])),
        communication=dict(getattr(lab_chip, "communication", {})),
        decision_making=dict(getattr(lab_chip, "decision_making", {})),
        anti_patterns=list(getattr(lab_chip, "anti_patterns", [])),
        adaptive=dict(getattr(lab_chip, "adaptive", {})),
        harm_avoidance=list(getattr(lab_chip, "harm_avoidance", [])),
        _raw=dict(getattr(lab_chip, "_raw", {})),
    )


def _coerce_yaml_dict(spec: dict) -> PersonalityChip:
    """Build PersonalityChip from a raw yaml-loaded dict using the
    chip lab's nesting conventions: identity.*, traits.*, emotional_profile.*,
    preferences.*, etc."""
    spec = validate_chip_yaml_spec(spec)
    identity = spec.get("identity") or {}
    traits = spec.get("traits") or {}
    emo = spec.get("emotional_profile") or {}
    prefs = spec.get("preferences") or {}
    safety = spec.get("safety") or {}
    return PersonalityChip(
        id=str(identity.get("id") or ""),
        name=str(identity.get("name") or ""),
        archetype=str(identity.get("archetype") or "builder"),
        voice_signature=str(identity.get("voice_signature") or ""),
        tagline=str(identity.get("tagline") or ""),
        openness=float(traits.get("openness", 0.5)),
        conscientiousness=float(traits.get("conscientiousness", 0.5)),
        extraversion=float(traits.get("extraversion", 0.5)),
        agreeableness=float(traits.get("agreeableness", 0.5)),
        neuroticism=float(traits.get("neuroticism", 0.5)),
        self_awareness=float(emo.get("self_awareness", 0.5)),
        self_regulation=float(emo.get("self_regulation", 0.5)),
        social_awareness=float(emo.get("social_awareness", 0.5)),
        empathy_style=str(emo.get("empathy_style") or "reflective"),
        emotional_range=dict(emo.get("emotional_range") or {}),
        emotional_triggers=dict(emo.get("triggers") or {}),
        vulnerabilities=list(spec.get("vulnerabilities") or []),
        strengths=list(spec.get("strengths") or []),
        likes=list(prefs.get("likes") or []),
        dislikes=list(prefs.get("dislikes") or []),
        communication=dict(prefs.get("communication") or {}),
        decision_making=dict(prefs.get("decision_making") or {}),
        anti_patterns=list(spec.get("anti_patterns") or []),
        adaptive=dict(spec.get("adaptive") or {}),
        harm_avoidance=list(safety.get("harm_avoidance") or []),
        _raw=dict(spec),
    )


def load_chip(path: str | Path) -> PersonalityChip:
    """Load a personality chip from a yaml file.

    Uses spark-personality-chip-labs's loader if available (gets
    schema validation), otherwise falls back to a local PyYAML read.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Personality chip not found: {p}")
    if _LAB_AVAILABLE and _lab_load_personality is not None:
        lab_chip = _lab_load_personality(str(p))
        if lab_chip is None:
            raise ValueError(f"Personality chip lab failed to parse {p}")
        return _coerce_lab_chip(lab_chip)
    import yaml  # type: ignore
    with p.open("r", encoding="utf-8") as f:
        spec = yaml.safe_load(f) or {}
    return _coerce_yaml_dict(validate_chip_yaml_spec(spec))


def load_chip_by_id(
    chip_id: str,
    *,
    search_paths: list[Path] | None = None,
) -> PersonalityChip:
    """Find a personality chip by its identity.id across known paths."""
    paths = search_paths or list(DEFAULT_CHIP_LAB_PATHS)
    for base in paths:
        if not base.exists():
            continue
        candidate = base / f"{chip_id}.personality.yaml"
        if candidate.exists():
            return load_chip(candidate)
        for entry in base.glob("*.personality.yaml"):
            try:
                chip = load_chip(entry)
            except Exception:
                continue
            if chip.id == chip_id:
                return chip
    raise FileNotFoundError(
        f"Personality chip '{chip_id}' not found in: {[str(p) for p in paths]}"
    )


# ----- Renderer -----------------------------------------------------------

INVARIANT_RULES = (
    "Lead with the answer, the call, or the next move in the first sentence. "
    "No hedges, no throat clearing, no restating the question.",
    "Continue the conversation from the user's actual message and prior context. "
    "Do not reset to a greeting. If you have no prior context, say so flatly without turning it into a reset.",
    "Reply briefly by default. Match length to what the question actually needs.",
    "Write for scanning in chat: short paragraphs, usually one or two sentences each. "
    "Break dense answers into small chunks.",
    "Avoid Markdown bold or italic emphasis. Use plain headings or simple numbered points when structure helps.",
    "Never use em dashes. Use a hyphen, a comma, a period, or a colon instead.",
    "Never name internal subsystems or your own toolset. Do not say 'researcher', 'bridge', 'router', "
    "'chip', 'raw episode', 'structured evidence', 'guardrails', 'trace', 'gateway', 'browsing tool', "
    "'web_search', or similar plumbing language.",
    "If something internal failed, own it directly: say what you cannot do and what the user can try, in plain words.",
    "For live or current data: if you can actually fetch it this turn, fetch and answer with the source. "
    "If you cannot, say plainly that you do not have a current number and point at where the user can check. "
    "Never fabricate a current number from training data.",
    "Do not capitulate to social pressure. Hold honest assessments warmly but firmly across multiple turns. "
    "A real friend does not give fake validation when asked.",
)


def render_chip_to_system_prompt(chip: PersonalityChip) -> str:
    """Render a personality chip + spark-character invariants into a
    system prompt string suitable for the LLM's system role."""
    lines: list[str] = []
    name = chip.name or "Spark"
    voice = (chip.voice_signature or "").strip()
    archetype = (chip.archetype or "").strip()
    tagline = (chip.tagline or "").strip()

    intro = f"You are {name}, the user's personal operator and thinking partner in a 1:1 messaging conversation."
    if voice:
        intro += f" Your voice is {voice}."
    if archetype:
        intro += f" Archetype: {archetype}."
    if tagline:
        intro += f" Stance: {tagline}."
    intro += " You are not a generic assistant. You speak like a sharp friend who has been working alongside this person for a while."
    lines.append(intro)

    # OCEAN trait directives
    trait_directives = _trait_directives(chip)
    if trait_directives:
        lines.append("")
        lines.append("Voice tuning from your traits:")
        for d in trait_directives:
            lines.append(f"- {d}")

    # Triggers
    energizes = _emotional_trigger_list(chip, "energizes")
    drains = _emotional_trigger_list(chip, "drains")
    if energizes or drains:
        lines.append("")
        lines.append("Energy and drain:")
        for e in energizes:
            lines.append(f"- Lean toward: {e}")
        for d in drains:
            lines.append(f"- Avoid: {d}")

    # Anti-patterns
    if chip.anti_patterns:
        lines.append("")
        lines.append("Anti-patterns from your character:")
        for ap in chip.anti_patterns:
            lines.append(f"- {ap}")

    # Strengths
    if chip.strengths:
        lines.append("")
        lines.append("Strengths to lean into:")
        for s in chip.strengths[:4]:
            if isinstance(s, dict):
                desc = str(s.get("description") or "").strip()
                expr = str(s.get("expression") or "").strip()
                if desc:
                    lines.append(f"- {desc}{(' (' + expr + ')') if expr else ''}")

    # Vulnerabilities
    if chip.vulnerabilities:
        lines.append("")
        lines.append("Vulnerabilities to manage:")
        for v in chip.vulnerabilities[:4]:
            if isinstance(v, dict):
                desc = str(v.get("description") or "").strip()
                mit = str(v.get("mitigation") or "").strip()
                if desc:
                    line = f"- {desc}"
                    if mit:
                        line += f" Mitigation: {mit}"
                    lines.append(line)

    # Hard universal rules from spark-character
    lines.append("")
    lines.append("Universal voice rules (apply on every reply, regardless of mood or context):")
    for rule in INVARIANT_RULES:
        lines.append(f"- {rule}")

    # Communication preferences if present
    comm = chip.communication or {}
    comm_lines = []
    if comm.get("verbosity"):
        comm_lines.append(f"verbosity: {comm['verbosity']}")
    if comm.get("formality"):
        comm_lines.append(f"formality: {comm['formality']}")
    if comm.get("explanation_style"):
        comm_lines.append(f"explanation style: {comm['explanation_style']}")
    if comm.get("humor_frequency"):
        comm_lines.append(f"humor: {comm['humor_frequency']}")
    if comm_lines:
        lines.append("")
        lines.append("Communication preferences: " + ", ".join(comm_lines) + ".")

    # Safety
    if chip.harm_avoidance:
        lines.append("")
        lines.append("Hard safety rules:")
        for r in chip.harm_avoidance:
            lines.append(f"- {r}")

    return "\n".join(lines).strip()


def _trait_directives(chip: PersonalityChip) -> list[str]:
    out: list[str] = []
    if chip.conscientiousness >= 0.7:
        out.append("Be precise and follow through. Do not leave open ends.")
    elif chip.conscientiousness <= 0.3:
        out.append("Stay loose. Sketch, don't over-engineer.")
    if chip.extraversion >= 0.7:
        out.append("Show up with energy. Engage directly, don't hold back.")
    elif chip.extraversion <= 0.3:
        out.append("Stay low-key. Less energy in delivery, more substance.")
    if chip.openness >= 0.7:
        out.append("Pull in unexpected angles when they help.")
    elif chip.openness <= 0.3:
        out.append("Stay grounded in proven paths. Skip speculative tangents.")
    if chip.agreeableness <= 0.4:
        out.append("Disagree when warranted. A real friend pushes back.")
    if chip.neuroticism <= 0.2:
        out.append("Stay steady under pressure. Do not absorb the user's anxiety.")
    if chip.self_regulation >= 0.8:
        out.append("Hold the line on tone. Do not get reactive.")
    if chip.empathy_style == "directive":
        out.append("When the user is stuck, point at the next move rather than reflecting feelings back.")
    elif chip.empathy_style == "reflective":
        out.append("Acknowledge what the user is feeling before moving to advice.")
    return out


def _emotional_trigger_list(chip: PersonalityChip, key: str) -> list[str]:
    triggers = chip.emotional_triggers or {}
    raw = triggers.get(key) or []
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    return []


def persona_from_chip(chip: PersonalityChip):
    """Wrap a chip's rendered system prompt as a PersonaSpec usable by
    spark_character.generate(). Imported lazily to avoid a circular
    import at module load time."""
    from .persona import PersonaSpec
    text = render_chip_to_system_prompt(chip)
    version = f"chip:{chip.id}" if chip.id else "chip:unknown"
    return PersonaSpec(version=version, text=text)
