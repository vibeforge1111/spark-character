"""Promote evolved personas back to spark-personality-chip-labs.

When evolve_persona ships a winning candidate, two things should happen:

1. spark-character/artifacts/persona.vN.md (already happens) - the
   internal flat-markdown artifact that spark-character generates from.
2. THIS MODULE: write a sidecar YAML into the chip lab's personalities
   directory so the chip lab registry sees the evolution.

The sidecar uses a clear marker so the chip lab can distinguish a
chip-lab-authored personality from a spark-character-evolved variant:

    schema: spark-personality-chip.v1
    spark_character_evolved:
      base_chip_id: founder-operator
      base_persona_version: v4
      new_persona_version: v5
      composite_score: 0.875
      promoted_at: 2026-04-25T17:30:00Z

Plus all the fields from the base chip carried forward, plus
`voice_rules_override` containing the evolved system-prompt markdown.

A spark-personality-chip-labs consumer can read the override field
and apply it on top of the base chip's standard fields when rendering.
A consumer that ignores the field gets the base chip behavior, which
is a safe fallback.

This is the v1 of registry integration. A deeper integration would
have evolve_persona produce a fully native chip YAML with mutated
trait values, but that requires the mutator to reason about OCEAN
numbers, which is a separate evolution problem.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .chip_loader import PersonalityChip

DEFAULT_LAB_PATH = Path(os.path.expanduser(
    "~/Desktop/spark-personality-chip-labs/personalities"
))


def find_chip_lab_path() -> Path | None:
    """Locate the chip lab personalities directory if installed locally."""
    candidates = [
        DEFAULT_LAB_PATH,
        Path("./personalities"),
        Path(os.path.expanduser("~/.spark/personalities")),
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def promote_evolved_persona_to_chip_lab(
    *,
    base_chip_id: str,
    base_persona_version: str,
    new_persona_version: str,
    persona_markdown: str,
    composite_score: float | None = None,
    lab_path: Path | None = None,
) -> Path | None:
    """Write a spark-character-evolved personality YAML into the chip lab.

    Returns the written path if successful, or None if the chip lab is
    not present or PyYAML is unavailable. Never raises on missing lab,
    so the evolve loop can call this unconditionally after a promotion.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        return None

    lab = lab_path or find_chip_lab_path()
    if lab is None:
        return None

    base_yaml_path = lab / f"{base_chip_id}.personality.yaml"
    base_spec: dict[str, Any] = {}
    if base_yaml_path.exists():
        try:
            base_spec = yaml.safe_load(base_yaml_path.read_text(encoding="utf-8")) or {}
        except Exception:
            base_spec = {}

    # Carry everything from the base chip forward, then mark this as an
    # evolved variant and embed the new voice rules.
    out: dict[str, Any] = dict(base_spec)
    out["schema"] = base_spec.get("schema", "spark-personality-chip.v1")
    identity = dict(base_spec.get("identity", {}))
    new_chip_id = f"{base_chip_id}-evolved-{new_persona_version.replace('.', '-')}"
    identity["id"] = new_chip_id
    if not identity.get("name"):
        identity["name"] = base_spec.get("identity", {}).get("name", base_chip_id)
    out["identity"] = identity
    out["spark_character_evolved"] = {
        "base_chip_id": base_chip_id,
        "base_persona_version": base_persona_version,
        "new_persona_version": new_persona_version,
        "composite_score": float(composite_score) if composite_score is not None else None,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out["voice_rules_override"] = persona_markdown.strip()

    target = lab / f"{new_chip_id}.personality.yaml"
    target.write_text(
        yaml.safe_dump(out, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def promote_evolved_chip_to_chip_lab(
    *,
    chip: PersonalityChip,
    base_chip_id: str,
    base_persona_version: str,
    new_persona_version: str,
    voice_rules_override: str | None = None,
    composite_score: float | None = None,
    delta_summary: dict[str, Any] | None = None,
    lab_path: Path | None = None,
) -> Path | None:
    """Promote a fully evolved PersonalityChip (with mutated trait values)
    back to the chip lab as a native chip yaml.

    Unlike promote_evolved_persona_to_chip_lab (which writes a sidecar
    with voice_rules_override on top of the unchanged base chip), this
    function writes a real chip yaml with the new trait values, new
    emotional_range entries, and optionally a system-prompt override.

    Returns the written path, or None if PyYAML isn't available or the
    chip lab is missing locally.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    from .trait_mutator import chip_to_yaml_dict  # local import to avoid cycle

    lab = lab_path or find_chip_lab_path()
    if lab is None:
        return None

    spec = chip_to_yaml_dict(chip)
    new_chip_id = f"{base_chip_id}-evolved-{new_persona_version.replace('.', '-')}"
    spec.setdefault("identity", {})
    spec["identity"]["id"] = new_chip_id
    spec["spark_character_evolved"] = {
        "base_chip_id": base_chip_id,
        "base_persona_version": base_persona_version,
        "new_persona_version": new_persona_version,
        "composite_score": float(composite_score) if composite_score is not None else None,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "delta_summary": delta_summary or {},
    }
    if voice_rules_override:
        spec["voice_rules_override"] = voice_rules_override.strip()

    target = lab / f"{new_chip_id}.personality.yaml"
    target.write_text(
        yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target
