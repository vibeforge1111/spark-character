"""Personality chip YAML loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

import spark_character.chip_loader as chip_loader


VALID_CHIP_YAML = """
schema: spark-personality-chip.v1
identity:
  id: founder-operator
  name: Founder Operator
  archetype: builder
traits:
  openness: 0.72
  conscientiousness: 0.88
  extraversion: 0.34
  agreeableness: 0.42
  neuroticism: 0.16
emotional_profile:
  self_awareness: 0.84
  self_regulation: 0.86
  social_awareness: 0.79
  empathy_style: directive
  emotional_range:
    curiosity: 0.91
  triggers:
    energizes:
      - clear stakes
preferences:
  likes:
    - direct answers
  communication:
    verbosity: concise
safety:
  harm_avoidance:
    - do not flatter
"""


def load_fallback_chip(path: Path, monkeypatch: pytest.MonkeyPatch) -> chip_loader.PersonalityChip:
    monkeypatch.setattr(chip_loader, "_LAB_AVAILABLE", False)
    monkeypatch.setattr(chip_loader, "_lab_load_personality", None)
    return chip_loader.load_chip(path)


def test_load_chip_validates_and_coerces_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "founder-operator.personality.yaml"
    path.write_text(VALID_CHIP_YAML, encoding="utf-8")

    chip = load_fallback_chip(path, monkeypatch)

    assert chip.id == "founder-operator"
    assert chip.name == "Founder Operator"
    assert chip.openness == 0.72
    assert chip.empathy_style == "directive"
    assert chip.emotional_range["curiosity"] == 0.91


def test_load_chip_rejects_missing_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bad.personality.yaml"
    path.write_text("schema: spark-personality-chip.v1\ntraits:\n  openness: 0.5\n", encoding="utf-8")

    with pytest.raises(ValueError, match="identity"):
        load_fallback_chip(path, monkeypatch)


def test_load_chip_rejects_out_of_range_trait(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bad.personality.yaml"
    path.write_text(
        VALID_CHIP_YAML.replace("openness: 0.72", "openness: 1.72"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="traits.openness"):
        load_fallback_chip(path, monkeypatch)


def test_load_chip_rejects_wrong_nested_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bad.personality.yaml"
    path.write_text(
        VALID_CHIP_YAML.replace("communication:\n    verbosity: concise", "communication: chatty"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="preferences.communication"):
        load_fallback_chip(path, monkeypatch)
