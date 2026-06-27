"""Trait mutator pure-function tests. No network."""

from __future__ import annotations

from pathlib import Path

import pytest

import spark_character.chip_loader as chip_loader
from spark_character.trait_mutator import (
    _apply_deltas,
    _clamp_dict,
    _parse_trait_response,
    chip_to_yaml_dict,
)


VALID_MUTATION_CHIP_YAML = """
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
    frustration: 0.22
vulnerabilities:
  - rushing under pressure
anti_patterns:
  - over-explaining simple answers
"""


def load_mutation_test_chip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    chip_path = tmp_path / "founder-operator.personality.yaml"
    chip_path.write_text(VALID_MUTATION_CHIP_YAML, encoding="utf-8")
    monkeypatch.setattr(chip_loader, "_LAB_AVAILABLE", False)
    monkeypatch.setattr(chip_loader, "_lab_load_personality", None)
    return chip_loader.load_chip(chip_path)


def test_clamp_dict_filters_unknown_keys() -> None:
    out = _clamp_dict(
        {"openness": 0.05, "fakekey": 0.99, "neuroticism": -0.03},
        ("openness", "neuroticism"),
        max_delta=0.10,
    )
    assert out == {"openness": 0.05, "neuroticism": -0.03}


def test_clamp_dict_clamps_out_of_range() -> None:
    out = _clamp_dict(
        {"openness": 0.50, "neuroticism": -0.50},
        ("openness", "neuroticism"),
        max_delta=0.10,
    )
    assert out == {"openness": 0.10, "neuroticism": -0.10}


def test_clamp_dict_rejects_non_finite_deltas() -> None:
    out = _clamp_dict(
        {"openness": float("nan"), "neuroticism": float("inf"), "agreeableness": "-Infinity"},
        ("openness", "neuroticism", "agreeableness"),
        max_delta=0.10,
    )
    assert out == {}


def test_clamp_dict_drops_zero_deltas() -> None:
    out = _clamp_dict({"openness": 0.0}, ("openness",), max_delta=0.10)
    assert out == {}


def test_apply_deltas_clamps_to_unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chip = load_mutation_test_chip(tmp_path, monkeypatch)
    new_chip = _apply_deltas(
        chip,
        trait_deltas={"agreeableness": -0.50, "neuroticism": 0.10},
        profile_deltas={"self_regulation": 0.02},
        range_deltas={"frustration": -0.05, "curiosity": 0.05},
    )
    # agreeableness was 0.42 - 0.50 = -0.08 -> clamped to 0.0
    assert new_chip.agreeableness == 0.0
    # neuroticism was 0.16 + 0.10 = 0.26
    assert new_chip.neuroticism == 0.26
    # self_regulation was 0.86 + 0.02 = 0.88
    assert new_chip.self_regulation == 0.88
    # range frustration was 0.22 - 0.05 = 0.17
    assert new_chip.emotional_range["frustration"] == 0.17


def test_parse_trait_response_handles_code_fence() -> None:
    raw = """```json
{"reasoning": "test", "deltas": {"openness": 0.05}}
```"""
    parsed = _parse_trait_response(raw)
    assert parsed["reasoning"] == "test"
    assert parsed["deltas"] == {"openness": 0.05}


def test_parse_trait_response_handles_preamble() -> None:
    raw = "Sure, here is the JSON:\n{\"deltas\": {\"openness\": 0.03}}"
    parsed = _parse_trait_response(raw)
    assert parsed["deltas"] == {"openness": 0.03}


def test_parse_trait_response_returns_none_on_garbage() -> None:
    assert _parse_trait_response("") is None
    assert _parse_trait_response("not json at all") is None


def test_chip_to_yaml_dict_round_trip_preserves_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chip = load_mutation_test_chip(tmp_path, monkeypatch)
    spec = chip_to_yaml_dict(chip)
    # required top-level fields
    assert spec["schema"] == "spark-personality-chip.v1"
    assert spec["identity"]["id"] == chip.id
    assert spec["traits"]["openness"] == chip.openness
    # carry-forward fields from _raw
    assert "vulnerabilities" in spec
    assert "anti_patterns" in spec
    # emotional profile populated
    assert spec["emotional_profile"]["self_awareness"] == chip.self_awareness


def test_chip_to_yaml_dict_after_mutation_reflects_new_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chip = load_mutation_test_chip(tmp_path, monkeypatch)
    new_chip = _apply_deltas(
        chip,
        trait_deltas={"openness": 0.05},
        profile_deltas={},
        range_deltas={},
    )
    spec = chip_to_yaml_dict(new_chip)
    assert spec["traits"]["openness"] == round(chip.openness + 0.05, 3)
    # other traits unchanged
    assert spec["traits"]["conscientiousness"] == chip.conscientiousness
