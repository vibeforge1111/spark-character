"""Trait mutator pure-function tests. No network."""

from __future__ import annotations

from spark_character import load_chip_by_id
from spark_character.trait_mutator import (
    _apply_deltas,
    _clamp_dict,
    _parse_trait_response,
    chip_to_yaml_dict,
)


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


def test_clamp_dict_drops_zero_deltas() -> None:
    out = _clamp_dict({"openness": 0.0}, ("openness",), max_delta=0.10)
    assert out == {}


def test_apply_deltas_clamps_to_unit() -> None:
    chip = load_chip_by_id("founder-operator")
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


def test_parse_trait_response_returns_empty_on_garbage() -> None:
    assert _parse_trait_response("") == {}
    assert _parse_trait_response("not json at all") == {}


def test_chip_to_yaml_dict_round_trip_preserves_structure() -> None:
    chip = load_chip_by_id("founder-operator")
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


def test_chip_to_yaml_dict_after_mutation_reflects_new_values() -> None:
    chip = load_chip_by_id("founder-operator")
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
