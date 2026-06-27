"""Registry promotion artifact tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from spark_character.chip_loader import PersonalityChip
from spark_character.registry import (
    ChipLabNotFoundError,
    ChipLabPromotionError,
    PyYamlMissingError,
    _personality_yaml_path,
    promote_evolved_chip_to_chip_lab,
    promote_evolved_persona_to_chip_lab,
)


BASE_CHIP_YAML = """
schema: spark-personality-chip.v1
identity:
  id: founder-operator
  name: Founder Operator
traits:
  openness: 0.72
emotional_profile:
  self_awareness: 0.84
preferences:
  likes: []
"""


def test_persona_sidecar_promotion_does_not_export_scores(tmp_path: Path) -> None:
    (tmp_path / "founder-operator.personality.yaml").write_text(BASE_CHIP_YAML, encoding="utf-8")

    target = promote_evolved_persona_to_chip_lab(
        base_chip_id="founder-operator",
        base_persona_version="v8",
        new_persona_version="v9",
        persona_markdown="New voice rules",
        composite_score=0.99,
        lab_path=tmp_path,
    )

    assert target is not None
    spec = yaml.safe_load(target.read_text(encoding="utf-8"))
    evolved = spec["spark_character_evolved"]
    assert evolved["promotion_result"] == "accepted"
    assert "composite_score" not in evolved
    assert "delta_summary" not in evolved


def test_persona_sidecar_promotion_rejects_malformed_base_yaml(tmp_path: Path) -> None:
    (tmp_path / "founder-operator.personality.yaml").write_text("identity: [", encoding="utf-8")

    with pytest.raises(ValueError, match="Base personality chip YAML is invalid: founder-operator.personality.yaml"):
        promote_evolved_persona_to_chip_lab(
            base_chip_id="founder-operator",
            base_persona_version="v8",
            new_persona_version="v9",
            persona_markdown="New voice rules",
            lab_path=tmp_path,
        )


def test_persona_sidecar_promotion_does_not_swallow_unexpected_validation_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "founder-operator.personality.yaml").write_text(BASE_CHIP_YAML, encoding="utf-8")

    def broken_validator(_spec):
        raise RuntimeError("unexpected validator bug")

    monkeypatch.setattr("spark_character.registry.validate_chip_yaml_spec", broken_validator)

    with pytest.raises(RuntimeError, match="unexpected validator bug"):
        promote_evolved_persona_to_chip_lab(
            base_chip_id="founder-operator",
            base_persona_version="v8",
            new_persona_version="v9",
            persona_markdown="New voice rules",
            lab_path=tmp_path,
        )


def test_chip_promotion_does_not_export_scores_or_delta_summary(tmp_path: Path) -> None:
    chip = PersonalityChip(
        id="founder-operator",
        name="Founder Operator",
        openness=0.72,
        _raw={"schema": "spark-personality-chip.v1", "identity": {"id": "founder-operator", "name": "Founder Operator"}},
    )

    target = promote_evolved_chip_to_chip_lab(
        chip=chip,
        base_chip_id="founder-operator",
        base_persona_version="v8",
        new_persona_version="v9",
        composite_score=0.99,
        delta_summary={"weak_axis": "t7"},
        lab_path=tmp_path,
    )

    assert target is not None
    spec = yaml.safe_load(target.read_text(encoding="utf-8"))
    evolved = spec["spark_character_evolved"]
    assert evolved["promotion_result"] == "accepted"
    assert "composite_score" not in evolved
    assert "delta_summary" not in evolved


def test_persona_sidecar_promotion_rejects_chip_id_path_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / "escaped.personality.yaml"

    with pytest.raises(ValueError):
        promote_evolved_persona_to_chip_lab(
            base_chip_id="../escaped",
            base_persona_version="v8",
            new_persona_version="v9",
            persona_markdown="New voice rules",
            lab_path=tmp_path,
        )

    assert not outside.exists()


@pytest.mark.parametrize("chip_id", ["", " ", "\t\n"])
def test_personality_yaml_path_rejects_blank_chip_id(tmp_path: Path, chip_id: str) -> None:
    with pytest.raises(ValueError, match="chip id is required"):
        _personality_yaml_path(tmp_path, chip_id)


def test_chip_promotion_rejects_chip_id_path_escape(tmp_path: Path) -> None:
    chip = PersonalityChip(id="founder-operator", name="Founder Operator")
    outside = tmp_path.parent / "escaped-evolved-v9.personality.yaml"

    with pytest.raises(ValueError):
        promote_evolved_chip_to_chip_lab(
            chip=chip,
            base_chip_id="../escaped",
            base_persona_version="v8",
            new_persona_version="v9",
            lab_path=tmp_path,
        )

    assert not outside.exists()


def test_persona_promotion_raises_pyyaml_missing(monkeypatch) -> None:
    """PyYAML import failure raises PyYamlMissingError, not None."""
    import builtins
    real_import = builtins.__import__

    def no_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)

    with pytest.raises(PyYamlMissingError, match="PyYAML is required"):
        promote_evolved_persona_to_chip_lab(
            base_chip_id="founder-operator",
            base_persona_version="v8",
            new_persona_version="v9",
            persona_markdown="New voice rules",
            lab_path=Path("/nonexistent"),
        )


def test_persona_promotion_raises_chip_lab_not_found(monkeypatch) -> None:
    """Missing chip lab raises ChipLabNotFoundError, not None."""
    monkeypatch.setattr(
        "spark_character.registry.find_chip_lab_path", lambda: None
    )

    with pytest.raises(ChipLabNotFoundError, match="Chip lab directory not found"):
        promote_evolved_persona_to_chip_lab(
            base_chip_id="founder-operator",
            base_persona_version="v8",
            new_persona_version="v9",
            persona_markdown="New voice rules",
        )


def test_chip_promotion_raises_pyyaml_missing(monkeypatch) -> None:
    """Chip variant also raises PyYamlMissingError when yaml is missing."""
    import builtins
    real_import = builtins.__import__

    def no_yaml(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_yaml)

    chip = PersonalityChip(id="founder-operator", name="Founder Operator")

    with pytest.raises(PyYamlMissingError, match="PyYAML is required"):
        promote_evolved_chip_to_chip_lab(
            chip=chip,
            base_chip_id="founder-operator",
            base_persona_version="v8",
            new_persona_version="v9",
            lab_path=Path("/nonexistent"),
        )


def test_chip_promotion_raises_chip_lab_not_found(monkeypatch) -> None:
    """Chip variant also raises ChipLabNotFoundError when lab is missing."""
    monkeypatch.setattr(
        "spark_character.registry.find_chip_lab_path", lambda: None
    )

    chip = PersonalityChip(id="founder-operator", name="Founder Operator")

    with pytest.raises(ChipLabNotFoundError, match="Chip lab directory not found"):
        promote_evolved_chip_to_chip_lab(
            chip=chip,
            base_chip_id="founder-operator",
            base_persona_version="v8",
            new_persona_version="v9",
        )


def test_promotion_exceptions_are_chip_lab_promotion_errors() -> None:
    """Both specific exceptions inherit from ChipLabPromotionError."""
    assert issubclass(PyYamlMissingError, ChipLabPromotionError)
    assert issubclass(ChipLabNotFoundError, ChipLabPromotionError)
