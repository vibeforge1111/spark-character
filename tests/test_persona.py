"""Persona + critic artifact loading tests."""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from spark_character import load_critic, load_persona, set_latest_persona_version
import spark_character.persona as persona_module
from spark_character.scoring import score_persona


def test_load_persona_v1() -> None:
    persona = load_persona("v1")
    assert persona.version == "v1"
    text = persona.system_prompt
    assert "Spark" in text
    assert "Never use em dashes" in text
    assert "researcher" in text.lower()


def test_load_critic_v1() -> None:
    critic = load_critic("v1")
    assert critic.version == "v1"
    text = critic.system_prompt
    assert "PASS" in text
    assert "em dash" in text.lower()
    assert "Avoid Markdown bold/italic emphasis" in text
    assert "paragraphs short" in text


def test_latest_persona_has_chat_scanning_rules() -> None:
    persona = load_persona()
    assert persona.version == "v8"
    text = persona.system_prompt
    assert "short paragraphs" in text
    assert "Avoid Markdown bold or italic emphasis" in text
    assert "Break dense answers into small chunks" in text


def test_persona_text_has_no_em_dash() -> None:
    """The persona spec itself must follow the no-em-dash rule it teaches.

    P3/P2 are deliberately not asserted here: the spec quotes example
    failure phrases ("How can I help today?", "researcher", "raw episode")
    so the scorers fire on the spec text by design. The point is that
    no generated reply passes through the spec scorer, only the model
    output does.
    """
    persona = load_persona("v1")
    score = score_persona(persona.system_prompt)
    assert score.p1_em_dash == 1.0
    assert score.p4_lead == 1.0


def test_latest_persona_pointer_rejects_malformed_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pointer = tmp_path / "persona.latest.txt"
    pointer.write_text("../secret\n", encoding="utf-8")
    monkeypatch.setattr(persona_module, "LATEST_POINTER", pointer)
    monkeypatch.setattr(persona_module, "ARTIFACTS_DIR", tmp_path)

    with pytest.raises(ValueError, match="vN"):
        persona_module.resolve_latest_persona_version()


def test_set_latest_persona_version_logs_and_protects_pointer(tmp_path: Path) -> None:
    (tmp_path / "persona.v9.md").write_text("Persona v9", encoding="utf-8")
    pointer = tmp_path / "persona.latest.txt"
    log_path = tmp_path / "persona.pointer.log"

    set_latest_persona_version(
        "v9",
        actor="test",
        reason="promotion",
        pointer_path=pointer,
        log_path=log_path,
        artifacts_dir=tmp_path,
    )

    assert pointer.read_text(encoding="utf-8") == "v9\n"
    assert not (pointer.stat().st_mode & stat.S_IWUSR)
    record = json.loads(log_path.read_text(encoding="utf-8").splitlines()[-1])
    assert record["actor"] == "test"
    assert record["reason"] == "promotion"
    assert record["current"] == "v9"


def test_set_latest_persona_version_requires_existing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        set_latest_persona_version("v9", pointer_path=tmp_path / "persona.latest.txt", artifacts_dir=tmp_path)
