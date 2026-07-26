"""Persona comparison must keep missing evidence distinct from real zeroes."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_character_compare_personas", REPO_ROOT / "evals" / "compare_personas.py"
)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def test_defaults_follow_the_latest_two_numeric_persona_versions(monkeypatch, tmp_path: Path) -> None:
    for version in ("v2", "v9", "v10"):
        (tmp_path / f"persona.{version}.md").write_text(f"# {version}\n", encoding="utf-8")
    monkeypatch.setattr(compare, "ARTIFACTS_DIR", tmp_path)
    assert compare._default_versions() == ("v9", "v10")


def test_active_pointer_is_the_baseline_when_a_newer_candidate_exists(monkeypatch, tmp_path: Path) -> None:
    for version in ("v8", "v9", "v10"):
        (tmp_path / f"persona.{version}.md").write_text(f"# {version}\n", encoding="utf-8")
    (tmp_path / "persona.latest.txt").write_text("v9\n", encoding="utf-8")
    monkeypatch.setattr(compare, "ARTIFACTS_DIR", tmp_path)
    assert compare._default_versions() == ("v9", "v10")


def test_unknown_version_names_available_versions_without_leaking_path(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "persona.v3.md").write_text("# v3\n", encoding="utf-8")
    monkeypatch.setattr(compare, "ARTIFACTS_DIR", tmp_path)
    try:
        compare._load("typo")
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("unknown persona version was accepted")
    assert "available versions: v3" in message
    assert str(tmp_path) not in message


def test_missing_tiers_are_none_with_explicit_counts(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "persona.v1.md").write_text("# v1\n\nBe direct.\n", encoding="utf-8")
    monkeypatch.setattr(compare, "ARTIFACTS_DIR", tmp_path)

    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(compare, "generate", fail)
    monkeypatch.setattr(compare, "run_probe", fail)
    monkeypatch.setattr(compare, "run_deep_probe", fail)
    logs: list[str] = []
    row = compare.score("v1", SimpleNamespace(), max_tokens=10, log=logs.append)

    assert all(row[tier] is None for tier in ("t1", "t2", "t3", "t6", "t7", "t8"))
    assert row["score_counts"] == {"t1": 0, "t2": 0, "t3": 0, "t6": 0, "t7": 0, "t8": 0}
    assert row["score_error_counts"]["generate"] == len(compare.PROMPTS)
    assert row["score_error_counts"]["t3"] == len(compare.PROBES)
    assert any("generation failed (RuntimeError)" in message for message in logs)
    assert all("provider unavailable" not in message for message in logs)
    assert compare.composite(row, (0.2, 0.3, 0.2, 0.1, 0.1, 0.1)) is None


def test_json_mode_emits_one_machine_readable_document(monkeypatch, capsys) -> None:
    monkeypatch.setattr(compare.ProviderSpec, "from_env", lambda: SimpleNamespace(model="test-model"))
    monkeypatch.setattr(compare, "_default_versions", lambda: ("v9", "v10"))

    def complete(version: str, *_args, **_kwargs) -> dict:
        value = 0.8 if version == "v9" else 0.9
        return {
            "version": version,
            "t1": value, "t2": value, "t3": value,
            "t6": value, "t7": value, "t8": value,
            "t8_per_probe": [],
            "score_counts": {tier: 1 for tier in ("t1", "t2", "t3", "t6", "t7", "t8")},
        }

    monkeypatch.setattr(compare, "score", complete)
    monkeypatch.setattr(sys, "argv", ["compare_personas.py", "--json"])
    assert compare.main() == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "complete"
    assert payload["baseline"]["version"] == "v9"
    assert payload["candidate"]["version"] == "v10"
    assert payload["delta"] == 0.1


def test_incomplete_comparison_returns_two_without_a_verdict(monkeypatch, capsys) -> None:
    monkeypatch.setattr(compare.ProviderSpec, "from_env", lambda: SimpleNamespace(model="test-model"))
    monkeypatch.setattr(compare, "_default_versions", lambda: ("v9", "v10"))

    def incomplete(version: str, *_args, **_kwargs) -> dict:
        return {
            "version": version,
            "t1": None, "t2": None, "t3": None,
            "t6": None, "t7": None, "t8": None,
            "t8_per_probe": [],
            "score_counts": {tier: 0 for tier in ("t1", "t2", "t3", "t6", "t7", "t8")},
        }

    monkeypatch.setattr(compare, "score", incomplete)
    monkeypatch.setattr(sys, "argv", ["compare_personas.py", "--json"])
    assert compare.main() == 2
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "incomplete"
    assert payload["delta"] is None
    assert "winner" not in payload
