"""Evaluation gates must fail closed when provider or judge coverage is partial."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


full_pulse = _load("spark_character_full_pulse", "evals/full_pulse.py")
live_pulse = _load("spark_character_live_pulse", "evals/live_pulse.py")
evolve = _load("spark_character_evolve", "evals/evolve.py")


def test_evolve_baseline_is_zero_when_every_prompt_fails(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(evolve, "generate", fail)
    overall, rows = evolve.baseline_score(object(), object())

    assert overall == 0.0
    assert len(rows) == len(evolve.PROMPTS)
    assert all("error" in row for row in rows)


def test_evolve_persona_artifact_write_is_atomic_and_cleans_failed_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "persona.v9.md"
    artifact.write_text("old persona", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        assert source.parent == artifact.parent
        assert destination == artifact
        assert source.read_text(encoding="utf-8") == "new persona"
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(evolve.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        evolve._write_persona_artifact(artifact, "new persona")

    assert artifact.read_text(encoding="utf-8") == "old persona"
    assert list(tmp_path.glob(".persona.v9.md.*.tmp")) == []


def test_evolve_persona_artifact_write_replaces_complete_file(tmp_path: Path) -> None:
    artifact = tmp_path / "nested" / "persona.v9.md"

    evolve._write_persona_artifact(artifact, "new persona")

    assert artifact.read_text(encoding="utf-8") == "new persona"
    assert list(artifact.parent.glob(".persona.v9.md.*.tmp")) == []


def test_evolve_promotes_artifact_before_pointer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, str]] = []
    baseline = SimpleNamespace(version="v8", system_prompt="# baseline")
    scores = iter([(0.5, []), (0.8, [])])
    monkeypatch.setattr(evolve.ProviderSpec, "from_env", lambda: SimpleNamespace())
    monkeypatch.setattr(evolve, "find_latest_persona", lambda: (8, baseline))
    monkeypatch.setattr(evolve, "baseline_score", lambda *_args: next(scores))
    monkeypatch.setattr(evolve, "diagnose_weaknesses", lambda _rows: ["synthetic"])
    monkeypatch.setattr(evolve, "mutate_persona", lambda *_args: "# winning persona")
    monkeypatch.setattr(evolve, "ARTIFACTS_DIR", tmp_path)
    monkeypatch.setattr(
        evolve,
        "_write_persona_artifact",
        lambda path, text: events.append(("artifact", f"{path.name}:{text}")),
    )
    monkeypatch.setattr(
        evolve,
        "set_latest_persona_version",
        lambda version, **_kwargs: events.append(("pointer", version)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--candidates", "1", "--out", str(tmp_path / "result.json")],
    )

    assert evolve.main() == 0
    assert events == [
        ("artifact", "persona.v9.md:# winning persona"),
        ("pointer", "v9"),
    ]


def test_evolve_preserves_candidate_failure_summary_in_new_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    baseline = SimpleNamespace(version="v8", system_prompt="# baseline")
    attempts = iter([RuntimeError("secret provider detail"), "# candidate two"])
    scores = iter([(0.5, []), (0.4, [])])
    monkeypatch.setattr(evolve.ProviderSpec, "from_env", lambda: SimpleNamespace())
    monkeypatch.setattr(evolve, "find_latest_persona", lambda: (8, baseline))
    monkeypatch.setattr(evolve, "baseline_score", lambda *_args: next(scores))
    monkeypatch.setattr(evolve, "diagnose_weaknesses", lambda _rows: ["synthetic"])

    def mutate(*_args):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(evolve, "mutate_persona", mutate)
    output = tmp_path / "missing" / "nested" / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["evolve.py", "--candidates", "2", "--dry-run", "--out", str(output)],
    )

    assert evolve.main() == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["candidate_attempts"] == 2
    assert payload["candidate_error_count"] == 1
    assert payload["candidate_errors"] == [{"index": 1, "error_type": "RuntimeError"}]
    assert "secret provider detail" not in output.read_text(encoding="utf-8")


def test_full_pulse_counts_explicit_and_missing_scores() -> None:
    rows = [{"score": 0.9}, {"error": "judge unavailable"}]
    assert full_pulse._missing_score_count(rows, expected=3, score_key="score") == 2


def test_full_pulse_counts_missing_t2_judge_rows() -> None:
    rows = [{"score": 0.9}]
    assert full_pulse._missing_score_count(rows, expected=len(full_pulse.T1_PROMPTS), score_key="score") == len(full_pulse.T1_PROMPTS) - 1


def test_full_pulse_docstring_matches_all_required_exit_tiers() -> None:
    documentation = full_pulse.__doc__ or ""
    assert "T2/T3/T4/T6/T7/T8/T9 means each >= 0.6" in documentation
    assert "T11 mean >= 0.6 when --include-sustained is set" in documentation


def test_live_pulse_counts_partial_prompt_coverage() -> None:
    rows = [{"score": {"mean": 1.0}}, {"error": "provider unavailable"}]
    assert live_pulse._missing_score_count(rows, expected=3, score_key="score") == 2
