"""Evaluation gates must fail closed when provider or judge coverage is partial."""

from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_full_pulse_counts_explicit_and_missing_scores() -> None:
    rows = [{"score": 0.9}, {"error": "judge unavailable"}]
    assert full_pulse._missing_score_count(rows, expected=3, score_key="score") == 2


def test_full_pulse_counts_missing_t2_judge_rows() -> None:
    rows = [{"score": 0.9}]
    assert full_pulse._missing_score_count(rows, expected=len(full_pulse.T1_PROMPTS), score_key="score") == len(full_pulse.T1_PROMPTS) - 1


def test_live_pulse_counts_partial_prompt_coverage() -> None:
    rows = [{"score": {"mean": 1.0}}, {"error": "provider unavailable"}]
    assert live_pulse._missing_score_count(rows, expected=3, score_key="score") == 2
