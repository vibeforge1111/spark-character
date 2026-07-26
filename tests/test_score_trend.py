"""Exact score-trend behavior, including CLI consumers and UTC output."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "evals" / "score_trend.py"
SPEC = importlib.util.spec_from_file_location("spark_character_score_trend", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
score_trend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_trend)


def test_compare_back_one_uses_the_previous_run() -> None:
    rows = [{"t1_mean": 0.2}, {"t1_mean": 0.5}, {"t1_mean": 0.9}]
    summary = score_trend._summarize(rows, compare_back=1)
    assert summary["t1_mean"]["delta_vs_compare"] == 0.4


def test_compare_back_two_uses_two_runs_before_current() -> None:
    rows = [{"t1_mean": 0.2}, {"t1_mean": 0.5}, {"t1_mean": 0.9}]
    summary = score_trend._summarize(rows, compare_back=2)
    assert summary["t1_mean"]["delta_vs_compare"] == 0.7


def test_compare_back_does_not_substitute_an_insufficient_or_missing_run() -> None:
    insufficient = score_trend._summarize([{"t1_mean": 0.2}, {"t1_mean": 0.9}], compare_back=2)
    missing = score_trend._summarize([{"t1_mean": 0.2}, {}, {"t1_mean": 0.9}], compare_back=1)
    assert insufficient["t1_mean"]["delta_vs_compare"] is None
    assert missing["t1_mean"]["delta_vs_compare"] is None


def test_table_labels_timestamps_as_utc() -> None:
    table = score_trend._format_table([{"ts": 0, "persona_version": "v1", "tier": "full"}])
    assert "when (UTC)" in table
    assert "1970-01-01 00:00:00Z" in table


def test_json_is_single_line_when_stdout_is_captured(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps({"ts": 1, "t1_mean": 0.8}) + "\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--history-file", str(history), "--json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert len(result.stdout.splitlines()) == 1
    assert json.loads(result.stdout)["summary"]["t1_mean"]["latest"] == 0.8


def test_non_positive_window_arguments_are_rejected() -> None:
    for flag in ("--last", "--compare-back"):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), flag, "0"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "expected a positive integer" in result.stderr
