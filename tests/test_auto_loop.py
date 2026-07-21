from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spark_character_auto_loop", REPO_ROOT / "evals" / "auto_loop.py")
assert SPEC is not None and SPEC.loader is not None
auto_loop = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auto_loop)


def test_save_state_is_atomic_and_cleans_failed_temp_file(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    original = {"cycle_count": 1}
    auto_loop._save_state(state_path, original)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(auto_loop.os, "replace", fail_replace)

    with pytest.raises(OSError, match="synthetic replace failure"):
        auto_loop._save_state(state_path, {"cycle_count": 2})

    assert json.loads(state_path.read_text(encoding="utf-8")) == original
    assert list(tmp_path.glob(".state.json.*.tmp")) == []


def test_instance_lock_rejects_a_second_owner_and_can_be_reacquired(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    first = auto_loop._acquire_instance_lock(state_path)
    try:
        with pytest.raises(RuntimeError, match="already owns"):
            auto_loop._acquire_instance_lock(state_path)
    finally:
        first.close()

    reacquired = auto_loop._acquire_instance_lock(state_path)
    reacquired.close()


def test_refresh_consumers_uses_exact_local_tree_and_surfaces_failure(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 7)

    monkeypatch.setattr(auto_loop.subprocess, "run", fake_run)
    args = SimpleNamespace(consumer_pythons="python-a, python-b")

    assert auto_loop.maybe_refresh_consumers(args, tmp_path) is False
    assert len(calls) == 2
    assert all(command[-2] == str(tmp_path.resolve()) for command in calls)
    assert all("git+" not in " ".join(command) and "@master" not in " ".join(command) for command in calls)
    output = capsys.readouterr().out
    assert "exit 7" in output
    assert "refresh is not confirmed" in output


@pytest.mark.parametrize(
    ("returncode", "stdout", "expected"),
    [
        (3, "failed", (False, False, "failed")),
        (0, "finished without promotion", (True, False, "finished without promotion")),
        (0, "PROMOTED: v8", (True, True, "PROMOTED: v8")),
    ],
)
def test_run_evolve_cycle_distinguishes_failure_from_no_promotion(
    tmp_path: Path, monkeypatch, returncode: int, stdout: str, expected: tuple[bool, bool, str]
) -> None:
    monkeypatch.setattr(
        auto_loop.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], returncode, stdout=stdout, stderr=""),
    )
    args = SimpleNamespace(
        candidates=3,
        weights="0.2,0.5,0.3",
        sib_home="synthetic-home",
        audit_limit=200,
        dry_run=False,
        evolve_timeout=30,
    )

    assert auto_loop.run_evolve_cycle(args, tmp_path) == expected


def test_cycle_decision_bootstraps_once_and_then_waits_for_threshold() -> None:
    assert auto_loop._should_fire_cycle({"last_audit_count": 0, "cycle_count": 0}, 0, 25)
    assert not auto_loop._should_fire_cycle({"last_audit_count": 0, "cycle_count": 1}, 0, 25)
    assert not auto_loop._should_fire_cycle({"last_audit_count": 10, "cycle_count": 1}, 34, 25)
    assert auto_loop._should_fire_cycle({"last_audit_count": 10, "cycle_count": 1}, 35, 25)


def test_interval_is_clamped_to_one_minute() -> None:
    assert auto_loop._bounded_interval_seconds(-1) == 60
    assert auto_loop._bounded_interval_seconds(59) == 60
    assert auto_loop._bounded_interval_seconds(60) == 60
    assert auto_loop._bounded_interval_seconds(61) == 61


def test_failed_evolution_does_not_advance_audit_cursor(tmp_path: Path, monkeypatch) -> None:
    state_path = tmp_path / "state.json"
    heartbeat_path = tmp_path / "heartbeat.txt"
    auto_loop._save_state(
        state_path,
        {
            "last_audit_count": 10,
            "cycle_count": 1,
            "loop_starts": 0,
            "last_cycle_phase": "complete",
        },
    )
    monkeypatch.setattr(auto_loop, "count_llm_replies", lambda _home: 35)
    monkeypatch.setattr(auto_loop, "resolve_latest_persona_version", lambda: "v7")
    monkeypatch.setattr(auto_loop, "run_evolve_cycle", lambda _args, _root: (False, False, "failed"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "auto_loop.py",
            "--sib-home",
            "synthetic-home",
            "--once",
            "--state-file",
            str(state_path),
            "--heartbeat-file",
            str(heartbeat_path),
        ],
    )

    assert auto_loop.main() == 1
    state = auto_loop._load_state(state_path)
    assert state["last_audit_count"] == 10
    assert state["cycle_count"] == 2
    assert state["last_cycle_phase"] == "failed"
