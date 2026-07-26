"""Lowest-tier watcher state persistence tests."""

from __future__ import annotations

import importlib.util
import logging
import subprocess
from pathlib import Path


def load_lowest_tier_watch():
    module_path = Path(__file__).resolve().parents[1] / "evals" / "lowest_tier_watch.py"
    spec = importlib.util.spec_from_file_location("lowest_tier_watch", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_save_state_replaces_temp_file(tmp_path: Path) -> None:
    watcher = load_lowest_tier_watch()
    state_path = tmp_path / "state.json"

    watcher._save_state(state_path, {"fires_total": 1})

    assert state_path.read_text(encoding="utf-8") == '{\n  "fires_total": 1\n}'
    assert not (tmp_path / "state.json.tmp").exists()


def test_save_state_removes_temp_file_when_replace_fails(tmp_path: Path, monkeypatch) -> None:
    watcher = load_lowest_tier_watch()
    state_path = tmp_path / "state.json"

    def fail_replace(*_args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(watcher.os, "replace", fail_replace)
    try:
        watcher._save_state(state_path, {"fires_total": 1})
    except OSError:
        pass
    else:
        raise AssertionError("replace failure was swallowed")
    assert not (tmp_path / "state.json.tmp").exists()


def test_save_state_removes_partial_temp_file_when_write_fails(tmp_path: Path, monkeypatch) -> None:
    watcher = load_lowest_tier_watch()
    state_path = tmp_path / "state.json"

    def fail_write(path: Path, *_args, **_kwargs):
        path.touch()
        raise OSError("synthetic write failure")

    monkeypatch.setattr(Path, "write_text", fail_write)
    try:
        watcher._save_state(state_path, {"fires_total": 1})
    except OSError:
        pass
    else:
        raise AssertionError("write failure was swallowed")
    assert not (tmp_path / "state.json.tmp").exists()


def test_heartbeat_failure_is_visible_without_exposing_path(tmp_path: Path, caplog) -> None:
    watcher = load_lowest_tier_watch()
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("x", encoding="utf-8")
    heartbeat = blocked_parent / "heartbeat.txt"
    with caplog.at_level(logging.WARNING, logger="lowest_tier_watch"):
        watcher._write_heartbeat(heartbeat, "checking")
    assert "heartbeat write failed" in caplog.text
    assert "phase=checking" in caplog.text
    assert str(tmp_path) not in caplog.text


def test_fire_evolution_returns_timeout_failure(tmp_path: Path, monkeypatch) -> None:
    watcher = load_lowest_tier_watch()

    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=watcher.EVOLUTION_SUBPROCESS_TIMEOUT_SECONDS)

    monkeypatch.setattr(watcher.subprocess, "run", fake_run)

    promoted, message = watcher.fire_evolution(
        target_tier="t1_mean",
        sib_home=None,
        consumer_pythons=None,
        candidates=2,
        repo_root=tmp_path,
        dry_run=False,
    )

    assert promoted is False
    assert "[timeout]" in message
    assert str(watcher.EVOLUTION_SUBPROCESS_TIMEOUT_SECONDS) in message
