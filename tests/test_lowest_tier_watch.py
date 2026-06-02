"""Lowest-tier watcher state persistence tests."""

from __future__ import annotations

import importlib.util
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
