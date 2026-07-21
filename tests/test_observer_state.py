"""Observer state must remain ordered, bounded, recoverable, and observable."""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_character_observer", REPO_ROOT / "evals" / "observer.py"
)
assert SPEC is not None and SPEC.loader is not None
observer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = observer
SPEC.loader.exec_module(observer)


def test_seen_file_load_preserves_order_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "seen.json"
    path.write_text(json.dumps({"seen_trace_refs": ["a", "b", "a", 7, "c"]}), encoding="utf-8")
    assert list(observer._load_seen(path)) == ["a", "b", "c"]


def test_seen_file_save_keeps_the_newest_five_thousand(tmp_path: Path) -> None:
    path = tmp_path / "seen.json"
    seen = dict.fromkeys(f"trace-{index}" for index in range(5010))
    observer._save_seen(path, seen)
    saved = json.loads(path.read_text(encoding="utf-8"))["seen_trace_refs"]
    assert len(saved) == 5000
    assert saved[0] == "trace-10"
    assert saved[-1] == "trace-5009"


def test_observation_recovery_is_ordered_deduplicated_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    rows = [json.dumps({"trace_ref": f"trace-{index}"}) for index in range(5002)]
    rows.extend(["not-json", json.dumps({"trace_ref": "trace-5001"}), json.dumps({"trace_ref": 9})])
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    recovered = observer._load_seen_from_observations(path)
    assert len(recovered) == 5000
    assert next(iter(recovered)) == "trace-2"
    assert list(recovered)[-1] == "trace-5001"


def test_merge_seen_moves_recovered_recent_refs_to_the_tail() -> None:
    merged = observer._merge_seen(
        dict.fromkeys(["old", "shared"]),
        dict.fromkeys(["new", "shared"]),
    )
    assert list(merged) == ["old", "new", "shared"]


def test_heartbeat_failure_is_visible_without_exposing_path(tmp_path: Path, caplog) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("x", encoding="utf-8")
    heartbeat = blocked_parent / "heartbeat.txt"
    with caplog.at_level(logging.WARNING, logger="spark_character_observer"):
        observer._write_heartbeat(heartbeat, "tailing")
    assert "heartbeat write failed" in caplog.text
    assert "phase=tailing" in caplog.text
    assert str(tmp_path) not in caplog.text
