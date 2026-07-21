from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "evals" / "live_tail.py"
SPEC = importlib.util.spec_from_file_location("live_tail", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
live_tail = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(live_tail)


def test_main_expands_sib_home_before_selecting_gateway_log(tmp_path: Path, monkeypatch, capsys) -> None:
    expanded = tmp_path / "spark-home"
    monkeypatch.setattr(live_tail.Path, "expanduser", lambda _path: expanded)
    monkeypatch.setattr(live_tail, "follow_jsonl", lambda _path: iter(()))
    monkeypatch.setattr(sys, "argv", ["live_tail.py", "--sib-home", "~/.spark"])

    assert live_tail.main() is None
    assert str(expanded / "logs" / "gateway-outbound.jsonl") in capsys.readouterr().out
