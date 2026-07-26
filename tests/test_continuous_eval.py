from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_character_continuous_eval", REPO_ROOT / "evals" / "continuous_eval.py"
)
assert SPEC is not None and SPEC.loader is not None
continuous_eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(continuous_eval)


def test_load_history_keeps_last_valid_objects_with_bounded_memory(tmp_path: Path) -> None:
    history = tmp_path / "history.jsonl"
    history.write_text(
        '\n'.join(['{"n": 1}', 'not-json', '[2]', '{"n": 2}', '', '{"n": 3}']) + '\n',
        encoding="utf-8",
    )

    assert continuous_eval._load_history(history, limit=2) == [{"n": 2}, {"n": 3}]
    assert continuous_eval._load_history(history, limit=0) == []
    assert continuous_eval._load_history(history, limit=-1) == []


def test_compute_baseline_excludes_non_finite_and_boolean_values() -> None:
    history = [
        {"score": math.nan},
        {"score": math.inf},
        {"score": True},
        {"score": 0.6},
        {"score": 0.8},
    ]

    assert continuous_eval._compute_baseline(history, axis="score") == 0.7


def test_full_eval_records_per_tier_failures(monkeypatch) -> None:
    monkeypatch.setattr(continuous_eval, "run_fast_eval", lambda *_args, **_kwargs: {"tier": "fast"})
    for name in (
        "PROBES",
        "STABILITY_SCENARIOS",
        "T6_EMOTIONAL_ATTUNEMENT_PROBES",
        "T7_MEMORY_COHERENCE_PROBES",
        "T8_INITIATIVE_PROBES",
        "T9_AESTHETIC_FINGERPRINT_PROBES",
        "T11_SUSTAINED_ATTACK_SCENARIOS",
        "T13_HUMANE_DEPTH_PROBES",
        "T14_MEMORABILITY_PROBES",
    ):
        monkeypatch.setattr(continuous_eval, name, [object()])

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic probe failure")

    monkeypatch.setattr(continuous_eval, "run_probe", fail)
    monkeypatch.setattr(continuous_eval, "run_deep_probe", fail)
    monkeypatch.setattr(continuous_eval, "run_stability_scenario", fail)

    result = continuous_eval.run_full_eval(SimpleNamespace(), SimpleNamespace(), include_sustained=True)

    assert result["probe_failures"] == {
        "t3": 1,
        "t4": 1,
        "t6": 1,
        "t7": 1,
        "t8": 1,
        "t9": 1,
        "t13": 1,
        "t14": 1,
        "t11": 1,
    }


def test_unknown_provider_and_missing_key_are_named(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["continuous_eval.py", "--providers", "typo,zai", "--once"])

    assert continuous_eval.main() == 2
    output = capsys.readouterr().out
    assert "unknown provider" in output
    assert "ZAI_API_KEY is not set" in output


def test_failed_full_eval_waits_for_full_interval_before_retry(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    clock = iter([2000.0, 2000.1, 2001.0, 2001.1])
    provider = SimpleNamespace(model="synthetic-model")
    persona = SimpleNamespace(version="v-test")
    monkeypatch.setattr(continuous_eval, "resolve_provider", lambda _name: provider)
    monkeypatch.setattr(continuous_eval, "load_persona", lambda **_kwargs: persona)
    monkeypatch.setattr(continuous_eval.time, "time", lambda: next(clock))
    monkeypatch.setattr(continuous_eval.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(continuous_eval._logger, "exception", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(continuous_eval, "_write_heartbeat", lambda *_args: None)

    def fail_full(*_args, **_kwargs):
        calls.append("full")
        raise RuntimeError("synthetic full failure")

    def stop_fast(*_args, **_kwargs):
        calls.append("fast")
        raise KeyboardInterrupt

    monkeypatch.setattr(continuous_eval, "run_full_eval", fail_full)
    monkeypatch.setattr(continuous_eval, "run_fast_eval", stop_fast)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "continuous_eval.py",
            "--providers",
            "zai",
            "--fast-interval",
            "60",
            "--full-interval",
            "1000",
            "--history-file",
            str(tmp_path / "history.jsonl"),
            "--heartbeat-file",
            str(tmp_path / "heartbeat.txt"),
        ],
    )

    assert continuous_eval.main() == 0
    assert calls == ["full", "fast"]


def test_audit_failure_is_visible_without_reflecting_private_details(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    history = tmp_path / "history.jsonl"
    provider = SimpleNamespace(model="synthetic-model")
    persona = SimpleNamespace(version="v-test")
    monkeypatch.setattr(continuous_eval, "resolve_provider", lambda _name: provider)
    monkeypatch.setattr(continuous_eval, "load_persona", lambda **_kwargs: persona)
    monkeypatch.setattr(
        continuous_eval,
        "run_full_eval",
        lambda *_args, **_kwargs: {"tier": "full", "t1_mean": 1.0},
    )

    def fail_audit(_home: str):
        raise PermissionError("/private/operator/home/audit.db")

    monkeypatch.setattr(continuous_eval, "AuditMiner", SimpleNamespace(from_sib_home=fail_audit))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "continuous_eval.py",
            "--providers",
            "zai",
            "--once",
            "--sib-home",
            "/private/operator/home",
            "--history-file",
            str(history),
            "--heartbeat-file",
            str(tmp_path / "heartbeat.txt"),
        ],
    )

    assert continuous_eval.main() == 0
    row = json.loads(history.read_text(encoding="utf-8"))
    assert row["production_audit_error"] == {
        "type": "PermissionError",
        "message": "production audit evidence unavailable",
    }
    combined = capsys.readouterr().out + history.read_text(encoding="utf-8")
    assert "/private/operator" not in combined


def test_cli_help_is_complete_and_inert_dry_run_is_rejected() -> None:
    help_result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "evals" / "continuous_eval.py"), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0
    for flag in ("--fast-interval", "--full-interval", "--regression-threshold", "--history-file", "--heartbeat-file"):
        assert flag in help_result.stdout

    dry_run = subprocess.run(
        [sys.executable, str(REPO_ROOT / "evals" / "continuous_eval.py"), "--dry-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert dry_run.returncode == 2
    assert "unrecognized arguments: --dry-run" in dry_run.stderr
