from pathlib import Path

from spark_character import audit_miner
from spark_character.audit_miner import (
    AuditFailure,
    AuditFindings,
    AuditMiner,
    _detect_failures,
)


def _failure_kinds(text: str) -> set[str]:
    return {kind for kind, _detail in _detect_failures(text)}


def test_detects_markdown_emphasis_in_reply_preview():
    kinds = _failure_kinds("Short answer: **yes**, mission control first is the right call.")

    assert "markdown_emphasis" in kinds


def test_detects_dense_opening_in_reply_preview():
    text = (
        "Mission control first is the right call because it lets you observe active work, "
        "inspect failures, intervene quickly, and learn from each run before expanding into canvas work"
    )

    kinds = _failure_kinds(text)

    assert "dense_opening" in kinds


def test_does_not_flag_short_scannable_reply_as_dense():
    text = "Mission control first is the right call.\n\nThen canvas has a place to report progress."

    kinds = _failure_kinds(text)

    assert "dense_opening" not in kinds


def test_audit_miner_module_docstring_uses_platform_safe_home_example():
    doc = audit_miner.__doc__ or ""

    assert "C:/Users" not in doc
    assert 'Path.home() / ".spark" / "sib-home"' in doc


def test_from_sib_home_accepts_path_home_style(tmp_path: Path):
    miner = AuditMiner.from_sib_home(tmp_path)

    assert miner.log_path == tmp_path / "logs" / "gateway-outbound.jsonl"


def test_diagnose_lines_do_not_include_reply_preview():
    findings = AuditFindings(
        rows_scanned=1,
        llm_rows=1,
        failures_by_kind={"dense_opening": 1},
        failures=[
            AuditFailure(
                kind="dense_opening",
                detail="long single-paragraph preview with few sentence breaks",
                route="provider_execution",
                chip="founder-operator",
                preview="Private reply text should not be copied into diagnose lines.",
                recorded_at="2026-05-13T00:00:00Z",
            )
        ],
    )

    lines = findings.diagnose_lines()

    assert "Private reply text" not in lines[0]
    assert "dense_opening" in lines[0]
    assert "provider_execution" in lines[0]
