import json
import re
from pathlib import Path

import pytest

from spark_character import audit_miner
from spark_character.audit_miner import (
    AuditFailure,
    AuditFindings,
    AuditMiner,
    _detect_failures,
    _iter_lines_reverse,
)


def _failure_kinds(text: str) -> set[str]:
    return {kind for kind, _detail in _detect_failures(text)}


def test_detects_markdown_emphasis_in_reply_preview():
    kinds = _failure_kinds("Short answer: **yes**, mission control first is the right call.")

    assert "markdown_emphasis" in kinds


def test_detects_dash_family_in_reply_preview():
    failures = _detect_failures("Ship v2\u2013v3 after the release audit.")

    assert ("em_dash", "1 occurrences") in failures


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


def test_audit_miner_docstring_is_self_contained_and_platform_safe():
    doc = audit_miner.__doc__ or ""

    assert "C:/Users" not in doc
    assert "from pathlib import Path" in doc
    assert 'Path.home() / ".spark" / "sib-home"' in doc


def test_iter_lines_reverse_preserves_unicode_and_line_boundaries(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes("first\nsecond \U0001f680\r\nthird".encode())

    assert list(_iter_lines_reverse(path, chunk_size=5)) == ["third", "second \U0001f680", "first"]


def test_recent_findings_streams_without_path_read_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps({"routing_decision": "provider_execution", "response_preview": "Ship it."}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()))

    findings = AuditMiner(path).recent_findings(limit=1)

    assert findings.rows_scanned == 1
    assert findings.llm_rows == 1


def test_recent_findings_scans_past_filtered_tail_without_semantic_loss(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    target = {
        "telegram_user_id": "target",
        "routing_decision": "provider_execution",
        "response_preview": "A useful reply \u2014 with a detectable dash.",
    }
    other = {
        "telegram_user_id": "other",
        "routing_decision": "provider_execution",
        "response_preview": "Other user.",
    }
    lines = [json.dumps(target), *(json.dumps(other) for _ in range(500))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    findings = AuditMiner(path).recent_findings(limit=1, only_user="target")

    assert findings.rows_scanned == 1
    assert findings.failures_by_kind == {"em_dash": 1}


def test_recent_findings_skips_non_object_json_rows(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    valid = {"routing_decision": "provider_execution", "response_preview": "Ship it."}
    path.write_text(json.dumps(valid) + "\n[]\nnull\n", encoding="utf-8")

    findings = AuditMiner(path).recent_findings(limit=1)

    assert findings.rows_scanned == 1


def test_detect_failures_searches_reset_and_hedge_once(monkeypatch: pytest.MonkeyPatch):
    class CountingPattern:
        def __init__(self) -> None:
            self.pattern = re.compile("hello", re.IGNORECASE)
            self.calls = 0

        def search(self, text: str):
            self.calls += 1
            return self.pattern.search(text)

    reset = CountingPattern()
    hedge = CountingPattern()
    monkeypatch.setattr(audit_miner, "RESET_PATTERN", reset)
    monkeypatch.setattr(audit_miner, "HEDGE_PATTERN", hedge)

    failures = _detect_failures("hello there.")

    assert {kind for kind, _detail in failures} >= {"reset", "hedge_opener"}
    assert reset.calls == 1
    assert hedge.calls == 1


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

    joined = "\n".join(lines)
    assert "Private reply text" not in joined
    assert "dense_opening" in lines[0]
    assert "provider_execution" in lines[0]
