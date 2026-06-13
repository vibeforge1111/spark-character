import json

from spark_character.audit_miner import AuditFailure, AuditFindings, AuditMiner, _detect_failures


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


def test_recent_findings_falls_back_to_gateway_trace_when_outbound_empty(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "gateway-outbound.jsonl").write_text("", encoding="utf-8")
    trace_row = {
        "recorded_at": "2026-06-13T00:00:00Z",
        "event": "telegram_update_processed",
        "telegram_user_id": "user-1",
        "routing_decision": "provider_execution",
        "active_chip_key": "founder-operator",
        "response_preview": "Short answer: **yes**, mission control first is the right call.",
        "guardrail_actions": ["replace_em_dashes"],
    }
    (logs_dir / "gateway-trace.jsonl").write_text(json.dumps(trace_row) + "\n", encoding="utf-8")

    findings = AuditMiner.from_sib_home(tmp_path).recent_findings(limit=10)

    assert findings.rows_scanned == 1
    assert findings.llm_rows == 1
    assert findings.failures_by_kind["markdown_emphasis"] == 1
    assert findings.failures_by_kind["em_dash"] == 1
    assert any("outbound audit source empty or stale" in warning for warning in findings.warnings)
    assert any("using gateway trace fallback" in warning for warning in findings.warnings)


def test_recent_findings_warns_when_audit_sources_are_missing(tmp_path):
    findings = AuditMiner.from_sib_home(tmp_path).recent_findings(limit=10)

    assert findings.rows_scanned == 0
    assert any("outbound audit source missing" in warning for warning in findings.warnings)
    assert any("gateway trace fallback missing" in warning for warning in findings.warnings)
    assert "warning:" in findings.summary()


def test_gateway_trace_fallback_respects_only_user_filter(tmp_path):
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    (logs_dir / "gateway-outbound.jsonl").write_text("", encoding="utf-8")
    trace_rows = [
        {
            "recorded_at": "2026-06-13T00:00:00Z",
            "event": "telegram_update_processed",
            "telegram_user_id": "user-1",
            "routing_decision": "provider_execution",
            "response_preview": "Short answer: **yes**.",
        },
        {
            "recorded_at": "2026-06-13T00:00:01Z",
            "event": "telegram_update_processed",
            "telegram_user_id": "user-2",
            "routing_decision": "provider_execution",
            "response_preview": "Clean reply.",
        },
    ]
    (logs_dir / "gateway-trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in trace_rows) + "\n",
        encoding="utf-8",
    )

    findings = AuditMiner.from_sib_home(tmp_path).recent_findings(limit=10, only_user="user-1")

    assert findings.rows_scanned == 1
    assert findings.llm_rows == 1
    assert findings.failures_by_kind["markdown_emphasis"] == 1
