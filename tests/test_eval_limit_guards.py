import json
from pathlib import Path

from evals import continuous_eval
from evals import lowest_tier_watch
from evals import observations_digest
from evals import observer
from evals import score_trend
from spark_character.audit_miner import AuditMiner


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_eval_history_limit_zero_returns_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    _write_jsonl(path, [{"run": 1}, {"run": 2}])

    assert continuous_eval._load_history(path, limit=0) == []
    assert continuous_eval._load_history(path, limit=-1) == []
    assert score_trend._load(path, limit=0) == []
    assert score_trend._load(path, limit=-1) == []
    assert lowest_tier_watch._load(path, limit=0) == []
    assert lowest_tier_watch._load(path, limit=-1) == []


def test_observation_limit_zero_returns_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "observations.jsonl"
    _write_jsonl(path, [{"observation": 1}, {"observation": 2}])

    assert observations_digest._load(path, limit=0) == []
    assert observations_digest._load(path, limit=-1) == []
    assert observer.replay_recent_rows(path, limit=0) == []
    assert observer.replay_recent_rows(path, limit=-1) == []


def test_audit_miner_limit_zero_does_not_read_log_rows(tmp_path: Path) -> None:
    log = tmp_path / "gateway-outbound.jsonl"
    _write_jsonl(
        log,
        [
            {
                "routing_decision": "provider_execution",
                "response_preview": "This should not be scanned when limit is zero.",
            }
        ],
    )

    findings = AuditMiner(log).recent_findings(limit=0)

    assert findings.rows_scanned == 0
    assert findings.llm_rows == 0
    assert findings.failures == []

    negative_findings = AuditMiner(log).recent_findings(limit=-1)

    assert negative_findings.rows_scanned == 0
    assert negative_findings.llm_rows == 0
    assert negative_findings.failures == []
