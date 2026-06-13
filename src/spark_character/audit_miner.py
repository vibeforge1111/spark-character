"""Audit miner: read SIB outbound traces and surface local failure modes.

The Spark Intelligence Builder writes one JSON line per outbound bot
reply to logs/gateway-outbound.jsonl in each home directory. Each line
includes the route, chip, decision, and a 160-char response_preview.
On bridge-driven live lanes that file may be empty; the miner then falls
back to logs/gateway-trace.jsonl, which carries the same redacted preview
and guardrail labels for processed Telegram updates.

This module reads that log, runs T1 surface-mechanic scoring on the
preview text, and aggregates summary findings so the evolution loop can
target observed failure classes instead of just synthetic probe prompts.

Limitations:
- The audit log only stores a 160-char preview, not the full reply.
  T1 patterns (em dash, plumbing leaks, hedge openers, greeting
  resets) usually fire in the first 160 chars, but T2/T3 judge
  scoring needs full text and is intentionally out of scope here.
- We skip canned fallbacks (Noted, mission_control_direct, etc.)
  because their text is hand-written and not voice-evolvable.

Usage:
    miner = AuditMiner.from_sib_home(Path.home() / ".spark" / "sib-home")
    findings = miner.recent_findings(limit=50)
    print(findings.summary())
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .output_sanitizer import EM_DASH_FAMILY
from .scoring import (
    HEDGE_PATTERN,
    PLUMBING_PATTERN,
    RESET_PATTERN,
    _first_sentence,
)

MARKDOWN_EMPHASIS_PATTERN = re.compile(
    r"(?<!\\)(?:\*\*\*[^*\n]+?\*\*\*|\*\*[^*\n]+?\*\*|__[^_\n]+?__)"
)

DENSE_OPENING_MIN_CHARS = 135


# Routes whose reply text comes from the LLM. Anything else (memory
# observations, mission control, error fallback shapers, runtime
# commands) has hand-written text that the persona cannot improve.
LLM_ROUTES = frozenset(
    {
        "provider_fallback_chat",
        "provider_fallback_chat+manual_recommended",
        "provider_execution",
        "provider_execution+manual_recommended",
        "browser_search_provider_chat+manual_recommended",
    }
)


@dataclass(frozen=True)
class AuditFailure:
    kind: str
    detail: str
    route: str
    chip: str | None
    preview: str
    recorded_at: str


@dataclass
class AuditFindings:
    rows_scanned: int = 0
    llm_rows: int = 0
    failures_by_kind: dict[str, int] = field(default_factory=dict)
    failures: list[AuditFailure] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Audit miner findings",
            f"  rows scanned: {self.rows_scanned}",
            f"  LLM-generated rows: {self.llm_rows}",
        ]
        for warning in self.warnings:
            lines.append(f"  warning: {warning}")
        if not self.failures_by_kind:
            lines.append("  no T1 failures detected on recent LLM replies")
            return "\n".join(lines)
        lines.append("  failures by kind:")
        for kind, count in sorted(self.failures_by_kind.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {kind}: {count}")
        return "\n".join(lines)

    def diagnose_lines(self, max_per_kind: int = 3) -> list[str]:
        """Format failures as evolution-ready diagnose lines without raw replies."""
        out: list[str] = []
        seen: dict[str, int] = {}
        for f in self.failures:
            seen[f.kind] = seen.get(f.kind, 0) + 1
            if seen[f.kind] > max_per_kind:
                continue
            out.append(
                f"observed T1 {f.kind} on route {f.route} "
                f"chip {f.chip or 'none'}: {f.detail}"
            )
        return out


@dataclass
class AuditMiner:
    log_path: Path
    trace_path: Path | None = None

    @classmethod
    def from_sib_home(cls, home: str | Path) -> "AuditMiner":
        logs_dir = Path(home) / "logs"
        return cls(
            log_path=logs_dir / "gateway-outbound.jsonl",
            trace_path=logs_dir / "gateway-trace.jsonl",
        )

    def recent_findings(
        self,
        *,
        limit: int = 100,
        only_user: str | None = None,
    ) -> AuditFindings:
        rows, warnings = self._recent_rows(limit=limit, only_user=only_user)

        findings = AuditFindings(rows_scanned=len(rows), warnings=warnings)
        for row in rows:
            route = str(row.get("routing_decision") or "")
            if route not in LLM_ROUTES:
                continue
            preview = str(row.get("response_preview") or "").strip()
            if not preview:
                continue
            findings.llm_rows += 1
            chip = row.get("active_chip_key")
            recorded_at = str(row.get("recorded_at") or "")
            for kind, detail in _detect_failures(preview):
                fail = AuditFailure(
                    kind=kind,
                    detail=detail,
                    route=route,
                    chip=str(chip) if chip else None,
                    preview=preview,
                    recorded_at=recorded_at,
                )
                findings.failures.append(fail)
                findings.failures_by_kind[kind] = findings.failures_by_kind.get(kind, 0) + 1
            for kind, detail in _detect_action_failures(row):
                fail = AuditFailure(
                    kind=kind,
                    detail=detail,
                    route=route,
                    chip=str(chip) if chip else None,
                    preview=preview,
                    recorded_at=recorded_at,
                )
                findings.failures.append(fail)
                findings.failures_by_kind[kind] = findings.failures_by_kind.get(kind, 0) + 1
        return findings

    def _recent_rows(self, *, limit: int, only_user: str | None) -> tuple[list[dict[str, Any]], list[str]]:
        warnings: list[str] = []
        rows = _read_recent_jsonl_rows(self.log_path, limit=limit, only_user=only_user)
        if rows:
            return rows, warnings
        if not self.log_path.exists():
            warnings.append(f"outbound audit source missing: {self.log_path}")
        else:
            warnings.append(f"outbound audit source empty or stale: {self.log_path}")
        if self.trace_path is None:
            return rows, warnings
        trace_rows = [
            _gateway_trace_to_outbound_row(row)
            for row in _read_recent_jsonl_rows(self.trace_path, limit=limit, only_user=only_user)
            if str(row.get("event") or "") == "telegram_update_processed"
        ]
        trace_rows = [row for row in trace_rows if row]
        if trace_rows:
            warnings.append(f"using gateway trace fallback: {self.trace_path}")
            return trace_rows[:limit], warnings
        if not self.trace_path.exists():
            warnings.append(f"gateway trace fallback missing: {self.trace_path}")
        else:
            warnings.append(f"gateway trace fallback had no processed Telegram rows: {self.trace_path}")
        return [], warnings


def _detect_failures(text: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    dash_count = sum(text.count(ch) for ch in EM_DASH_FAMILY)
    if dash_count:
        out.append(("em_dash", f"{dash_count} occurrences"))
    markdown_matches = MARKDOWN_EMPHASIS_PATTERN.findall(text)
    if markdown_matches:
        out.append(("markdown_emphasis", f"{len(markdown_matches)} markdown emphasis markers"))
    if _looks_like_dense_opening(text):
        out.append(("dense_opening", "long single-paragraph preview with few sentence breaks"))
    matches = sorted({m.lower() for m in PLUMBING_PATTERN.findall(text)})
    if matches:
        out.append(("plumbing", ",".join(matches)))
    if RESET_PATTERN.search(text):
        match = RESET_PATTERN.search(text)
        out.append(("reset", match.group(0)[:60] if match else ""))
    first = _first_sentence(text)
    if HEDGE_PATTERN.search(first):
        match = HEDGE_PATTERN.search(first)
        out.append(("hedge_opener", match.group(0)[:60] if match else ""))
    return out


def _detect_action_failures(row: dict[str, Any]) -> list[tuple[str, str]]:
    actions = row.get("guardrail_actions")
    if not isinstance(actions, list):
        return []
    normalized = {str(action) for action in actions}
    out: list[tuple[str, str]] = []
    if "replace_em_dashes" in normalized:
        out.append(("em_dash", "guardrail action replace_em_dashes"))
    if "strip_voice_caption_chunk_markers" in normalized:
        out.append(("voice_caption_chunk_markers", "guardrail action strip_voice_caption_chunk_markers"))
    return out


def _looks_like_dense_opening(text: str) -> bool:
    """Detect scan-hostile openings within the audit log's 160-char preview."""
    preview = text.strip()
    if len(preview) < DENSE_OPENING_MIN_CHARS:
        return False
    if "\n" in preview:
        return False
    sentence_breaks = len(re.findall(r"[.!?]", preview))
    return sentence_breaks <= 1


def _read_recent_jsonl_rows(path: Path, *, limit: int, only_user: str | None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    rows: list[dict[str, Any]] = []
    for raw in reversed(lines):
        raw = raw.strip()
        if not raw:
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if only_user and str(row.get("telegram_user_id") or "") != only_user:
            continue
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _gateway_trace_to_outbound_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "recorded_at": row.get("recorded_at"),
        "event": row.get("event"),
        "channel_id": row.get("channel_id"),
        "update_id": row.get("update_id"),
        "telegram_user_id": row.get("telegram_user_id"),
        "chat_id": row.get("chat_id"),
        "session_id": row.get("session_id"),
        "bridge_mode": row.get("bridge_mode"),
        "routing_decision": row.get("routing_decision"),
        "active_chip_key": row.get("active_chip_key"),
        "active_chip_task_type": row.get("active_chip_task_type"),
        "trace_ref": row.get("trace_ref"),
        "delivery_ok": row.get("delivery_ok"),
        "delivery_error": row.get("delivery_error"),
        "guardrail_actions": row.get("guardrail_actions"),
        "response_preview": row.get("response_preview"),
        "response_length": row.get("response_length"),
        "user_message_preview": row.get("user_message_preview"),
        "user_message_length": row.get("user_message_length"),
    }
