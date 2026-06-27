"""Audit miner: read SIB's gateway-outbound.jsonl and surface local failure
modes from Telegram conversations.

The Spark Intelligence Builder writes one JSON line per outbound bot
reply to logs/gateway-outbound.jsonl in each home directory. Each line
includes the route, chip, decision, and a 160-char response_preview.

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
from collections import deque
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

    def summary(self) -> str:
        lines = [
            f"Audit miner findings",
            f"  rows scanned: {self.rows_scanned}",
            f"  LLM-generated rows: {self.llm_rows}",
        ]
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

    @classmethod
    def from_sib_home(cls, home: str | Path) -> "AuditMiner":
        log = Path(home) / "logs" / "gateway-outbound.jsonl"
        return cls(log_path=log)

    def recent_findings(
        self,
        *,
        limit: int = 100,
        only_user: str | None = None,
    ) -> AuditFindings:
        if not self.log_path.exists():
            return AuditFindings()
        # Stream the tail of the file instead of reading the entire
        # unbounded gateway-outbound.jsonl into memory.  ``deque`` with
        # *maxlen* keeps only the most-recent lines as we iterate, so
        # peak memory is O(limit) rather than O(file_size).
        headroom = max(limit * 3, 200)
        with open(self.log_path, encoding="utf-8", errors="replace") as fh:
            tail: deque[str] = deque(fh, maxlen=headroom)
        # Most recent first
        rows: list[dict[str, Any]] = []
        for raw_line in reversed(tail):
            raw_line = raw_line.rstrip("\n\r")
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if only_user and str(row.get("telegram_user_id") or "") != only_user:
                continue
            rows.append(row)
            if len(rows) >= limit:
                break

        findings = AuditFindings(rows_scanned=len(rows))
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
        return findings


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


def _looks_like_dense_opening(text: str) -> bool:
    """Detect scan-hostile openings within the audit log's 160-char preview."""
    preview = text.strip()
    if len(preview) < DENSE_OPENING_MIN_CHARS:
        return False
    if "\n" in preview:
        return False
    sentence_breaks = len(re.findall(r"[.!?]", preview))
    return sentence_breaks <= 1
