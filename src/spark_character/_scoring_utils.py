"""Shared, evidence-preserving parsing for LLM judge scores."""

from __future__ import annotations

import re


_SCORE_LINE_PATTERN = re.compile(r"SCORE\s*=\s*(\d+)", re.IGNORECASE)
_STANDALONE_SCORE_PATTERN = re.compile(r"\b([0-9]|10)\b")


class JudgeScoreUnavailable(ValueError):
    """Raised when a judge response contains no supported score evidence."""


def parse_judge_score(text: str) -> int:
    """Return a clamped 0-10 judge score without inventing a midpoint."""
    if text:
        match = _SCORE_LINE_PATTERN.search(text)
        if match:
            return max(0, min(10, int(match.group(1))))
        fallback = _STANDALONE_SCORE_PATTERN.search(text)
        if fallback:
            return int(fallback.group(1))
    raise JudgeScoreUnavailable("judge response did not contain a supported 0-10 score")
