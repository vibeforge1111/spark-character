"""Shared score parsing for LLM judge responses.

Multiple modules (voice_judge, probes, stability, deeper_probes) parse
SCORE=<integer 0-10> from judge LLM output. This module extracts that
logic into a single canonical implementation so parsing behavior stays
consistent across all tiers.

Duplication was found in:
  - voice_judge.py  (_parse_score, lines 66-75)
  - probes.py       (_parse_score, lines 225-234)
  - stability.py    (_parse_score, lines 333-342)
  - deeper_probes.py (_parse_score, lines 592-601)

All four copies were identical. If the parsing logic ever needs to
change (e.g., accepting SCORE=10.5 or handling SCORE: 7), it must be
updated in all four places. Centralizing eliminates that risk.
"""

from __future__ import annotations

import re


def parse_judge_score(text: str, default: int = 5) -> int:
    """Extract a 0-10 integer score from a judge LLM response.

    Priority:
      1. SCORE=<integer> format (exact match)
      2. Any standalone integer 0-10 in the text
      3. default (5) when no score can be extracted

    Args:
        text: Raw judge LLM output.
        default: Score returned when no parseable score is found.

    Returns:
        Integer in [0, 10].
    """
    if not text:
        return default
    match = re.search(r"SCORE\s*=\s*(\d+)", text, re.IGNORECASE)
    if match:
        return max(0, min(10, int(match.group(1))))
    digits = re.findall(r"\b([0-9]|10)\b", text)
    if digits:
        return max(0, min(10, int(digits[0])))
    return default
