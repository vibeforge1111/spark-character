"""Tests for evals/evolve_persona.py pure helpers.

evolve_persona is the multi-tier (T1+T2+T3, optionally +T6+T7+T8) persona
evolution CLI. The fitness composite, the diagnose-from-failures shaper,
the mutator-output sanitizer (which strips chain-of-thought + extracts
markdown specs from code fences), and the --audit-limit positive-int
validator are all pure helpers, but ship without test coverage. A
silent regression in any of them would warp evolution decisions or
write a half-thought reasoning trace into persona.vN.md without any
test catching it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest


# evals/ is a script directory, not a package — add it explicitly.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "evals"))

import evolve_persona as ep  # noqa: E402


# ----- composite() weighted fitness -----


def test_composite_three_weight_combines_t1_t2_t3_only() -> None:
    scores = {"t1_mean": 0.8, "t2_mean": 0.7, "t3_mean": 0.9}
    # 0.2 * 0.8 + 0.5 * 0.7 + 0.3 * 0.9 = 0.16 + 0.35 + 0.27 = 0.78
    assert ep.composite(scores, (0.2, 0.5, 0.3)) == 0.78


def test_composite_six_weight_extends_to_deeper_tiers() -> None:
    scores = {
        "t1_mean": 0.8, "t2_mean": 0.7, "t3_mean": 0.9,
        "t6_mean": 0.6, "t7_mean": 0.5, "t8_mean": 0.4,
    }
    result = ep.composite(scores, (0.1, 0.2, 0.1, 0.2, 0.2, 0.2))
    # 0.08 + 0.14 + 0.09 + 0.12 + 0.10 + 0.08 = 0.61
    assert result == 0.61


def test_composite_missing_axes_default_to_zero() -> None:
    # Three-weight call against a scores dict missing t3_mean — missing key
    # contributes 0, not raises.
    assert ep.composite({"t1_mean": 1.0, "t2_mean": 1.0}, (0.5, 0.5, 0.2)) == 1.0


# ----- _sanitize_mutator_output() -----


def test_sanitize_mutator_output_extracts_fenced_markdown_spec() -> None:
    raw = (
        "Some reasoning preamble that the mutator emitted.\n\n"
        "```markdown\n"
        "# Spark persona v9\n"
        "Body line.\n"
        "```\n"
    )
    cleaned = ep._sanitize_mutator_output(raw)
    assert cleaned.startswith("# Spark persona v9")
    assert "reasoning preamble" not in cleaned


def test_sanitize_mutator_output_keeps_heading_when_no_code_fence() -> None:
    raw = "Some analysis...\n\n# Spark persona v9\nBody."
    cleaned = ep._sanitize_mutator_output(raw)
    assert cleaned.startswith("# Spark persona v9")
    assert "Some analysis" not in cleaned


def test_sanitize_mutator_output_handles_empty_input() -> None:
    assert ep._sanitize_mutator_output("") == ""
    assert ep._sanitize_mutator_output("   \n  ") == ""


def test_sanitize_mutator_output_last_resort_returns_raw_when_no_heading() -> None:
    raw = "Just some text with no heading."
    assert ep._sanitize_mutator_output(raw) == raw


# ----- _positive_int() argparse type -----


def test_positive_int_accepts_positive_decimal_string() -> None:
    assert ep._positive_int("7") == 7


def test_positive_int_rejects_zero_and_negative() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        ep._positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        ep._positive_int("-3")


def test_positive_int_rejects_non_integer_string() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        ep._positive_int("abc")


# ----- diagnose() -----


def test_diagnose_empty_scores_returns_default_guidance_line() -> None:
    lines = ep.diagnose({})
    assert len(lines) == 1
    assert "sharper warmth" in lines[0]


def test_diagnose_surfaces_t1_t2_t3_failures_with_capped_count() -> None:
    scores = {
        "failures_t1": [f"on prompt {i}: em_dash" for i in range(5)],
        "failures_t2": [(f"p{i}", 0.4, "preview") for i in range(5)],
        "failures_t3": [(f"probe-{i}", 0.5) for i in range(5)],
    }
    lines = ep.diagnose(scores)
    # The function caps T1 at 3, T2 at 4, T3 at 3.
    t1_count = sum(1 for l in lines if l.startswith("T1 mechanics:"))
    t2_count = sum(1 for l in lines if l.startswith("T2 distinctiveness"))
    t3_count = sum(1 for l in lines if l.startswith("T3 trait probe"))
    assert t1_count == 3
    assert t2_count == 4
    assert t3_count == 3


# ----- _format_score_line() -----


def test_format_score_line_includes_deeper_when_marker_set() -> None:
    scores = {
        "t1_mean": 0.8, "t2_mean": 0.7, "t3_mean": 0.9,
        "t6_mean": 0.6, "t7_mean": 0.5, "t8_mean": 0.4,
        "deeper_included": True,
    }
    line = ep._format_score_line("test", scores, 0.78)
    assert "T1=0.8" in line
    assert "T6=0.6" in line
    assert "T7=0.5" in line
    assert "T8=0.4" in line
    assert "composite=0.78" in line


def test_format_score_line_omits_deeper_when_marker_false() -> None:
    scores = {
        "t1_mean": 0.8, "t2_mean": 0.7, "t3_mean": 0.9,
        "deeper_included": False,
    }
    line = ep._format_score_line("baseline v5", scores, 0.78)
    assert "T1=" in line and "T2=" in line and "T3=" in line
    assert "T6=" not in line
    assert "T8=" not in line
