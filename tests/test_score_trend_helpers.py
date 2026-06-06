"""Tests for evals/score_trend.py trend arrow + summarizer + table.

score_trend.py reads evals/_score_history.jsonl and prints a per-tier
table + trend arrows + N-runs-back deltas. The arrow direction (epsilon
band), the summarizer's compare_back-vs-fallback semantics, and the
table empty-state render are all pure helpers but ship with no direct
test coverage. A regression to the epsilon band silently mislabels
trends; a regression to compare_back fallback silently drops historical
context.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "evals"))

import score_trend as st  # noqa: E402


# ----- _arrow -----


def test_arrow_returns_up_for_above_epsilon_delta() -> None:
    assert st._arrow(0.9, 0.7) == "^"


def test_arrow_returns_down_for_below_negative_epsilon() -> None:
    assert st._arrow(0.7, 0.9) == "v"


def test_arrow_returns_flat_within_epsilon_band() -> None:
    # default eps = 0.02 — a 0.01 delta is flat.
    assert st._arrow(0.81, 0.80) == "="
    assert st._arrow(0.80, 0.81) == "="


def test_arrow_returns_space_when_either_value_is_none() -> None:
    assert st._arrow(None, 0.5) == " "
    assert st._arrow(0.5, None) == " "


def test_arrow_respects_custom_epsilon() -> None:
    # With eps=0.5 a 0.3 delta is inside the band.
    assert st._arrow(0.7, 0.4, eps=0.5) == "="


# ----- _summarize -----


def test_summarize_returns_latest_window_mean_delta_for_each_tier() -> None:
    rows = [
        {"t1_mean": 0.7, "t2_mean": 0.5},
        {"t1_mean": 0.8, "t2_mean": 0.6},
        {"t1_mean": 0.9, "t2_mean": 0.7},
    ]
    s = st._summarize(rows, compare_back=2)
    assert s["t1_mean"]["latest"] == 0.9
    assert s["t1_mean"]["window_mean"] == 0.8
    # compare_back=2 means latest minus the row 2 places back -> 0.9 - 0.8 = 0.1
    assert s["t1_mean"]["delta_vs_compare"] == 0.1
    assert s["t1_mean"]["n_samples"] == 3


def test_summarize_falls_back_to_first_value_when_compare_back_exceeds_history() -> None:
    rows = [
        {"t1_mean": 0.5},
        {"t1_mean": 0.9},
    ]
    # compare_back=10 with only 2 samples -> fall back to values[0]
    s = st._summarize(rows, compare_back=10)
    assert s["t1_mean"]["delta_vs_compare"] == 0.4


def test_summarize_skips_tier_with_no_numeric_samples() -> None:
    # All rows have non-numeric t1 values.
    rows = [
        {"t1_mean": "not-a-number", "t2_mean": 0.5},
        {"t1_mean": None, "t2_mean": 0.6},
    ]
    s = st._summarize(rows, compare_back=1)
    assert "t1_mean" not in s
    assert s["t2_mean"]["n_samples"] == 2


def test_summarize_filters_non_numeric_values_per_tier() -> None:
    rows = [
        {"t1_mean": 0.7},
        {"t1_mean": "broken"},
        {"t1_mean": 0.9},
    ]
    s = st._summarize(rows, compare_back=1)
    # The "broken" row contributes nothing — n_samples is 2.
    assert s["t1_mean"]["n_samples"] == 2


# ----- _format_table -----


def test_format_table_handles_empty_input_with_explicit_message() -> None:
    assert st._format_table([]) == "no rows yet"


def test_format_table_includes_tier_header_columns() -> None:
    rows = [{"ts": 0, "persona_version": "v1", "tier": "T1", "t1_mean": 0.9}]
    table = st._format_table(rows)
    # Header should include the eight tier labels (T1..T9 minus T5).
    for label in ("T1", "T2", "T3", "T4", "T6", "T7", "T8", "T9"):
        assert label in table


def test_format_table_renders_dash_for_missing_tier_values() -> None:
    rows = [{"ts": 0, "persona_version": "v1", "tier": "T1", "t1_mean": 0.9}]
    table = st._format_table(rows)
    # Most tiers are missing -> appear as the dash placeholder.
    assert "-" in table
