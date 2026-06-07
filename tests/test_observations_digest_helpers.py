"""Tests for evals/observations_digest.py pure helpers.

observations_digest reads evals/_observations.jsonl, aggregates score
means + top patterns + evolution-target counts, and prints a digest with
a "next move" suggestion. The aggregation and suggest-next-move helpers
are pure functions over plain dicts but ship with no direct test
coverage. A regression to either silently changes the operator-facing
guidance without any CI signal.
"""

from __future__ import annotations

import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "evals"))

import observations_digest as od  # noqa: E402


# ----- _digest aggregation -----


def test_digest_empty_rows_returns_zero_shape() -> None:
    result = od._digest([])
    assert result["n_observations"] == 0
    assert result["score_means"] == {}
    assert result["top_patterns"] == []
    assert result["recommendation_tiers"] == {}
    assert result["evolution_targets"] == {}
    assert result["recent_rewrites"] == []


def test_digest_filters_non_numeric_score_values() -> None:
    rows = [
        {"scores": {"t1": 0.9, "t2": "not-a-number"}},
        {"scores": {"t1": 0.7, "t2": None}},
    ]
    result = od._digest(rows)
    # Only the t1 axis collected numeric values; t2 had none, so it is
    # not present in the means dict at all.
    assert "t1" in result["score_means"]
    assert "t2" not in result["score_means"]
    assert result["score_means"]["t1"] == 0.8


def test_digest_aggregates_pattern_counts_and_returns_top_5() -> None:
    rows = [{"pattern": "em_dash"}] * 6 + [{"pattern": "reset"}] * 2
    result = od._digest(rows)
    # Counter.most_common(5) so top entries appear with their counts.
    assert ("em_dash", 6) in result["top_patterns"]
    assert ("reset", 2) in result["top_patterns"]


def test_digest_aggregates_evolution_targets_and_recommendation_tiers() -> None:
    rows = [
        {"recommendation_tier": "fire_now", "evolution_target": "t2"},
        {"recommendation_tier": "fire_now", "evolution_target": "t2"},
        {"recommendation_tier": "consider_evolution", "evolution_target": "t1"},
    ]
    result = od._digest(rows)
    assert result["recommendation_tiers"] == {"fire_now": 2, "consider_evolution": 1}
    assert result["evolution_targets"] == {"t2": 2, "t1": 1}


def test_digest_skips_too_short_rewrite_suggestions() -> None:
    rows = [
        {"rewrite_suggestion": "tiny"},  # 4 chars, below the 12-char gate
        {"rewrite_suggestion": "this is a real concrete rewrite", "ts": 0},
    ]
    result = od._digest(rows)
    assert len(result["recent_rewrites"]) == 1
    assert "concrete rewrite" in result["recent_rewrites"][0]["rewrite"]


def test_digest_strips_whitespace_only_pattern_and_target_values() -> None:
    rows = [
        {"pattern": "   ", "evolution_target": "  ", "recommendation_tier": "  "},
        {"pattern": "real", "evolution_target": "t1", "recommendation_tier": "fire_now"},
    ]
    result = od._digest(rows)
    assert ("real", 1) in result["top_patterns"]
    assert "  " not in result["top_patterns"]
    assert result["evolution_targets"] == {"t1": 1}
    assert result["recommendation_tiers"] == {"fire_now": 1}


# ----- _suggest_next_move -----


def test_suggest_next_move_targets_strongest_evolution_signal_when_3plus() -> None:
    digest = {
        "evolution_targets": {"t2": 4, "t1": 1},
        "recommendation_tiers": {},
        "score_means": {},
    }
    suggestion = od._suggest_next_move(digest)
    assert "T2 distinctiveness" in suggestion
    assert "(t2)" in suggestion
    assert "4 times" in suggestion


def test_suggest_next_move_flags_weak_axis_when_no_target_signal() -> None:
    digest = {
        "evolution_targets": {},
        "recommendation_tiers": {},
        "score_means": {"t1": 0.9, "t2": 0.5},
    }
    suggestion = od._suggest_next_move(digest)
    assert "weakest axis t2" in suggestion
    assert "below 0.6" in suggestion


def test_suggest_next_move_recommends_evolution_when_two_plus_fire_now() -> None:
    digest = {
        "evolution_targets": {},
        "recommendation_tiers": {"fire_now": 3},
        "score_means": {"t1": 0.9, "t2": 0.9},  # not below 0.6 so we skip to tier
    }
    suggestion = od._suggest_next_move(digest)
    assert "fire_now" in suggestion


def test_suggest_next_move_default_when_no_strong_signal() -> None:
    digest = {
        "evolution_targets": {"t2": 1},  # below 3 threshold
        "recommendation_tiers": {"fire_now": 1},  # below 2 threshold
        "score_means": {"t1": 0.8, "t2": 0.8},  # all above 0.6
    }
    suggestion = od._suggest_next_move(digest)
    assert "nothing urgent" in suggestion
