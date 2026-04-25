"""Tests for the voice-rule post-output sanitizer."""

from spark_character import (
    EM_DASH_FAMILY,
    replace_em_dashes,
    sanitize_voice_output,
)
from spark_character.scoring import score_persona


def test_em_dash_family_replaced_with_hyphen():
    text = "Two chips routing live \u2014 X Content and Startup."
    out = replace_em_dashes(text)
    assert "\u2014" not in out
    assert " - " in out


def test_all_em_dash_family_chars_handled():
    for ch in EM_DASH_FAMILY:
        text = f"alpha {ch} beta"
        out = replace_em_dashes(text)
        assert ch not in out
        assert "alpha - beta" == out


def test_collapses_double_spaces_introduced_by_replacement():
    text = "alpha \u2014 beta"
    out = replace_em_dashes(text)
    assert "  " not in out


def test_unaffected_text_unchanged():
    text = "regular hyphen - stays. no em dashes here."
    assert replace_em_dashes(text) == text


def test_empty_input_passthrough():
    assert replace_em_dashes("") == ""
    assert sanitize_voice_output("") == ""


def test_sanitized_output_passes_t1_em_dash_check():
    text = "Yeah \u2014 fair point. Here's the straight version."
    raw_score = score_persona(text)
    assert raw_score.p1_em_dash == 0.0
    cleaned = sanitize_voice_output(text)
    cleaned_score = score_persona(cleaned)
    assert cleaned_score.p1_em_dash == 1.0


def test_sanitize_preserves_non_em_dash_voice_violations():
    text = "Great question \u2014 let me think. The router decides."
    cleaned = sanitize_voice_output(text)
    cleaned_score = score_persona(cleaned)
    assert cleaned_score.p1_em_dash == 1.0
    assert cleaned_score.p4_lead == 0.0
    assert cleaned_score.p2_plumbing < 1.0
