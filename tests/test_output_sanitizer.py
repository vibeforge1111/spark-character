"""Tests for the voice-rule post-output sanitizer."""

from spark_character import (
    EM_DASH_FAMILY,
    replace_em_dashes,
    sanitize_voice_output,
    strip_markdown_emphasis,
)
from spark_character.output_sanitizer import (
    rewrite_spawner_surface_standalone_question,
    strip_format_controls,
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


def test_unicode_dash_punctuation_family_handled():
    for ch in ("\u2010", "\u2011", "\u2e3a", "\u2e3b"):
        text = f"alpha {ch} beta"
        out = replace_em_dashes(text)
        assert ch not in out
        assert out == "alpha - beta"


def test_strip_format_controls_removes_bidi_and_zero_width_marks():
    text = "alpha\u200bbeta\u202egamma"
    assert strip_format_controls(text) == "alphabetagamma"


def test_sanitize_removes_format_controls():
    text = "alpha\u200b \u2014 beta"
    cleaned = sanitize_voice_output(text)
    assert "\u200b" not in cleaned
    assert "\u2014" not in cleaned
    assert cleaned == "alpha - beta"


def test_collapses_double_spaces_introduced_by_replacement():
    text = "alpha \u2014 beta"
    out = replace_em_dashes(text)
    assert "  " not in out


def test_unaffected_text_unchanged():
    text = "regular hyphen - stays. no em dashes here."
    assert replace_em_dashes(text) == text


def test_ascii_hyphenated_identifiers_are_unchanged():
    text = "`startup-operator` session-123 domain-chip-voice-comms anti-filler"
    assert replace_em_dashes(text) == text
    assert sanitize_voice_output(text) == text


def test_empty_input_passthrough():
    assert replace_em_dashes("") == ""
    assert sanitize_voice_output("") == ""


def test_strip_markdown_emphasis_preserves_bullets():
    text = "* **Lean dashboard first** - ship it fast"
    assert strip_markdown_emphasis(text) == "* Lean dashboard first - ship it fast"


def test_strip_markdown_emphasis_handles_single_character_content():
    cases = {
        "**x**": "x",
        "***x***": "x",
        "__x__": "x",
        "Before **x** after": "Before x after",
        "Before ***x*** after": "Before x after",
        "Before __x__ after": "Before x after",
    }
    for text, expected in cases.items():
        assert strip_markdown_emphasis(text) == expected


def test_strip_markdown_emphasis_preserves_empty_and_unpaired_markers():
    for text in ("****", "******", "__", "**open", "open**", "* bullet"):
        assert strip_markdown_emphasis(text) == text


def test_strip_markdown_emphasis_does_not_silently_skip_long_valid_content():
    content = "x" * 10_001
    assert strip_markdown_emphasis(f"**{content}**") == content


def test_sanitize_removes_bold_markers():
    text = "Short answer: **yes**.\n\n**Two directions:**"
    assert sanitize_voice_output(text) == "Short answer: yes.\n\nTwo directions:"


def test_rewrites_stale_standalone_spawner_surface_question():
    text = (
        "Spawner Kanban and Canvas notes:\n\n"
        "My pick: start with Live State Sync.\n\n"
        "- Are you thinking this runs locally as a standalone page, or lives inside the existing Spawner UI routes?"
    )
    cleaned = rewrite_spawner_surface_standalone_question(text)

    assert "standalone page" not in cleaned.lower()
    assert "Since this lives inside the existing Spawner UI routes" in cleaned
    assert "Kanban state accuracy, Canvas execution state, or Telegram relay messaging" in cleaned


def test_sanitize_removes_bold_and_stale_standalone_question():
    text = (
        "1. **Live State Sync**\n\n"
        "Spawner Kanban and Canvas.\n\n"
        "- Are you thinking this runs locally as a standalone page, or lives inside the existing Spawner UI routes?"
    )
    cleaned = sanitize_voice_output(text)

    assert "**" not in cleaned
    assert "standalone page" not in cleaned.lower()
    assert "1. Live State Sync" in cleaned
    assert "Since this lives inside the existing Spawner UI routes" in cleaned


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


def test_replace_em_dashes_preserves_unrelated_whitespace():
    # Regression for the global double-space collapse that mangled list
    # indentation and aligned text on every sanitized reply. Spacing that is
    # not adjacent to a replaced dash must be left intact.
    text = "Plan:\n  1. ship it\n  2. measure — then iterate"
    out = replace_em_dashes(text)
    assert out == "Plan:\n  1. ship it\n  2. measure - then iterate"
    # A pre-existing double space with no dash nearby must survive.
    assert replace_em_dashes("a  b") == "a  b"


def test_replace_em_dashes_still_single_spaces_around_dash():
    assert replace_em_dashes("word — word") == "word - word"
    assert replace_em_dashes("word—word") == "word - word"
