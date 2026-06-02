from spark_character.critic import _interpret


# ----- PASS detection -----

def test_pass_alone_is_pass():
    result = _interpret("original draft", "PASS")
    assert result.rewritten is False
    assert result.final == "original draft"


def test_pass_lowercase_is_pass():
    # PASS_TOKEN check is case-insensitive via .upper()
    result = _interpret("original draft", "pass")
    assert result.rewritten is False
    assert result.final == "original draft"


def test_pass_with_trailing_newline_is_pass():
    result = _interpret("original draft", "PASS\n")
    assert result.rewritten is False
    assert result.final == "original draft"


def test_pass_with_trailing_commentary_is_pass():
    # Critic meta-text on subsequent lines is ignored; only the first line
    # is checked. The original draft is returned unchanged.
    result = _interpret("original draft", "PASS\nThis draft reads cleanly.")
    assert result.rewritten is False
    assert result.final == "original draft"


def test_final_equals_draft_on_pass():
    draft = "The original content stays intact."
    result = _interpret(draft, "PASS\nextra content here")
    assert result.final == draft


def test_empty_response_returns_original_draft():
    result = _interpret("draft text", "")
    assert result.rewritten is False
    assert result.final == "draft text"


# ----- Rewrite detection — boundary and edge cases -----

def test_actual_rewrite_detected():
    result = _interpret("old draft", "This is a completely new rewritten version.")
    assert result.rewritten is True


def test_pass_colon_prefix_is_a_rewrite():
    # "PASS: ..." is not equal to the PASS token — must be treated as a rewrite.
    result = _interpret("draft", "PASS: the draft is acceptable, but consider this change.")
    assert result.rewritten is True


def test_multiline_rewrite_not_confused_with_pass():
    # A rewrite whose first line is not PASS must not be treated as a pass.
    result = _interpret("old draft", "Here is the improved version.\nWith more detail.")
    assert result.rewritten is True


def test_pass_token_in_non_first_position_is_a_rewrite():
    # PASS on a line other than the first is a rewrite, not a pass.
    result = _interpret("draft text", "This version is better.\nPASS")
    assert result.rewritten is True
