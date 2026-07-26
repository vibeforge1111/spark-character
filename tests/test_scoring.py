"""Pure-function scoring tests. No network, no provider."""

from __future__ import annotations

from spark_character.scoring import _first_sentence, score_persona


def test_em_dash_fails_p1() -> None:
    score = score_persona("Two chips routing live: \u2014 X Content \u2014 tweet eval.")
    assert score.p1_em_dash == 0.0


def test_dash_family_fails_p1() -> None:
    score = score_persona("Revenue grew from 10\u2013100 per day.")
    assert score.p1_em_dash == 0.0


def test_hyphen_passes_p1() -> None:
    score = score_persona("Two chips routing live: - X Content - tweet eval.")
    assert score.p1_em_dash == 1.0


def test_plumbing_terms_fail_p2() -> None:
    score = score_persona("Spark Researcher returned no concrete guidance.")
    assert score.p2_plumbing < 1.0
    assert "spark researcher" in score.p2_hits


def test_clean_text_passes_p2() -> None:
    score = score_persona("My read: ship the redesign now. Three reasons.")
    assert score.p2_plumbing == 1.0
    assert score.p2_hits == ()


def test_canned_greeting_fails_p3() -> None:
    score = score_persona("Hey! How can I help you today?")
    assert score.p3_reset == 0.0


def test_followup_passes_p3() -> None:
    score = score_persona("Got it. My read on the launch: ship.")
    assert score.p3_reset == 1.0


def test_hedge_opener_fails_p4() -> None:
    score = score_persona("Great question! That depends on the runway.")
    assert score.p4_lead == 0.0


def test_lead_with_call_passes_p4() -> None:
    score = score_persona("Ship it. Three reasons follow.")
    assert score.p4_lead == 1.0


def test_robotic_text_drops_p5() -> None:
    text = "As an AI language model, I don't have personal feelings, but I hope this helps!"
    score = score_persona(text)
    assert score.p5_voice < 0.7


def test_warm_direct_text_high_p5() -> None:
    score = score_persona("Got it. My call: ship now.")
    assert score.p5_voice >= 0.9


def test_passed_aggregates_all_axes() -> None:
    score = score_persona("Got it. My call: ship the redesign now. Three reasons follow.")
    assert score.passed
    assert score.mean >= 0.95


def test_failed_when_any_axis_below_threshold() -> None:
    score = score_persona("Great question \u2014 how can I help today?")
    assert not score.passed


def test_first_sentence_does_not_split_after_honorific() -> None:
    assert _first_sentence("Mr. Smith built it. Three reasons follow.") == "Mr. Smith built it."


def test_cjk_reply_receives_the_same_verbosity_penalty() -> None:
    score = score_persona("\u4f60" * 201)
    assert score.p5_voice < 1.0
    assert score.p5_reason == "verbose=201w"


def test_hangul_reply_receives_the_same_verbosity_penalty() -> None:
    score = score_persona("\uac00" * 201)
    assert score.p5_voice < 1.0
    assert score.p5_reason == "verbose=201w"
