"""Shared judge-score parsing tests. No network, no provider."""

from __future__ import annotations

import pytest

from spark_character import JudgeScoreUnavailable as PublicJudgeScoreUnavailable
from spark_character._scoring_utils import JudgeScoreUnavailable, parse_judge_score
from spark_character.deeper_probes import _parse_score as parse_deep_score
from spark_character.probes import _parse_score as parse_probe_score
from spark_character.stability import _parse_score as parse_stability_score
from spark_character.voice_judge import _parse_score as parse_voice_score


def test_missing_score_exception_is_public() -> None:
    assert PublicJudgeScoreUnavailable is JudgeScoreUnavailable


@pytest.mark.parametrize(
    ("response", "expected"),
    [("SCORE=0", 0), ("score = 10", 10), ("The result is 7.", 7), ("SCORE=99", 10)],
)
def test_parse_judge_score_accepts_supported_forms(response: str, expected: int) -> None:
    assert parse_judge_score(response) == expected


@pytest.mark.parametrize("response", ["", "no score available", "SCORE=unknown"])
def test_parse_judge_score_rejects_missing_evidence(response: str) -> None:
    with pytest.raises(JudgeScoreUnavailable, match="judge response"):
        parse_judge_score(response)


@pytest.mark.parametrize(
    "parser",
    [parse_probe_score, parse_deep_score, parse_stability_score, parse_voice_score],
)
def test_all_judges_share_the_missing_score_contract(parser) -> None:
    with pytest.raises(JudgeScoreUnavailable):
        parser("judge failed without a score")
