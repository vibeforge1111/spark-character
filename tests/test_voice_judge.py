from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import spark_character.voice_judge as voice_judge
from spark_character.provider import ProviderSpec
from spark_character.voice_judge import (
    VoiceCorpusUnavailable,
    _format_examples,
    _load_corpus,
    score_distinctiveness,
    score_distinctiveness_async,
)


def test_load_corpus_returns_entries(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"entries": [{"text": "sharp, warm reply"}]}) + "\n", encoding="utf-8")

    assert _load_corpus(path) == [{"text": "sharp, warm reply"}]


def test_load_corpus_returns_empty_for_malformed_or_missing_file(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")

    assert _load_corpus(malformed) == []
    assert _load_corpus(tmp_path / "missing.json") == []


@pytest.mark.parametrize("payload", [[], "text", 1, None, {"entries": {}}, {"entries": "bad"}])
def test_load_corpus_rejects_invalid_root_and_entries_shapes(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert _load_corpus(path) == []


def test_load_corpus_filters_invalid_or_blank_examples(tmp_path: Path) -> None:
    path = tmp_path / "corpus.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"text": " valid reply "},
                    {},
                    {"text": "   "},
                    {"text": 42},
                    "not an object",
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _load_corpus(path) == [{"text": " valid reply "}]


def test_format_examples_skips_missing_text_without_blank_numbered_rows() -> None:
    rendered = _format_examples("A", [{}, {"text": ""}, {"text": "actual"}])

    assert rendered == "[Voice A]\nA1. actual"


def test_empty_corpus_is_unavailable_and_does_not_call_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        voice_judge,
        "call_provider",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provider called")),
    )

    with pytest.raises(VoiceCorpusUnavailable, match="golden and foil"):
        score_distinctiveness(
            "reply",
            provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
            golden_path=tmp_path / "missing-golden.json",
            foil_path=tmp_path / "missing-foil.json",
        )


def test_one_sided_corpus_is_unavailable(tmp_path: Path) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps({"entries": [{"text": "valid"}]}), encoding="utf-8")

    with pytest.raises(VoiceCorpusUnavailable, match="foil"):
        score_distinctiveness(
            "reply",
            provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
            golden_path=golden,
            foil_path=tmp_path / "missing-foil.json",
        )


def test_empty_corpus_async_is_unavailable_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def unexpected_provider(**_kwargs: object) -> str:
        raise AssertionError("provider called")

    monkeypatch.setattr(voice_judge, "call_provider_async", unexpected_provider)

    with pytest.raises(VoiceCorpusUnavailable):
        asyncio.run(
            score_distinctiveness_async(
                "reply",
                provider=ProviderSpec("https://api.z.ai/v1/", "model", "key"),
                golden_path=tmp_path / "missing-golden.json",
                foil_path=tmp_path / "missing-foil.json",
            )
        )
