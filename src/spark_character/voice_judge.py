"""T2 distinctiveness scorer.

Loads the golden and foil voice corpora, samples K examples of each,
shows a judge LLM both signatures, asks it to score how Spark-like a
candidate reply is on a 0-10 scale. Returns 0.0-1.0 normalized.

Cheap variant: judge with the same provider used for generation.
Stronger variant: pass a separate judge_provider.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ._scoring_utils import parse_judge_score
from .persona import ARTIFACTS_DIR
from .provider import ProviderSpec, call_provider, call_provider_async


CORPUS_DIR = ARTIFACTS_DIR / "voice_corpus"
GOLDEN_DEFAULT = "golden.v2.json"
FOIL_DEFAULT = "foil.v2.json"

JUDGE_SYSTEM = (
    "You are a voice signature judge for an AI agent named Spark. "
    "You will see two voice signatures (A and B), each illustrated by "
    "5 example replies. Then you will see a candidate reply. "
    "Your job is to rate how much the candidate sounds like Voice A "
    "versus Voice B on a 0-10 scale, where 10 = clearly Voice A, "
    "5 = ambiguous, 0 = clearly Voice B. "
    "Score the voice and posture, not the factual content of the reply. "
    "Return exactly one line in the format SCORE=<integer 0-10>. "
    "Do not add any other text."
)


@dataclass(frozen=True)
class DistinctivenessScore:
    raw: int
    score: float
    judge_response: str

    @property
    def passed(self) -> bool:
        return self.score >= 0.6


class VoiceCorpusUnavailable(RuntimeError):
    """Raised when a distinctiveness comparison lacks either voice corpus."""


def _load_corpus(path: Path) -> list[dict]:
    try:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"voice_judge corpus parse failed for {path}: {exc}") from exc
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    entries = data.get("entries")
    if not isinstance(entries, list):
        return []
    return [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("text"), str)
        and entry["text"].strip()
    ]


def _format_examples(label: str, samples: list[dict]) -> str:
    lines = [f"[Voice {label}]"]
    texts = [
        sample["text"]
        for sample in samples
        if isinstance(sample, dict)
        and isinstance(sample.get("text"), str)
        and sample["text"].strip()
    ]
    for i, text in enumerate(texts, 1):
        lines.append(f"{label}{i}. {text}")
    return "\n".join(lines)


def _require_comparable_corpora(golden: list[dict], foil: list[dict]) -> None:
    missing = [label for label, corpus in (("golden", golden), ("foil", foil)) if not corpus]
    if missing:
        raise VoiceCorpusUnavailable(
            "Voice distinctiveness requires non-empty golden and foil corpora; "
            f"unavailable: {', '.join(missing)}."
        )


def _parse_score(text: str) -> int:
    return parse_judge_score(text)


def _build_judge_prompt(
    *,
    reply: str,
    golden_samples: list[dict],
    foil_samples: list[dict],
) -> str:
    return (
        f"{_format_examples('A', golden_samples)}\n\n"
        f"{_format_examples('B', foil_samples)}\n\n"
        "[Candidate reply]\n"
        f"{reply}\n\n"
        "Return SCORE=<integer 0-10> only."
    )


def score_distinctiveness(
    reply: str,
    *,
    provider: ProviderSpec,
    samples_per_side: int = 5,
    golden_path: Path | None = None,
    foil_path: Path | None = None,
    rng: random.Random | None = None,
) -> DistinctivenessScore:
    rng = rng or random.Random(7)
    golden = _load_corpus(golden_path or (CORPUS_DIR / GOLDEN_DEFAULT))
    foil = _load_corpus(foil_path or (CORPUS_DIR / FOIL_DEFAULT))
    _require_comparable_corpora(golden, foil)
    g = rng.sample(golden, min(samples_per_side, len(golden)))
    f = rng.sample(foil, min(samples_per_side, len(foil)))
    user_prompt = _build_judge_prompt(reply=reply, golden_samples=g, foil_samples=f)
    response = call_provider(
        provider=provider,
        system_prompt=JUDGE_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=120,
        temperature=0.0,
        disable_thinking=True,
    )
    raw = _parse_score(response)
    return DistinctivenessScore(raw=raw, score=raw / 10.0, judge_response=response)


async def score_distinctiveness_async(
    reply: str,
    *,
    provider: ProviderSpec,
    samples_per_side: int = 5,
    golden_path: Path | None = None,
    foil_path: Path | None = None,
    rng: random.Random | None = None,
) -> DistinctivenessScore:
    rng = rng or random.Random(7)
    golden = _load_corpus(golden_path or (CORPUS_DIR / GOLDEN_DEFAULT))
    foil = _load_corpus(foil_path or (CORPUS_DIR / FOIL_DEFAULT))
    _require_comparable_corpora(golden, foil)
    g = rng.sample(golden, min(samples_per_side, len(golden)))
    f = rng.sample(foil, min(samples_per_side, len(foil)))
    user_prompt = _build_judge_prompt(reply=reply, golden_samples=g, foil_samples=f)
    response = await call_provider_async(
        provider=provider,
        system_prompt=JUDGE_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=120,
        temperature=0.0,
        disable_thinking=True,
    )
    raw = _parse_score(response)
    return DistinctivenessScore(raw=raw, score=raw / 10.0, judge_response=response)
