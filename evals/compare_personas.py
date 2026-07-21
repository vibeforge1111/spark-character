"""Head-to-head persona comparison without flipping the latest pointer.

Useful when you have hand-crafted a candidate and want to see whether
it actually beats the current baseline before promoting. The full
evolve_persona.py loop generates candidates with the LLM mutator;
this skips that and scores two existing markdown specs directly.

Usage:
  python evals/compare_personas.py
  python evals/compare_personas.py --baseline v7 --candidate v8 \
      --weights 0.20,0.30,0.20,0.10,0.10,0.10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean as mean_
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    PROBES,
    PersonaSpec,
    ProviderSpec,
    T6_EMOTIONAL_ATTUNEMENT_PROBES,
    T7_MEMORY_COHERENCE_PROBES,
    T8_INITIATIVE_PROBES,
    generate,
    run_deep_probe,
    run_probe,
    score_distinctiveness,
    score_persona,
)
from spark_character.persona import ARTIFACTS_DIR  # noqa: E402


PROMPTS = [
    "List three things I should focus on as a founder this week.",
    "Should I prioritize fundraising or shipping the product first?",
    "The web search just failed. What happened?",
    "What can you actually help me with right now?",
    "Yeah let's keep going on the launch plan from earlier.",
    "hey",
    "where are we",
    "Should I raise now or wait six months?",
    "What does TVL mean in DeFi?",
    "I'm anxious about the launch tomorrow.",
    "Quick gut check: ship the redesign or hold a week?",
]


def _available_versions() -> list[str]:
    versions: list[tuple[int, str]] = []
    for path in ARTIFACTS_DIR.glob("persona.v*.md"):
        value = path.name.removeprefix("persona.v").removesuffix(".md")
        if value.isdigit():
            versions.append((int(value), f"v{value}"))
    return [version for _, version in sorted(versions)]


def _default_versions() -> tuple[str, str]:
    versions = _available_versions()
    if len(versions) < 2:
        raise ValueError("compare_personas requires at least two persona.vN.md artifacts")
    pointer = ARTIFACTS_DIR / "persona.latest.txt"
    try:
        active = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        active = ""
    if active in versions and active != versions[-1]:
        return active, versions[-1]
    return versions[-2], versions[-1]


def _load(version: str) -> PersonaSpec:
    path = ARTIFACTS_DIR / f"persona.{version}.md"
    if not path.is_file():
        available = _available_versions()
        suffix = f"; available versions: {', '.join(available)}" if available else "; no persona versions available"
        raise ValueError(f"persona {version!r} not found{suffix}")
    text = path.read_text(encoding="utf-8")
    return PersonaSpec(version=version, text=text)


def score(
    version: str,
    provider: ProviderSpec,
    *,
    max_tokens: int,
    log: Callable[[str], None] = print,
) -> dict:
    persona = _load(version)
    log(f"[{version}] scoring T1+T2+T3+T6+T7+T8 ...")

    t1_scores: list[float] = []
    t2_scores: list[float] = []
    score_error_counts = {tier: 0 for tier in ("generate", "t2", "t3", "t6", "t7", "t8")}
    for prompt in PROMPTS:
        try:
            r = generate(prompt, provider=provider, persona=persona, max_tokens=max_tokens)
            t1_scores.append(score_persona(r.final).mean)
            try:
                t2_scores.append(score_distinctiveness(r.final, provider=provider).score)
            except Exception as exc:
                score_error_counts["t2"] += 1
                log(f"  T2 scoring failed ({type(exc).__name__})")
        except Exception as exc:
            score_error_counts["generate"] += 1
            log(f"  generation failed ({type(exc).__name__})")

    t3_scores: list[float] = []
    for probe in PROBES:
        try:
            t3_scores.append(run_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens).score)
        except Exception as exc:
            score_error_counts["t3"] += 1
            log(f"  T3 probe failed ({type(exc).__name__})")

    t6_scores: list[float] = []
    for probe in T6_EMOTIONAL_ATTUNEMENT_PROBES:
        try:
            t6_scores.append(run_deep_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens).score)
        except Exception as exc:
            score_error_counts["t6"] += 1
            log(f"  T6 probe failed ({type(exc).__name__})")

    t7_scores: list[float] = []
    for probe in T7_MEMORY_COHERENCE_PROBES:
        try:
            t7_scores.append(run_deep_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens).score)
        except Exception as exc:
            score_error_counts["t7"] += 1
            log(f"  T7 probe failed ({type(exc).__name__})")

    t8_scores: list[float] = []
    t8_per_probe: list[tuple[str, float]] = []
    for probe in T8_INITIATIVE_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens)
            t8_scores.append(r.score)
            t8_per_probe.append((probe.id, r.score))
        except Exception as exc:
            score_error_counts["t8"] += 1
            log(f"  T8 probe failed ({type(exc).__name__})")

    scores = {
        "t1": t1_scores,
        "t2": t2_scores,
        "t3": t3_scores,
        "t6": t6_scores,
        "t7": t7_scores,
        "t8": t8_scores,
    }
    return {
        "version": version,
        **{tier: round(mean_(values), 3) if values else None for tier, values in scores.items()},
        "t8_per_probe": t8_per_probe,
        "score_counts": {tier: len(values) for tier, values in scores.items()},
        "score_error_counts": score_error_counts,
    }


def composite(row: dict, weights: tuple[float, ...]) -> float | None:
    tiers = ("t1", "t2", "t3", "t6", "t7", "t8")
    if any(row.get(tier) is None for tier in tiers):
        return None
    w1, w2, w3, w6, w7, w8 = weights
    return round(
        w1 * row["t1"] + w2 * row["t2"] + w3 * row["t3"]
        + w6 * row["t6"] + w7 * row["t7"] + w8 * row["t8"],
        4,
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", help="Persona version to score as baseline (default: previous available)")
    parser.add_argument("--candidate", help="Persona version to score as candidate (default: latest available)")
    parser.add_argument("--max-tokens", type=_positive_int, default=600,
                        help="Per-call max output tokens for the upstream LLM (must be a positive integer)")
    parser.add_argument(
        "--weights",
        default="0.20,0.30,0.20,0.10,0.10,0.10",
        help="comma-separated weights for T1,T2,T3,T6,T7,T8",
    )
    parser.add_argument("--json", action="store_true", help="Emit one machine-readable comparison document")
    args = parser.parse_args()

    try:
        weights = tuple(float(w) for w in args.weights.split(","))
    except ValueError:
        parser.error("--weights must contain numeric values")
    if len(weights) != 6:
        parser.error("--weights must have 6 values (T1,T2,T3,T6,T7,T8)")

    try:
        if args.baseline is not None and args.candidate is not None:
            baseline, candidate = args.baseline, args.candidate
        else:
            default_baseline, default_candidate = _default_versions()
            baseline = args.baseline or default_baseline
            candidate = args.candidate or default_candidate
    except ValueError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, separators=(",", ":")))
        else:
            print(f"compare_personas: {exc}", file=sys.stderr)
        return 2

    provider = ProviderSpec.from_env()
    stream = sys.stderr if args.json else sys.stdout

    def log(message: str) -> None:
        print(message, file=stream, flush=True)

    log(f"=== persona compare | baseline={baseline} candidate={candidate} model={provider.model} ===")

    try:
        base = score(baseline, provider, max_tokens=args.max_tokens, log=log)
        cand = score(candidate, provider, max_tokens=args.max_tokens, log=log)
    except ValueError as exc:
        if args.json:
            print(json.dumps({"status": "invalid", "error": str(exc)}, separators=(",", ":")))
        else:
            print(f"compare_personas: {exc}", file=sys.stderr)
        return 2

    base_c = composite(base, weights)
    cand_c = composite(cand, weights)
    delta = round(cand_c - base_c, 4) if base_c is not None and cand_c is not None else None
    complete = delta is not None
    payload = {
        "status": "complete" if complete else "incomplete",
        "model": provider.model,
        "weights": weights,
        "baseline": base,
        "candidate": cand,
        "baseline_composite": base_c,
        "candidate_composite": cand_c,
        "delta": delta,
    }
    if complete:
        payload["winner"] = candidate if delta > 0 else baseline

    if args.json:
        print(json.dumps(payload, separators=(",", ":")))
        return 0 if complete else 2

    print("\n=== verdict ===")
    print(f"[{baseline}] T1={base['t1']} T2={base['t2']} T3={base['t3']} T6={base['t6']} T7={base['t7']} T8={base['t8']} composite={base_c}")
    print(f"[{candidate}] T1={cand['t1']} T2={cand['t2']} T3={cand['t3']} T6={cand['t6']} T7={cand['t7']} T8={cand['t8']} composite={cand_c}")
    if not complete:
        print("\ncomparison incomplete: one or more tiers have no score evidence")
        return 2
    print(f"\nT8 per-probe (target axis):")
    for pid, s in cand["t8_per_probe"]:
        baseline_s = next((bs for bp, bs in base["t8_per_probe"] if bp == pid), None)
        baseline_str = f"{baseline_s:.2f}" if baseline_s is not None else "n/a"
        print(f"  {pid:<32} baseline={baseline_str} candidate={s:.2f}")

    print(f"\ndelta (candidate - baseline): {delta:+}")
    if delta > 0:
        print(f"=> candidate {candidate} WINS")
        return 0
    print(f"=> baseline {baseline} holds")
    # Exit 0 on a completed comparison regardless of verdict (POSIX: the exit
    # code signals run success/failure, not a domain result). The verdict is on
    # stdout: "candidate ... WINS" vs "baseline ... holds". No CI/Makefile/
    # wrapper in this repo branches on the old exit-1 baseline-holds signal.
    return 0


if __name__ == "__main__":
    sys.exit(main())
