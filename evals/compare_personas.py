"""Head-to-head persona comparison without flipping the latest pointer.

Useful when you have hand-crafted a candidate and want to see whether
it actually beats the current baseline before promoting. The full
evolve_persona.py loop generates candidates with the LLM mutator;
this skips that and scores two existing markdown specs directly.

Usage:
  python evals/compare_personas.py --baseline v5 --candidate v6
  python evals/compare_personas.py --baseline v5 --candidate v6 \
      --weights 0.20,0.30,0.20,0.10,0.10,0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean as mean_

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


def _load(version: str) -> PersonaSpec:
    path = ARTIFACTS_DIR / f"persona.{version}.md"
    text = path.read_text(encoding="utf-8")
    return PersonaSpec(version=version, text=text)


def score(version: str, provider: ProviderSpec, *, max_tokens: int) -> dict:
    persona = _load(version)
    print(f"\n[{version}] scoring T1+T2+T3+T6+T7+T8 ...", flush=True)

    t1_scores: list[float] = []
    t2_scores: list[float] = []
    for prompt in PROMPTS:
        try:
            r = generate(prompt, provider=provider, persona=persona, max_tokens=max_tokens)
            t1_scores.append(score_persona(r.final).mean)
            try:
                t2_scores.append(score_distinctiveness(r.final, provider=provider).score)
            except Exception:
                pass
        except Exception as exc:
            print(f"  generate error on {prompt[:40]!r}: {exc}")

    t3_scores: list[float] = []
    for probe in PROBES:
        try:
            t3_scores.append(run_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens).score)
        except Exception:
            pass

    t6_scores: list[float] = []
    for probe in T6_EMOTIONAL_ATTUNEMENT_PROBES:
        try:
            t6_scores.append(run_deep_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens).score)
        except Exception:
            pass

    t7_scores: list[float] = []
    for probe in T7_MEMORY_COHERENCE_PROBES:
        try:
            t7_scores.append(run_deep_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens).score)
        except Exception:
            pass

    t8_scores: list[float] = []
    t8_per_probe: list[tuple[str, float]] = []
    for probe in T8_INITIATIVE_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona, max_tokens=max_tokens)
            t8_scores.append(r.score)
            t8_per_probe.append((probe.id, r.score))
        except Exception:
            pass

    return {
        "version": version,
        "t1": round(mean_(t1_scores), 3) if t1_scores else 0.0,
        "t2": round(mean_(t2_scores), 3) if t2_scores else 0.0,
        "t3": round(mean_(t3_scores), 3) if t3_scores else 0.0,
        "t6": round(mean_(t6_scores), 3) if t6_scores else 0.0,
        "t7": round(mean_(t7_scores), 3) if t7_scores else 0.0,
        "t8": round(mean_(t8_scores), 3) if t8_scores else 0.0,
        "t8_per_probe": t8_per_probe,
    }


def composite(row: dict, weights: tuple[float, ...]) -> float:
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
    parser.add_argument("--baseline", default="v5")
    parser.add_argument("--candidate", default="v6")
    parser.add_argument("--max-tokens", type=_positive_int, default=600,
                        help="Per-call max output tokens for the upstream LLM (must be a positive integer)")
    parser.add_argument(
        "--weights",
        default="0.20,0.30,0.20,0.10,0.10,0.10",
        help="comma-separated weights for T1,T2,T3,T6,T7,T8",
    )
    args = parser.parse_args()

    weights = tuple(float(w) for w in args.weights.split(","))
    if len(weights) != 6:
        print("--weights must have 6 values (T1,T2,T3,T6,T7,T8)")
        return 2

    provider = ProviderSpec.from_env()
    print(f"=== persona compare | baseline={args.baseline} candidate={args.candidate} model={provider.model} ===")

    base = score(args.baseline, provider, max_tokens=args.max_tokens)
    cand = score(args.candidate, provider, max_tokens=args.max_tokens)

    base_c = composite(base, weights)
    cand_c = composite(cand, weights)
    delta = round(cand_c - base_c, 4)

    print("\n=== verdict ===")
    print(f"[{args.baseline}] T1={base['t1']} T2={base['t2']} T3={base['t3']} T6={base['t6']} T7={base['t7']} T8={base['t8']} composite={base_c}")
    print(f"[{args.candidate}] T1={cand['t1']} T2={cand['t2']} T3={cand['t3']} T6={cand['t6']} T7={cand['t7']} T8={cand['t8']} composite={cand_c}")
    print(f"\nT8 per-probe (target axis):")
    for pid, s in cand["t8_per_probe"]:
        baseline_s = next((bs for bp, bs in base["t8_per_probe"] if bp == pid), None)
        baseline_str = f"{baseline_s:.2f}" if baseline_s is not None else "n/a"
        print(f"  {pid:<32} baseline={baseline_str} candidate={s:.2f}")

    print(f"\ndelta (candidate - baseline): {delta:+}")
    if delta > 0:
        print(f"=> candidate {args.candidate} WINS")
        return 0
    print(f"=> baseline {args.baseline} holds")
    return 1


if __name__ == "__main__":
    sys.exit(main())
