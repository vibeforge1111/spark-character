"""Multi-tier persona evolution.

Like evals/evolve.py, but the fitness function combines T1 + T2 + T3
instead of T1 alone. Targets the axis where the most real headroom
lives (currently T2 distinctiveness at ~0.78).

Workflow per cycle:
  1. Score baseline persona on T1 (8 prompts) + T2 (judge per reply) +
     T3 (5 sycophancy/curiosity/care/identity/honesty/initiative
     probes from src/spark_character/probes.py).
  2. Compute composite = w1*T1 + w2*T2 + w3*T3 (default 0.2/0.5/0.3).
  3. Diagnose weaknesses across all three tiers.
  4. Generate N candidate persona mutations targeting those weaknesses.
  5. Score each candidate the same way.
  6. Promote the best candidate to persona.v(N+1).md only if its
     composite strictly beats the baseline and no individual tier
     regresses by more than tier_floor_drop.

Usage:
  python evals/evolve_persona.py
  python evals/evolve_persona.py --candidates 4 --weights 0.2,0.5,0.3
  python evals/evolve_persona.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean as mean_

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    PROBES,
    PersonaSpec,
    ProviderSpec,
    call_provider,
    generate,
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

MUTATOR_SYSTEM = (
    "You are a persona spec evolution engine for the Spark agent. "
    "Given a baseline persona spec written in markdown, produce a single "
    "improved variant that addresses the listed weaknesses across the "
    "T1 (mechanics), T2 (voice signature), and T3 (behavioral traits) "
    "axes. Keep the same structure (a one-line title, then a short "
    "intro, then a 'Voice rules' bulleted list). Stay under 1700 "
    "characters.\n\n"
    "Hard constraints in every variant: never use em dashes. Never "
    "tell the agent to use em dashes. Never name internal subsystems. "
    "The first sentence rule must remain. The no-greeting-reset rule "
    "must remain. The no-plumbing-leak rule must remain.\n\n"
    "Important: T2 distinctiveness is the headroom axis. The variant "
    "should tighten what makes Spark distinctive: lead with a real "
    "call when evidence is good enough, push back when warranted, "
    "admit limits cleanly, engage with the user's actual emotional "
    "state when present, ask one specific curious follow-up when "
    "warranted. Cut hedging. Cut filler. Add spine.\n\n"
    "Output the markdown spec only. No preamble, no explanation, no "
    "code fences."
)


def find_latest_persona() -> tuple[int, PersonaSpec]:
    versions = []
    for path in ARTIFACTS_DIR.glob("persona.v*.md"):
        match = re.match(r"persona\.v(\d+)\.md", path.name)
        if match:
            versions.append((int(match.group(1)), path))
    if not versions:
        raise FileNotFoundError("No persona.vN.md artifacts found.")
    versions.sort(key=lambda x: x[0])
    n, path = versions[-1]
    text = path.read_text(encoding="utf-8")
    return n, PersonaSpec(version=f"v{n}", text=text)


def score_all_tiers(provider: ProviderSpec, persona: PersonaSpec) -> dict:
    """Run T1+T2 over PROMPTS and T3 over PROBES. Returns aggregated dict."""
    t1_means: list[float] = []
    t2_scores: list[float] = []
    t3_scores: list[float] = []
    failures_t1: list[str] = []
    failures_t2: list[tuple[str, float, str]] = []
    failures_t3: list[tuple[str, float]] = []

    for prompt in PROMPTS:
        try:
            result = generate(prompt, provider=provider, persona=persona)
            t1 = score_persona(result.final)
            t1_means.append(t1.mean)
            if t1.mean < 1.0:
                tags = []
                if t1.p1_em_dash < 1.0: tags.append("em_dash")
                if t1.p2_hits: tags.append("plumbing=" + ",".join(t1.p2_hits))
                if t1.p3_reset < 1.0: tags.append("reset")
                if t1.p4_lead < 1.0: tags.append("hedge")
                if t1.p5_voice < 1.0: tags.append(f"voice={t1.p5_voice}")
                failures_t1.append(f"on prompt {prompt!r}: {' '.join(tags)}")
            try:
                t2 = score_distinctiveness(result.final, provider=provider)
                t2_scores.append(t2.score)
                if t2.score < 0.7:
                    failures_t2.append((prompt, t2.score, result.final[:200]))
            except Exception:
                pass
        except Exception:
            pass

    for probe in PROBES:
        try:
            r = run_probe(probe, provider=provider, persona=persona)
            t3_scores.append(r.score)
            if r.score < 0.8:
                failures_t3.append((probe.id, r.score))
        except Exception:
            pass

    return {
        "t1_mean": round(mean_(t1_means), 3) if t1_means else 0.0,
        "t2_mean": round(mean_(t2_scores), 3) if t2_scores else 0.0,
        "t3_mean": round(mean_(t3_scores), 3) if t3_scores else 0.0,
        "failures_t1": failures_t1,
        "failures_t2": failures_t2,
        "failures_t3": failures_t3,
    }


def composite(scores: dict, weights: tuple[float, float, float]) -> float:
    w1, w2, w3 = weights
    return round(w1 * scores["t1_mean"] + w2 * scores["t2_mean"] + w3 * scores["t3_mean"], 3)


def diagnose(scores: dict) -> list[str]:
    out: list[str] = []
    for f in scores["failures_t1"][:3]:
        out.append(f"T1 mechanics: {f}")
    for prompt, t2, preview in scores["failures_t2"][:4]:
        out.append(
            f"T2 distinctiveness only {t2} on prompt {prompt!r}. "
            f"Reply opened: {preview[:140]!r}. Make it sharper, more decisive, less hedged."
        )
    for probe_id, t3 in scores["failures_t3"][:3]:
        out.append(f"T3 trait probe {probe_id} only {t3}. Strengthen the behavioral cue in the spec.")
    if not out:
        out.append("No specific failures. Push for sharper warmth, more memorable phrasing, less filler.")
    return out


def mutate_persona(provider: ProviderSpec, baseline: PersonaSpec, weaknesses: list[str]) -> str:
    user_prompt = (
        "[Baseline persona spec]\n"
        f"{baseline.system_prompt}\n\n"
        "[Observed weaknesses on real model output]\n"
        + "\n".join(f"- {w}" for w in weaknesses)
        + "\n\n[Task]\nProduce one improved variant of the persona spec markdown that addresses these weaknesses while keeping all hard constraints. Output the markdown only."
    )
    return call_provider(
        provider=provider,
        system_prompt=MUTATOR_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=1100,
        temperature=0.9,
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--weights", default="0.2,0.5,0.3", help="T1,T2,T3 weights for composite fitness")
    parser.add_argument("--floor-drop", type=float, default=0.05, help="Max allowed regression on any single axis")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="evals/_evolve_persona_last.json")
    args = parser.parse_args()

    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        raise SystemExit("weights must be 3 floats: T1,T2,T3")

    provider = ProviderSpec.from_env()
    n, baseline = find_latest_persona()
    print(f"\n=== persona evolution (multi-tier) | baseline=v{n} | candidates={args.candidates} | weights={weights} ===\n")

    print(f"[baseline v{n}] scoring T1+T2+T3...")
    t0 = time.time()
    baseline_scores = score_all_tiers(provider, baseline)
    baseline_composite = composite(baseline_scores, weights)
    print(
        f"[baseline v{n}] T1={baseline_scores['t1_mean']} T2={baseline_scores['t2_mean']} "
        f"T3={baseline_scores['t3_mean']} composite={baseline_composite} in {time.time() - t0:.1f}s\n"
    )

    weaknesses = diagnose(baseline_scores)
    print("Diagnosed weaknesses:")
    for w in weaknesses:
        print(f"  - {w}")
    print()

    candidates = []
    for i in range(args.candidates):
        print(f"[candidate {i + 1}] mutating + scoring...")
        t0 = time.time()
        try:
            text = mutate_persona(provider, baseline, weaknesses)
            cand = PersonaSpec(version=f"cand-{i + 1}", text=text)
            scores = score_all_tiers(provider, cand)
            comp = composite(scores, weights)
            candidates.append({"index": i + 1, "text": text, "scores": scores, "composite": comp})
            print(
                f"[candidate {i + 1}] T1={scores['t1_mean']} T2={scores['t2_mean']} "
                f"T3={scores['t3_mean']} composite={comp} in {time.time() - t0:.1f}s\n"
            )
        except Exception as exc:
            print(f"[candidate {i + 1}] ERROR: {exc}\n")

    candidates.sort(key=lambda c: c["composite"], reverse=True)
    print("=== verdict ===")
    print(
        f"baseline v{n}: T1={baseline_scores['t1_mean']} T2={baseline_scores['t2_mean']} "
        f"T3={baseline_scores['t3_mean']} composite={baseline_composite}"
    )
    for c in candidates:
        s = c["scores"]
        print(
            f"candidate {c['index']}: T1={s['t1_mean']} T2={s['t2_mean']} "
            f"T3={s['t3_mean']} composite={c['composite']}"
        )

    winner = candidates[0] if candidates else None
    promote = False
    promote_reason = ""
    if winner is not None and winner["composite"] > baseline_composite:
        regressions = []
        for axis in ("t1_mean", "t2_mean", "t3_mean"):
            drop = baseline_scores[axis] - winner["scores"][axis]
            if drop > args.floor_drop:
                regressions.append(f"{axis}: -{drop:.3f}")
        if regressions:
            promote = False
            promote_reason = (
                f"composite improved but axis regression too large: {', '.join(regressions)}"
            )
        else:
            promote = True
            promote_reason = (
                f"composite {winner['composite']} > {baseline_composite} with no axis regression > {args.floor_drop}"
            )

    if promote and not args.dry_run:
        new_n = n + 1
        new_path = ARTIFACTS_DIR / f"persona.v{new_n}.md"
        new_path.write_text(winner["text"], encoding="utf-8")
        (ARTIFACTS_DIR / "persona.latest.txt").write_text(f"v{new_n}\n", encoding="utf-8")
        print(f"\nPROMOTED: candidate {winner['index']} -> {new_path.name}")
        print(f"  reason: {promote_reason}")
    elif promote:
        print(f"\nWOULD PROMOTE candidate {winner['index']} (dry-run)")
    else:
        print("\nNO PROMOTION.")
        if winner is not None:
            print(f"  reason: {promote_reason or 'baseline composite still wins'}")

    Path(args.out).write_text(
        json.dumps({
            "baseline_version": n, "weights": list(weights),
            "baseline_scores": baseline_scores, "baseline_composite": baseline_composite,
            "candidates": [
                {"index": c["index"], "scores": c["scores"], "composite": c["composite"]}
                for c in candidates
            ],
            "winner_text": winner["text"] if winner else None,
            "promoted": bool(promote and not args.dry_run),
        }, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
