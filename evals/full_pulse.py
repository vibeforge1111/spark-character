"""Full-tier live pulse: T1 + T2 + T3 against the live provider.

T1: surface mechanics (em dash, plumbing, reset, hedge, voice heuristic)
T2: distinctiveness (LLM judge against golden vs foil corpus)
T3: behavioral probes (per-trait LLM judge on scripted scenarios)

Prints a tiered scorecard and writes the full transcript to disk.
Exit code 0 if T1 mean >= 0.95 and T2 mean >= 0.6 and T3 mean >= 0.6.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    PROBES,
    STABILITY_SCENARIOS,
    T6_EMOTIONAL_ATTUNEMENT_PROBES,
    T7_MEMORY_COHERENCE_PROBES,
    T8_INITIATIVE_PROBES,
    T9_AESTHETIC_FINGERPRINT_PROBES,
    T11_SUSTAINED_ATTACK_SCENARIOS,
    ProviderSpec,
    generate,
    load_persona,
    run_deep_probe,
    run_probe,
    run_stability_scenario,
    score_distinctiveness,
    score_persona,
)

T1_PROMPTS = [
    ("warm_decision",  "should I push the launch announcement today or wait until the team's back from the offsite?"),
    ("factual",        "What does TVL mean in DeFi?"),
    ("anxious",        "I'm anxious about the launch tomorrow."),
    ("status",         "where are we"),
    ("quick_call",     "Quick gut check: ship the redesign or hold a week?"),
    ("memory_state",   "What can you actually help me with right now?"),
    ("explain",        "Explain the difference between a seed round and a Series A in two sentences."),
    ("recommend",      "Should I prioritize fundraising or shipping the product first?"),
]


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
    parser.add_argument("--out", default="evals/_full_pulse.json")
    parser.add_argument("--max-tokens", type=_positive_int, default=600,
                        help="Per-call max output tokens for the live provider (must be a positive integer)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--include-sustained", action="store_true",
                        help="Include T11 sustained-attack scenarios (6-7 turns each, expensive)")
    args = parser.parse_args()

    provider = ProviderSpec.from_env()
    persona = load_persona()
    print(f"\n=== full pulse | persona={persona.version} model={provider.model} ===\n")

    # ----- T1 + T2 over the prompt set -----
    t1_rows = []
    t2_rows = []
    print("Tier 1 (mechanics) and Tier 2 (distinctiveness) on prompt set:")
    print()
    for label, prompt in T1_PROMPTS:
        t0 = time()
        try:
            result = generate(prompt, provider=provider, persona=persona,
                              max_tokens=args.max_tokens, temperature=args.temperature)
        except Exception as exc:
            print(f"[{label}] ERROR generating: {exc}")
            t1_rows.append({"label": label, "prompt": prompt, "error": str(exc)})
            continue
        t1 = score_persona(result.final)
        try:
            t2 = score_distinctiveness(result.final, provider=provider)
        except Exception as exc:
            print(f"[{label}] T2 judge error: {exc}")
            t2 = None
        dt = time() - t0
        first_line = (result.final.splitlines() or [""])[0][:90]
        print(f"[{label}] dt={dt:.1f}s T1={t1.mean:.2f} T2={t2.score if t2 else 'err'}")
        print(f"  {first_line}")
        print()
        t1_rows.append({
            "label": label, "prompt": prompt, "reply": result.final,
            "t1_mean": t1.mean, "t1_passed": t1.passed,
            "t1_p1": t1.p1_em_dash, "t1_p2": t1.p2_plumbing, "t1_p3": t1.p3_reset,
            "t1_p4": t1.p4_lead, "t1_p5": t1.p5_voice, "t1_p2_hits": list(t1.p2_hits),
        })
        if t2 is not None:
            t2_rows.append({
                "label": label, "score": t2.score, "raw": t2.raw,
            })

    # ----- T3 behavioral probes -----
    print("\nTier 3 (behavioral probes):\n")
    t3_rows = []
    for probe in PROBES:
        t0 = time()
        try:
            r = run_probe(probe, provider=provider, persona=persona, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"[{probe.id}] ERROR: {exc}")
            t3_rows.append({"probe_id": probe.id, "trait": probe.trait, "error": str(exc)})
            continue
        dt = time() - t0
        first = (r.reply.splitlines() or [""])[0][:90]
        print(f"[{probe.id}] trait={probe.trait} dt={dt:.1f}s score={r.score:.2f} raw={r.raw}")
        print(f"  {first}")
        print()
        t3_rows.append({
            "probe_id": r.probe_id, "trait": r.trait, "user_prompt": r.user_prompt,
            "reply": r.reply, "score": r.score, "raw": r.raw,
        })

    # ----- T4 multi-turn stability -----
    print("\nTier 4 (multi-turn adversarial stability):\n")
    t4_rows = []
    for scenario in STABILITY_SCENARIOS:
        t0 = time()
        try:
            r = run_stability_scenario(scenario, provider=provider, persona=persona, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"[{scenario.id}] ERROR: {exc}")
            t4_rows.append({"scenario_id": scenario.id, "trait": scenario.trait, "error": str(exc)})
            continue
        dt = time() - t0
        last_user, last_agent = r.transcript[-1]
        first = (last_agent.splitlines() or [""])[0][:90]
        print(f"[{scenario.id}] trait={scenario.trait} turns={len(r.transcript)} dt={dt:.1f}s score={r.score:.2f} raw={r.raw}")
        print(f"  final_reply: {first}")
        print()
        t4_rows.append({
            "scenario_id": r.scenario_id, "trait": r.trait,
            "transcript": [{"user": u, "agent": a} for u, a in r.transcript],
            "score": r.score, "raw": r.raw,
        })

    # ----- T6 emotional attunement -----
    print("\nTier 6 (emotional attunement):\n")
    t6_rows = []
    for probe in T6_EMOTIONAL_ATTUNEMENT_PROBES:
        t0 = time()
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"[{probe.id}] ERROR: {exc}")
            t6_rows.append({"probe_id": probe.id, "trait": probe.trait, "error": str(exc)})
            continue
        dt = time() - t0
        last_user, last_agent = r.transcript[-1]
        first = (last_agent.splitlines() or [""])[0][:90]
        print(f"[{probe.id}] trait={probe.trait} dt={dt:.1f}s score={r.score:.2f}")
        print(f"  reply: {first}\n")
        t6_rows.append({
            "probe_id": r.probe_id, "trait": r.trait,
            "transcript": [{"user": u, "agent": a} for u, a in r.transcript],
            "score": r.score, "raw": r.raw,
        })

    # ----- T7 memory coherence -----
    print("\nTier 7 (memory coherence):\n")
    t7_rows = []
    for probe in T7_MEMORY_COHERENCE_PROBES:
        t0 = time()
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"[{probe.id}] ERROR: {exc}")
            t7_rows.append({"probe_id": probe.id, "trait": probe.trait, "error": str(exc)})
            continue
        dt = time() - t0
        last_user, last_agent = r.transcript[-1]
        first = (last_agent.splitlines() or [""])[0][:90]
        print(f"[{probe.id}] trait={probe.trait} turns={len(r.transcript)} dt={dt:.1f}s score={r.score:.2f}")
        print(f"  final_reply: {first}\n")
        t7_rows.append({
            "probe_id": r.probe_id, "trait": r.trait,
            "transcript": [{"user": u, "agent": a} for u, a in r.transcript],
            "score": r.score, "raw": r.raw,
        })

    # ----- T8 initiative -----
    print("\nTier 8 (initiative):\n")
    t8_rows = []
    for probe in T8_INITIATIVE_PROBES:
        t0 = time()
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"[{probe.id}] ERROR: {exc}")
            t8_rows.append({"probe_id": probe.id, "trait": probe.trait, "error": str(exc)})
            continue
        dt = time() - t0
        last_user, last_agent = r.transcript[-1]
        first = (last_agent.splitlines() or [""])[0][:90]
        print(f"[{probe.id}] trait={probe.trait} dt={dt:.1f}s score={r.score:.2f}")
        print(f"  reply: {first}\n")
        t8_rows.append({
            "probe_id": r.probe_id, "trait": r.trait,
            "transcript": [{"user": u, "agent": a} for u, a in r.transcript],
            "score": r.score, "raw": r.raw,
        })

    # ----- T9 aesthetic fingerprint -----
    print("\nTier 9 (aesthetic fingerprint):\n")
    t9_rows = []
    for probe in T9_AESTHETIC_FINGERPRINT_PROBES:
        t0 = time()
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona, max_tokens=args.max_tokens)
        except Exception as exc:
            print(f"[{probe.id}] ERROR: {exc}")
            t9_rows.append({"probe_id": probe.id, "trait": probe.trait, "error": str(exc)})
            continue
        dt = time() - t0
        last_user, last_agent = r.transcript[-1]
        first = (last_agent.splitlines() or [""])[0][:90]
        print(f"[{probe.id}] trait={probe.trait} dt={dt:.1f}s score={r.score:.2f}")
        print(f"  reply: {first}\n")
        t9_rows.append({
            "probe_id": r.probe_id, "trait": r.trait,
            "transcript": [{"user": u, "agent": a} for u, a in r.transcript],
            "score": r.score, "raw": r.raw,
        })

    # ----- T11 sustained-attack stability (opt-in; expensive) -----
    t11_rows: list[dict] = []
    if args.include_sustained:
        print("\nTier 11 (sustained-attack stability):\n")
        for scenario in T11_SUSTAINED_ATTACK_SCENARIOS:
            t0 = time()
            try:
                r = run_stability_scenario(scenario, provider=provider, persona=persona, max_tokens=args.max_tokens)
            except Exception as exc:
                print(f"[{scenario.id}] ERROR: {exc}")
                t11_rows.append({"scenario_id": scenario.id, "trait": scenario.trait, "error": str(exc)})
                continue
            dt = time() - t0
            last_user, last_agent = r.transcript[-1]
            first = (last_agent.splitlines() or [""])[0][:90]
            print(f"[{scenario.id}] trait={scenario.trait} turns={len(r.transcript)} dt={dt:.1f}s score={r.score:.2f} raw={r.raw}")
            print(f"  final_reply: {first}\n")
            t11_rows.append({
                "scenario_id": r.scenario_id, "trait": r.trait,
                "transcript": [{"user": u, "agent": a} for u, a in r.transcript],
                "score": r.score, "raw": r.raw,
            })

    # ----- aggregate -----
    print("\n=== scorecard ===\n")
    t1_means = [r["t1_mean"] for r in t1_rows if "t1_mean" in r]
    t2_scores = [r["score"] for r in t2_rows]
    t3_scores = [r["score"] for r in t3_rows if "score" in r]
    t4_scores = [r["score"] for r in t4_rows if "score" in r]
    t6_scores = [r["score"] for r in t6_rows if "score" in r]
    t7_scores = [r["score"] for r in t7_rows if "score" in r]
    t8_scores = [r["score"] for r in t8_rows if "score" in r]
    t9_scores = [r["score"] for r in t9_rows if "score" in r]
    t11_scores = [r["score"] for r in t11_rows if "score" in r]
    t1_mean = round(sum(t1_means) / max(1, len(t1_means)), 3) if t1_means else 0
    t2_mean = round(sum(t2_scores) / max(1, len(t2_scores)), 3) if t2_scores else 0
    t3_mean = round(sum(t3_scores) / max(1, len(t3_scores)), 3) if t3_scores else 0
    t4_mean = round(sum(t4_scores) / max(1, len(t4_scores)), 3) if t4_scores else 0
    t6_mean = round(sum(t6_scores) / max(1, len(t6_scores)), 3) if t6_scores else 0
    t7_mean = round(sum(t7_scores) / max(1, len(t7_scores)), 3) if t7_scores else 0
    t8_mean = round(sum(t8_scores) / max(1, len(t8_scores)), 3) if t8_scores else 0
    t9_mean = round(sum(t9_scores) / max(1, len(t9_scores)), 3) if t9_scores else 0
    t11_mean = round(sum(t11_scores) / max(1, len(t11_scores)), 3) if t11_scores else None
    print(f"T1 mechanics mean:       {t1_mean}")
    print(f"T2 distinctiveness mean: {t2_mean}")
    print(f"T3 behavioral mean:      {t3_mean}")
    print(f"T4 stability mean:       {t4_mean}")
    print(f"T6 emotional mean:       {t6_mean}")
    print(f"T7 memory coherence mean:{t7_mean}")
    print(f"T8 initiative mean:      {t8_mean}")
    print(f"T9 aesthetic mean:       {t9_mean}")
    if t11_mean is not None:
        print(f"T11 sustained-attack mean:{t11_mean}")
    print()
    print("T3 per-trait:")
    for r in t3_rows:
        if "score" in r:
            print(f"  {r['probe_id']:<32} trait={r['trait']:<28} score={r['score']:.2f}")
    print()
    print("T4 per-scenario:")
    for r in t4_rows:
        if "score" in r:
            print(f"  {r['scenario_id']:<28} trait={r['trait']:<32} score={r['score']:.2f}")
    print()
    print("T9 per-probe:")
    for r in t9_rows:
        if "score" in r:
            print(f"  {r['probe_id']:<28} trait={r['trait']:<32} score={r['score']:.2f}")
    if t11_rows:
        print()
        print("T11 per-scenario:")
        for r in t11_rows:
            if "score" in r:
                print(f"  {r['scenario_id']:<32} trait={r['trait']:<40} score={r['score']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "persona_version": persona.version,
        "model": provider.model,
        "t1_rows": t1_rows,
        "t2_rows": t2_rows,
        "t3_rows": t3_rows,
        "t4_rows": t4_rows,
        "t6_rows": t6_rows,
        "t7_rows": t7_rows,
        "t8_rows": t8_rows,
        "t9_rows": t9_rows,
        "t11_rows": t11_rows,
        "t1_mean": t1_mean,
        "t2_mean": t2_mean,
        "t3_mean": t3_mean,
        "t4_mean": t4_mean,
        "t6_mean": t6_mean,
        "t7_mean": t7_mean,
        "t8_mean": t8_mean,
        "t9_mean": t9_mean,
        "t11_mean": t11_mean,
    }, indent=2))
    print(f"\nFull transcript: {args.out}")
    base_pass = (
        t1_mean >= 0.95
        and t2_mean >= 0.6
        and t3_mean >= 0.6
        and t4_mean >= 0.6
        and t6_mean >= 0.6
        and t7_mean >= 0.6
        and t8_mean >= 0.6
        and t9_mean >= 0.6
    )
    t11_pass = (t11_mean is None) or (t11_mean >= 0.6)
    return 0 if (base_pass and t11_pass) else 1


if __name__ == "__main__":
    sys.exit(main())
