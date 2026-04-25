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
    ProviderSpec,
    generate,
    load_persona,
    run_probe,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="evals/_full_pulse.json")
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=0.7)
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

    # ----- aggregate -----
    print("\n=== scorecard ===\n")
    t1_means = [r["t1_mean"] for r in t1_rows if "t1_mean" in r]
    t2_scores = [r["score"] for r in t2_rows]
    t3_scores = [r["score"] for r in t3_rows if "score" in r]
    t1_mean = round(sum(t1_means) / max(1, len(t1_means)), 3) if t1_means else 0
    t2_mean = round(sum(t2_scores) / max(1, len(t2_scores)), 3) if t2_scores else 0
    t3_mean = round(sum(t3_scores) / max(1, len(t3_scores)), 3) if t3_scores else 0
    print(f"T1 mechanics mean:       {t1_mean}")
    print(f"T2 distinctiveness mean: {t2_mean}")
    print(f"T3 behavioral mean:      {t3_mean}")
    print()
    print("T3 per-trait:")
    for r in t3_rows:
        if "score" in r:
            print(f"  {r['probe_id']:<32} trait={r['trait']:<28} score={r['score']:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "persona_version": persona.version,
        "model": provider.model,
        "t1_rows": t1_rows,
        "t2_rows": t2_rows,
        "t3_rows": t3_rows,
        "t1_mean": t1_mean,
        "t2_mean": t2_mean,
        "t3_mean": t3_mean,
    }, indent=2))
    print(f"\nFull transcript: {args.out}")
    return 0 if (t1_mean >= 0.95 and t2_mean >= 0.6 and t3_mean >= 0.6) else 1


if __name__ == "__main__":
    sys.exit(main())
