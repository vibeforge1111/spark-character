"""Persona evolution cycle.

Workflow per cycle:
  1. Load the current best persona (highest version on disk).
  2. Ask the provider to mutate it into N candidate variants.
  3. For each candidate plus the baseline, run the live pulse and score
     with P1-P5.
  4. If a candidate's overall mean strictly beats the baseline's, write
     it as the next persona.vN.md and update the symlink-equivalent
     pointer in artifacts/persona.latest.txt.
  5. Print a verdict.

Conservative on purpose: candidates have to win on real LLM output, not
just on heuristics. Easy to roll back by deleting the latest pointer.

Usage:
  python evals/evolve.py                  # 3 candidates, default temp
  python evals/evolve.py --candidates 5
  python evals/evolve.py --dry-run        # do not write a new artifact
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean as mean_

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    PersonaSpec,
    ProviderSpec,
    call_provider,
    generate,
    score_persona,
)
from spark_character.persona import ARTIFACTS_DIR, set_latest_persona_version  # noqa: E402

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
    "improved variant that addresses the listed weaknesses. Keep the "
    "same structure (a one-line title, then a short intro, then a 'Voice "
    "rules' bulleted list). Stay under 1500 characters. "
    "Hard constraints in every variant: never use em dashes. Never tell "
    "the agent to use em dashes. Never name internal subsystems. The "
    "first sentence rule must remain. The no-greeting-reset rule must "
    "remain. The no-plumbing-leak rule must remain.\n\n"
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


def baseline_score(provider: ProviderSpec, persona: PersonaSpec) -> tuple[float, list[dict]]:
    rows = []
    means_per_axis = {k: [] for k in ("p1_em_dash", "p2_plumbing", "p3_reset", "p4_lead", "p5_voice")}
    for prompt in PROMPTS:
        try:
            result = generate(prompt, provider=provider, persona=persona)
            score = score_persona(result.final)
            rows.append({"prompt": prompt, "reply": result.final, "mean": score.mean})
            for k in means_per_axis:
                means_per_axis[k].append(getattr(score, k))
        except Exception as exc:
            rows.append({"prompt": prompt, "error": str(exc)})
    overall = round(mean_([mean_(v) for v in means_per_axis.values() if v]), 3)
    return overall, rows


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
        max_tokens=900,
        temperature=0.9,
    ).strip()


def diagnose_weaknesses(rows: list[dict]) -> list[str]:
    out: list[str] = []
    for row in rows:
        if "error" in row or row["mean"] >= 1.0:
            continue
        score = score_persona(row["reply"])
        if score.p1_em_dash < 1.0:
            out.append(f"em-dash in reply to: {row['prompt']!r}")
        if score.p2_hits:
            out.append(f"plumbing leak ({','.join(score.p2_hits)}) in reply to: {row['prompt']!r}")
        if score.p3_reset < 1.0:
            out.append(f"greeting reset in reply to: {row['prompt']!r}")
        if score.p4_lead < 1.0:
            out.append(f"hedge opener in reply to: {row['prompt']!r}")
        if score.p5_voice < 0.7:
            out.append(f"voice low ({score.p5_voice}) in reply to: {row['prompt']!r}")
    return out[:8] or ["No specific failures, push for sharper warmth and brevity."]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="evals/_evolve_last.json")
    args = parser.parse_args()

    provider = ProviderSpec.from_env()
    n, baseline = find_latest_persona()
    print(f"\n=== persona evolution cycle | baseline=v{n} | candidates={args.candidates} ===\n")

    print(f"[baseline v{n}] scoring...")
    t0 = time.time()
    baseline_overall, baseline_rows = baseline_score(provider, baseline)
    print(f"[baseline v{n}] overall={baseline_overall} in {time.time() - t0:.1f}s\n")

    weaknesses = diagnose_weaknesses(baseline_rows)
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
            overall, rows = baseline_score(provider, cand)
            candidates.append({"index": i + 1, "text": text, "overall": overall, "rows": rows})
            print(f"[candidate {i + 1}] overall={overall} in {time.time() - t0:.1f}s\n")
        except Exception as exc:
            print(f"[candidate {i + 1}] ERROR: {exc}\n")

    candidates.sort(key=lambda c: c["overall"], reverse=True)
    print("=== verdict ===")
    print(f"baseline v{n}: {baseline_overall}")
    for c in candidates:
        print(f"candidate {c['index']}: {c['overall']}")

    winner = candidates[0] if candidates else None
    promote = winner is not None and winner["overall"] > baseline_overall
    if promote and not args.dry_run:
        new_n = n + 1
        new_path = ARTIFACTS_DIR / f"persona.v{new_n}.md"
        new_path.write_text(winner["text"], encoding="utf-8")
        set_latest_persona_version(
            f"v{new_n}",
            actor="evals/evolve.py",
            reason=f"candidate {winner['index']} score {winner['overall']} > baseline {baseline_overall}",
        )
        print(f"\nPROMOTED: candidate {winner['index']} -> {new_path.name} ({winner['overall']} > {baseline_overall})")
    elif promote:
        print(f"\nWOULD PROMOTE candidate {winner['index']} (dry-run)")
    else:
        print("\nNO PROMOTION: baseline still wins.")

    Path(args.out).write_text(
        json.dumps({
            "baseline_version": n,
            "baseline_overall": baseline_overall,
            "candidates": [{"index": c["index"], "overall": c["overall"]} for c in candidates],
            "winner_text": winner["text"] if winner else None,
            "promoted": bool(promote and not args.dry_run),
        }, indent=2)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
