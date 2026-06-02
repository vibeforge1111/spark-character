"""Live persona pulse.

Fires a fixed prompt set through the real provider, scores each reply
with P1-P5, prints a scorecard. Useful before/after evolution cycles
to confirm voice quality is moving in the right direction.

Usage:

    # Default: uses ZAI_API_KEY / ZAI_BASE_URL / ZAI_MODEL from env.
    python evals/live_pulse.py

    # With critic-rewrite pass enabled:
    python evals/live_pulse.py --critic

    # Against a specific provider env:
    ZAI_API_KEY=... ZAI_MODEL=glm-5.1 python evals/live_pulse.py

The exit code is 0 if the mean across all axes is >= 0.9, else 1.
That gives CI-style gating if you want it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import time

# Allow running from repo root without install
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    ProviderSpec,
    generate,
    generate_with_critique,
    load_critic,
    load_persona,
    score_persona,
)


PROMPTS = [
    ("P1_list",        "List three things I should focus on as a founder this week."),
    ("P1_explain",     "Explain the difference between a seed round and a Series A in two sentences."),
    ("P1_recommend",   "Should I prioritize fundraising or shipping the product first?"),
    ("P2_error_path",  "The web search just failed. What happened?"),
    ("P2_capability",  "What can you actually help me with right now?"),
    ("P3_followup",    "Yeah let's keep going on the launch plan from earlier."),
    ("P3_short",       "hey"),
    ("P3_status",      "where are we"),
    ("P4_decision",    "Should I raise now or wait six months?"),
    ("P4_factual",     "What does TVL mean in DeFi?"),
    ("P5_anxious",     "I'm anxious about the launch tomorrow."),
    ("P5_quick",       "Quick gut check: ship the redesign or hold a week?"),
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
    parser = argparse.ArgumentParser(description="spark-character live persona pulse")
    parser.add_argument("--critic", action="store_true", help="Run the critic-rewrite pass after generation")
    parser.add_argument("--max-tokens", type=_positive_int, default=600, help="Maximum tokens per reply (must be a positive integer)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--out", default="evals/_pulse_last.json")
    args = parser.parse_args()

    provider = ProviderSpec.from_env()
    persona = load_persona()
    critic = load_critic() if args.critic else None
    print(
        f"\n=== spark-character live pulse | persona={persona.version} "
        f"critic={'v1' if args.critic else 'off'} model={provider.model} ===\n"
    )

    rows = []
    for label, prompt in PROMPTS:
        t0 = time()
        try:
            if args.critic and critic is not None:
                result = generate_with_critique(
                    prompt, provider=provider, persona=persona, critic=critic,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                )
            else:
                result = generate(
                    prompt, provider=provider, persona=persona,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                )
        except Exception as exc:
            print(f"[{label}] ERROR: {exc}\n")
            rows.append({"label": label, "prompt": prompt, "error": str(exc)})
            continue
        dt = time() - t0
        score = score_persona(result.final)
        rows.append({
            "label": label, "prompt": prompt, "draft": result.draft,
            "final": result.final, "rewritten": result.rewritten,
            "score": {
                "p1_em_dash": score.p1_em_dash,
                "p2_plumbing": score.p2_plumbing,
                "p2_hits": list(score.p2_hits),
                "p3_reset": score.p3_reset,
                "p4_lead": score.p4_lead,
                "p5_voice": score.p5_voice,
                "p5_reason": score.p5_reason,
                "mean": score.mean,
                "passed": score.passed,
            },
            "dt_s": round(dt, 2),
        })
        first_line = result.final.splitlines()[0] if result.final else "(empty)"
        flags = _flags(score)
        marker = " [REWRITTEN]" if result.rewritten else ""
        print(f"[{label}] dt={dt:.1f}s mean={score.mean}{marker}")
        print(f"  reply: {first_line[:100]}")
        print(f"  flags: {flags}\n")

    print("\n=== scorecard ===\n")
    cols = ["p1_em_dash", "p2_plumbing", "p3_reset", "p4_lead", "p5_voice"]
    print(f"{'label':<15} " + " ".join(f"{c:<13}" for c in cols))
    means: dict[str, float] = {c: 0.0 for c in cols}
    counted = 0
    for r in rows:
        if "score" not in r:
            print(f"{r['label']:<15} ERROR")
            continue
        s = r["score"]
        print(f"{r['label']:<15} " + " ".join(f"{s[c]:<13}" for c in cols))
        for c in cols:
            means[c] += s[c]
        counted += 1
    if counted:
        for c in cols:
            means[c] = round(means[c] / counted, 3)
    overall_mean = round(sum(means.values()) / max(1, len(cols)), 3)
    print(f"\nMEANS: {means}")
    print(f"OVERALL: {overall_mean}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "means": means, "overall": overall_mean}, indent=2))
    print(f"\nFull transcript: {out_path}")

    return 0 if overall_mean >= 0.9 else 1


def _flags(score) -> str:
    parts = []
    if score.p1_em_dash < 1.0:
        parts.append("EM_DASH")
    if score.p2_hits:
        parts.append(f"PLUMBING={','.join(score.p2_hits)}")
    if score.p3_reset < 1.0:
        parts.append("RESET")
    if score.p4_lead < 1.0:
        parts.append("HEDGE")
    if score.p5_voice < 0.7:
        parts.append(f"VOICE_LOW({score.p5_voice})")
    return " ".join(parts) if parts else "clean"


if __name__ == "__main__":
    sys.exit(main())
