"""T5 cross-provider voice consistency.

Spark is provider-agnostic. The same persona spec gets injected as the
system prompt regardless of backend (Z.AI, OpenAI, MiniMax, Anthropic,
Ollama). T5 measures whether the resulting voice is actually
consistent across backends.

Workflow:
  1. Resolve a list of providers from env (Z.AI + OpenAI by default).
  2. Run the same prompt set against each provider.
  3. Score every reply on T1 mechanics + T2 distinctiveness.
  4. For each axis: report per-provider mean and pairwise drift.
  5. Voice-pair-judge: ask a judge to compare each pair of replies
     (same prompt, different providers) and rate how much they sound
     like the same agent. 0-10 scale.

Output:
  - per-provider scorecard (so we know each one stands on its own)
  - cross-provider drift on T1, T2
  - same-agent judge mean (0-1)

Usage:
  python evals/cross_provider.py
  python evals/cross_provider.py --providers zai,openai
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean as mean_
from time import time

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    ProviderSpec,
    call_provider,
    generate,
    load_persona,
    score_distinctiveness,
    score_persona,
)


PROMPTS = [
    "Should I prioritize fundraising or shipping the product first?",
    "What does TVL mean in DeFi?",
    "I'm anxious about the launch tomorrow.",
    "where are we",
    "Quick gut check: ship the redesign or hold a week?",
    "What can you actually help me with right now?",
]


@dataclass(frozen=True)
class ProviderProfile:
    name: str
    spec: ProviderSpec


def resolve_providers(names: list[str]) -> list[ProviderProfile]:
    out: list[ProviderProfile] = []
    for name in names:
        n = name.lower().strip()
        if n == "zai":
            api_key = os.environ.get("ZAI_API_KEY")
            if not api_key:
                continue
            out.append(ProviderProfile(
                name="zai",
                spec=ProviderSpec(
                    base_url=os.environ.get("ZAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4/"),
                    model=os.environ.get("ZAI_MODEL", "glm-5.1"),
                    api_key=api_key,
                ),
            ))
        elif n == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                continue
            out.append(ProviderProfile(
                name="openai",
                spec=ProviderSpec(
                    base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/"),
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    api_key=api_key,
                ),
            ))
        elif n == "minimax":
            api_key = os.environ.get("MINIMAX_API_KEY")
            if not api_key:
                continue
            out.append(ProviderProfile(
                name="minimax",
                spec=ProviderSpec(
                    base_url=os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1/"),
                    model=os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7"),
                    api_key=api_key,
                ),
            ))
    return out


SAME_AGENT_JUDGE_SYSTEM = (
    "You are a voice consistency judge for an AI agent named Spark. "
    "You will see two replies to the same prompt, written by Spark "
    "running on two different language model backends. "
    "Score 0-10 how much these two replies sound like the same agent: "
    "10 = unmistakably the same character, voice register, posture, "
    "and stance, just different words. 5 = related but distinct "
    "voices. 0 = clearly different agents. "
    "Score the voice signature, not the literal text overlap. "
    "Return exactly one line: SCORE=<integer 0-10>. No other text."
)


def same_agent_score(*, prompt: str, reply_a: str, reply_b: str, judge: ProviderSpec) -> tuple[int, str]:
    user = (
        f"[Prompt]\n{prompt}\n\n"
        f"[Reply on backend A]\n{reply_a}\n\n"
        f"[Reply on backend B]\n{reply_b}\n\n"
        "Return SCORE=<integer 0-10> only."
    )
    response = call_provider(
        provider=judge,
        system_prompt=SAME_AGENT_JUDGE_SYSTEM,
        user_prompt=user,
        max_tokens=120,
        temperature=0.0,
        disable_thinking=True,
    )
    return _parse_score(response), response


def _parse_score(text: str) -> int:
    if not text:
        return 5
    match = re.search(r"SCORE\s*=\s*(\d+)", text, re.IGNORECASE)
    if match:
        return max(0, min(10, int(match.group(1))))
    digits = re.findall(r"\b([0-9]|10)\b", text)
    if digits:
        return max(0, min(10, int(digits[0])))
    return 5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", default="zai,openai")
    parser.add_argument("--judge", default="zai", help="Provider to use as the same-agent judge")
    parser.add_argument("--out", default="evals/_cross_provider.json")
    args = parser.parse_args()

    requested = [p.strip() for p in args.providers.split(",") if p.strip()]
    profiles = resolve_providers(requested)
    if len(profiles) < 2:
        print(f"Need at least 2 providers. Resolved: {[p.name for p in profiles]}")
        return 2
    judge_profiles = resolve_providers([args.judge])
    if not judge_profiles:
        print(f"Judge provider '{args.judge}' has no API key set.")
        return 2
    judge = judge_profiles[0].spec

    persona = load_persona()
    print(f"\n=== T5 cross-provider | persona={persona.version} ===")
    print(f"providers: {[p.name for p in profiles]}  models: {[p.spec.model for p in profiles]}")
    print(f"judge: {judge_profiles[0].name} model={judge.model}\n")

    # Generate per-provider replies
    per_provider: dict[str, list[dict]] = {p.name: [] for p in profiles}
    for prompt in PROMPTS:
        print(f"prompt: {prompt}")
        for p in profiles:
            t0 = time()
            try:
                result = generate(prompt, provider=p.spec, persona=persona)
                t1 = score_persona(result.final)
                try:
                    t2 = score_distinctiveness(result.final, provider=p.spec)
                    t2_score = t2.score
                except Exception:
                    t2_score = None
                dt = time() - t0
                per_provider[p.name].append({
                    "prompt": prompt, "reply": result.final,
                    "t1": t1.mean, "t2": t2_score, "dt_s": round(dt, 1),
                })
                print(f"  [{p.name}] T1={t1.mean} T2={t2_score} dt={dt:.1f}s :: {result.final.splitlines()[0][:80] if result.final else ''}")
            except Exception as exc:
                per_provider[p.name].append({"prompt": prompt, "error": str(exc)})
                print(f"  [{p.name}] ERROR: {exc}")
        print()

    # Same-agent judge across each pair
    print("\nsame-agent judge across pairs:\n")
    pair_scores: list[dict] = []
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            same_scores: list[int] = []
            for k, prompt in enumerate(PROMPTS):
                row_a = per_provider[a.name][k]
                row_b = per_provider[b.name][k]
                if "reply" not in row_a or "reply" not in row_b:
                    continue
                try:
                    raw, judge_resp = same_agent_score(
                        prompt=prompt,
                        reply_a=row_a["reply"],
                        reply_b=row_b["reply"],
                        judge=judge,
                    )
                    same_scores.append(raw)
                    pair_scores.append({
                        "a": a.name, "b": b.name, "prompt": prompt, "raw": raw,
                    })
                    print(f"  [{a.name} vs {b.name}] {prompt[:60]} -> SCORE={raw}")
                except Exception as exc:
                    print(f"  [{a.name} vs {b.name}] {prompt[:60]} -> ERROR: {exc}")
            if same_scores:
                m = round(mean_(same_scores) / 10.0, 3)
                print(f"  >> same-agent mean ({a.name} vs {b.name}): {m}")
                print()

    # Aggregate
    print("\n=== T5 scorecard ===\n")
    for p in profiles:
        rows = [r for r in per_provider[p.name] if "t1" in r]
        t1m = round(mean_(r["t1"] for r in rows), 3) if rows else 0.0
        t2_vals = [r["t2"] for r in rows if r.get("t2") is not None]
        t2m = round(mean_(t2_vals), 3) if t2_vals else 0.0
        print(f"  {p.name:<10} T1={t1m} T2={t2m}")

    pair_means: dict[tuple[str, str], float] = {}
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            scores = [r["raw"] for r in pair_scores if r["a"] == a.name and r["b"] == b.name]
            if scores:
                pair_means[(a.name, b.name)] = round(mean_(scores) / 10.0, 3)
    print()
    for (a, b), m in pair_means.items():
        print(f"  same-agent({a},{b}) = {m}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "persona_version": persona.version,
        "providers": [p.name for p in profiles],
        "models": {p.name: p.spec.model for p in profiles},
        "per_provider": per_provider,
        "pair_scores": pair_scores,
        "pair_means": {f"{a}|{b}": v for (a, b), v in pair_means.items()},
    }, indent=2))
    print(f"\nFull transcript: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
