"""Aggregate observer.py observations into a digest with recommendations.

Reads evals/_observations.jsonl, computes:

- Mean tier scores from the meta-observer (t1, t2, t3, t13)
- Top recurring 'pattern' strings flagged 2+ times
- Top recommendation_tier 'fire_now' / 'consider_evolution' targets
- Most common evolution_target axis with sample observations
- Recent specific 'rewrite_suggestion' lines that are concrete enough
  to act on

Prints a digest sized for human reading. Suggests one specific evolution
move based on the strongest signal in the window.

Usage:

    python evals/observations_digest.py
    python evals/observations_digest.py --last 100
    python evals/observations_digest.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean as mean_

OBSERVATIONS_FILE_DEFAULT = Path("evals/_observations.jsonl")


def _load(path: Path, *, limit: int) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def _digest(rows: list[dict]) -> dict:
    score_axes = {"t1": [], "t2": [], "t3": [], "t13": []}
    patterns: Counter = Counter()
    rec_tiers: Counter = Counter()
    evolution_targets: Counter = Counter()
    rewrites: list[dict] = []
    recurring_chips: Counter = Counter()
    recurring_routes: Counter = Counter()

    for r in rows:
        scores = r.get("scores") or {}
        for k in score_axes:
            v = scores.get(k)
            if isinstance(v, (int, float)):
                score_axes[k].append(float(v))
        pat = (r.get("pattern") or "").strip()
        if pat:
            patterns[pat] += 1
        tier = (r.get("recommendation_tier") or "").strip()
        if tier:
            rec_tiers[tier] += 1
        target = (r.get("evolution_target") or "").strip()
        if target:
            evolution_targets[target] += 1
        rewrite = (r.get("rewrite_suggestion") or "").strip()
        if rewrite and len(rewrite) > 12:
            rewrites.append({
                "ts": r.get("ts"),
                "rewrite": rewrite,
                "trace_ref": r.get("trace_ref"),
                "preview": (r.get("reply_preview") or "")[:120],
            })
        chip = r.get("chip")
        if chip:
            recurring_chips[str(chip)] += 1
        route = r.get("route")
        if route:
            recurring_routes[str(route)] += 1

    means = {}
    for k, vals in score_axes.items():
        if vals:
            means[k] = round(mean_(vals), 2)

    return {
        "n_observations": len(rows),
        "score_means": means,
        "top_patterns": patterns.most_common(5),
        "recommendation_tiers": dict(rec_tiers),
        "evolution_targets": dict(evolution_targets),
        "recurring_chips": recurring_chips.most_common(5),
        "recurring_routes": recurring_routes.most_common(5),
        "recent_rewrites": rewrites[-5:],
    }


def _suggest_next_move(digest: dict) -> str:
    """One specific suggestion based on the strongest signal."""
    targets = digest.get("evolution_targets") or {}
    tiers = digest.get("recommendation_tiers") or {}
    means = digest.get("score_means") or {}

    # Strongest signal: any target with >= 3 observations
    if targets:
        most_target, count = max(targets.items(), key=lambda kv: kv[1])
        if count >= 3:
            tier_label = {
                "t1": "T1 mechanics", "t2": "T2 distinctiveness",
                "t3": "T3 behavioral", "t13": "T13 humane depth",
            }.get(most_target, most_target)
            return (
                f"Suggested next move: fire an evolution targeting {tier_label} "
                f"({most_target}). The observer flagged it {count} times in "
                f"the recent window.\n"
                f"  python -u evals/lowest_tier_watch.py --once  "
                f"# or evolve_persona.py with weights skewed to {most_target}"
            )

    # If lowest mean axis < 0.6, suggest targeting it
    if means:
        weakest = min(means.items(), key=lambda kv: kv[1])
        if weakest[1] < 0.6:
            return (
                f"Suggested next move: weakest axis {weakest[0]} mean {weakest[1]} "
                f"is below 0.6. Run lowest_tier_watch."
            )

    if tiers.get("fire_now", 0) >= 2:
        return (
            "Suggested next move: 2+ 'fire_now' recommendations in window. "
            "Trigger evolution now via auto_loop or evolve_persona."
        )

    return (
        "Suggested next move: nothing urgent. Keep continuous_eval running, "
        "let the score history fill, and check back."
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations-file", default=str(OBSERVATIONS_FILE_DEFAULT))
    parser.add_argument("--last", type=int, default=50)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    obs_path = Path(args.observations_file)
    rows = _load(obs_path, limit=args.last)
    digest = _digest(rows)

    if args.json:
        # Well-known CLI UX pattern (jq, kubectl, gh): pretty-print when a
        # human is at the terminal; emit compact single-line JSON when stdout
        # is a pipe / file redirect so `| jq`, `| grep`, ndjson pipelines, and
        # line-oriented shell loops can consume the digest without the extra
        # whitespace from indent=2. JSON content is byte-equivalent on both
        # branches; only whitespace differs.
        if getattr(sys.stdout, "isatty", lambda: False)():
            print(json.dumps(digest, indent=2))
        else:
            print(json.dumps(digest, separators=(",", ":")))
        return 0

    if not rows:
        print(f"No observations at {obs_path}. Run observer.py first.")
        return 0

    print(f"=== observer digest, last {len(rows)} observations ===\n")
    print("Score means (meta-LLM judge):")
    for k, v in digest["score_means"].items():
        bar = "#" * int(v)
        print(f"  {k.upper():<5} {v:>5.2f} {bar}")
    print()

    if digest["top_patterns"]:
        print("Top recurring patterns:")
        for pat, count in digest["top_patterns"]:
            print(f"  ({count}x) {pat[:140]}")
        print()

    if digest["evolution_targets"]:
        print("Evolution targets surfaced by the observer:")
        for target, count in sorted(digest["evolution_targets"].items(), key=lambda kv: -kv[1]):
            print(f"  {target:<6} {count}")
        print()

    if digest["recommendation_tiers"]:
        print("Recommendation tiers:")
        for tier, count in digest["recommendation_tiers"].items():
            print(f"  {tier:<22} {count}")
        print()

    if digest["recent_rewrites"]:
        print("Recent concrete rewrite suggestions:")
        for r in digest["recent_rewrites"]:
            ts = r.get("ts")
            when = (
                time.strftime("%Y-%m-%d %H:%M", time.gmtime(int(ts)))
                if ts
                else "?"
            )
            print(f"  [{when}] preview: {r['preview']}")
            print(f"    rewrite: {r['rewrite'][:200]}")
            print()

    print("---")
    print(_suggest_next_move(digest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
