"""Read _score_history.jsonl and print trends.

Reads the score history written by continuous_eval and presents:

- Last N runs as a per-tier table
- Per-tier mean over the last N runs
- Largest improvements and largest regressions across the window
- Trend arrows: up, down, flat (vs the run from --compare-back ago)

Usage:

    python evals/score_trend.py
    python evals/score_trend.py --last 30 --compare-back 10
    python evals/score_trend.py --json    # machine-readable
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean as mean_

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))


HISTORY_FILE_DEFAULT = Path("evals/_score_history.jsonl")
TIER_KEYS = (
    "t1_mean", "t2_mean", "t3_mean", "t4_mean",
    "t6_mean", "t7_mean", "t8_mean", "t9_mean",
)


def _load(history_path: Path, *, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    if not history_path.exists():
        return []
    rows: list[dict] = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def _arrow(current: float, previous: float, eps: float = 0.02) -> str:
    if current is None or previous is None:
        return " "
    delta = current - previous
    if delta > eps:
        return "^"
    if delta < -eps:
        return "v"
    return "="


def _format_table(rows: list[dict]) -> str:
    if not rows:
        return "no rows yet"
    header_keys = ["ts", "persona", "tier"] + list(TIER_KEYS)
    header = "{:<19} {:<22} {:<6} ".format("when", "persona", "tier") + " ".join(
        f"{k.replace('_mean','').upper():>5}" for k in TIER_KEYS
    )
    lines = [header]
    lines.append("-" * len(header))
    for r in rows:
        ts = r.get("ts")
        when = (
            time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(ts))) if ts else "?"
        )
        persona = (r.get("persona_version") or "")[:22]
        tier = (r.get("tier") or "")[:6]
        cells = []
        for k in TIER_KEYS:
            v = r.get(k)
            cells.append(f"{v:>5}" if isinstance(v, (int, float)) else f"{'-':>5}")
        lines.append("{:<19} {:<22} {:<6} ".format(when, persona, tier) + " ".join(cells))
    return "\n".join(lines)


def _summarize(rows: list[dict], *, compare_back: int) -> dict:
    """For each tier: latest value, mean over window, delta vs N runs back."""
    summary: dict[str, dict] = {}
    for k in TIER_KEYS:
        values = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        if not values:
            continue
        latest = values[-1]
        window_mean = round(mean_(values), 3)
        compared = values[-compare_back] if len(values) > compare_back else (values[0] if values else None)
        delta = round(latest - compared, 3) if compared is not None else None
        summary[k] = {
            "latest": latest,
            "window_mean": window_mean,
            "delta_vs_compare": delta,
            "n_samples": len(values),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-file", default=str(HISTORY_FILE_DEFAULT))
    parser.add_argument("--last", type=int, default=15, help="Show this many most-recent runs")
    parser.add_argument("--compare-back", type=int, default=10, help="Compare current to N runs back for trend")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    history_path = Path(args.history_file)
    rows = _load(history_path, limit=args.last)

    if args.json:
        print(json.dumps({
            "rows": rows,
            "summary": _summarize(rows, compare_back=args.compare_back),
        }, indent=2))
        return 0

    if not rows:
        print(f"No score history at {history_path}. Run continuous_eval first.")
        return 0

    print(f"=== last {len(rows)} runs from {history_path} ===\n")
    print(_format_table(rows))

    print("\n=== summary (per tier, over the window) ===\n")
    summary = _summarize(rows, compare_back=args.compare_back)
    print("{:<8} {:>8} {:>10} {:>14} {:>8}".format("tier", "latest", "win_mean", "delta_vs_-N", "samples"))
    print("-" * 55)
    for k, s in summary.items():
        delta = s.get("delta_vs_compare")
        delta_str = f"{delta:+.3f}" if delta is not None else "  -  "
        arrow = "^" if (delta or 0) > 0.02 else ("v" if (delta or 0) < -0.02 else "=")
        print("{:<8} {:>8} {:>10} {:>14} {:>8}".format(
            k.replace("_mean", "").upper(),
            s["latest"],
            s["window_mean"],
            f"{delta_str} {arrow}",
            s["n_samples"],
        ))

    regressions = [r for r in rows if r.get("regressions")]
    if regressions:
        print(f"\n=== {len(regressions)} regression event(s) in window ===")
        for r in regressions[-5:]:
            ts = r.get("ts")
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(int(ts))) if ts else "?"
            print(f"\n[{when}] {r.get('persona_version')}/{r.get('tier')}")
            for line in r["regressions"]:
                print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
