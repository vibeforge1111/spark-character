"""Lowest-tier evolution trigger.

Reads the score history, finds the weakest tier in the most recent
window, and fires a targeted evolution cycle on that tier even when
the overall composite is healthy. The principle: "always more room
for improvement." Even at 0.95 overall, the worst axis is the leverage
point for the next improvement.

Pairs with continuous_eval (which writes the history) and evolve_persona
(which performs the cycle). This script is the connective tissue.

Usage:

    # Watch + trigger forever
    python -u evals/lowest_tier_watch.py \\
        --sib-home <home> \\
        --consumer-pythons "C:/Python313/python.exe,..."

    # Single check
    python -u evals/lowest_tier_watch.py --once --dry-run

The watcher fires evolution when:
  1. There are at least --min-runs samples per tier
  2. The lowest-tier mean has held for at least --min-streak runs
  3. Time since the last evolution is at least --cooldown-seconds

It picks weights so the evolution targets the lowest tier.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean as mean_

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

HISTORY_FILE_DEFAULT = Path("evals/_score_history.jsonl")
STATE_FILE_DEFAULT = Path("evals/_lowest_tier_watch_state.json")
HEARTBEAT_FILE_DEFAULT = Path("evals/_lowest_tier_watch_heartbeat.txt")

TIER_KEYS = (
    "t1_mean", "t2_mean", "t3_mean", "t4_mean",
    "t6_mean", "t7_mean", "t8_mean", "t9_mean", "t13_mean",
)

# Weight presets per target tier. The mutator gets pulled toward the
# weak axis without abandoning the others. Weights here are 6-tuple
# T1,T2,T3,T6,T7,T8 since evolve_persona's --include-deeper accepts that.
TIER_TARGETED_WEIGHTS = {
    "t1_mean":  "0.40,0.20,0.10,0.10,0.10,0.10",
    "t2_mean":  "0.10,0.50,0.10,0.10,0.10,0.10",
    "t3_mean":  "0.10,0.20,0.40,0.10,0.10,0.10",
    "t6_mean":  "0.10,0.20,0.10,0.40,0.10,0.10",
    "t7_mean":  "0.10,0.20,0.10,0.10,0.40,0.10",
    "t8_mean":  "0.10,0.20,0.10,0.10,0.10,0.40",
    # t4, t9, t13 don't yet feed evolve_persona's composite directly,
    # so we still fire a generic balanced cycle when those are weakest.
    "t4_mean":  "0.20,0.30,0.30,0.10,0.05,0.05",
    "t9_mean":  "0.20,0.40,0.20,0.10,0.05,0.05",
    "t13_mean": "0.10,0.30,0.30,0.10,0.10,0.10",
}


def _load(path: Path, *, limit: int = 200) -> list[dict]:
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


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"last_fired_at": 0, "last_target_tier": None, "fires_total": 0}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"last_fired_at": 0, "last_target_tier": None, "fires_total": 0}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _write_heartbeat(path: Path, phase: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{int(time.time())} {phase}\n", encoding="utf-8")
    except Exception:
        pass


def find_lowest_tier(history: list[dict], *, min_runs: int) -> tuple[str | None, float, int]:
    """Return (axis, mean_score, n_samples) for the weakest tier with
    at least min_runs samples. Returns (None, 1.0, 0) if nothing
    qualifies."""
    weakest = (None, 1.0, 0)
    for axis in TIER_KEYS:
        values = [r.get(axis) for r in history if isinstance(r.get(axis), (int, float))]
        if len(values) < min_runs:
            continue
        m = round(mean_(values[-min_runs:]), 3)
        if m < weakest[1]:
            weakest = (axis, m, len(values))
    return weakest


def streak_held(history: list[dict], axis: str, *, min_streak: int) -> bool:
    """True if the axis has been the lowest of the recent runs for at
    least min_streak consecutive runs."""
    if not axis:
        return False
    recent = [r for r in history[-min_streak:] if isinstance(r.get(axis), (int, float))]
    if len(recent) < min_streak:
        return False
    for row in recent:
        scores = {k: row.get(k) for k in TIER_KEYS if isinstance(row.get(k), (int, float))}
        if not scores:
            continue
        local_lowest = min(scores, key=lambda k: scores[k])
        if local_lowest != axis:
            return False
    return True


def fire_evolution(
    *,
    target_tier: str,
    sib_home: str | None,
    consumer_pythons: str | None,
    candidates: int,
    repo_root: Path,
    dry_run: bool,
) -> tuple[bool, str]:
    weights = TIER_TARGETED_WEIGHTS.get(target_tier, "0.20,0.40,0.20,0.10,0.05,0.05")
    cmd = [
        sys.executable, "-u",
        str(repo_root / "evals" / "evolve_persona.py"),
        "--candidates", str(candidates),
        "--weights", weights,
        "--include-deeper",
    ]
    if sib_home:
        cmd.extend(["--sib-home", sib_home])
    if dry_run:
        cmd.append("--dry-run")
    print(f"[lowest_tier_watch] firing evolution targeting {target_tier} weights={weights}", flush=True)
    if dry_run:
        print(f"[lowest_tier_watch] dry-run, would run: {' '.join(cmd)}", flush=True)
        return False, ""
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root)
    log_tail = (result.stdout or "")[-3000:]
    print(log_tail, flush=True)
    promoted = "PROMOTED:" in (result.stdout or "")
    if promoted and consumer_pythons:
        for py in [p.strip() for p in consumer_pythons.split(",") if p.strip()]:
            try:
                subprocess.run(
                    [py, "-m", "pip", "install", "--upgrade", "--force-reinstall",
                     "--no-deps",
                     "git+https://github.com/vibeforge1111/spark-character.git@master",
                     "-q"],
                    timeout=180,
                    check=False,
                )
            except Exception as exc:
                print(f"[lowest_tier_watch] consumer refresh failed for {py}: {exc}", flush=True)
    return promoted, log_tail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-file", default=str(HISTORY_FILE_DEFAULT))
    parser.add_argument("--state-file", default=str(STATE_FILE_DEFAULT))
    parser.add_argument("--heartbeat-file", default=str(HEARTBEAT_FILE_DEFAULT))
    parser.add_argument("--check-interval", type=int, default=3600,
                        help="Seconds between watch checks")
    parser.add_argument("--cooldown-seconds", type=int, default=21600,
                        help="Minimum seconds between firings (default 6h)")
    parser.add_argument("--min-runs", type=int, default=3,
                        help="Min samples per tier before considering it")
    parser.add_argument("--min-streak", type=int, default=2,
                        help="How many recent runs the axis must be lowest")
    parser.add_argument("--sib-home", default=None)
    parser.add_argument("--consumer-pythons", default=None)
    parser.add_argument("--candidates", type=int, default=2)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    history_path = Path(args.history_file)
    state_path = Path(args.state_file)
    heartbeat_path = Path(args.heartbeat_file)
    repo_root = _REPO_ROOT

    while True:
        try:
            _write_heartbeat(heartbeat_path, "checking")
            history = _load(history_path, limit=200)
            state = _load_state(state_path)

            now = time.time()
            since_last = now - state.get("last_fired_at", 0)
            print(
                f"[lowest_tier_watch] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"history={len(history)} fires_total={state.get('fires_total', 0)} "
                f"since_last_fire={int(since_last)}s",
                flush=True,
            )

            axis, score, n = find_lowest_tier(history, min_runs=args.min_runs)
            if axis is None:
                print(
                    f"[lowest_tier_watch] no tier has >= {args.min_runs} samples yet",
                    flush=True,
                )
            else:
                print(
                    f"[lowest_tier_watch] lowest tier: {axis} mean={score} (n={n})",
                    flush=True,
                )
                if since_last < args.cooldown_seconds:
                    print(
                        f"[lowest_tier_watch] cooldown active "
                        f"({int(since_last)}s < {args.cooldown_seconds}s), skipping",
                        flush=True,
                    )
                elif not streak_held(history, axis, min_streak=args.min_streak):
                    print(
                        f"[lowest_tier_watch] {axis} has not been lowest for "
                        f"{args.min_streak} consecutive runs, skipping",
                        flush=True,
                    )
                else:
                    _write_heartbeat(heartbeat_path, f"firing:{axis}")
                    promoted, _log = fire_evolution(
                        target_tier=axis,
                        sib_home=args.sib_home,
                        consumer_pythons=args.consumer_pythons,
                        candidates=args.candidates,
                        repo_root=repo_root,
                        dry_run=args.dry_run,
                    )
                    state["last_fired_at"] = int(now)
                    state["last_target_tier"] = axis
                    state["fires_total"] = int(state.get("fires_total", 0)) + 1
                    state["last_promoted"] = bool(promoted)
                    _save_state(state_path, state)
                    print(
                        f"[lowest_tier_watch] fired targeting {axis}, promoted={promoted}",
                        flush=True,
                    )

            if args.once:
                return 0
            _write_heartbeat(heartbeat_path, "sleeping")
            time.sleep(max(60, args.check_interval))
        except KeyboardInterrupt:
            print(
                f"[lowest_tier_watch] interrupted by operator — state preserved at {state_path}; re-run with the same --state-file to resume tracking from this point.",
                flush=True,
            )
            return 0
        except Exception as exc:
            print(f"[lowest_tier_watch] error: {exc}", flush=True)
            if args.once:
                return 1
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
