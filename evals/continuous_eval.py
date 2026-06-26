"""Continuous evaluation daemon.

Different from auto_loop. auto_loop fires evolution cycles when
summarized audit failures accumulate. This runs the rubric periodically and
logs scores over time, so you can see whether voice quality is
trending up, holding, or regressing.

Tiered cadence:

- Fast tier (every --fast-interval seconds, default 1800s = 30 min):
    T1 mechanics + T2 distinctiveness on the standard prompt set.
    ~12 LLM calls per run.

- Full tier (every --full-interval seconds, default 21600s = 6 hours):
    Everything: T1+T2+T3+T4+T6+T7+T8+T9 via full_pulse semantics.
    ~50 LLM calls per run.

Each run appends one JSONL line to _score_history.jsonl with timestamp,
persona version, model, every tier mean, and per-probe scores when
applicable. score_trend.py reads that file to show trends.

Regression detection: each run compares against a rolling baseline of
the last N successful runs. When any tier mean drops > --regression-
threshold below the baseline, a REGRESSION line is logged with the
specific axis and the delta.

Usage:

    python -u evals/continuous_eval.py
    python -u evals/continuous_eval.py --fast-interval 1200 --full-interval 14400
    python -u evals/continuous_eval.py --once  # run one fast + one full and exit

Heartbeat file: evals/_continuous_eval_heartbeat.txt (epoch + phase),
similar to auto_loop's. External monitors can watch the modtime.

Soft-fails: any single eval error is logged and the loop continues.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import logging
from pathlib import Path
from statistics import mean as mean_

_logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

import os

from spark_character import (  # noqa: E402
    PROBES,
    STABILITY_SCENARIOS,
    T6_EMOTIONAL_ATTUNEMENT_PROBES,
    T7_MEMORY_COHERENCE_PROBES,
    T8_INITIATIVE_PROBES,
    T9_AESTHETIC_FINGERPRINT_PROBES,
    T11_SUSTAINED_ATTACK_SCENARIOS,
    T13_HUMANE_DEPTH_PROBES,
    T14_MEMORABILITY_PROBES,
    AuditMiner,
    ProviderSpec,
    generate,
    load_persona,
    run_deep_probe,
    run_probe,
    run_stability_scenario,
    score_distinctiveness,
    score_persona,
)


PROVIDER_DEFAULTS = {
    "zai": {
        "api_key_env": "ZAI_API_KEY",
        "base_url_env": "ZAI_BASE_URL",
        "model_env": "ZAI_MODEL",
        "default_base": "https://api.z.ai/api/coding/paas/v4/",
        "default_model": "glm-5.1",
    },
    "minimax": {
        "api_key_env": "MINIMAX_API_KEY",
        "base_url_env": "MINIMAX_BASE_URL",
        "model_env": "MINIMAX_MODEL",
        "default_base": "https://api.minimax.io/v1/",
        "default_model": "MiniMax-M2.7",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "model_env": "OPENAI_MODEL",
        "default_base": "https://api.openai.com/v1/",
        "default_model": "gpt-4o-mini",
    },
}


def resolve_provider(name: str) -> ProviderSpec | None:
    cfg = PROVIDER_DEFAULTS.get(name.lower().strip())
    if not cfg:
        return None
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        return None
    return ProviderSpec(
        base_url=os.environ.get(cfg["base_url_env"], cfg["default_base"]),
        model=os.environ.get(cfg["model_env"], cfg["default_model"]),
        api_key=api_key,
    )


HISTORY_FILE_DEFAULT = Path("evals/_score_history.jsonl")
HEARTBEAT_FILE_DEFAULT = Path("evals/_continuous_eval_heartbeat.txt")


FAST_PROMPTS = [
    ("warm_decision",  "should I push the launch announcement today or wait until the team's back from the offsite?"),
    ("factual",        "What does TVL mean in DeFi?"),
    ("anxious",        "I'm anxious about the launch tomorrow."),
    ("status",         "where are we"),
    ("quick_call",     "Quick gut check: ship the redesign or hold a week?"),
    ("memory_state",   "What can you actually help me with right now?"),
]


def _write_heartbeat(path: Path, phase: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{int(time.time())} {phase}\n", encoding="utf-8")
    except Exception:
        pass


def _append_history(history_path: Path, row: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _load_history(history_path: Path, *, limit: int = 100) -> list[dict]:
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


def _compute_baseline(history: list[dict], *, axis: str, last_n: int = 5) -> float | None:
    """Rolling mean over the last `last_n` runs that have this axis."""
    values = [r.get(axis) for r in history[-last_n:] if isinstance(r.get(axis), (int, float))]
    if not values:
        return None
    return round(mean_(values), 3)


def run_fast_eval(provider: ProviderSpec, persona) -> dict:
    """T1 + T2 on the fast prompt set."""
    rows = []
    for label, prompt in FAST_PROMPTS:
        try:
            result = generate(prompt, provider=provider, persona=persona)
            t1 = score_persona(result.final)
            try:
                t2 = score_distinctiveness(result.final, provider=provider)
                t2_score = t2.score
            except Exception:
                t2_score = None
            rows.append({"label": label, "t1_mean": t1.mean, "t2": t2_score})
        except Exception as exc:
            rows.append({"label": label, "error": str(exc)})

    t1_means = [r["t1_mean"] for r in rows if "t1_mean" in r]
    t2_scores = [r["t2"] for r in rows if r.get("t2") is not None]
    return {
        "tier": "fast",
        "t1_mean": round(mean_(t1_means), 3) if t1_means else 0.0,
        "t2_mean": round(mean_(t2_scores), 3) if t2_scores else 0.0,
        "rows": rows,
    }


def run_full_eval(provider: ProviderSpec, persona, *, include_sustained: bool = False) -> dict:
    """T1+T2+T3+T4+T6+T7+T8+T9+T13+T14 in one pass; T11 if include_sustained."""
    fast = run_fast_eval(provider, persona)

    t3_scores: list[float] = []
    for probe in PROBES:
        try:
            r = run_probe(probe, provider=provider, persona=persona)
            t3_scores.append(r.score)
        except Exception:
            pass

    t4_scores: list[float] = []
    for scenario in STABILITY_SCENARIOS:
        try:
            r = run_stability_scenario(scenario, provider=provider, persona=persona)
            t4_scores.append(r.score)
        except Exception:
            pass

    t6_scores: list[float] = []
    for probe in T6_EMOTIONAL_ATTUNEMENT_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona)
            t6_scores.append(r.score)
        except Exception:
            pass

    t7_scores: list[float] = []
    for probe in T7_MEMORY_COHERENCE_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona)
            t7_scores.append(r.score)
        except Exception:
            pass

    t8_scores: list[float] = []
    for probe in T8_INITIATIVE_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona)
            t8_scores.append(r.score)
        except Exception:
            pass

    t9_scores: list[float] = []
    for probe in T9_AESTHETIC_FINGERPRINT_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona)
            t9_scores.append(r.score)
        except Exception:
            pass

    t13_scores: list[float] = []
    for probe in T13_HUMANE_DEPTH_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona)
            t13_scores.append(r.score)
        except Exception:
            pass

    t14_scores: list[float] = []
    for probe in T14_MEMORABILITY_PROBES:
        try:
            r = run_deep_probe(probe, provider=provider, persona=persona)
            t14_scores.append(r.score)
        except Exception:
            pass

    t11_scores: list[float] = []
    if include_sustained:
        for scenario in T11_SUSTAINED_ATTACK_SCENARIOS:
            try:
                r = run_stability_scenario(scenario, provider=provider, persona=persona)
                t11_scores.append(r.score)
            except Exception:
                pass

    out = {
        **fast,
        "tier": "full",
        "t3_mean": round(mean_(t3_scores), 3) if t3_scores else 0.0,
        "t4_mean": round(mean_(t4_scores), 3) if t4_scores else 0.0,
        "t6_mean": round(mean_(t6_scores), 3) if t6_scores else 0.0,
        "t7_mean": round(mean_(t7_scores), 3) if t7_scores else 0.0,
        "t8_mean": round(mean_(t8_scores), 3) if t8_scores else 0.0,
        "t9_mean": round(mean_(t9_scores), 3) if t9_scores else 0.0,
        "t13_mean": round(mean_(t13_scores), 3) if t13_scores else 0.0,
        "t14_mean": round(mean_(t14_scores), 3) if t14_scores else 0.0,
    }
    if include_sustained:
        out["t11_mean"] = round(mean_(t11_scores), 3) if t11_scores else 0.0
    return out


def detect_regressions(history: list[dict], current: dict, *, threshold: float = 0.10) -> list[str]:
    out: list[str] = []
    for axis in ("t1_mean", "t2_mean", "t3_mean", "t4_mean", "t6_mean", "t7_mean", "t8_mean", "t9_mean", "t11_mean", "t13_mean", "t14_mean"):
        if axis not in current:
            continue
        baseline = _compute_baseline(history, axis=axis, last_n=5)
        if baseline is None:
            continue
        delta = baseline - float(current[axis])
        if delta > threshold:
            out.append(f"REGRESSION {axis}: {current[axis]} (baseline {baseline}, delta -{round(delta, 3)})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-interval", type=int, default=1800)
    parser.add_argument("--full-interval", type=int, default=21600)
    parser.add_argument("--once", action="store_true",
                        help="Run one fast + one full and exit. Useful for cron.")
    parser.add_argument("--regression-threshold", type=float, default=0.10)
    parser.add_argument("--history-file", default=str(HISTORY_FILE_DEFAULT))
    parser.add_argument("--heartbeat-file", default=str(HEARTBEAT_FILE_DEFAULT))
    parser.add_argument(
        "--providers",
        default="zai",
        help="Comma-separated provider names to rotate through "
        "(zai,codex,minimax,openai). Each fast/full cycle uses the "
        "next provider in the list, so cross-provider drift gets "
        "exercised continuously without a separate run.",
    )
    parser.add_argument(
        "--sib-home",
        default=None,
        help="Path to a Spark Intelligence Builder home. When set, "
        "each cycle also reports current summarized T1 failure counts "
        "from the audit miner alongside the eval scores.",
    )
    parser.add_argument(
        "--include-sustained",
        action="store_true",
        help="Include T11 sustained-attack scenarios in full evals "
        "(6-7 turns each, ~3x cost of a normal full eval). "
        "Recommended for daily/weekly cycles, not hourly.",
    )
    args = parser.parse_args()

    history_path = Path(args.history_file)
    heartbeat_path = Path(args.heartbeat_file)

    provider_names = [p.strip() for p in args.providers.split(",") if p.strip()]
    providers: list[tuple[str, ProviderSpec]] = []
    for name in provider_names:
        spec = resolve_provider(name)
        if spec is None:
            print(f"[continuous_eval] skipping {name!r}: API key not set in env", flush=True)
            continue
        providers.append((name, spec))
    if not providers:
        print("[continuous_eval] no providers resolved. Set at least one API key.", flush=True)
        return 2

    persona = load_persona()
    print(
        f"[continuous_eval] starting | persona={persona.version} "
        f"providers={[n for n, _ in providers]} "
        f"fast_interval={args.fast_interval}s full_interval={args.full_interval}s",
        flush=True,
    )
    _write_heartbeat(heartbeat_path, "boot")

    last_full = 0.0
    cycle_idx = 0

    while True:
        try:
            now = time.time()
            run_full = (now - last_full) >= args.full_interval
            phase = "full_eval" if run_full else "fast_eval"
            # Rotate through configured providers per cycle
            provider_name, provider = providers[cycle_idx % len(providers)]
            cycle_idx += 1
            _write_heartbeat(heartbeat_path, f"{phase}:{provider_name}")
            t0 = time.time()
            print(
                f"\n[continuous_eval] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"starting {phase} on {provider_name} ({provider.model})",
                flush=True,
            )
            try:
                # Provider-specific persona (chip + provider overlay) for honest scoring
                persona_for_run = load_persona(provider_kind=provider_name)
                result = (
                    run_full_eval(provider, persona_for_run, include_sustained=args.include_sustained)
                    if run_full
                    else run_fast_eval(provider, persona_for_run)
                )
                if run_full:
                    last_full = now
            except Exception as exc:
                print(f"[continuous_eval] eval error on {provider_name}: {exc}", flush=True)
                _logger.exception("eval error on %s", provider_name)
                _write_heartbeat(heartbeat_path, "error")
                if args.once:
                    return 1
                time.sleep(min(60, args.fast_interval))
                continue
            dt = time.time() - t0
            row = {
                "ts": int(now),
                "persona_version": persona_for_run.version,
                "provider": provider_name,
                "model": provider.model,
                "duration_s": round(dt, 1),
                **{k: v for k, v in result.items() if k != "rows"},
            }
            # Add production audit signal if sib_home is configured
            if args.sib_home:
                try:
                    miner = AuditMiner.from_sib_home(args.sib_home)
                    findings = miner.recent_findings(limit=200)
                    row["production_audit"] = {
                        "rows_scanned": findings.rows_scanned,
                        "llm_rows": findings.llm_rows,
                        "failures_by_kind": dict(findings.failures_by_kind),
                    }
                except Exception:
                    pass
            history = _load_history(history_path)
            # Per-provider baseline so regressions reflect drift on the
            # same backend, not noise from cross-provider differences
            same_provider_history = [r for r in history if r.get("provider") == provider_name]
            regressions = detect_regressions(same_provider_history, row, threshold=args.regression_threshold)
            if regressions:
                row["regressions"] = regressions
            _append_history(history_path, row)
            scorecard = " ".join(
                f"{k}={row[k]}"
                for k in ("t1_mean", "t2_mean", "t3_mean", "t4_mean", "t6_mean", "t7_mean", "t8_mean", "t9_mean", "t11_mean", "t13_mean", "t14_mean")
                if k in row
            )
            print(f"[continuous_eval] {phase} done in {dt:.1f}s :: {scorecard}", flush=True)
            for r in regressions:
                print(f"[continuous_eval] {r}", flush=True)
            if args.once:
                # one fast + one full if we just did fast; else done
                if not run_full:
                    last_full = 0  # force the next loop iteration to run full
                    continue
                return 0
            sleep_seconds = max(60, args.fast_interval)
            _write_heartbeat(heartbeat_path, "sleeping")
            time.sleep(sleep_seconds)
        except KeyboardInterrupt:
            print("[continuous_eval] interrupted by operator", flush=True)
            return 0
        except Exception as exc:
            print(f"[continuous_eval] outer error: {exc}", flush=True)
            _logger.exception("outer eval loop error")
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
