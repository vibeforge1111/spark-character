"""Continuous Telegram-grounded auto-improvement loop.

Watches a Spark Intelligence Builder home for new outbound replies. When
enough new live replies have accumulated, runs a multi-tier evolution
cycle seeded from summarized audit failures and ships the winner if one
beats the active baseline. Designed to keep Spark's voice improving from
observed failure classes without manual triggers.

Run continuously (e.g. on a long-running console or as a Windows
scheduled task pointing at a python -u shim):

    python -u evals/auto_loop.py \\
        --sib-home "C:/Users/USER/Desktop/spark-intelligence-builder/.tmp-home-live-telegram-real" \\
        --interval-seconds 1800 \\
        --new-replies-threshold 25 \\
        --candidates 3 \\
        --weights 0.2,0.5,0.3

A single check:

    python evals/auto_loop.py --once ...

The loop never deletes or rolls back. Each promoted candidate becomes
persona.v(N+1).md and persona.latest.txt is updated to point at it.
v1 is preserved for diff or manual rollback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import AuditMiner  # noqa: E402
from spark_character.persona import resolve_latest_persona_version  # noqa: E402


STATE_FILE_DEFAULT = Path("evals/_auto_loop_state.json")
HEARTBEAT_FILE_DEFAULT = Path("evals/_auto_loop_heartbeat.txt")


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {
            "last_evolved_at": 0,
            "last_audit_count": 0,
            "last_persona_version": None,
            "last_cycle_phase": "idle",
            "last_cycle_started_at": 0,
            "last_promoted_at": 0,
            "loop_starts": 0,
            "cycle_count": 0,
            "promotion_count": 0,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "last_evolved_at": 0,
            "last_audit_count": 0,
            "last_persona_version": None,
            "last_cycle_phase": "idle",
            "last_cycle_started_at": 0,
            "last_promoted_at": 0,
            "loop_starts": 0,
            "cycle_count": 0,
            "promotion_count": 0,
        }


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(state, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _acquire_instance_lock(state_path: Path):
    """Hold one process-wide lock beside the state file for this loop."""
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise RuntimeError(f"Another character auto-loop already owns {state_path.name}.") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    return handle


def _write_heartbeat(path: Path, phase: str) -> None:
    """Touch a heartbeat file with current epoch + phase. External
    monitors can watch the modtime and the contents to detect a hung
    daemon. Called before/after every loop iteration and around evolve
    subprocess invocations."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{int(time.time())} {phase}\n", encoding="utf-8")
    except Exception:
        pass


def count_llm_replies(sib_home: str) -> int:
    miner = AuditMiner.from_sib_home(sib_home)
    findings = miner.recent_findings(limit=10_000)
    return findings.llm_rows


def run_evolve_cycle(args, repo_root: Path) -> tuple[bool, bool, str]:
    """Run evolve_persona.py. Return (success, promoted, log_tail)."""
    cmd = [
        sys.executable, "-u",
        str(repo_root / "evals" / "evolve_persona.py"),
        "--candidates", str(args.candidates),
        "--weights", args.weights,
        "--sib-home", args.sib_home,
        "--audit-limit", str(args.audit_limit),
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"\n[auto_loop] firing evolve cycle: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root, timeout=args.evolve_timeout)
    log_tail = (result.stdout or "")[-4000:]
    if result.returncode != 0:
        print("[auto_loop] evolve subprocess returned non-zero")
        print(log_tail)
        return False, False, log_tail
    print(log_tail)
    promoted = "PROMOTED:" in result.stdout
    return True, promoted, log_tail


def maybe_refresh_consumers(args, repo_root: Path = _REPO_ROOT) -> bool:
    if not args.consumer_pythons:
        return True
    pythons = [p.strip() for p in args.consumer_pythons.split(",") if p.strip()]
    package_source = str(repo_root.resolve())
    all_refreshed = True
    for py in pythons:
        try:
            print(f"[auto_loop] refreshing consumer: {py}", flush=True)
            result = subprocess.run(
                [py, "-m", "pip", "install", "--upgrade", "--force-reinstall", "--no-deps", package_source, "-q"],
                check=False,
                timeout=180,
            )
            if result.returncode != 0:
                all_refreshed = False
                print(
                    f"[auto_loop] consumer refresh failed for {py} "
                    f"(exit {result.returncode}); refresh is not confirmed.",
                    flush=True,
                )
        except Exception as exc:
            all_refreshed = False
            print(f"[auto_loop] consumer refresh failed for {py}: {exc}")
    return all_refreshed


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {parsed}")
    return parsed


def _should_fire_cycle(state: dict, current_audit_count: int, threshold: int) -> bool:
    last_audit_count = int(state.get("last_audit_count", 0))
    cycle_count = int(state.get("cycle_count", 0))
    return current_audit_count - last_audit_count >= threshold or (
        last_audit_count == 0 and cycle_count == 0
    )


def _bounded_interval_seconds(requested: int) -> int:
    return max(60, requested)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sib-home", required=True, help="Spark Intelligence Builder home containing outbound audit evidence")
    parser.add_argument("--interval-seconds", type=int, default=1800, help="Seconds between checks; values below 60 are reported and clamped")
    parser.add_argument("--new-replies-threshold", type=int, default=25, help="New audited replies required before evolution")
    parser.add_argument("--candidates", type=int, default=3, help="Mutation candidates per evolution cycle")
    parser.add_argument("--weights", default="0.2,0.5,0.3", help="Composite T1,T2,T3 scoring weights")
    parser.add_argument("--audit-limit", type=_positive_int, default=200, help="Number of recent audit failures to seed each evolve cycle (must be a positive integer)")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate without promoting a persona")
    parser.add_argument("--once", action="store_true", help="Run a single check then exit")
    parser.add_argument("--state-file", default=str(STATE_FILE_DEFAULT), help="Persistent loop state JSON path")
    parser.add_argument("--heartbeat-file", default=str(HEARTBEAT_FILE_DEFAULT), help="Operator heartbeat path")
    parser.add_argument("--evolve-timeout", type=int, default=2400, help="Maximum evolution subprocess seconds")
    parser.add_argument(
        "--consumer-pythons",
        default="",
        help="Comma-separated python interpreters to force-refresh "
        "spark-character on after a promotion (e.g. system Python + spark-cli venv). "
        "Each one installs the exact local candidate tree being evaluated.",
    )
    args = parser.parse_args()

    state_path = Path(args.state_file)
    heartbeat_path = Path(args.heartbeat_file)
    repo_root = _REPO_ROOT
    _instance_lock = _acquire_instance_lock(state_path)

    # Bump loop_starts so external monitors can see daemon restarts
    boot_state = _load_state(state_path)
    if boot_state.get("last_cycle_phase") not in ("idle", "complete", None):
        print(
            f"[auto_loop] WARNING last cycle ended in phase "
            f"{boot_state.get('last_cycle_phase')!r} at "
            f"{boot_state.get('last_cycle_started_at')!r}. Likely killed mid-cycle.",
            flush=True,
        )
    boot_state["loop_starts"] = int(boot_state.get("loop_starts", 0)) + 1
    boot_state["last_cycle_phase"] = "idle"
    _save_state(state_path, boot_state)
    _write_heartbeat(heartbeat_path, "boot")

    while True:
        try:
            _write_heartbeat(heartbeat_path, "loop_check")
            state = _load_state(state_path)
            current = count_llm_replies(args.sib_home)
            new_replies = current - state.get("last_audit_count", 0)
            active = resolve_latest_persona_version()
            print(
                f"[auto_loop] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
                f"sib_home={args.sib_home} active={active} "
                f"audit_total={current} new_since_last={new_replies} "
                f"threshold={args.new_replies_threshold}",
                flush=True,
            )
            should_fire = _should_fire_cycle(state, current, args.new_replies_threshold)
            if should_fire:
                state["last_cycle_phase"] = "evolving"
                state["last_cycle_started_at"] = int(time.time())
                state["cycle_count"] = int(state.get("cycle_count", 0)) + 1
                _save_state(state_path, state)
                _write_heartbeat(heartbeat_path, "evolving")
                success, promoted, _log = run_evolve_cycle(args, repo_root)
                if not success:
                    state["last_cycle_phase"] = "failed"
                    state["last_cycle_error"] = "evolve subprocess returned non-zero"
                    _save_state(state_path, state)
                    _write_heartbeat(heartbeat_path, "evolve_failed")
                    if args.once:
                        return 1
                    time.sleep(60)
                    continue
                state["last_evolved_at"] = int(time.time())
                state["last_audit_count"] = current
                state["last_persona_version"] = resolve_latest_persona_version()
                state["last_cycle_phase"] = "complete"
                state.pop("last_cycle_error", None)
                _save_state(state_path, state)
                _write_heartbeat(heartbeat_path, "post_evolve")
                if promoted:
                    state["last_promoted_at"] = int(time.time())
                    state["promotion_count"] = int(state.get("promotion_count", 0)) + 1
                    _save_state(state_path, state)
                    print(f"[auto_loop] promoted to {state['last_persona_version']}")
                    _write_heartbeat(heartbeat_path, "refreshing_consumers")
                    if not maybe_refresh_consumers(args, repo_root):
                        state["last_cycle_phase"] = "consumer_refresh_failed"
                        state["last_cycle_error"] = "one or more consumer refreshes failed"
                        _save_state(state_path, state)
                        _write_heartbeat(heartbeat_path, "consumer_refresh_failed")
                        if args.once:
                            return 1
                else:
                    print("[auto_loop] no promotion this cycle")
            else:
                print("[auto_loop] threshold not met, skipping")
            if args.once:
                return 0
            actual_interval = _bounded_interval_seconds(args.interval_seconds)
            if actual_interval != args.interval_seconds:
                print(
                    f"[auto_loop] --interval-seconds={args.interval_seconds} clamped to 60",
                    flush=True,
                )
            time.sleep(actual_interval)
        except KeyboardInterrupt:
            print("[auto_loop] interrupted by operator")
            return 0
        except Exception as exc:
            print(f"[auto_loop] error: {exc}", flush=True)
            if args.once:
                return 1
            time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
