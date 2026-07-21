"""Live tail of Spark Telegram replies with T1 failure flagging.

Watches a Spark Intelligence Builder home's gateway-outbound.jsonl
file. Each new outbound reply is parsed, classified, and printed
with any T1 failures inlined. Useful for spot-checking what Spark
is actually shipping to users in real time.

Usage:

    python -u evals/live_tail.py --sib-home <home>

Press Ctrl+C to exit.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character.audit_miner import LLM_ROUTES, _detect_failures  # noqa: E402


def follow_jsonl(path: Path, *, poll_seconds: float = 1.0):
    """Generator: yield each new JSON line appended to a file."""
    while not path.exists():
        time.sleep(poll_seconds)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, 2)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_seconds)
                continue
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def format_row(row: dict) -> str:
    route = str(row.get("routing_decision") or "")
    chip = str(row.get("active_chip_key") or "none")
    user = str(row.get("telegram_user_id") or "?")
    preview = str(row.get("response_preview") or "").strip()
    at = str(row.get("recorded_at") or "")
    flags: list[str] = []
    if route in LLM_ROUTES:
        for kind, detail in _detect_failures(preview):
            flags.append(f"{kind.upper()}({detail[:40]})")
    flag_str = " ".join(flags) if flags else "clean"
    return (
        f"\n[{at}] user={user} route={route} chip={chip}"
        f"\n  flags: {flag_str}"
        f"\n  preview: {preview[:200]}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sib-home", required=True)
    parser.add_argument("--include-non-llm", action="store_true",
                        help="Also print non-LLM routes (memory observations, mission control, "
                        "fallback shapers). By default these are skipped.")
    args = parser.parse_args()

    log = Path(args.sib_home).expanduser() / "logs" / "gateway-outbound.jsonl"
    print(f"[live_tail] watching {log}", flush=True)
    if not log.exists():
        print(f"[live_tail] file does not exist yet, waiting...", flush=True)
    try:
        for row in follow_jsonl(log):
            route = str(row.get("routing_decision") or "")
            if not args.include_non_llm and route not in LLM_ROUTES:
                continue
            print(format_row(row), flush=True)
    except KeyboardInterrupt:
        print("\n[live_tail] interrupted by operator", flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
