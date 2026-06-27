"""Live observer: watch Spark Telegram replies and surface recommendations.

This is the "plug Claude (or any meta-LLM) into the chat as an observer"
piece. It does NOT interfere with Spark's live replies. It tails the
gateway-outbound.jsonl audit log and, for each new LLM-generated reply,
runs a meta-observation LLM call that asks:

  - What landed well?
  - What missed or could be sharper?
  - What pattern is worth flagging across the recent set?
  - Specific evolution suggestion when warranted.

Observations are written to evals/_observations.jsonl with timestamp,
trace_ref, route, chip, the reply, the meta-LLM's structured analysis,
and a recommendation tier ("flag", "consider_evolution", "fire_now").

The observer LLM defaults to whichever provider you have configured
(Z.AI by default), but you can point it at any HTTP-compatible backend
via --observer-provider. The meta-prompt makes it score against the
T1-T8 + T13 axes and surface specific phrasing observations.

Pair with observations_digest.py to aggregate and print recommendations.

Usage:

    python -u evals/observer.py \\
      --sib-home C:/Users/.../tmp-home-live-telegram-real

    python -u evals/observer.py --once  # single pass over recent rows
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import ProviderSpec, call_provider  # noqa: E402
from spark_character.audit_miner import LLM_ROUTES, _detect_failures  # noqa: E402


OBSERVATIONS_FILE_DEFAULT = Path("evals/_observations.jsonl")
SEEN_FILE_DEFAULT = Path("evals/_observer_seen.json")
HEARTBEAT_FILE_DEFAULT = Path("evals/_observer_heartbeat.txt")


META_OBSERVER_SYSTEM = (
    "You are an observer for Spark, an AI agent. You see Spark's reply "
    "to its user and the user's preceding message. Your job is to "
    "evaluate the reply honestly and surface what should improve.\n\n"
    "Score the reply on these dimensions, each 0-10:\n"
    "- t1_mechanics: em dashes (-1 for any), plumbing leaks (-1 each), "
    "  greeting reset (-2), hedge opener (-1), verbose filler (-1)\n"
    "- t2_distinctiveness: does it sound like Spark (sharp friend "
    "  operator) or generic helper?\n"
    "- t3_behavioral: did it engage warmly with substance, push back "
    "  when warranted, lead with the answer?\n"
    "- t13_humane_depth: does it feel like someone the user wants to "
    "  keep talking to? Specific observations, callbacks, vulnerability "
    "  about own limits, warmth that survives directness?\n\n"
    "Then surface:\n"
    "- ONE specific thing that landed well\n"
    "- ONE specific thing that could be sharper, with a concrete "
    "  rewrite of the weakest sentence if applicable\n"
    "- ONE pattern worth flagging if you've seen the same thing before "
    "  in the recent set\n\n"
    "Output a single JSON object, no preamble, no code fence. Schema:\n"
    "{\n"
    "  \"scores\": {\"t1\": 8, \"t2\": 7, \"t3\": 9, \"t13\": 6},\n"
    "  \"landed_well\": \"...\",\n"
    "  \"could_be_sharper\": \"...\",\n"
    "  \"rewrite_suggestion\": \"...\" or null,\n"
    "  \"pattern\": \"...\" or null,\n"
    "  \"recommendation_tier\": \"flag\" | \"consider_evolution\" | \"fire_now\",\n"
    "  \"evolution_target\": \"t1\" | \"t2\" | \"t3\" | \"t13\" | null\n"
    "}\n\n"
    "Be terse. Be honest. Do not add commentary outside the JSON."
)


@dataclass(frozen=True)
class Observation:
    ts: int
    trace_ref: str
    route: str
    chip: str | None
    user_id: str | None
    reply_preview: str
    raw_observation: str
    parsed: dict


def _load_seen(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return set(data.get("seen_trace_refs", []))
    except Exception:
        return set()


def _save_seen(path: Path, seen: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cap stored IDs to last 5000 to bound state size
    capped = list(seen)[-5000:]
    path.write_text(
        json.dumps({"seen_trace_refs": capped}, indent=2),
        encoding="utf-8",
    )


def _append_observation(path: Path, obs: Observation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": obs.ts,
        "trace_ref": obs.trace_ref,
        "route": obs.route,
        "chip": obs.chip,
        "user_id": obs.user_id,
        "reply_preview": obs.reply_preview,
        "raw_observation": obs.raw_observation,
        **obs.parsed,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def _write_heartbeat(path: Path, phase: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{int(time.time())} {phase}\n", encoding="utf-8")
    except Exception:
        pass


def parse_observation_response(text: str) -> dict:
    """Extract the JSON object the meta-observer was asked to produce."""
    if not text:
        return {}
    raw = text.strip()
    if raw.startswith("```"):
        match = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
        if match:
            raw = match.group(1).strip()
    open_match = re.search(r"\{", raw)
    if not open_match:
        return {}
    try:
        return json.loads(raw[open_match.start():])
    except json.JSONDecodeError:
        depth = 0
        in_string = False
        escape = False
        start = open_match.start()
        for i in range(start, len(raw)):
            c = raw[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == """:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start:i + 1])
                    except json.JSONDecodeError:
                        return {}
    return {}


def observe_reply(
    *,
    provider: ProviderSpec,
    reply_text: str,
    user_message: str | None = None,
    route: str = "",
    chip: str | None = None,
) -> dict:
    user_prompt = (
        ("[User message that prompted the reply]\n"
         f"{user_message}\n\n" if user_message else "")
        + "[Spark's reply]\n"
        + f"{reply_text}\n\n"
        + (f"[Route]: {route}\n" if route else "")
        + (f"[Active chip]: {chip}\n" if chip else "")
        + "\nReturn the observation JSON object only."
    )
    raw = call_provider(
        provider=provider,
        system_prompt=META_OBSERVER_SYSTEM,
        user_prompt=user_prompt,
        max_tokens=400,
        temperature=0.4,
        disable_thinking=True,
    )
    parsed = parse_observation_response(raw)
    return {"raw": raw, "parsed": parsed}


def follow_audit_rows(
    log_path: Path,
    *,
    poll_seconds: float = 2.0,
    only_new: bool = True,
):
    """Generator yielding each new audit row appended to gateway-outbound.jsonl."""
    while not log_path.exists():
        time.sleep(poll_seconds)
    with log_path.open("r", encoding="utf-8") as f:
        if only_new:
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


def replay_recent_rows(log_path: Path, *, limit: int = 50) -> list[dict]:
    if not log_path.exists():
        return []
    rows: list[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sib-home", required=True)
    parser.add_argument("--observations-file", default=str(OBSERVATIONS_FILE_DEFAULT))
    parser.add_argument("--seen-file", default=str(SEEN_FILE_DEFAULT))
    parser.add_argument("--heartbeat-file", default=str(HEARTBEAT_FILE_DEFAULT))
    parser.add_argument("--once", action="store_true",
                        help="Process the last N rows and exit")
    parser.add_argument("--once-limit", type=int, default=10)
    parser.add_argument(
        "--observer-provider",
        default="zai",
        help="Provider for the meta-observer LLM. Defaults to Z.AI. "
        "Could also be openai or any HTTP-compat env-configured provider.",
    )
    parser.add_argument("--max-observations", type=int, default=0,
                        help="Stop after this many observations (0 = unbounded)")
    args = parser.parse_args()

    log_path = Path(args.sib_home) / "logs" / "gateway-outbound.jsonl"
    obs_path = Path(args.observations_file)
    seen_path = Path(args.seen_file)
    heartbeat_path = Path(args.heartbeat_file)

    # Resolve observer provider
    cfg = {
        "zai": ("ZAI_API_KEY", "ZAI_BASE_URL", "ZAI_MODEL",
                "https://api.z.ai/api/coding/paas/v4/", "glm-5.1"),
        "openai": ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_MODEL",
                   "https://api.openai.com/v1/", "gpt-4o-mini"),
        "minimax": ("MINIMAX_API_KEY", "MINIMAX_BASE_URL", "MINIMAX_MODEL",
                    "https://api.minimax.io/v1/", "MiniMax-M2.7"),
    }
    p_cfg = cfg.get(args.observer_provider.lower())
    if not p_cfg:
        print(f"[observer] unknown provider {args.observer_provider!r}")
        return 2
    api_key = os.environ.get(p_cfg[0])
    if not api_key:
        print(f"[observer] missing {p_cfg[0]} in env")
        return 2
    provider = ProviderSpec(
        base_url=os.environ.get(p_cfg[1], p_cfg[3]),
        model=os.environ.get(p_cfg[2], p_cfg[4]),
        api_key=api_key,
    )
    print(
        f"[observer] watching {log_path}\n"
        f"[observer] meta-LLM = {args.observer_provider} ({provider.model})\n"
        f"[observer] writing observations -> {obs_path}",
        flush=True,
    )
    _write_heartbeat(heartbeat_path, "boot")

    seen = _load_seen(seen_path)
    observed_count = 0

    def process_row(row: dict) -> None:
        nonlocal observed_count
        route = str(row.get("routing_decision") or "")
        if route not in LLM_ROUTES:
            return
        trace_ref = str(row.get("trace_ref") or "")
        if trace_ref and trace_ref in seen:
            return
        preview = (row.get("response_preview") or "").strip()
        if not preview or len(preview) < 60:
            return
        chip = row.get("active_chip_key")
        user_id = row.get("telegram_user_id")
        ts = int(time.time())
        _write_heartbeat(heartbeat_path, "observing")
        try:
            result = observe_reply(
                provider=provider,
                reply_text=preview,
                user_message=None,  # outbound log doesn't store inbound text
                route=route,
                chip=str(chip) if chip else None,
            )
        except Exception as exc:
            print(f"[observer] meta-LLM error: {exc}", flush=True)
            return
        parsed = result.get("parsed") or {}
        # Surface T1-detected mechanical issues from regex too
        regex_failures = _detect_failures(preview)
        if regex_failures:
            parsed.setdefault("regex_failures", [
                {"kind": k, "detail": d} for k, d in regex_failures
            ])
        obs = Observation(
            ts=ts, trace_ref=trace_ref, route=route,
            chip=str(chip) if chip else None,
            user_id=str(user_id) if user_id else None,
            reply_preview=preview,
            raw_observation=result.get("raw") or "",
            parsed=parsed,
        )
        _append_observation(obs_path, obs)
        observed_count += 1
        if trace_ref:
            seen.add(trace_ref)
        scores = parsed.get("scores", {})
        rec = parsed.get("recommendation_tier", "?")
        landed = (parsed.get("landed_well") or "")[:100]
        sharper = (parsed.get("could_be_sharper") or "")[:100]
        print(
            f"\n[observer] route={route} chip={chip} scores={scores} rec={rec}\n"
            f"  reply: {preview[:120]}\n"
            f"  landed: {landed}\n"
            f"  sharper: {sharper}",
            flush=True,
        )

    if args.once:
        recent = replay_recent_rows(log_path, limit=args.once_limit)
        print(f"[observer] one-pass over {len(recent)} recent rows", flush=True)
        for row in recent:
            process_row(row)
            if args.max_observations and observed_count >= args.max_observations:
                break
        _save_seen(seen_path, seen)
        return 0

    try:
        for row in follow_audit_rows(log_path, only_new=True):
            _write_heartbeat(heartbeat_path, "tailing")
            process_row(row)
            if args.max_observations and observed_count >= args.max_observations:
                print(f"[observer] hit max observations ({args.max_observations}), exiting", flush=True)
                break
            if observed_count and observed_count % 5 == 0:
                _save_seen(seen_path, seen)
    except KeyboardInterrupt:
        print("\n[observer] interrupted by operator", flush=True)
    _save_seen(seen_path, seen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
