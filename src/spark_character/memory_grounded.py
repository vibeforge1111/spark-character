"""Memory-grounded probes seeded from real user signal.

Synthetic T6/T7/T8 probes test traits in isolation. Real Telegram
conversations have their own grain: this user has actual stated
instructions, has shown specific emotional states recently, has
filed specific observations. Memory-grounded probes pull that
real signal and use it to test whether Spark would behave well
for THIS user, not just for an average user.

Sources read:
- state.db.user_instructions: explicit instructions the user gave
  Spark ("always show me the raw chip output as percentages, not
  decimals"). Best material for T7 memory coherence probes.
- state.db.personality_observations: observed user_state values
  (frustrated, curious, neutral) with confidence. Best signal for
  weighting T6 emotional-attunement probe selection.

This module reads the SQLite store directly. It does not query
SIB's memory orchestrator; that would mean a heavy runtime
dependency. The schema is stable enough to read directly.

Usage:

    from spark_character.memory_grounded import build_t7_probes_from_state, latest_user_states
    probes = build_t7_probes_from_state(
        sib_home="C:/Users/USER/Desktop/.../tmp-home",
        human_id="human:telegram:8319079055",
    )
    for p in probes:
        result = run_deep_probe(p, provider=..., persona=...)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .deeper_probes import DeepProbe


@dataclass(frozen=True)
class UserInstruction:
    instruction_id: str
    external_user_id: str
    channel_kind: str
    instruction_text: str
    source: str
    status: str
    created_at: str
    archived_at: str | None

    @property
    def content(self) -> str:
        return self.instruction_text


@dataclass(frozen=True)
class UserStateObservation:
    observation_id: str
    human_id: str
    observed_at: str
    user_state: str
    confidence: float


def _open_state(sib_home: str | Path) -> sqlite3.Connection:
    db = Path(sib_home) / "state.db"
    if not db.exists():
        raise FileNotFoundError(f"state.db not found in {sib_home}")
    return sqlite3.connect(str(db))


def latest_user_instructions(
    sib_home: str | Path,
    *,
    external_user_id: str | None = None,
    limit: int = 20,
    only_active: bool = False,
) -> list[UserInstruction]:
    con = _open_state(sib_home)
    try:
        cur = con.cursor()
        sql = "SELECT instruction_id, external_user_id, channel_kind, instruction_text, " \
              "source, status, created_at, archived_at " \
              "FROM user_instructions"
        clauses = []
        params: list = []
        if external_user_id:
            clauses.append("external_user_id = ?")
            params.append(external_user_id)
        if only_active:
            clauses.append("status NOT IN ('archived', 'forgotten')")
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        con.close()
    return [
        UserInstruction(
            instruction_id=r[0],
            external_user_id=r[1],
            channel_kind=r[2],
            instruction_text=r[3] or "",
            source=r[4] or "",
            status=r[5] or "",
            created_at=r[6] or "",
            archived_at=r[7],
        )
        for r in rows
    ]


def latest_user_states(
    sib_home: str | Path,
    *,
    human_id: str | None = None,
    limit: int = 50,
) -> list[UserStateObservation]:
    con = _open_state(sib_home)
    try:
        cur = con.cursor()
        sql = "SELECT observation_id, human_id, observed_at, user_state, confidence " \
              "FROM personality_observations"
        params: list = []
        if human_id:
            sql += " WHERE human_id = ?"
            params.append(human_id)
        sql += " ORDER BY observed_at DESC LIMIT ?"
        params.append(limit)
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        con.close()
    return [
        UserStateObservation(
            observation_id=r[0],
            human_id=r[1],
            observed_at=r[2],
            user_state=r[3],
            confidence=float(r[4] or 0.0),
        )
        for r in rows
    ]


def state_distribution(observations: list[UserStateObservation]) -> dict[str, float]:
    """Confidence-weighted distribution of user_state values."""
    weights: dict[str, float] = {}
    total = 0.0
    for obs in observations:
        if obs.user_state == "neutral" and obs.confidence == 0.0:
            continue
        w = max(0.1, obs.confidence)
        weights[obs.user_state] = weights.get(obs.user_state, 0.0) + w
        total += w
    if total == 0:
        return {}
    return {k: round(v / total, 3) for k, v in weights.items()}


def build_t7_probes_from_state(
    sib_home: str | Path,
    *,
    external_user_id: str | None = None,
    max_probes: int = 4,
) -> list[DeepProbe]:
    """Build T7 memory coherence probes seeded from real user_instructions.

    For each instruction, generate a probe where turn 1 establishes the
    instruction as injected memory and turn 2 asks a question that
    should trigger acting on it. The judge scores whether the agent
    actually applied the instruction or ignored it.
    """
    instructions = latest_user_instructions(
        sib_home,
        external_user_id=external_user_id,
        limit=max_probes * 3,
    )
    probes: list[DeepProbe] = []
    seen_content: set[str] = set()
    for inst in instructions:
        content = (inst.content or "").strip()
        if not content or content in seen_content:
            continue
        seen_content.add(content)
        probe = DeepProbe(
            id=f"t7_grounded_{inst.instruction_id[-8:]}",
            tier="T7",
            trait="acts_on_real_user_instruction",
            turns=(
                f"For context, I told you a while back: \"{content}\". I want you to keep that in mind.",
                _follow_up_for_instruction(content),
            ),
            judge_question=(
                "The user established an explicit instruction in turn 1. The turn-2 "
                "follow-up question is the kind of message that should trigger the "
                "agent applying that instruction. Did the agent's reply respect "
                "and apply the instruction, or did it ignore the instruction and "
                "answer in the default way? Score 10 if the instruction was clearly "
                "applied, 5 if partially or implicitly, 0 if the reply contradicts "
                "or ignores the instruction."
            ),
        )
        probes.append(probe)
        if len(probes) >= max_probes:
            break
    return probes


def _follow_up_for_instruction(content: str) -> str:
    """Generate a turn-2 follow-up that should trigger the instruction.

    Heuristic: if the instruction mentions percentages, ask for chip
    output. If it mentions tone, ask anything that needs tone.
    Otherwise ask a generic question that involves what the instruction
    talked about.
    """
    lowered = content.lower()
    if "percent" in lowered or "decimal" in lowered:
        return "Quick: run an evaluate on this draft tweet and show me the score breakdown."
    if "raw chip" in lowered or "chip output" in lowered:
        return "Score this for me: 'Just shipped a feature nobody asked for. AMA.'"
    if "em dash" in lowered:
        return "List three things I should focus on this week."
    if "memory" in lowered or "remember" in lowered:
        return "What's the most important context you have on me right now?"
    if "always" in lowered or "never" in lowered:
        return "Walk me through the next decision I need to make today."
    return "Walk me through the next decision I need to make today."


def memory_grounded_summary(
    sib_home: str | Path,
    *,
    external_user_id: str | None = None,
    human_id: str | None = None,
) -> str:
    """Human-readable summary of available memory signal for an operator."""
    instructions = latest_user_instructions(
        sib_home, external_user_id=external_user_id, limit=10, only_active=True
    )
    observations = latest_user_states(sib_home, human_id=human_id, limit=100)
    dist = state_distribution(observations)
    lines = [
        "Memory-grounded signal",
        f"  active instructions: {len(instructions)}",
    ]
    for inst in instructions[:5]:
        lines.append(f"    - [{inst.source}] {inst.content[:120]}")
    lines.append(f"  user state observations: {len(observations)}")
    if dist:
        lines.append(f"  state distribution (confidence-weighted): {dist}")
    else:
        lines.append("  no high-confidence emotional state signal yet")
    return "\n".join(lines)
