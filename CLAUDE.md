# Spark Character Agent Contract

This repo owns Spark's public character layer: persona artifacts, provider overlays, scoring, and character evolution gates. It does not own runtime memory, messaging ingress, mission execution, installer pins, or voice I/O transport.

## Ownership

- Own persona text, provider-safe overlays, prompt guards, output sanitizers, voice-quality scoring, and review-only character evolution artifacts.
- Treat `spark-voice-comms` as the owner of speech capture, playback, audio routing, and voice transport.
- Treat `spark-intelligence-builder` as the owner of runtime identity assembly, AOC, Route Confidence, authority, memory gates, and black-box traces.
- Treat `spark-telegram-bot` as a thin messaging adapter and relay, not a persona source of truth.
- Treat `spark-cli` as the installer and registry owner; do not edit registry pins from this repo.

## Privacy Boundaries

- Do not commit raw Telegram logs, raw transcripts, provider outputs, API keys, bot tokens, env values, private memory bodies, Builder black-box logs, or local Spark homes.
- Character evals may use redacted samples, synthetic fixtures, or metadata summaries. Private live samples must stay local and ignored.
- Generated candidates from private conversations are review-only until redacted, scored, and explicitly promoted.
- Persona artifacts and overlays should stay public-safe. If a candidate requires private context to understand, it does not belong in the public artifact.

## Change Rules

- Keep changes small and evidence-backed. Do not rewrite persona, scorer, provider, and evolution logic in one commit unless the user explicitly asks for a coordinated release.
- Preserve existing style, package layout, and public APIs unless a test proves the current behavior is unsafe.
- Prefer typed, inspectable contracts over hidden prompt magic.
- Do not add background daemons, network calls, or live provider evals to normal tests.
- Live provider scripts must remain opt-in and must read credentials only from the user's local secret layer.

## Verification

- Run `python -m pytest -q` for release-facing changes.
- Run `python -m compileall src tests` when touching package code, tests, or artifacts loaded by package code.
- For persona or overlay changes, include the smallest relevant scorer/eval evidence and state whether it is synthetic, redacted, or live.
- For public release curation, verify that `README.md`, `pyproject.toml`, `spark.toml`, and package data agree before requesting a registry pin.

## Release Discipline

- Branch from the current remote `master` for release curation.
- Commit only coherent, verified changes.
- Push normally; never force-push or rewrite history.
- After a release commit lands, update installer pins through `spark-cli` only.
- If a local worktree has unrelated dirty changes, preserve them and replay the intended patch onto a clean branch.

<!-- SPARK FLEET STANDARD BLOCK v1 — canonical source: spark-compete/fleet/AGENT_GUIDE.md.
     This same block is mirrored into every repo's AGENTS.md and CLAUDE.md. Keep in sync. -->
## How agents work in this repo (Claude, Codex, Gemini — every LLM)

Many agents and sessions work these repos at the same time. There is a tiny **automatic**
workflow that keeps you from colliding. **There are no human-review steps — CI is the only
gate, and it is automatic.** This is coordination, not bureaucracy: claim, work, PR.

### Start of work — one command, then just work normally
```
python3 ~/spark-compete/scripts/fleet.py claim <this-repo-path> <area> <task>
```
You get your **own private worktree + branch + a lease** on `<area>`, so no other agent
edits the same files. It prints the folder to `cd` into. Work there and commit as usual —
a pre-commit hook **auto-checks and renews your lease**; you never manage it by hand.

- `fleet board` — see who's working on what, right now
- `fleet handoff <agent> --note "..."` — pass your work to another agent (with context)
- `fleet release --here` — done (frees the area + removes the worktree)

### Landing work — fully automatic, no human approval
1. Open a PR to the default branch.
2. **CI is the gate.** When it's green, the PR merges. No human reviews anything.
3. Never push directly to the protected branch; never commit from the shared checkout —
   always from your worktree.

### The rules (enforced by CI, not by people)
Full ruleset: **`spark-cli/docs/harness-discipline/`** — `01_RULESET.md` (7 Prime
Directives · Red Lines RL-01..21 · Rules R-01..28) and `07_FLEET_DISCIPLINE.md` (this
workflow). The day-to-day essentials:
- A real fix targets the **root cause**, not a symptom (R-05).
- No regex / keyword / canned answer **owns authority** — it is evidence only (RL-01).
- A failure **surfaces** with a clear reason; it never becomes a fake success (RL-08).
- One worktree per task; PRs only; nothing bypasses the CI gate (F-01 / F-09).

That's the whole contract. The system handles coordination and the gate for you —
automatically, with no human in the loop.
<!-- END SPARK FLEET STANDARD BLOCK v1 -->
