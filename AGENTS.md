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
