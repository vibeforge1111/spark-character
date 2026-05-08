# spark-character

The voice and character of Spark agents. Provider-agnostic. Versioned. Evolvable from real conversations.

## What this is

Spark is a personal-operator agent that runs on top of swappable LLMs (Z.AI, MiniMax, OpenAI, Anthropic, Ollama). Anthropic can bake Claude's voice into the model weights through character training; spark-character solves the same problem at inference time so the voice survives provider swaps.

Three things live here:

1. **Engine** that loads a personality chip from spark-personality-chip-labs and renders it into a system prompt with the model-agnostic invariants (no em dashes, no plumbing leaks, lead with the answer, never reset to a greeting, never fabricate live data, hold honest assessments under social pressure).
2. **Eight tiers of evaluation** (T1 through T8 plus T5 cross-provider) that score real LLM output on mechanics, voice signature, behavioral traits, multi-turn stability, emotional attunement, memory coherence, and initiative.
3. **Evolution loop** that mutates the persona, scores candidates against the rubric, refuses regressions, and ships winners. The loop reads real Spark Telegram conversations from SIB's audit log so the evolution targets failures users actually experience.

## Architecture in one diagram

```
spark-personality-chip-labs (canonical schema, OCEAN + emotional profile + triggers)
   founder-operator.personality.yaml
                     │
                     ▼
        spark-character (engine, this repo)
        ┌──────────────────────────────────────┐
        │ chip_loader     → render to system   │
        │ scorers (T1-T8) → measure replies    │
        │ overlays        → per-provider tunes │
        │ critic          → rewrite gate       │
        │ evolve_persona  → multi-tier mutator │
        │ audit_miner     → production failure │
        │   reading                            │
        │ memory_grounded → real user signal   │
        │ registry        → promote back to    │
        │   chip lab                           │
        │ auto_loop       → continuous daemon  │
        └──────────────────────┬───────────────┘
                               │
                               ▼
                spark-intelligence-builder (runtime)
                advisory.py loads chip via spark-character,
                applies overlay, calls generate() with tools.
                               │
                               ▼
                        Production Telegram
```

Full architectural detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
Open gaps and planned evolutions in [docs/ROADMAP.md](docs/ROADMAP.md).

## Install

Normal Spark users get this module through the starter installer:

```bash
spark setup
```

That installs `spark-character` with the rest of the Telegram starter stack so Builder can use the persona runtime without a separate manual clone.

For local development:

```bash
pip install -e .
# or, with dev tools:
pip install -e .[dev]
```

## Quick start

### Generate from the canonical chip

```python
from spark_character import (
    ProviderSpec,
    load_chip_by_id,
    persona_from_chip,
    generate,
)

chip = load_chip_by_id("founder-operator")
persona = persona_from_chip(chip)
provider = ProviderSpec.from_env()  # reads ZAI_API_KEY / ZAI_BASE_URL / ZAI_MODEL

result = generate(
    "Should I raise now or wait six months?",
    provider=provider,
    persona=persona,
)
print(result.final)
```

### Generate with critic-rewrite pass

```python
from spark_character import generate_with_critique
result = generate_with_critique(prompt, provider=provider, persona=persona)
# critic only fires when the local scorers flag a violation; only accepts
# rewrites that strictly improve the persona score.
```

### Score an arbitrary reply

```python
from spark_character import score_persona
score = score_persona(some_reply_text)
print(score.passed, score.mean, score.p2_hits)
```

### Run the full T1-T8 pulse against your live provider

```bash
python -u evals/full_pulse.py
```

## The eight tiers

| Tier | What it measures | How |
|------|------------------|-----|
| **T1** mechanics | em dash, plumbing leaks, reset greetings, hedge openers, voice heuristic | regex (free) |
| **T2** distinctiveness | does the reply sound like Spark vs a generic helper? | LLM judge against `voice_corpus/golden.json` + `foil.json` |
| **T3** behavioral | sycophancy resistance, disagreement, curiosity, care, identity, honesty, initiative | 7 single-turn probes, judge per trait |
| **T4** stability | identity / honesty / boundary held across multi-turn pressure | 5 multi-turn adversarial scenarios |
| **T5** cross-provider | does Spark sound like the same agent on Z.AI vs Codex vs MiniMax? | same-prompt comparison + same-agent judge |
| **T6** emotional attunement | does the reply engage with stated emotional state with substance? | 5 scenarios, trait-specific judges |
| **T7** memory coherence | does the agent act on facts the user stated earlier? | 3 multi-turn probes, optionally seeded from real `user_instructions` |
| **T8** initiative | does the agent surface implicit problems users buried in literal questions? | 3 probes, judge per scenario |

## Evolution

```bash
# Multi-tier evolution cycle, 3 candidates, weighted T1+T2+T3
python -u evals/evolve_persona.py --candidates 3

# Production-grounded: read SIB audit log to seed weaknesses from real failures
python -u evals/evolve_persona.py \
  --candidates 3 \
  --sib-home /path/to/spark-intelligence-builder/.tmp-home-live-telegram-real

# With deeper tiers (T6/T7/T8) included in fitness function
python -u evals/evolve_persona.py --include-deeper

# With chip-loaded scoring (eval simulates active domain chips)
python -u evals/evolve_persona.py --chip-load xcontent,startup-yc
```

A promoted candidate writes:
- `src/spark_character/artifacts/persona.v(N+1).md` — internal flat artifact
- `~/Desktop/spark-personality-chip-labs/personalities/founder-operator-evolved-v(N+1).personality.yaml` — sidecar back to the chip lab registry

## Continuous improvement

```bash
python -u evals/auto_loop.py \
  --sib-home /path/to/spark-intelligence-builder/.tmp-home-live-telegram-real \
  --interval-seconds 1800 \
  --new-replies-threshold 25 \
  --candidates 3 \
  --consumer-pythons "C:/Python313/python.exe,C:/Users/USER/.spark/tools/spark-cli-venv/Scripts/python.exe"
```

The daemon polls the SIB outbound log every 30 min. When 25+ new replies have been recorded, it fires an evolution cycle seeded from production failures. If a candidate beats the baseline composite without regressing on any axis by more than 0.05, it's promoted, and (optionally) the consumer Python interpreters are force-refreshed so the new persona reaches production on the next gateway boot.

## Cross-provider voice consistency

```bash
python -u evals/cross_provider.py --providers zai,codex,minimax --judge zai
```

Same prompt set, three backends, same persona. A "same-agent judge" rates each cross-backend pair on a 0-10 scale of "do these sound like the same agent." Real numbers from a recent run with overlays active:

| Provider | T1 | T2 | Notes |
|---|---|---|---|
| Z.AI (glm-5.1) | 0.97 | 0.82 | strongest baseline |
| Codex (gpt-5.5) | 1.00 | 0.83 | strongest distinctiveness |
| MiniMax (M2.7) | 0.97 | 0.62 | helper-register drift |

Same-agent pair scores: zai-codex 0.67, zai-minimax 0.68, codex-minimax 0.62.

## Live tail

Watch what Spark is shipping to users in real time with T1 flags inlined:

```bash
python -u evals/live_tail.py --sib-home <home>
```

## Repo layout

```
src/spark_character/
  artifacts/
    persona.v{N}.md              # evolvable flat persona (legacy)
    persona.latest.txt           # active version pointer
    critic.v1.md                 # critic-rewriter spec
    overlays/
      zai.md, minimax.md, codex.md   # per-provider voice tuning
    voice_corpus/
      golden.v1.json, foil.v1.json   # T2 judge reference
  chip_loader.py        # load + render personality chip yaml
  chip_context.py       # synthetic active-chip context for eval
  persona.py            # PersonaSpec + overlay resolution
  critic.py             # critic-rewriter pipeline
  pipeline.py           # generate, generate_with_critique
  provider.py           # OpenAI-compat HTTP (sync + async)
  codex_provider.py     # codex CLI wrapper
  scoring.py            # T1 scorers (pure functions)
  voice_judge.py        # T2 distinctiveness judge
  probes.py             # T3 single-turn behavioral probes
  stability.py          # T4 multi-turn adversarial scenarios
  deeper_probes.py      # T6/T7/T8 probes
  audit_miner.py        # read SIB outbound log + classify failures
  memory_grounded.py    # build T7 probes from real user_instructions
  registry.py           # promote evolved chips back to chip lab
  harness_adapter.py    # plug into spark-self-evolving-harness
evals/
  full_pulse.py         # T1+T2+T3+T4 + T6/T7/T8 sections
  cross_provider.py     # T5 multi-backend comparison
  evolve.py             # T1-only evolution (legacy)
  evolve_persona.py     # multi-tier production-grounded evolution
  auto_loop.py          # continuous improvement daemon
  live_tail.py          # stream Telegram replies with T1 flags
  live_pulse.py         # 12-prompt scorecard
docs/
  ARCHITECTURE.md       # the unified architecture in detail
  ROADMAP.md            # open gaps + planned evolutions
tests/
  25 unit tests, no network
```

## License

MIT. See [LICENSE](./LICENSE).

Spark Swarm is AGPL-licensed. Other Spark repos are MIT unless their
LICENSE file says otherwise. Spark Pro hosted services, private corpuses,
brand assets, deployment secrets, and Pro drops are not included in
open-source licenses. Pro drops do not grant redistribution rights unless
a separate written license says so.
