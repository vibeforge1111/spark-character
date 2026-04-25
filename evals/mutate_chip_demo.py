"""Demo: chip-aware trait mutation against a real chip.

Pulls weaknesses from the audit miner (or synthetic), runs the LLM
trait mutator against the founder-operator chip, writes the result
as a real chip yaml back to spark-personality-chip-labs.

Use this to validate the trait_mutator pipeline end-to-end without
running a full evolution cycle.

Usage:
    python -u evals/mutate_chip_demo.py --base founder-operator
    python -u evals/mutate_chip_demo.py --base founder-operator --sib-home <home>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from spark_character import (  # noqa: E402
    AuditMiner,
    ProviderSpec,
    load_chip_by_id,
    mutate_trait_values,
    promote_evolved_chip_to_chip_lab,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="founder-operator")
    parser.add_argument("--sib-home", default=None)
    parser.add_argument("--audit-limit", type=int, default=200)
    parser.add_argument("--label", default="trait-demo")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    provider = ProviderSpec.from_env()
    chip = load_chip_by_id(args.base)
    print(f"\nbase chip: {chip.id} ({chip.name})")
    print(f"  OCEAN: O={chip.openness} C={chip.conscientiousness} "
          f"E={chip.extraversion} A={chip.agreeableness} N={chip.neuroticism}")
    print(f"  empathy_style: {chip.empathy_style}")
    print(f"  emotional_range: {chip.emotional_range}")
    print()

    weaknesses: list[str] = []
    if args.sib_home:
        miner = AuditMiner.from_sib_home(args.sib_home)
        findings = miner.recent_findings(limit=args.audit_limit)
        print(findings.summary())
        print()
        weaknesses = findings.diagnose_lines(max_per_kind=2)

    if not weaknesses:
        weaknesses = [
            "T2 distinctiveness: replies sometimes drift into helper register",
            "T3 sycophancy_resistance: agent capitulates after sustained user pressure",
            "T6 anxiety_engagement: agent acknowledges anxiety briefly then pivots to checklist",
        ]

    print("Weaknesses fed to trait mutator:")
    for w in weaknesses:
        print(f"  - {w}")
    print()

    print("Calling LLM trait mutator...")
    result = mutate_trait_values(
        provider=provider,
        chip=chip,
        weaknesses=weaknesses,
    )
    print()
    print(f"reasoning: {result.reasoning}")
    print(f"OCEAN deltas: {result.deltas}")
    print(f"emotional_profile deltas: {result.emotional_profile_deltas}")
    print(f"emotional_range deltas: {result.emotional_range_deltas}")
    print()
    print(f"new chip: {result.chip.id}")
    print(f"  OCEAN: O={result.chip.openness} C={result.chip.conscientiousness} "
          f"E={result.chip.extraversion} A={result.chip.agreeableness} N={result.chip.neuroticism}")
    print(f"  emotional_range: {result.chip.emotional_range}")
    print()

    if args.dry_run:
        print("(dry-run, not promoting)")
        return 0

    written = promote_evolved_chip_to_chip_lab(
        chip=result.chip,
        base_chip_id=args.base,
        base_persona_version="v0",
        new_persona_version=args.label,
        voice_rules_override=None,
        delta_summary={
            "ocean": result.deltas,
            "emotional_profile": result.emotional_profile_deltas,
            "emotional_range": result.emotional_range_deltas,
            "reasoning": result.reasoning,
        },
    )
    if written:
        print(f"PROMOTED to chip lab: {written}")
    else:
        print("Chip lab not found locally, skipped promotion")
    return 0


if __name__ == "__main__":
    sys.exit(main())
