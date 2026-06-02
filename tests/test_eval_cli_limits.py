from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_eval_script(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def test_eval_scripts_reject_non_positive_token_and_audit_limits() -> None:
    scripts = [
        ("evals/live_pulse.py", "--max-tokens", "0"),
        ("evals/full_pulse.py", "--max-tokens", "0"),
        ("evals/compare_personas.py", "--max-tokens", "0"),
        ("evals/auto_loop.py", "--sib-home", "unused", "--audit-limit", "0"),
        ("evals/mutate_chip_demo.py", "--audit-limit", "0"),
        ("evals/evolve_persona.py", "--audit-limit", "0"),
    ]

    for script, *args in scripts:
        flag_index = args.index("--max-tokens") if "--max-tokens" in args else args.index("--audit-limit")
        for invalid in ("0", "-1"):
            invalid_args = [*args]
            invalid_args[flag_index + 1] = invalid
            result = _run_eval_script(script, *invalid_args)

            assert result.returncode == 2, f"{script} unexpectedly accepted {invalid_args}: {result.stdout} {result.stderr}"
            assert "expected a positive integer" in result.stderr
