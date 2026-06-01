"""Smoke test: fire_evolution handles subprocess.TimeoutExpired gracefully.

Verifies that when the evolve subprocess times out, fire_evolution returns
(False, "[timeout] ...") instead of crashing or hanging.

Usage:
    python -m pytest tests/test_fire_evolution_timeout.py -v
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest


def test_fire_evolution_returns_safe_failure_on_timeout(tmp_path: Path) -> None:
    """When subprocess.run raises TimeoutExpired, fire_evolution must:
    1. NOT crash (no unhandled exception)
    2. Return (False, "[timeout] ...") — a safe failure tuple
    3. Log a clear timeout message
    """
    # Import the module under test
    from evals.lowest_tier_watch import fire_evolution

    # Mock subprocess.run to raise TimeoutExpired (simulates a hung process)
    timeout_exc = subprocess.TimeoutExpired(
        cmd=["python", "evals/evolve_persona.py", "--candidates", "3"],
        timeout=2400,
    )

    with patch("evals.lowest_tier_watch.subprocess.run", side_effect=timeout_exc):
        promoted, message = fire_evolution(
            target_tier="test_tier",
            sib_home=None,
            consumer_pythons=None,
            candidates=3,
            repo_root=tmp_path,
            dry_run=False,
        )

    # Must NOT be promoted
    assert promoted is False, f"Expected promoted=False on timeout, got {promoted}"

    # Must return a timeout message (safe failure, not empty/crash)
    assert "[timeout]" in message, f"Expected '[timeout]' in message, got: {message}"
    assert "2400" in message, f"Expected timeout duration in message, got: {message}"


def test_fire_evolution_dry_run_still_works(tmp_path: Path) -> None:
    """Dry-run mode should still work (no timeout, no subprocess call)."""
    from evals.lowest_tier_watch import fire_evolution

    promoted, message = fire_evolution(
        target_tier="test_tier",
        sib_home=None,
        consumer_pythons=None,
        candidates=3,
        repo_root=tmp_path,
        dry_run=True,
    )

    assert promoted is False
    assert message == ""


def test_fire_evolution_normal_success(tmp_path: Path) -> None:
    """When subprocess completes normally, fire_evolution should process the output."""
    from evals.lowest_tier_watch import fire_evolution

    # Mock subprocess.run to return a normal result (no promotion)
    mock_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="Some output\nNo promotion this time",
        stderr="",
    )

    with patch("evals.lowest_tier_watch.subprocess.run", return_value=mock_result):
        promoted, message = fire_evolution(
            target_tier="test_tier",
            sib_home=None,
            consumer_pythons=None,
            candidates=3,
            repo_root=tmp_path,
            dry_run=False,
        )

    assert promoted is False
    assert "PROMOTED:" not in message


def test_fire_evolution_normal_promotion(tmp_path: Path) -> None:
    """When subprocess output contains PROMOTED:, fire_evolution should return True."""
    from evals.lowest_tier_watch import fire_evolution

    mock_result = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="Evolution complete\nPROMOTED: new_candidate_v2",
        stderr="",
    )

    with patch("evals.lowest_tier_watch.subprocess.run", return_value=mock_result):
        promoted, message = fire_evolution(
            target_tier="test_tier",
            sib_home=None,
            consumer_pythons=None,
            candidates=3,
            repo_root=tmp_path,
            dry_run=False,
        )

    assert promoted is True
