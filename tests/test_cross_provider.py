from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_character_cross_provider", REPO_ROOT / "evals" / "cross_provider.py"
)
assert SPEC is not None and SPEC.loader is not None
cross_provider = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cross_provider
SPEC.loader.exec_module(cross_provider)


def test_unknown_provider_name_is_visible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cross_provider.resolve_providers([" typo-provider "]) == []

    error = capsys.readouterr().err
    assert "unknown provider 'typo-provider'" in error
    assert "known: codex, minimax, openai, zai" in error


def test_unavailable_codex_binary_is_visible(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cross_provider, "codex_available", lambda _spec: False)

    assert cross_provider.resolve_providers(["codex"]) == []
    assert "skipping 'codex' (binary not on PATH)" in capsys.readouterr().err
