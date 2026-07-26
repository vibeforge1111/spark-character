from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spark_character_observations_digest", REPO_ROOT / "evals" / "observations_digest.py"
)
assert SPEC is not None and SPEC.loader is not None
observations_digest = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observations_digest)


def test_digest_coerces_non_string_reply_preview() -> None:
    digest = observations_digest._digest(
        [
            {
                "rewrite_suggestion": "Use a clearer concrete response.",
                "reply_preview": {"structured": "preview"},
            }
        ]
    )

    assert digest["recent_rewrites"][0]["preview"] == "{'structured': 'preview'}"


def test_json_is_single_line_for_downstream_pipe(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(observations_digest, "_load", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(sys, "argv", ["observations_digest.py", "--json"])

    assert observations_digest.main() == 0

    output = capsys.readouterr().out
    assert output.count("\n") == 1
    assert json.loads(output)["n_observations"] == 0
