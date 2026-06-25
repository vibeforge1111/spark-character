"""set_latest_persona_version must not leak the artifacts_dir path.

Exercises the real FileNotFoundError path in
persona.set_latest_persona_version by pointing artifacts_dir at an empty
directory and asserting the surfaced message keeps the artifact filename
for debugging but does not embed the full directory path.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spark_character.persona import set_latest_persona_version


def test_set_latest_missing_artifact_does_not_leak_dir(tmp_path: Path) -> None:
    artifacts_dir = tmp_path / "secret-artifacts-7777"
    artifacts_dir.mkdir()
    pointer = tmp_path / "persona.latest.txt"
    log = tmp_path / "pointer.log"

    with pytest.raises(FileNotFoundError) as excinfo:
        set_latest_persona_version(
            "v9",
            pointer_path=pointer,
            log_path=log,
            artifacts_dir=artifacts_dir,
        )

    message = str(excinfo.value)
    # The full internal directory path must not be exposed...
    assert str(artifacts_dir) not in message
    assert "secret-artifacts-7777" not in message
    # ...but the artifact filename is retained for debugging.
    assert "persona.v9.md" in message
