"""_open_state must not leak sib_home in its FileNotFoundError.

Exercises the real redaction path in memory_grounded._open_state by
pointing it at a directory with no state.db and asserting the surfaced
message is generic (no sib_home path fragment).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from spark_character.memory_grounded import _open_state


def test_open_state_missing_db_does_not_leak_path(tmp_path: Path) -> None:
    sib_home = tmp_path / "home-secret-9999"
    sib_home.mkdir()

    with pytest.raises(FileNotFoundError) as excinfo:
        _open_state(sib_home)

    message = str(excinfo.value)
    assert "home-secret-9999" not in message
    assert str(sib_home) not in message


def test_open_state_message_is_generic(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        _open_state(tmp_path)
    assert str(excinfo.value) == "State database not found"
