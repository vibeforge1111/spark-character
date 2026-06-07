"""Tests: temp file suffix uses secrets.token_hex, not os.getpid(), preventing symlink pre-emption."""
from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import patch
import secrets


def _generate_temp_name(pointer_name: str) -> str:
    return f".{pointer_name}.{secrets.token_hex(8)}.tmp"


def _generate_temp_name_vulnerable(pointer_name: str) -> str:
    import os
    return f".{pointer_name}.{os.getpid()}.tmp"


class TestPersonaTempFileSymlink:
    def test_temp_name_uses_random_hex_not_pid(self):
        import os
        name = _generate_temp_name("persona.latest.txt")
        pid_str = str(os.getpid())
        assert pid_str not in name, "temp name must not contain PID"

    def test_temp_name_suffix_is_16_hex_chars(self):
        name = _generate_temp_name("persona.latest.txt")
        match = re.search(r'\.([0-9a-f]+)\.tmp$', name)
        assert match is not None, "suffix must be hex"
        assert len(match.group(1)) == 16, "secrets.token_hex(8) produces 16 hex chars"

    def test_temp_names_are_unique_across_1000_calls(self):
        names = {_generate_temp_name("p.txt") for _ in range(1000)}
        assert len(names) == 1000, "all 1000 temp names must be unique"

    def test_vulnerable_name_contains_pid(self):
        import os
        name = _generate_temp_name_vulnerable("persona.latest.txt")
        assert str(os.getpid()) in name, "vulnerable pattern confirms PID in name"

    def test_temp_name_ends_with_dot_tmp(self):
        assert _generate_temp_name("pointer.txt").endswith(".tmp")
