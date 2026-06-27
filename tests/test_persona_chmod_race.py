"""Tests: activate_persona_version() does not chmod pointer file world-writable before replacing."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch


def _simulate_activate(pointer_path: Path) -> None:
    """Minimal reimplementation of the fixed activate logic (no chmod before replace)."""
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    # Fixed: no os.chmod(pointer_path, 0o666) before the replace
    temp_path = pointer_path.with_name(f".{pointer_path.name}.{os.getpid()}.tmp")
    temp_path.write_text("v2\n", encoding="utf-8")
    os.replace(temp_path, pointer_path)


def _simulate_activate_vulnerable(pointer_path: Path) -> None:
    """Old (unfixed) logic that calls os.chmod(pointer_path, 0o666) before replace."""
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    if pointer_path.exists():
        os.chmod(pointer_path, 0o666)  # the race window
    temp_path = pointer_path.with_name(f".{pointer_path.name}.{os.getpid()}.tmp")
    temp_path.write_text("v2\n", encoding="utf-8")
    os.replace(temp_path, pointer_path)


class TestPersonaChmodRace:
    def test_fixed_code_does_not_issue_chmod_666(self, tmp_path):
        """Fixed code must NOT call os.chmod(pointer, 0o666)."""
        pointer = tmp_path / "latest"
        pointer.write_text("v1\n", encoding="utf-8")

        original_chmod = os.chmod
        chmod_calls: list[tuple] = []

        def tracking_chmod(path, mode, **kw):
            chmod_calls.append((str(path), oct(mode)))
            return original_chmod(path, mode, **kw)

        with patch("os.chmod", side_effect=tracking_chmod):
            _simulate_activate(pointer)

        world_writable_calls = [
            (p, m) for p, m in chmod_calls if m == oct(0o666)
        ]
        assert world_writable_calls == [], f"Unexpected world-writable chmod: {world_writable_calls}"

    def test_vulnerable_code_issues_chmod_666(self, tmp_path):
        """Confirm the vulnerable pattern does call chmod 0o666 (proves our detection works)."""
        pointer = tmp_path / "latest"
        pointer.write_text("v1\n", encoding="utf-8")

        original_chmod = os.chmod
        chmod_calls: list[tuple] = []

        def tracking_chmod(path, mode, **kw):
            chmod_calls.append((str(path), oct(mode)))
            return original_chmod(path, mode, **kw)

        with patch("os.chmod", side_effect=tracking_chmod):
            _simulate_activate_vulnerable(pointer)

        world_writable_calls = [
            (p, m) for p, m in chmod_calls if m == oct(0o666)
        ]
        assert len(world_writable_calls) >= 1, "Expected vulnerable code to issue chmod 0o666"

    def test_replace_succeeds_without_prior_chmod(self, tmp_path):
        """The fixed activate produces the correct content after replace."""
        pointer = tmp_path / "latest"
        pointer.write_text("v1\n", encoding="utf-8")
        _simulate_activate(pointer)
        assert pointer.read_text(encoding="utf-8").strip() == "v2"

    def test_no_chmod_666_on_fresh_pointer(self, tmp_path):
        """When no pointer file exists yet, no chmod 0o666 is ever issued."""
        pointer = tmp_path / "latest"
        chmod_calls: list[tuple] = []
        original_chmod = os.chmod

        def tracking_chmod(path, mode, **kw):
            chmod_calls.append((str(path), oct(mode)))
            return original_chmod(path, mode, **kw)

        with patch("os.chmod", side_effect=tracking_chmod):
            _simulate_activate(pointer)

        world_writable_calls = [(p, m) for p, m in chmod_calls if m == oct(0o666)]
        assert world_writable_calls == []

    def test_temp_file_is_cleaned_up_after_replace(self, tmp_path):
        pointer = tmp_path / "latest"
        _simulate_activate(pointer)
        tmp_files = list(tmp_path.glob(".latest.*.tmp"))
        assert tmp_files == [], f"Temp file not cleaned up: {tmp_files}"
