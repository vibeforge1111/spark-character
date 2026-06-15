"""Tests for codex binary path allowlist enforcement."""

from __future__ import annotations

import shutil
from unittest.mock import patch

import pytest

from spark_character.codex_provider import (
    CodexSpec,
    _resolve_codex_binary,
    codex_available,
)


class TestResolveCodexBinary:
    def test_bash_binary_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            _resolve_codex_binary("/bin/bash")

    def test_rm_binary_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            _resolve_codex_binary("/bin/rm")

    def test_arbitrary_unknown_binary_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            _resolve_codex_binary("evil-script")

    def test_python_binary_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            _resolve_codex_binary("python")

    def test_sh_binary_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            _resolve_codex_binary("sh")

    def test_codex_name_accepted_if_found_by_which(self) -> None:
        with patch("spark_character.codex_provider.shutil.which", return_value="/usr/local/bin/codex"):
            resolved = _resolve_codex_binary("codex")
        assert resolved == "/usr/local/bin/codex"

    def test_spark_codex_name_accepted_if_found_by_which(self) -> None:
        with patch("spark_character.codex_provider.shutil.which", return_value="/usr/bin/spark-codex"):
            resolved = _resolve_codex_binary("spark-codex")
        assert resolved == "/usr/bin/spark-codex"

    def test_codex_cmd_accepted_on_windows_path(self) -> None:
        with patch("spark_character.codex_provider.shutil.which", return_value="C:\\tools\\codex.cmd"):
            resolved = _resolve_codex_binary("codex.cmd")
        assert resolved == "C:\\tools\\codex.cmd"

    def test_missing_codex_binary_raises_file_not_found(self) -> None:
        with patch("spark_character.codex_provider.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="not found on PATH"):
                _resolve_codex_binary("codex")

    def test_missing_spark_codex_binary_raises_file_not_found(self) -> None:
        with patch("spark_character.codex_provider.shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="not found on PATH"):
                _resolve_codex_binary("spark-codex")

    def test_absolute_path_to_codex_accepted_if_basename_allowed(self) -> None:
        with patch("spark_character.codex_provider.shutil.which", return_value="/opt/codex/bin/codex"):
            resolved = _resolve_codex_binary("/opt/codex/bin/codex")
        assert resolved == "/opt/codex/bin/codex"

    def test_absolute_path_to_bash_rejected(self) -> None:
        with pytest.raises(ValueError, match="allowlist"):
            _resolve_codex_binary("/usr/bin/bash")


class TestCodexAvailable:
    def test_codex_available_returns_false_for_disallowed_binary(self) -> None:
        spec = CodexSpec(binary="/bin/bash")
        assert codex_available(spec) is False

    def test_codex_available_returns_false_when_binary_not_on_path(self) -> None:
        with patch("spark_character.codex_provider.shutil.which", return_value=None):
            spec = CodexSpec(binary="codex")
            assert codex_available(spec) is False

    def test_codex_available_returns_true_when_binary_found_and_returns_zero(self) -> None:
        import subprocess as _subprocess
        mock_result = _subprocess.CompletedProcess(args=[], returncode=0)
        with patch("spark_character.codex_provider.shutil.which", return_value="/usr/bin/codex"), \
             patch("spark_character.codex_provider.subprocess.run", return_value=mock_result):
            spec = CodexSpec(binary="codex")
            assert codex_available(spec) is True
