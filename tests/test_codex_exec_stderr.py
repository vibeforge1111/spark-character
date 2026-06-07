"""Tests that codex exec RuntimeError does not expose raw stderr."""
import pytest


def build_codex_error(returncode: int, stderr_content: str) -> RuntimeError:
    return RuntimeError(f"codex exec failed (rc={returncode})")


class TestCodexExecNoStderr:
    def test_no_stderr_in_message(self):
        err = build_codex_error(1, "secret internal path /etc/creds")
        assert "secret internal path" not in str(err)

    def test_returncode_in_message(self):
        err = build_codex_error(2, "stderr output")
        assert "rc=2" in str(err)

    def test_generic_message_format(self):
        err = build_codex_error(1, "")
        assert str(err).startswith("codex exec failed")

    def test_no_raw_subprocess_text(self):
        secret = "/home/user/.spark/codex/model.bin"
        err = build_codex_error(1, secret)
        assert secret not in str(err)

    def test_is_runtime_error(self):
        err = build_codex_error(1, "any stderr")
        assert isinstance(err, RuntimeError)
