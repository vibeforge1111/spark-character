"""call_codex must not leak raw subprocess stderr in its RuntimeError.

Exercises the real redaction path in codex_provider.call_codex by
monkeypatching subprocess.run to return a non-zero exit with a stderr
payload, then asserting the surfaced message keeps the return code but
drops the raw stderr text.
"""
from __future__ import annotations

import subprocess

import pytest

from spark_character import codex_provider
from spark_character.codex_provider import CodexSpec, call_codex


class _FakeCompleted:
    def __init__(self, returncode: int, stderr: bytes) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = b""


def test_call_codex_redacts_stderr_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "/home/operator/.spark/secret/model.bin internal trace"

    def fake_run(*_args, **_kwargs):
        return _FakeCompleted(returncode=7, stderr=secret.encode("utf-8"))

    monkeypatch.setattr(codex_provider, "_resolve_codex_binary", lambda binary: binary)
    monkeypatch.setattr(codex_provider.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        call_codex(
            spec=CodexSpec(binary="codex"),
            system_prompt="sys",
            user_prompt="hi",
        )

    message = str(excinfo.value)
    assert "rc=7" in message
    assert secret not in message
    assert "/home/operator" not in message


def test_call_codex_failure_message_is_generic(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args, **_kwargs):
        return _FakeCompleted(returncode=2, stderr=b"boom")

    monkeypatch.setattr(codex_provider, "_resolve_codex_binary", lambda binary: binary)
    monkeypatch.setattr(codex_provider.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        call_codex(spec=CodexSpec(binary="codex"), system_prompt="s", user_prompt="u")

    assert str(excinfo.value) == "codex exec failed (rc=2)"
