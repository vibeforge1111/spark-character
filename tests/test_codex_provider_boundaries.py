"""Codex subprocess authority and output-contract regressions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spark_character import codex_provider
from spark_character.codex_provider import (
    CodexSpec,
    _resolve_codex_binary,
    call_codex,
    codex_available,
)


@pytest.mark.parametrize("binary", ["/bin/bash", "/bin/rm", "python", "evil-script"])
def test_resolve_codex_binary_rejects_unowned_executables(binary: str) -> None:
    with pytest.raises(ValueError, match="allowed"):
        _resolve_codex_binary(binary)


def test_resolve_codex_binary_accepts_owned_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_provider.shutil, "which", lambda _candidate: "/opt/spark/bin/codex")

    assert _resolve_codex_binary("codex") == "/opt/spark/bin/codex"


def test_resolve_codex_binary_rejects_spoofed_allowed_basename(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(candidate: str) -> str:
        return "/tmp/codex" if candidate == "/tmp/codex" else "/opt/spark/bin/codex"

    monkeypatch.setattr(codex_provider.shutil, "which", fake_which)

    with pytest.raises(ValueError, match="same launcher"):
        _resolve_codex_binary("/tmp/codex")


def test_codex_available_rejects_disallowed_binary_without_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("disallowed binary reached subprocess")

    monkeypatch.setattr(codex_provider.subprocess, "run", unexpected_run)

    assert codex_available(CodexSpec(binary="/bin/bash")) is False


def test_call_codex_rejects_empty_output_without_stderr_leak(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_provider.shutil, "which", lambda _candidate: "/opt/spark/bin/codex")

    def fake_run(cmd: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        out_path = Path(cmd[cmd.index("--output-last-message") + 1])
        out_path.write_text("   ", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"sensitive-marker")

    monkeypatch.setattr(codex_provider.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="empty last-message") as exc_info:
        call_codex(spec=CodexSpec(binary="codex"), system_prompt="system", user_prompt="user")

    assert "sensitive-marker" not in str(exc_info.value)


def test_call_codex_names_missing_output_contract_and_remediation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(codex_provider.shutil, "which", lambda _candidate: "/opt/spark/bin/codex")
    monkeypatch.setattr(
        codex_provider.subprocess,
        "run",
        lambda cmd, **_kwargs: subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b""),
    )

    with pytest.raises(RuntimeError) as exc_info:
        call_codex(spec=CodexSpec(binary="codex"), system_prompt="system", user_prompt="user")

    message = str(exc_info.value)
    assert "last-message.txt" in message
    assert "Codex CLI" in message
    assert "/spark-character-codex-" not in message
