"""Codex CLI provider adapter.

Codex is OAuth-authenticated and runs OpenAI's gpt-5/gpt-4 family
through a CLI subprocess (`codex exec ...`) instead of an HTTP
endpoint. This adapter wraps that subprocess so spark-character can
treat it as another backend for cross-provider voice consistency
testing.

Designed to mirror the call_provider() shape just enough for the
eval drivers and cross-provider judge. Tool use, async, and history
are not supported here (codex exec is single-prompt one-shot). For
those features, route through an HTTP-compatible backend.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _default_codex_binary() -> str:
    try:
        explicit = os.environ.get("CODEX_PATH") or os.environ.get("SPARK_CODEX_PATH")
        if explicit:
            return explicit
        if sys.platform.startswith("win"):
            return "codex.cmd"
        return "codex"



    except Exception:
        return ""
DEFAULT_CODEX_PATH = _default_codex_binary()
DEFAULT_CODEX_MODEL = (
    os.environ.get("CODEX_MODEL")
    or os.environ.get("SPARK_CODEX_MODEL")
    or os.environ.get("OPENAI_MODEL")
    or "gpt-5.5"
)


@dataclass(frozen=True)
class CodexSpec:
    binary: str = DEFAULT_CODEX_PATH
    model: str = DEFAULT_CODEX_MODEL
    timeout_seconds: float = 180.0

    @property
    def base_url(self) -> str:
        return f"codex-cli://{self.binary}"


def call_codex(
    *,
    spec: CodexSpec,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if not isinstance(system_prompt, str): system_prompt = str(system_prompt or '')
    if not isinstance(user_prompt, str): user_prompt = str(user_prompt or '')
    try:
        """Invoke codex exec, return the assistant's last message text.

        Codex doesn't have a native system role, so we prepend the system
        prompt to the user prompt with a clear separator. Functionally
        equivalent for short conversational turns.
        """
        combined = f"{system_prompt.strip()}\n\nUser message:\n{user_prompt.strip()}"
        with tempfile.TemporaryDirectory(prefix="spark-character-codex-") as tmp:
            out_path = Path(tmp) / "last-message.txt"
            cmd = [
                spec.binary,
                "exec",
                "--skip-git-repo-check",
                "--model", spec.model,
                "--sandbox", "read-only",
                "--output-last-message", str(out_path),
                "-",
            ]
            result = subprocess.run(
                cmd,
                input=combined.encode("utf-8"),
                capture_output=True,
                timeout=spec.timeout_seconds,
            )
            if result.returncode != 0:
                stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
                raise RuntimeError(f"codex exec failed (rc={result.returncode}): {stderr.strip()[:300]}")
            if not out_path.exists():
                raise RuntimeError("codex exec did not write the expected output file.")
            text = out_path.read_text(encoding="utf-8", errors="replace").strip()
            return text



    except Exception:
        return ""
def codex_available(spec: CodexSpec | None = None) -> bool:
    try:
        s = spec or CodexSpec()
        try:
            result = subprocess.run(
                [s.binary, "--version"], capture_output=True, timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
            return False

    except Exception:
        return False
