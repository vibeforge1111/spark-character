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
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Hardcoded allowlist of expected codex binary names.
# Only these basenames (after stripping platform extensions) are permitted.
_ALLOWED_CODEX_BINARY_NAMES: frozenset[str] = frozenset(["codex", "spark-codex"])


def _resolve_codex_binary(candidate: str) -> str:
    """Resolve and validate a codex binary path against the allowlist.

    Accepts a bare name (looked up via shutil.which) or an absolute/relative
    path. Raises ValueError if the resolved binary name is not in the allowlist
    so that env-var-injected paths like /bin/bash or /bin/rm are rejected.
    """
    basename = Path(candidate).name
    # Strip common platform extensions (.cmd, .exe, .bat) before allowlist check.
    for ext in (".cmd", ".exe", ".bat"):
        if basename.lower().endswith(ext):
            basename = basename[: -len(ext)]
            break

    if basename not in _ALLOWED_CODEX_BINARY_NAMES:
        raise ValueError(
            f"Codex binary {candidate!r} is not in the allowlist of permitted binary names "
            f"({', '.join(sorted(_ALLOWED_CODEX_BINARY_NAMES))}). "
            "Set CODEX_PATH or SPARK_CODEX_PATH to 'codex' or 'spark-codex'."
        )

    # Resolve the actual binary path via PATH lookup.
    resolved = shutil.which(candidate)
    if resolved is None:
        raise FileNotFoundError(
            f"Codex binary {candidate!r} not found on PATH. "
            "Install codex or spark-codex before using the codex provider."
        )
    return resolved


def _default_codex_binary() -> str:
    explicit = os.environ.get("CODEX_PATH") or os.environ.get("SPARK_CODEX_PATH")
    if explicit:
        return explicit
    if sys.platform.startswith("win"):
        return "codex.cmd"
    return "codex"


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
    """Invoke codex exec, return the assistant's last message text.

    Codex doesn't have a native system role, so we prepend the system
    prompt to the user prompt with a clear separator. Functionally
    equivalent for short conversational turns.
    """
    combined = f"{system_prompt.strip()}\n\nUser message:\n{user_prompt.strip()}"
    binary = _resolve_codex_binary(spec.binary)
    with tempfile.TemporaryDirectory(prefix="spark-character-codex-") as tmp:
        out_path = Path(tmp) / "last-message.txt"
        cmd = [
            binary,
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


def codex_available(spec: CodexSpec | None = None) -> bool:
    s = spec or CodexSpec()
    try:
        binary = _resolve_codex_binary(s.binary)
        result = subprocess.run(
            [binary, "--version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, ValueError, subprocess.TimeoutExpired, PermissionError):
        return False
