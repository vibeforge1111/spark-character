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


# Directories from which user-supplied binary paths are allowed.
_ALLOWED_BIN_DIRS: frozenset[str] = frozenset({
    str(Path(p).resolve())
    for p in (
        "/usr/bin",
        "/usr/local/bin",
        "/usr/local/codex",
    )
    if Path(p).is_dir()
})


def _validate_binary(path: str) -> str:
    """Resolve and validate a binary path from an environment variable.

    Security constraints:
    * The path must resolve to an existing, regular file.
    * The file must be executable (``os.X_OK``).
    * For **absolute** paths the directory must be in the allow-list.
    * For **bare names** the binary must be discoverable via ``shutil.which()``
      in the current ``$PATH`` and the resolved location must also pass
      the directory check above.

    Returns the resolved absolute path on success.
    Raises ``ValueError`` with a descriptive message on failure.
    """
    if not path or not path.strip():
        raise ValueError("codex binary path must not be empty")

    candidate = Path(path)

    # Bare command name (no /) — resolve via PATH.
    if not candidate.is_absolute():
        found = shutil.which(path)
        if found is None:
            raise ValueError(
                f"codex binary '{path}' not found on $PATH"
            )
        candidate = Path(found)

    resolved = candidate.resolve()

    if not resolved.exists():
        raise ValueError(f"codex binary does not exist: {resolved}")

    if not resolved.is_file():
        raise ValueError(f"codex binary path is not a regular file: {resolved}")

    if not os.access(resolved, os.X_OK):
        raise ValueError(f"codex binary is not executable: {resolved}")

    parent = str(resolved.parent)
    if parent not in _ALLOWED_BIN_DIRS:
        raise ValueError(
            f"codex binary directory '{parent}' is not in the "
            f"allowed list: {sorted(_ALLOWED_BIN_DIRS)}"
        )

    return str(resolved)


def _default_codex_binary() -> str:
    explicit = os.environ.get("CODEX_PATH") or os.environ.get("SPARK_CODEX_PATH")
    if explicit:
        return _validate_binary(explicit)
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


def codex_available(spec: CodexSpec | None = None) -> bool:
    s = spec or CodexSpec()
    try:
        result = subprocess.run(
            [s.binary, "--version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False
