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


def _explicit_codex_path() -> str | None:
    """Return the operator-supplied codex path (env), expanded, or None.

    Only CODEX_PATH / SPARK_CODEX_PATH are treated as explicit paths; the
    platform fallbacks ("codex" / "codex.cmd") resolve through PATH and are
    not validated here.
    """
    explicit = os.environ.get("CODEX_PATH") or os.environ.get("SPARK_CODEX_PATH")
    if explicit:
        return os.path.expanduser(explicit)
    return None


def _default_codex_binary() -> str:
    # Resolution only — never raises. The isfile validation for an explicit
    # env-supplied path is deferred to call time (validate_codex_binary), so
    # that merely importing this module with a stale CODEX_PATH does not crash
    # eval drivers that don't even use the codex backend.
    explicit = _explicit_codex_path()
    if explicit:
        return explicit
    if sys.platform.startswith("win"):
        return "codex.cmd"
    return "codex"


def validate_codex_binary(binary: str) -> None:
    """Raise FileNotFoundError if an explicitly-configured codex path is bad.

    Guards against arbitrary binary execution via a malicious/stale
    CODEX_PATH / SPARK_CODEX_PATH env var. Only validates when the resolved
    binary matches the explicit env path; bare PATH lookups ("codex") are
    left for subprocess to resolve so this stays a no-op in the common case.
    """
    explicit = _explicit_codex_path()
    if explicit and binary == explicit and not os.path.isfile(binary):
        raise FileNotFoundError(f"Codex binary not found: {binary}")


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
    # Defer the env-path validation to call time so import never crashes on a
    # stale CODEX_PATH; this still blocks executing an explicit, non-file path.
    validate_codex_binary(spec.binary)
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
        try:
            result = subprocess.run(
                cmd,
                input=combined.encode("utf-8"),
                capture_output=True,
                timeout=spec.timeout_seconds,
            )
        except FileNotFoundError as exc:
            # Guards the eval/judge driver against a raw stack trace when the
            # codex CLI is not installed or CODEX_PATH points at a removed
            # binary. Preserves the operator's next move (install codex or
            # set CODEX_PATH) instead of leaking the OSError text.
            raise RuntimeError(
                f"codex binary not found at {spec.binary!r}. Install the codex CLI "
                f"or set CODEX_PATH / SPARK_CODEX_PATH to its absolute path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            # Closes the silent-hang window when codex exec exceeds the
            # configured timeout; surfaces the actual budget so the operator
            # can raise CodexSpec.timeout_seconds rather than guess.
            raise RuntimeError(
                f"codex exec timed out after {spec.timeout_seconds:.0f}s. "
                f"Increase CodexSpec.timeout_seconds or check that the codex "
                f"CLI is responsive."
            ) from exc
        if result.returncode != 0:
            # Redact raw stderr (may carry internal paths / prompt fragments);
            # keep the return code so operators can still triage.
            raise RuntimeError(f"codex exec failed (rc={result.returncode})")
        if not out_path.exists():
            raise RuntimeError("codex exec did not write the expected output file.")
        text = out_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            raise RuntimeError(
                "codex exec exited 0 but produced an empty last-message file; "
                "the caller cannot distinguish this from a successful empty reply. "
                f"stderr tail: {stderr.strip()[:300]}"
            )
        return text


def codex_available(spec: CodexSpec | None = None) -> bool:
    s = spec or CodexSpec()
    try:
        validate_codex_binary(s.binary)
        result = subprocess.run(
            [s.binary, "--version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False
