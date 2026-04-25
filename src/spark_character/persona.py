"""Persona spec loading.

The persona is a markdown artifact under artifacts/. The harness mutates
that file to evolve voice; everything else reads from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
LATEST_POINTER = ARTIFACTS_DIR / "persona.latest.txt"
DEFAULT_PERSONA_VERSION = "v1"


@dataclass(frozen=True)
class PersonaSpec:
    version: str
    text: str

    @property
    def system_prompt(self) -> str:
        return self.text.strip()


def resolve_latest_persona_version() -> str:
    """Resolve which persona version is currently active.

    Priority:
      1. persona.latest.txt pointer (written by the evolution loop on
         promotion).
      2. Highest-numbered persona.vN.md file on disk.
      3. DEFAULT_PERSONA_VERSION as the final fallback.
    """
    if LATEST_POINTER.exists():
        text = LATEST_POINTER.read_text(encoding="utf-8").strip()
        if text:
            return text
    versions: list[int] = []
    for path in ARTIFACTS_DIR.glob("persona.v*.md"):
        try:
            n = int(path.stem.split(".v", 1)[-1])
            versions.append(n)
        except ValueError:
            continue
    if versions:
        return f"v{max(versions)}"
    return DEFAULT_PERSONA_VERSION


def load_persona(version: str | None = None) -> PersonaSpec:
    """Load a persona artifact. Pass a version like 'v1' / 'v2' to pin
    explicitly, or omit to auto-resolve to the active one."""
    resolved = version or resolve_latest_persona_version()
    path = ARTIFACTS_DIR / f"persona.{resolved}.md"
    if not path.exists():
        raise FileNotFoundError(f"Persona artifact not found: {path}")
    return PersonaSpec(version=resolved, text=path.read_text(encoding="utf-8"))


def load_persona_from_path(path: str | Path) -> PersonaSpec:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Persona artifact not found: {p}")
    version = p.stem.split(".", 1)[-1] if "." in p.stem else "custom"
    return PersonaSpec(version=version, text=p.read_text(encoding="utf-8"))
