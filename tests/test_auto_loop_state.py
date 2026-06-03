"""Regression tests for the narrowed state-persistence except clauses in
evals/auto_loop.py.

The two helpers under test, `_load_state` and `_write_heartbeat`, each wrap
a small disk operation in a narrow except clause. The narrowing replaces a
bare `except Exception:` with `except (OSError, json.JSONDecodeError):` and
`except OSError:` respectively, so that programmer bugs (AttributeError,
TypeError, NameError) inside the try block raise out of the helper instead
of being folded into the documented default-state dict or silent pass.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


_REPO_ROOT = Path(__file__).resolve().parents[1]
_AUTO_LOOP_PATH = _REPO_ROOT / "evals" / "auto_loop.py"


def _load_auto_loop_module():
    """Import evals/auto_loop.py as a module without invoking main()."""
    sys.path.insert(0, str(_REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(
        "_auto_loop_under_test", _AUTO_LOOP_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AutoLoopStateTests(unittest.TestCase):
    """Cover the narrow-tuple contract on _load_state and _write_heartbeat."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_auto_loop_module()

    def test_load_state_returns_default_on_malformed_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text("{not valid json", encoding="utf-8")
            state = self.mod._load_state(state_file)
            # Documented default-state contract on JSONDecodeError preserved.
            self.assertEqual(state.get("last_evolved_at"), 0)
            self.assertEqual(state.get("cycle_count"), 0)
            self.assertIn("last_persona_version", state)

    def test_load_state_lets_attribute_errors_propagate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_file = Path(tmp) / "state.json"
            state_file.write_text("{}", encoding="utf-8")

            def _boom(*_args, **_kwargs):
                # Programmer-bug substitute -- a typo'd attribute access on
                # the json module after a refactor would surface as this.
                raise AttributeError(
                    "simulated typo in json module after refactor"
                )

            with patch.object(self.mod.json, "loads", side_effect=_boom):
                with self.assertRaises(AttributeError):
                    self.mod._load_state(state_file)

    def test_write_heartbeat_silently_passes_on_oserror(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Point the heartbeat path inside a regular file so that the
            # parent.mkdir() inside _write_heartbeat fails with OSError
            # (NotADirectoryError is an OSError subclass).
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            hb = blocker / "heartbeat.txt"
            # Documented silent-pass contract on OSError preserved.
            self.mod._write_heartbeat(hb, "phase")
            self.assertFalse(hb.exists())

    def test_write_heartbeat_lets_attribute_errors_propagate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            hb = Path(tmp) / "heartbeat.txt"

            def _boom(self, *_args, **_kwargs):
                raise AttributeError(
                    "simulated typo in Path.write_text wrapper"
                )

            with patch.object(Path, "write_text", _boom):
                with self.assertRaises(AttributeError):
                    self.mod._write_heartbeat(hb, "phase")


if __name__ == "__main__":
    unittest.main()
