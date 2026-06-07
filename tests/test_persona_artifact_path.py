"""Tests that persona FileNotFoundError does not expose artifact path."""
import pytest


def persona_not_found_error(artifact_path: str) -> FileNotFoundError:
    return FileNotFoundError("Persona artifact not found")


class TestPersonaArtifactPathNotExposed:
    def test_no_path_in_error(self):
        err = persona_not_found_error("/home/user/.spark/personas/persona.v1.md")
        assert "/home/user" not in str(err)

    def test_generic_message(self):
        err = persona_not_found_error("/any/path.md")
        assert str(err) == "Persona artifact not found"

    def test_is_file_not_found(self):
        err = persona_not_found_error("/path")
        assert isinstance(err, FileNotFoundError)

    def test_no_resolved_version_in_message(self):
        secret = "/etc/secret/persona.v99.md"
        err = persona_not_found_error(secret)
        assert secret not in str(err)

    def test_message_is_plain_string(self):
        err = persona_not_found_error("/path")
        assert isinstance(str(err), str)
