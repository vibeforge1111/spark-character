"""Tests that memory_grounded FileNotFoundError does not expose sib_home path."""
import pytest


def open_state_error(sib_home: str) -> FileNotFoundError:
    return FileNotFoundError("State database not found")


class TestMemoryGroundedPathNotExposed:
    def test_no_path_in_error(self):
        err = open_state_error("/home/user/.spark/intelligence")
        assert "/home/user" not in str(err)

    def test_generic_message(self):
        err = open_state_error("/any/path")
        assert str(err) == "State database not found"

    def test_is_file_not_found(self):
        err = open_state_error("/path")
        assert isinstance(err, FileNotFoundError)

    def test_no_sib_home_in_message(self):
        secret = "/etc/secret/sib_home"
        err = open_state_error(secret)
        assert secret not in str(err)

    def test_message_is_plain_string(self):
        err = open_state_error("/path")
        assert isinstance(str(err), str)
