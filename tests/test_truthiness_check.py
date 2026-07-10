"""Dummy test for idiomatic truthiness check migration."""
import pytest


def test_truthiness_check():
    """Placeholder: verify that not X replaces len(X)==0."""
    # TODO: implement truthiness check test
    assert True


def test_empty_list_truthiness():
    """Placeholder: verify that empty list is falsy."""
    assert not []
    assert not ()
    assert not {}
    assert not ""
    assert not set()
