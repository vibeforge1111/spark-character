"""Synthetic chip-context injection helpers."""

from __future__ import annotations

from spark_character.chip_context import (
    CHIP_KEY_ALIASES,
    SYNTHETIC_CHIP_CONTEXTS,
    attach_chip_context,
    chip_context_for,
    known_chip_keys,
)


def test_chip_context_for_returns_empty_when_no_keys_given() -> None:
    assert chip_context_for([]) == ""


def test_chip_context_for_returns_block_for_known_key() -> None:
    block = chip_context_for(["xcontent"])
    assert block.startswith("[Domain chip active: x-content]")
    assert "hidden background context" in block


def test_chip_context_for_resolves_aliases_to_canonical_keys() -> None:
    via_alias = chip_context_for(["domain-chip-xcontent"])
    via_canonical = chip_context_for(["xcontent"])
    assert via_alias == via_canonical
    for alias, canonical in CHIP_KEY_ALIASES.items():
        assert canonical in SYNTHETIC_CHIP_CONTEXTS
        assert chip_context_for([alias]) == SYNTHETIC_CHIP_CONTEXTS[canonical]


def test_chip_context_for_normalizes_case_and_whitespace() -> None:
    assert chip_context_for(["  Xcontent  "]) == SYNTHETIC_CHIP_CONTEXTS["xcontent"]


def test_chip_context_for_deduplicates_same_chip_in_multiple_forms() -> None:
    block = chip_context_for(["xcontent", "domain-chip-xcontent", "XCONTENT"])
    assert block.count("[Domain chip active: x-content]") == 1


def test_chip_context_for_joins_distinct_chips_with_blank_line() -> None:
    block = chip_context_for(["xcontent", "startup-yc"])
    parts = block.split("\n\n")
    assert any(part.startswith("[Domain chip active: x-content]") for part in parts)
    assert any(part.startswith("[Domain chip active: startup-yc]") for part in parts)


def test_chip_context_for_ignores_unknown_and_blank_keys() -> None:
    assert chip_context_for(["totally-fake-chip"]) == ""
    assert chip_context_for(["", "   ", None]) == ""  # type: ignore[list-item]
    assert chip_context_for(["unknown", "xcontent"]) == SYNTHETIC_CHIP_CONTEXTS["xcontent"]


def test_attach_chip_context_returns_message_unchanged_when_no_chips_match() -> None:
    message = "What is the weather?"
    assert attach_chip_context(message, []) == message
    assert attach_chip_context(message, ["unknown-chip"]) == message


def test_attach_chip_context_prepends_context_then_user_message_marker() -> None:
    attached = attach_chip_context("Draft a tweet about AGI.", ["xcontent"])
    assert attached.startswith("[Domain chip active: x-content]")
    assert "[User message]\nDraft a tweet about AGI." in attached
    assert "\n\n[User message]\n" in attached


def test_known_chip_keys_returns_sorted_canonical_keys() -> None:
    keys = known_chip_keys()
    assert keys == sorted(keys)
    assert set(keys) == set(SYNTHETIC_CHIP_CONTEXTS)
