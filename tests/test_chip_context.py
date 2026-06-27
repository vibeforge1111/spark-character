"""Synthetic chip context tests.

chip_context.py builds the context blocks an eval driver attaches so
candidates get scored on something closer to the production chip-router
path. The miss-vs-hit, alias resolution, and dedup behavior all matter
because eval cycles silently degrade when these helpers return the wrong
shape.
"""

from __future__ import annotations

from spark_character import chip_context


def test_chip_context_for_empty_input_returns_empty_string() -> None:
    assert chip_context.chip_context_for([]) == ""


def test_chip_context_for_unknown_keys_returns_empty_string() -> None:
    assert chip_context.chip_context_for(["does-not-exist", "also-nope"]) == ""


def test_chip_context_for_resolves_alias_to_canonical_block() -> None:
    aliased = chip_context.chip_context_for(["domain-chip-xcontent"])
    canonical = chip_context.chip_context_for(["xcontent"])
    assert aliased == canonical
    assert "x-content" in aliased


def test_chip_context_for_dedupes_repeats_and_aliases() -> None:
    out = chip_context.chip_context_for([
        "xcontent",
        "domain-chip-xcontent",
        "x-content",
        "XCONTENT",
    ])
    # Only the xcontent block should appear once.
    assert out.count("[Domain chip active: x-content]") == 1


def test_chip_context_for_joins_multiple_blocks_with_blank_line() -> None:
    out = chip_context.chip_context_for(["xcontent", "startup-yc"])
    assert "[Domain chip active: x-content]" in out
    assert "[Domain chip active: startup-yc]" in out
    # Blocks joined by blank line separator (two newlines).
    assert "\n\n" in out


def test_chip_context_for_ignores_blank_and_none_entries() -> None:
    out = chip_context.chip_context_for(["", "   ", "xcontent"])
    assert out.startswith("[Domain chip active: x-content]")


def test_attach_chip_context_passes_through_when_no_known_keys() -> None:
    msg = "what's the latest on the deploy?"
    assert chip_context.attach_chip_context(msg, []) == msg
    assert chip_context.attach_chip_context(msg, ["unknown"]) == msg


def test_attach_chip_context_prepends_context_and_user_marker() -> None:
    msg = "score this tweet"
    out = chip_context.attach_chip_context(msg, ["xcontent"])
    assert out.endswith("[User message]\n" + msg)
    assert out.startswith("[Domain chip active: x-content]")


def test_known_chip_keys_is_sorted_and_contains_expected_canonical_set() -> None:
    keys = chip_context.known_chip_keys()
    assert keys == sorted(keys)
    # Spot-check a stable subset; full list will grow.
    for required in ("xcontent", "startup-yc", "spark-browser"):
        assert required in keys
