from __future__ import annotations

from spark_character.prompt_guard import scan_prompt_text, sanitize_prompt_text

# scan_invisible_unicode is the steganography boundary: it flags invisible /
# non-rendering characters used to smuggle hidden instructions an LLM reads but
# a human reviewer cannot see. The original set covered zero-width chars and the
# bidi embedding/override controls, but missed several invisible classes that
# are active prompt-injection vectors:
#   - the Unicode Tags block (U+E0000–U+E007F): "ASCII smuggling"
#   - the bidi isolate controls (U+2066–U+2069): Trojan-Source family
#   - the directional marks LRM/RLM (U+200E/U+200F)
#   - soft hyphen (U+00AD), combining grapheme joiner (U+034F)
#   - the invisible math operators (U+2061–U+2064)


def _categories(text: str) -> set[str]:
    return {f.category for f in scan_prompt_text(text)}


def test_flags_unicode_tags_block_ascii_smuggling() -> None:
    smuggled = "hello" + chr(0xE0041) + chr(0xE0042) + " world"  # tag 'A','B'
    assert "invisible-unicode" in _categories(smuggled)
    # and sanitize must neutralize it (the raw tag char must not survive)
    assert chr(0xE0041) not in sanitize_prompt_text(smuggled)


def test_flags_bidi_isolates_trojan_source() -> None:
    for cp in (0x2066, 0x2067, 0x2068, 0x2069):
        text = "value" + chr(cp) + "hidden"
        assert "invisible-unicode" in _categories(text), f"U+{cp:04X} not flagged"


def test_flags_directional_marks_and_other_invisibles() -> None:
    for cp in (0x200E, 0x200F, 0x00AD, 0x034F, 0x2061, 0x2062, 0x2063, 0x2064):
        text = "a" + chr(cp) + "b"
        assert "invisible-unicode" in _categories(text), f"U+{cp:04X} not flagged"


def test_existing_zero_width_coverage_still_works() -> None:
    # regression guard: shapes already covered must keep being flagged
    for cp in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x202E):
        assert "invisible-unicode" in _categories("a" + chr(cp) + "b")


def test_plain_text_not_flagged_as_invisible() -> None:
    # ordinary visible text (incl. an em dash) must not trip the invisible scan
    for text in ("normal persona text", "use an em dash — like so", "café résumé"):
        assert "invisible-unicode" not in _categories(text)
