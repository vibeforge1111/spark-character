"""Voice-rule post-processors for LLM output.

These run after generation, before delivery. Use them when the model
won't honor a rule from the system prompt no matter how loudly we ask.

Currently scoped to em-dash family substitution: production telemetry
shows ~50% of LLM-generated replies contain em dashes despite an explicit
persona rule against them. Prompt-layer correction has been insufficient,
so we apply a deterministic post-output fix at the runtime boundary.

Other voice violations (plumbing leaks, reset openers, hedge openers)
require regeneration rather than substitution, so they are not handled
here. Leave those to the critic.
"""

from __future__ import annotations

EM_DASH_FAMILY = (
    "\u2014",  # em dash
    "\u2013",  # en dash
    "\u2012",  # figure dash
    "\u2015",  # horizontal bar
    "\u2212",  # minus sign (rare in prose, but models do emit it)
)


def replace_em_dashes(text: str, replacement: str = " - ") -> str:
    """Replace all em-dash-family characters with a hyphen.

    Default replacement is " - " (space-hyphen-space) to match the
    typographic role an em dash usually plays as a parenthetical
    separator. The function then collapses any double spaces this
    introduces, so existing single-spaced "word — word" becomes
    "word - word", not "word  -  word".
    """
    if not text:
        return text
    out = text
    for ch in EM_DASH_FAMILY:
        out = out.replace(ch, replacement)
    while "  " in out:
        out = out.replace("  ", " ")
    return out


def sanitize_voice_output(text: str) -> str:
    """Apply all voice post-processors that are safe to run in production."""
    return replace_em_dashes(text)
