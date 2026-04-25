"""Synthetic chip context for scoring.

In production, when a user message lands, SIB's chip_router activates
domain chips (e.g. domain-chip-xcontent, startup-yc). Each active chip
injects context into the prompt. The model then replies with that
context active. This is the path users actually experience.

When evaluation runs without that context, the model is on an easier
path than production. spark-character's audit miner showed this gap
concretely: 88 of 164 production LLM replies break the no-em-dash
rule, almost all on chip-active routes (domain-chip-xcontent,
startup-yc). Synthetic probes scored T1=1.0 on the same persona.

This module provides short, representative chip context blocks that
can be injected into eval prompts so candidates are scored on
something closer to the production path. Not as good as wiring the
real chip system in (which would mean depending on SIB), but enough
to surface chip-driven failure modes during evolution cycles.

Usage:
    context = chip_context_for(["xcontent", "startup-yc"])
    # then pass to generate(prompt + "\n\n" + context, ...) in eval
"""

from __future__ import annotations


SYNTHETIC_CHIP_CONTEXTS: dict[str, str] = {
    "xcontent": (
        "[Domain chip active: x-content]\n"
        "x-content scores tweets and threads on hook quality, novelty (0-10), "
        "intensity (0-10), audience fit, format probe, and final verdict "
        "(accept / conditional_accept / reject) with confidence. Treat this "
        "as hidden background context the user does not need to see by name. "
        "If the user asks about a tweet or post, evaluate it against these "
        "axes but speak about the work, not the chip."
    ),
    "startup-yc": (
        "[Domain chip active: startup-yc]\n"
        "startup-yc provides YC-style founder framing: focus is finding a "
        "problem worth solving, building something people want, and not "
        "dying before you do. Bias toward decisive recommendations grounded "
        "in user pull. Treat this as hidden context. Do not name it as a "
        "chip in the reply."
    ),
    "spark-browser": (
        "[Domain chip active: spark-browser]\n"
        "spark-browser provides live web browsing. If the user asks for "
        "current data and you can fetch it via the browser, fetch and "
        "answer with the source. Treat the browser session as a capability "
        "you have, not a tool to name."
    ),
    "spark-personality-chip-labs": (
        "[Domain chip active: spark-personality-chip-labs]\n"
        "Personality chip lab is loaded. Behave consistently with the "
        "loaded personality chip's traits, voice signature, and "
        "anti-patterns. Do not narrate the chip lab, just be the chip."
    ),
    "spark-swarm": (
        "[Domain chip active: spark-swarm]\n"
        "Spark swarm is available for multi-agent coordination on harder "
        "tasks. You can mention escalating to other agents only if the "
        "user explicitly asks how something gets done; otherwise, just do "
        "the work."
    ),
    "domain-chip-voice-comms": (
        "[Domain chip active: voice-comms]\n"
        "Voice-comms chip is active. The user may be communicating via "
        "voice transcription. Keep replies short, declarative, and "
        "easy to listen to."
    ),
    "domain-chip-memory": (
        "[Domain chip active: domain-chip-memory]\n"
        "Memory chip is loaded with this user's recent observations and "
        "stated facts. Use that context naturally when it bears on the "
        "current message. Do not enumerate what you remember unless asked."
    ),
}


CHIP_KEY_ALIASES: dict[str, str] = {
    "domain-chip-xcontent": "xcontent",
    "domain-chip-x-content": "xcontent",
    "x-content": "xcontent",
    "domain-chip-spark-browser": "spark-browser",
    "domain-chip-startup-yc": "startup-yc",
}


def chip_context_for(chip_keys: list[str]) -> str:
    """Return a context string for the given list of chip keys, or
    empty string if none of them are recognized."""
    if not chip_keys:
        return ""
    seen: set[str] = set()
    blocks: list[str] = []
    for raw in chip_keys:
        key = (raw or "").strip().lower()
        key = CHIP_KEY_ALIASES.get(key, key)
        if not key or key in seen:
            continue
        seen.add(key)
        block = SYNTHETIC_CHIP_CONTEXTS.get(key)
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "\n\n".join(blocks)


def attach_chip_context(user_message: str, chip_keys: list[str]) -> str:
    """Return a user message with chip context injected as a preamble.
    Mimics SIB's _build_contextual_task pattern of attaching chip
    guidance before the literal user message."""
    ctx = chip_context_for(chip_keys)
    if not ctx:
        return user_message
    return f"{ctx}\n\n[User message]\n{user_message}"


def known_chip_keys() -> list[str]:
    return sorted(SYNTHETIC_CHIP_CONTEXTS.keys())
