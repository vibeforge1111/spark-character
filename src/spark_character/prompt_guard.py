"""Prompt-boundary guards for persona and chip-authored system text."""

from __future__ import annotations

import re
from dataclasses import dataclass

INVISIBLE_UNICODE_CHARS = {
    "\u200b": "ZERO WIDTH SPACE",
    "\u200c": "ZERO WIDTH NON-JOINER",
    "\u200d": "ZERO WIDTH JOINER",
    "\u2060": "WORD JOINER",
    "\ufeff": "BYTE ORDER MARK",
    "\u202a": "LEFT-TO-RIGHT EMBEDDING",
    "\u202b": "RIGHT-TO-LEFT EMBEDDING",
    "\u202c": "POP DIRECTIONAL FORMATTING",
    "\u202d": "LEFT-TO-RIGHT OVERRIDE",
    "\u202e": "RIGHT-TO-LEFT OVERRIDE",
}
PROMPT_BOUNDARY_PREFIX = r"(?:^|[:-]\s*)"
STORED_PROMPT_INJECTION_PATTERNS = (
    (
        "instruction-override",
        re.compile(
            PROMPT_BOUNDARY_PREFIX
            + r"(ignore|disregard|forget|dismiss|abandon)\s+(all\s+)?"
            r"(previous|prior|above|earlier|preceding)\s+instructions?\b",
            re.I,
        ),
    ),
    (
        "system-prompt-override",
        re.compile(
            PROMPT_BOUNDARY_PREFIX
            + r"(system|developer|admin)\s+(prompt|message|instruction|directive)s?\b"
            r".*\b(override|replace|ignore|disregard)\b",
            re.I,
        ),
    ),
    (
        "hidden-html",
        re.compile(
            r"<!--|<\s*(?:div|span)[^>]*(?:display\s*:\s*none|visibility\s*:\s*hidden)",
            re.I,
        ),
    ),
    (
        "secret-exfiltration",
        re.compile(
            r"\b(curl|wget|fetch|requests?\.get)\b"
            r".*\b(\.env|secret|token|api[_-]?key|password|credentials)\b",
            re.I,
        ),
    ),
    (
        "secret-file-request",
        re.compile(
            r"\b(read|open|print|cat|get-content|type|more|less)\b"
            r".*(\.env|secrets\.local\.json|id_rsa|\.ssh|api[_-]?key|credentials)\b",
            re.I,
        ),
    ),
    (
        "private-key",
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.I),
    ),
    # -- extended patterns (search-result / general prompt-injection) --
    (
        "role-impersonation",
        re.compile(
            r"(?:you\s+are\s+now\s+|act\s+as\s+|pretend\s+(?:you\s+are|to\s+be)\s+|"
            r"from\s+now\s+on\s+you\s+(?:are|will|must)\s+|your\s+new\s+role\s+is\s+|"
            r"embody\s+the\s+role\s+of\s+|take\s+on\s+the\s+persona\s+of\s+)",
            re.I,
        ),
    ),
    (
        "message-boundary-injection",
        re.compile(
            r"\b(?:System\s*:\s*|Human\s*:\s*|Assistant\s*:\s*|User\s*:\s*|AI\s*:\s*|"
            r"\[INST\]|<<SYS>>|<\|system\|>|<\|endoftext\|>|"
            r"<\|im_start\|>|<\|im_end\|>|</s>|<s>)",
            re.I,
        ),
    ),
    (
        "instruction-continuation",
        re.compile(
            r"(?:new\s+instructions?\s*:|override\s*:|updated\s+instructions?\s*:|"
            r"disregard\s+the\s+above|the\s+above\s+was\s+(?:wrong|incorrect)|"
            r"corrected\s+instructions?\s*:|revised\s+instructions?\s*:)",
            re.I,
        ),
    ),
    (
        "persona-hijack",
        re.compile(
            r"(?:you\s+are\s+an\s+AI\s+that\s+|you\s+have\s+been\s+(?:given|assigned)\s+"
            r"(?:the\s+task\s+of|the\s+role\s+of)|your\s+only\s+(?:job|purpose|directive)\s+is\s+to|"
            r"ignore\s+all\s+safety\s+(?:guidelines|rules|restrictions|filters)|"
            r"bypass\s+(?:all\s+)?(?:safety|content|security)\s+(?:filters?|restrictions?|guidelines?))",
            re.I,
        ),
    ),
    ("hidden-html", re.compile(r"<!--|<\s*(?:div|span)[^>]*(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.I)),
    ("secret-exfiltration", re.compile(r"\b(curl|wget|fetch)\b.*(?<!\w)(\.env|secret|token|api[_-]?key|password)\b", re.I)),
    ("secret-file-request", re.compile(r"\b(read|open|print|cat|get-content)\b.*(\.env|secrets\.local\.json|id_rsa|\.ssh|api[_-]?key)\b", re.I)),
    ("private-key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----", re.I)),
)


@dataclass(frozen=True)
class PromptGuardFinding:
    category: str
    detail: str


def scan_invisible_unicode(text: str) -> list[PromptGuardFinding]:
    findings: list[PromptGuardFinding] = []
    for char, name in INVISIBLE_UNICODE_CHARS.items():
        if char in text:
            findings.append(PromptGuardFinding("invisible-unicode", f"U+{ord(char):04X} {name}"))
    return findings


def scan_stored_prompt_injection(text: str) -> list[PromptGuardFinding]:
    findings: list[PromptGuardFinding] = []
    for category, pattern in STORED_PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            findings.append(PromptGuardFinding(category, "prompt text matched a stored-injection pattern"))
    return findings


def scan_prompt_text(text: str) -> list[PromptGuardFinding]:
    return [*scan_invisible_unicode(text), *scan_stored_prompt_injection(text)]


def sanitize_prompt_text(text: str) -> str:
    if not text:
        return text
    sanitized = text
    for char, name in INVISIBLE_UNICODE_CHARS.items():
        sanitized = sanitized.replace(char, _invisible_marker(char, name))
    output_lines: list[str] = []
    for line in sanitized.splitlines():
        matched_category = None
        for category, pattern in STORED_PROMPT_INJECTION_PATTERNS:
            if pattern.search(line):
                matched_category = category
                break
        if matched_category:
            output_lines.append(f"[blocked stored prompt-injection content: {matched_category}]")
            output_lines.extend(_line_invisible_markers(line))
        else:
            output_lines.append(line)
    return "\n".join(output_lines)


def _invisible_marker(char: str, name: str) -> str:
    return f"[blocked invisible unicode U+{ord(char):04X} {name}]"


def _line_invisible_markers(line: str) -> list[str]:
    markers: list[str] = []
    for char, name in INVISIBLE_UNICODE_CHARS.items():
        marker = _invisible_marker(char, name)
        if marker in line:
            markers.append(marker)
    return markers
