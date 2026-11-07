"""Input guardrail 2: prompt-injection detection (spec Task 10).

A transparent, deterministic pattern detector. It flags deliberate attempts to override instructions,
hijack the assistant's role, extract hidden prompts, pull bulk customer data, or forge prompt structure. It
does NOT claim complete protection: paraphrased, obfuscated or non-English attacks can get through. That is why
the output-side groundedness guardrail also checks every fact against the actual tool outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_INSTR = r"(?:instructions?|rules?|prompts?|guidelines?|directions?|polic(?:y|ies)|guardrails?)"
PATTERNS: dict[str, re.Pattern[str]] = {
    "instruction_override": re.compile(
        rf"\b(?:ignore|disregard|forget|override|bypass|skip)\b[^.?!]{{0,40}}\b{_INSTR}\b"
        rf"|\bnew {_INSTR}\s*:|\bfrom now on,? (?:you|ignore|answer)",
        re.I,
    ),
    "role_hijack": re.compile(
        r"\byou are now (?:an?|the|in|acting|operating|dan)\b|\byou are no longer\b|\bpretend (?:to be|you are)\b"
        r"|\bact as (?:an? )?(?:admin|administrator|"
        r"developer|system|root|dan|jailbr\w*|unrestricted)\b|\b(?:developer|god|dan|jailbreak) mode\b",
        re.I,
    ),
    "prompt_exfiltration": re.compile(
        r"\b(?:reveal|show|print|repeat|output|display|tell me|leak)\b[^.?!]{0,30}\b(?:system prompt|hidden "
        r"(?:prompt|instructions?)|your (?:instructions|prompt|rules)|initial prompt)\b",
        re.I,
    ),
    "bulk_data_request": re.compile(
        r"\b(?:list|dump|export|show|give me|print)\b[^.?!]{0,25}\b(?:all|every|each)\b[^.?!]{0,15}"
        r"\b(?:orders|customers|records|users|phone numbers|cards)\b|\bdump (?:the )?(?:database|db|table)\b",
        re.I,
    ),
    "forged_structure": re.compile(
        r"<\|im_(?:start|end)\|>|\[/?INST\]|<<\s*SYS\s*>>|^\s*(?:system|assistant)\s*:|###\s*(?:system|instruction)"
        r"|\bthis is the context you're working with\b|\bobservation\s*:|\bfinal answer\s*:|\baction input\s*:",
        re.I | re.M,
    ),
}

BLOCKED_REPLY = (
    "I can only help with Nykaa orders and policies. Your message looked like an attempt to change how I work, "
    "so I did not process it. Please rephrase your question."
)


@dataclass(frozen=True)
class InjectionResult:
    detected: bool
    categories: tuple[str, ...]


def detect_injection(text: str) -> InjectionResult:
    hits = tuple(name for name, pattern in PATTERNS.items() if pattern.search(text))
    return InjectionResult(bool(hits), hits)
