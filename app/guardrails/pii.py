"""Input guardrail 1: mask the spec's fixed-format PII (L85-86): phone numbers and payment-card last-4 digits.

Applied before anything reaches the crew, the session history or (Phase 9) the logs, so all of them see the
same masked text. Free-text names and addresses have no reliable pattern and are out of scope (spec L87-89).
Order ids (ORD-1234), rupee amounts and day counts are left intact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

PHONE_TOKEN = "[PHONE]"
CARD_LAST4_TOKEN = "[CARD-LAST4]"
CARD_TOKEN = "[CARD]"

# Full card numbers first (13-19 digits, optional single spaces or hyphens), so their tail is not seen as a phone.
_FULL_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
# Indian mobile numbers: optional +91 / 91 / 0 prefix, 10 digits starting 6-9, optional one space/hyphen inside.
_PHONE = re.compile(r"(?<![\w-])(?:\+?91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?![\w-])")
# Last-4 needs card context so ordinary 4-digit numbers (years, ORD-1234) are untouched.
_CARD_LAST4 = re.compile(
    r"(?P<lead>\b(?:card|visa|mastercard|rupay|amex|debit|credit)\b[^.?!\n]{0,40}?"
    r"(?:ending(?: in| with)?|last (?:4|four)(?: digits)?(?: (?:are|is|were))?|no\.?|number)[\s:#-]*"
    r"|\blast (?:4|four) digits\b[^.?!\n]{0,30}?\b(?:card|visa|mastercard|rupay|amex)\b(?: (?:are|is|were))?[\s:#-]*"
    r"|(?:[x*•]{4}[ -]?){1,3})"
    r"(?P<digits>\d{4})(?!\d)",
    re.I,
)


@dataclass(frozen=True)
class MaskResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def masked(self) -> bool:
        return any(self.counts.values())


def mask_pii(text: str) -> MaskResult:
    counts = {"card_number": 0, "phone": 0, "card_last4": 0}

    def sub(pattern: re.Pattern[str], replacement, key: str, value: str) -> str:
        new, n = pattern.subn(replacement, value)
        counts[key] += n
        return new

    text = sub(_FULL_CARD, CARD_TOKEN, "card_number", text)
    text = sub(_PHONE, PHONE_TOKEN, "phone", text)
    text = sub(_CARD_LAST4, lambda m: m.group("lead") + CARD_LAST4_TOKEN, "card_last4", text)
    return MaskResult(text, counts)
