"""Metadata-aware query analysis for retrieval (Phase 3 review fix).

Root cause it addresses: dense MiniLM similarity is dominated by surface wording. For "My hair dryer
stopped working after three months. Is it covered?", KB-01's "Electronics such as hair dryers ... can be
returned within 7 days" scores 0.51, while KB-05's "Electronics come with a 12-month manufacturer warranty"
scores 0.19. Only KB-01 names hair dryers, and the query never says "warranty"; appending "warranty" alone
still leaves KB-01 ahead (0.55 vs 0.39). So the intent has to be applied through metadata:

1. Product -> category: product nouns map to the five catalogue categories, so the query can match
   category-level policy text ("Electronics").
2. Warranty routing: a query that names a product AND either says warranty/guarantee or describes a
   malfunction after weeks/months/years of use is restricted to topic="warranty" chunks (Chroma `where`)
   and expanded with its category and "warranty". Elapsed time in days is not a cue, because return windows
   are stated in days, so "stopped working after five days" stays a return or damaged-item question.

3. Escalation routing (Phase 7 remediation, E12): a query with an explicit escalation cue (escalate,
   senior specialist, grievance officer, Level 1-3, supervisor, manager) and NO cue for another policy topic
   is restricted to topic="escalation_matrix" chunks. Measured cause: for "My order ORD-1022 was delayed.
   Will my complaint be escalated to a senior specialist?", KB-10's damaged-item chunk took the third
   retrieval slot and displaced the KB-12 window that completes the best-matching sentence. The generator then
   preferred KB-10's complete sentence. No expansion is added, so a routed query whose top chunk was already
   KB-12 keeps exactly the same similarity (the calibration values do not move). Mixed questions (e.g. return +
   escalation) are not routed, because a single-topic filter would drop the other intent.

Queries that are not routed are searched exactly as before, with no expansion. Only confusions observed in
evaluation (return vs warranty; damaged-item vs escalation) are encoded, each with a frozen regression set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Electronics, Home and footwear examples include the KB's own wording (KB-01/KB-05); the rest are common
# catalogue nouns. A term also matches its plural ("dryers").
PRODUCT_TERMS: dict[str, tuple[str, ...]] = {
    "Electronics": ("hair dryer", "dryer", "straightener", "trimmer", "shaver", "epilator", "curler"),
    "Home": ("bedding", "bedsheet", "decor", "kitchenware", "mixer", "appliance", "cookware"),
    "Footwear": ("shoe", "sneaker", "sandal", "boot", "slipper", "footwear"),
    "Apparel": ("dress", "shirt", "kurta", "jeans", "saree"),
    "Beauty": ("lipstick", "cream", "perfume", "serum", "shampoo", "nail polish"),
}
_PRODUCT_RE = {
    category: re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")s?\b", re.I)
    for category, terms in PRODUCT_TERMS.items()
}
_EXPLICIT_WARRANTY = re.compile(r"\b(warrant(y|ies)|guarantee[sd]?)\b", re.I)
_MALFUNCTION = re.compile(
    r"\b(stopped working|no longer \w+|not working|broke( down)?|broken|came apart|fell apart|faulty|repair(ed)?)\b", re.I
)
_ELAPSED_USE = re.compile(
    r"\b(a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|few|several|\d+)\s+(week|month|year)s?\b"
    r"|\blast (month|year)\b",
    re.I,
)


_ESCALATION = re.compile(
    r"\b(escalat\w*|senior (?:support )?specialists?|grievance officer|level [123]|supervisor|manager)\b", re.I
)
# Cues of the OTHER eleven policy topics. Escalation's own words ("complaint", "unresolved") are not listed.
_OTHER_POLICY_TOPIC = re.compile(
    r"\b(return|refund|cancel|exchang|swap|size|warrant|guarantee|repair|deliver|dispatch|ship|pick ?up|courier|"
    r"point|loyalty|pay|cod\b|cash on delivery|damag|broke|claim|international|abroad|customs)\w*",
    re.I,
)
EXPANDED_TOPICS = {"warranty"}  # only the warranty route needs expansion (category bridging + intent word)


@dataclass(frozen=True)
class QueryAnalysis:
    query: str
    categories: tuple[str, ...]
    topic: str | None  # metadata topic the search is restricted to, or None

    @property
    def search_text(self) -> str:
        if self.topic not in EXPANDED_TOPICS:
            return self.query
        return " ".join([self.query, *self.categories, self.topic])

    @property
    def where(self) -> dict[str, str] | None:
        return {"topic": self.topic} if self.topic else None


def analyse_query(query: str) -> QueryAnalysis:
    categories = tuple(c for c, pattern in _PRODUCT_RE.items() if pattern.search(query))
    warranty = bool(categories) and bool(
        _EXPLICIT_WARRANTY.search(query) or (_MALFUNCTION.search(query) and _ELAPSED_USE.search(query))
    )
    if warranty:
        return QueryAnalysis(query, categories, "warranty")
    if _ESCALATION.search(query) and not _OTHER_POLICY_TOPIC.search(query):
        return QueryAnalysis(query, categories, "escalation_matrix")
    return QueryAnalysis(query, categories, None)
