"""Knowledge-base documents: schema, loader and the shared sentence splitter (spec Task 2).

The same KBDocument model validates the bundled policies, documents added at runtime via
POST /add-document, and the input to both chunking strategies. Run
`python -m app.rag.documents` to print the inventory.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import ROOT

KB_PATH = ROOT / "data" / "knowledge_base" / "policies.json"

# Spec L79-83, in the spec's order. Keys are the stable `topic` metadata values.
REQUIRED_TOPICS: dict[str, str] = {
    "return_window": "Return window by product category",
    "cod_refund": "COD refund timelines",
    "delivery_sla": "Delivery SLAs",
    "reverse_pickup": "Reverse-pickup eligibility",
    "warranty": "Warranty terms by category",
    "order_cancellation": "Order-cancellation policy",
    "loyalty_points": "Loyalty-points redemption policy",
    "payment_failure": "Payment-failure/retry policy",
    "size_exchange": "Size-exchange policy",
    "damaged_item_claim": "Damaged-item claim process",
    "international_shipping": "International shipping restrictions",
    "escalation_matrix": "Customer-support escalation matrix",
}
MIN_SENTENCES, MAX_SENTENCES = 2, 5

Category = Literal["Apparel", "Electronics", "Home", "Footwear", "Beauty", "All"]

# A sentence ends at . ! or ? followed by whitespace and an upper-case letter, digit, ₹ or opening
# quote/bracket. Decimals ("4.5") and lower-case continuations ("e.g. the") therefore never split.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9₹\"'(])")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


class KBDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str = Field(pattern=r"^KB-\d{2,}$")
    title: str = Field(min_length=3)
    topic: str = Field(pattern=r"^[a-z][a-z_]*$")
    categories: list[Category] = Field(min_length=1)
    simulated: bool = True
    text: str

    @field_validator("text")
    @classmethod
    def _two_to_five_sentences(cls, text: str) -> str:
        n = len(split_sentences(text))
        if not MIN_SENTENCES <= n <= MAX_SENTENCES:
            raise ValueError(f"document text must have {MIN_SENTENCES}-{MAX_SENTENCES} sentences, got {n}")
        return text.strip()

    def chroma_metadata(self) -> dict[str, str | bool]:
        """Flat scalar metadata, the shape ChromaDB stores per chunk; doc_id maps chunks back to documents."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "topic": self.topic,
            "categories": ",".join(self.categories),
            "simulated": self.simulated,
        }


class KnowledgeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    disclaimer: str = Field(min_length=20)
    documents: list[KBDocument] = Field(min_length=1)


def load_knowledge_base(path: Path = KB_PATH) -> list[KBDocument]:
    kb = KnowledgeBase.model_validate_json(path.read_text(encoding="utf-8"))
    ids = [d.doc_id for d in kb.documents]
    if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
        raise ValueError(f"duplicate doc_id values: {duplicates}")
    return kb.documents


def main() -> None:
    docs = load_knowledge_base()
    print(KnowledgeBase.model_validate_json(KB_PATH.read_text(encoding="utf-8")).disclaimer)
    print(f"\n{len(docs)} documents loaded from {KB_PATH.relative_to(ROOT)}\n")
    print(f"{'doc_id':<7} {'topic':<23} {'sent':>4}  {'categories':<41} title")
    for d in docs:
        print(f"{d.doc_id:<7} {d.topic:<23} {len(split_sentences(d.text)):>4}  {','.join(d.categories):<41} {d.title}")
    covered = {d.topic for d in docs}
    missing = [label for key, label in REQUIRED_TOPICS.items() if key not in covered]
    print(f"\nRequired topics covered: {len(REQUIRED_TOPICS) - len(missing)}/{len(REQUIRED_TOPICS)}"
          + (f"  MISSING: {missing}" if missing else ""))
    print(f"All documents marked simulated: {all(d.simulated for d in docs)}")


if __name__ == "__main__":
    main()
