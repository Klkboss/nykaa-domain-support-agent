"""Threshold calibration and document-level precision/recall (spec Tasks 4-5)."""

from __future__ import annotations

from typing import Any


def dedupe(ids: list[str]) -> list[str]:
    """Order-preserving de-duplication (chunks -> parent documents)."""
    return list(dict.fromkeys(ids))


def precision_recall(retrieved_doc_ids: list[str], relevant_doc_ids: list[str]) -> dict[str, Any]:
    """Document-level precision and recall, with the arithmetic spelled out."""
    if not relevant_doc_ids:
        raise ValueError("relevant_doc_ids must not be empty")
    retrieved = dedupe(retrieved_doc_ids)
    relevant = set(relevant_doc_ids)
    hits = [d for d in retrieved if d in relevant]
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len(hits) / len(relevant)
    return {
        "retrieved_docs": retrieved,
        "relevant_docs": sorted(relevant),
        "relevant_retrieved": hits,
        "precision": precision,
        "recall": recall,
        "precision_arithmetic": f"{len(hits)}/{len(retrieved)} = {precision:.3f}",
        "recall_arithmetic": f"{len(hits)}/{len(relevant)} = {recall:.3f}",
    }


def choose_threshold(in_scope_scores: list[float], out_of_scope_scores: list[float]) -> dict[str, float]:
    """Midpoint of the gap between the two observed clusters. Raises if they overlap."""
    if len(in_scope_scores) < 3 or len(out_of_scope_scores) < 2:
        raise ValueError("spec needs >= 3 in-scope and >= 2 out-of-scope measurements")
    lowest_in, highest_out = min(in_scope_scores), max(out_of_scope_scores)
    if highest_out >= lowest_in:
        raise ValueError(f"clusters overlap: max out-of-scope {highest_out:.4f} >= min in-scope {lowest_in:.4f}")
    threshold = round((lowest_in + highest_out) / 2, 3)
    return {"min_in_scope": lowest_in, "max_out_of_scope": highest_out,
            "gap": lowest_in - highest_out, "threshold": threshold}
