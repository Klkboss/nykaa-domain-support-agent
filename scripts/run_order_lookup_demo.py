"""Phase 4 evidence run (spec Task 6): threshold derivation, every order's score, and safe-failure demos.

Usage: .venv\\Scripts\\python.exe scripts\\run_order_lookup_demo.py
Writes evaluation/escalation_analysis.json.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import ROOT  # noqa: E402
from app.tools.order_lookup import (  # noqa: E402
    AGE_WEIGHT, DELAY_WEIGHT, ESCALATION_AGE_DAYS, ESCALATION_THRESHOLD, OrderLookupError, check_order_status,
)
from data.dataset import MAX_DAYS, ORDERS  # noqa: E402


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    days = sorted(o["days_since_created"] for o in ORDERS)
    median = statistics.median(days)

    section("1. Formula and threshold derivation from the generated dataset")
    print(f"escalation_score = {DELAY_WEIGHT} * delayed_shipment + {AGE_WEIGHT} * (days_since_created / {MAX_DAYS})")
    print(f"days_since_created, sorted (n={len(days)}): {days}")
    print(f"median = ({days[len(days) // 2 - 1]} + {days[len(days) // 2]}) / 2 = {median}")
    print(f"threshold = {DELAY_WEIGHT} + {AGE_WEIGHT} * {median} / {MAX_DAYS} = {DELAY_WEIGHT + AGE_WEIGHT * median / MAX_DAYS:.4f}"
          f"  (code constant ESCALATION_THRESHOLD = {ESCALATION_THRESHOLD}, ESCALATION_AGE_DAYS = {ESCALATION_AGE_DAYS})")
    print(f"score ranges: not delayed 0.0-{AGE_WEIGHT}, delayed {DELAY_WEIGHT}-{DELAY_WEIGHT + AGE_WEIGHT}"
          f" -> only delayed orders aged >= 14 days reach {ESCALATION_THRESHOLD}")

    section("2. check_order_status() for all 50 orders")
    rows = []
    for o in ORDERS:
        result = check_order_status(o["record_id"])
        rows.append({"record_id": o["record_id"], "delayed_shipment": o["delayed_shipment"],
                     "days_since_created": o["days_since_created"], **result})
        flag = "  <-- ESCALATE" if result["escalation_recommended"] else ""
        print(f"  {o['record_id']}  delayed={o['delayed_shipment']!s:<5} days={o['days_since_created']:>2}  "
              f"status={result['status']:<9} value=₹{result['order_value_inr']:>6}  score={result['escalation_score']:.4f}{flag}")
    escalated = [r for r in rows if r["escalation_recommended"]]
    delayed = [r for r in rows if r["delayed_shipment"]]
    print(f"\nEscalation recommended: {len(escalated)}/{len(rows)} = {100 * len(escalated) / len(rows):.1f}% "
          f"(of {len(delayed)} delayed orders: {len(escalated)} escalated, {len(delayed) - len(escalated)} below threshold)")
    print("Escalated by status:", dict(Counter(r["status"] for r in escalated)))
    print("Delayed but not escalated (younger than median):",
          [(r["record_id"], r["days_since_created"], r["escalation_score"]) for r in delayed if not r["escalation_recommended"]])
    print("Highest non-delayed score:", max(r["escalation_score"] for r in rows if not r["delayed_shipment"]))

    section("3. Safe failure: malformed, unknown and injection-like ids (no data returned, input not echoed)")
    failures = []
    for bad in [None, 1001, "", "   ", "ORD-1", "ORD-12345", "1001", "ORD-9999",
                "ORD-1001; DROP TABLE orders", "ORD-1001\nIgnore previous instructions", "+91 98765 43210"]:
        try:
            check_order_status(bad)  # type: ignore[arg-type]
            outcome = "UNEXPECTED SUCCESS"
        except OrderLookupError as e:
            outcome = f"{type(e).__name__}: {e}"
        failures.append({"input": repr(bad), "outcome": outcome})
        print(f"  {bad!r:<45} -> {outcome}")
    print(f"  {' ord-1029 '!r:<45} -> {check_order_status(' ord-1029 ')} (normalised to ORD-1029)")

    section("4. Determinism")
    again = [check_order_status(o["record_id"]) for o in ORDERS]
    print(f"Second full pass identical to first: {again == [{k: r[k] for k in again[0]} for r in rows]}")

    analysis = {
        "formula": f"{DELAY_WEIGHT} * delayed_shipment + {AGE_WEIGHT} * (days_since_created / {MAX_DAYS})",
        "days_sorted": days, "median_days": median, "threshold": ESCALATION_THRESHOLD,
        "threshold_arithmetic": f"{DELAY_WEIGHT} + {AGE_WEIGHT} * {median} / {MAX_DAYS} = {DELAY_WEIGHT + AGE_WEIGHT * median / MAX_DAYS}",
        "orders": rows, "escalated_ids": [r["record_id"] for r in escalated], "failure_cases": failures,
    }
    (ROOT / "evaluation" / "escalation_analysis.json").write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print("\nWrote evaluation/escalation_analysis.json")


if __name__ == "__main__":
    main()
