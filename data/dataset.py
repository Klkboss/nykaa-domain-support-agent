"""Seeded, deterministic order dataset for the Nykaa support agent (spec Task 1).

Run `python data/dataset.py` to print the statistics, validate them and write data/orders.json.
Records are never edited after generation: if a constraint fails, change SEED or the weights.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

SEED = 42
N_RECORDS = 50
DELAY_PROBABILITY = 0.20

CATEGORY_WEIGHTS: dict[str, float] = {
    "Beauty": 0.30, "Apparel": 0.20, "Footwear": 0.15, "Home": 0.15, "Electronics": 0.20,
}
STATUS_WEIGHTS: dict[str, float] = {
    "Delivered": 0.35, "Shipped": 0.20, "Placed": 0.15, "Returned": 0.15, "Refunded": 0.15,
}
# Whole-rupee price bands per category, from mass-market cosmetics up to premium beauty devices.
AMOUNT_RANGES_INR: dict[str, tuple[int, int]] = {
    "Beauty": (199, 4_999), "Apparel": (399, 5_999), "Footwear": (499, 7_999),
    "Home": (299, 9_999), "Electronics": (999, 24_999),
}
MAX_DAYS = 30
MIN_PER_CATEGORY = 3
DELAYED_BAND = (0.10, 0.30)

REQUIRED_CATEGORIES = ("Apparel", "Electronics", "Home", "Footwear", "Beauty")
REQUIRED_STATUSES = ("Placed", "Shipped", "Delivered", "Returned", "Refunded")
FIELD_TYPES: dict[str, type] = {
    "record_id": str, "category": str, "status": str,
    "order_value_inr": int, "days_since_created": int, "delayed_shipment": bool,
}

OUTPUT_PATH = Path(__file__).with_name("orders.json")


def generate_orders(seed: int = SEED, n: int = N_RECORDS) -> list[dict[str, Any]]:
    rng = random.Random(seed)  # local generator: nothing else in the process can shift the sequence
    categories, cat_weights = list(CATEGORY_WEIGHTS), list(CATEGORY_WEIGHTS.values())
    statuses, status_weights = list(STATUS_WEIGHTS), list(STATUS_WEIGHTS.values())
    orders = []
    for i in range(n):
        category = rng.choices(categories, cat_weights)[0]
        low, high = AMOUNT_RANGES_INR[category]
        orders.append({
            "record_id": f"ORD-{1001 + i}",
            "category": category,
            "status": rng.choices(statuses, status_weights)[0],
            "order_value_inr": rng.randint(low, high),
            "days_since_created": rng.randint(0, MAX_DAYS),
            "delayed_shipment": rng.random() < DELAY_PROBABILITY,
        })
    return orders


def dataset_stats(orders: list[dict[str, Any]]) -> dict[str, Any]:
    delayed = sum(o["delayed_shipment"] for o in orders)
    return {
        "total": len(orders),
        "per_category": dict(Counter(o["category"] for o in orders).most_common()),
        "per_status": dict(Counter(o["status"] for o in orders).most_common()),
        "delayed_count": delayed,
        "delayed_pct": round(100 * delayed / len(orders), 2),
    }


def validate(orders: list[dict[str, Any]]) -> dict[str, Any]:
    """Raise ValueError listing every violated constraint; return stats when all pass."""
    errors: list[str] = []
    if len(orders) < 40:
        errors.append(f"need >=40 records, got {len(orders)}")
    ids = [o.get("record_id") for o in orders]
    if len(set(ids)) != len(ids):
        errors.append("record_id values are not unique")
    for o in orders:
        rid = o.get("record_id")
        if set(o) != set(FIELD_TYPES):
            errors.append(f"{rid}: fields {sorted(o)} != {sorted(FIELD_TYPES)}")
            continue
        # bool is a subclass of int, so exact type checks keep True out of the int fields.
        bad_types = [f for f, t in FIELD_TYPES.items() if type(o[f]) is not t]
        if bad_types:
            errors.append(f"{rid}: wrong type for {bad_types}")
            continue  # range checks below assume correct types
        if o["category"] not in AMOUNT_RANGES_INR:
            errors.append(f"{rid}: unknown category {o['category']}")
        else:
            low, high = AMOUNT_RANGES_INR[o["category"]]
            if not low <= o["order_value_inr"] <= high:
                errors.append(f"{rid}: order_value_inr {o['order_value_inr']} outside {low}-{high}")
        if o["status"] not in STATUS_WEIGHTS:
            errors.append(f"{rid}: unknown status {o['status']}")
        if not 0 <= o["days_since_created"] <= MAX_DAYS:
            errors.append(f"{rid}: days_since_created {o['days_since_created']} outside 0-{MAX_DAYS}")
    if errors:
        raise ValueError("; ".join(errors))

    stats = dataset_stats(orders)
    for category in REQUIRED_CATEGORIES:
        if stats["per_category"].get(category, 0) < MIN_PER_CATEGORY:
            errors.append(f"category {category} has {stats['per_category'].get(category, 0)} < {MIN_PER_CATEGORY}")
    for status in REQUIRED_STATUSES:
        if stats["per_status"].get(status, 0) < 1:
            errors.append(f"status {status} never appears")
    low_pct, high_pct = (100 * b for b in DELAYED_BAND)
    if not low_pct <= stats["delayed_pct"] <= high_pct:
        errors.append(f"delayed_shipment {stats['delayed_pct']}% outside {low_pct:.0f}-{high_pct:.0f}%")
    if errors:
        raise ValueError("; ".join(errors))
    return stats


ORDERS: list[dict[str, Any]] = generate_orders()


def main() -> None:
    stats = validate(ORDERS)
    print(f"Seed: {SEED} | records: {stats['total']} | delay probability: {DELAY_PROBABILITY}")
    print(f"Category weights: {CATEGORY_WEIGHTS}")
    print(f"Status weights:   {STATUS_WEIGHTS}")
    print(f"Amount ranges (INR): {AMOUNT_RANGES_INR}")
    print("\nCount per category (required >=3 each):")
    for category in REQUIRED_CATEGORIES:
        print(f"  {category:<12} {stats['per_category'].get(category, 0):>3}")
    print("\nCount per status (required >=1 each):")
    for status in REQUIRED_STATUSES:
        print(f"  {status:<12} {stats['per_status'].get(status, 0):>3}")
    print(f"\ndelayed_shipment=True: {stats['delayed_count']}/{stats['total']} = {stats['delayed_pct']}% (required 10-30%)")
    days = [o["days_since_created"] for o in ORDERS]
    values = [o["order_value_inr"] for o in ORDERS]
    print(f"days_since_created range: {min(days)}-{max(days)} | order_value_inr range: {min(values)}-{max(values)}")
    OUTPUT_PATH.write_text(json.dumps(ORDERS, indent=2) + "\n", encoding="utf-8")
    print(f"\nVALIDATION PASSED. Wrote {len(ORDERS)} records to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
