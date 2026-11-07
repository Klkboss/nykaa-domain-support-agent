"""Order lookup tool with a designed escalation score (spec Task 6).

    escalation_score = 0.6 * delayed_shipment + 0.4 * (days_since_created / 30)

- delayed_shipment (0 or 1) is the explicit SLA-breach signal, so it carries the larger weight: every
  delayed order (0.6-1.0) outranks every non-delayed order (0.0-0.4).
- days_since_created / 30 is the normalised recency (age) signal. 30 is the dataset's maximum age, so the
  term is in [0, 1]. At a fixed delay flag, an older order scores higher.

Escalation is recommended when score >= ESCALATION_THRESHOLD = 0.6 + 0.4 * (13.5 / 30) = 0.78, where 13.5
is the median days_since_created of the generated ORDERS. In words: escalate a delayed order once it is
older than the typical order (14+ days). A non-delayed order never crosses the threshold on age alone.
"""

from __future__ import annotations

import re
from typing import Any

from data.dataset import MAX_DAYS, ORDERS

DELAY_WEIGHT = 0.6
AGE_WEIGHT = 0.4
ESCALATION_AGE_DAYS = 13.5  # median days_since_created of ORDERS (checked by tests)
ESCALATION_THRESHOLD = round(DELAY_WEIGHT + AGE_WEIGHT * ESCALATION_AGE_DAYS / MAX_DAYS, 4)  # 0.78

RECORD_ID_RE = re.compile(r"ORD-[0-9]{4}")  # not \d: that also matches non-ASCII digits such as ١
_ORDERS_BY_ID: dict[str, dict[str, Any]] = {o["record_id"]: o for o in ORDERS}


class OrderLookupError(Exception):
    """Base class. Messages never echo raw caller input."""


class InvalidRecordIdError(OrderLookupError, ValueError):
    pass


class OrderNotFoundError(OrderLookupError, LookupError):
    pass


def normalise_record_id(record_id: object) -> str:
    if not isinstance(record_id, str):
        raise InvalidRecordIdError(f"record_id must be a string, got {type(record_id).__name__}")
    candidate = record_id.strip().upper()
    if not RECORD_ID_RE.fullmatch(candidate):
        raise InvalidRecordIdError("record_id must look like ORD-1234 (ORD- followed by 4 digits)")
    return candidate


def escalation_score(delayed_shipment: bool, days_since_created: int) -> float:
    """Raises instead of guessing when an input is missing or invalid."""
    if type(delayed_shipment) is not bool:
        raise ValueError("delayed_shipment must be a bool")
    if type(days_since_created) is not int or not 0 <= days_since_created <= MAX_DAYS:
        raise ValueError(f"days_since_created must be an int in 0-{MAX_DAYS}")
    return DELAY_WEIGHT * delayed_shipment + AGE_WEIGHT * days_since_created / MAX_DAYS


def check_order_status(record_id: str) -> dict[str, Any]:
    """Return status, order_value_inr, escalation_score and the threshold decision for one order."""
    key = normalise_record_id(record_id)
    order = _ORDERS_BY_ID.get(key)
    if order is None:
        raise OrderNotFoundError(f"No order found with record_id {key}")  # key is validated, safe to echo
    score = escalation_score(order["delayed_shipment"], order["days_since_created"])
    return {
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": round(score, 4),
        "escalation_recommended": score >= ESCALATION_THRESHOLD,
    }
