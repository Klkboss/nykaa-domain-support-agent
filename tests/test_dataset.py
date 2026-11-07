"""Spec Task 1: reproducibility, structural thresholds and validator negative cases."""

from __future__ import annotations

import copy
import json

import pytest

from data import dataset
from data.dataset import (
    AMOUNT_RANGES_INR, FIELD_TYPES, ORDERS, OUTPUT_PATH, REQUIRED_CATEGORIES, REQUIRED_STATUSES,
    SEED, generate_orders, validate,
)


def test_generation_is_deterministic_and_matches_module_list():
    assert generate_orders(SEED) == generate_orders(SEED) == ORDERS
    assert generate_orders(SEED + 1) != ORDERS  # the seed actually drives the data


def test_global_random_state_does_not_affect_generation():
    import random

    random.seed(999)
    random.random()
    assert generate_orders() == ORDERS


def test_record_count_and_unique_ids():
    assert len(ORDERS) == 50
    ids = [o["record_id"] for o in ORDERS]
    assert len(set(ids)) == 50
    assert ids[0] == "ORD-1001" and ids[-1] == "ORD-1050"


def test_field_types_and_ranges():
    for o in ORDERS:
        assert set(o) == set(FIELD_TYPES)
        for field, expected in FIELD_TYPES.items():
            assert type(o[field]) is expected, (o["record_id"], field)
        assert 0 <= o["days_since_created"] <= 30
        low, high = AMOUNT_RANGES_INR[o["category"]]
        assert low <= o["order_value_inr"] <= high


def test_spec_thresholds():
    stats = validate(ORDERS)
    assert all(stats["per_category"][c] >= 3 for c in REQUIRED_CATEGORIES)
    assert all(stats["per_status"][s] >= 1 for s in REQUIRED_STATUSES)
    assert 10 <= stats["delayed_pct"] <= 30


def test_saved_json_matches_generator():
    assert json.loads(OUTPUT_PATH.read_text(encoding="utf-8")) == ORDERS


# --- The validator must reject each kind of violation (proves the checks are real). ---

def _broken(mutate) -> list[dict]:
    orders = copy.deepcopy(ORDERS)
    mutate(orders)
    return orders


@pytest.mark.parametrize("mutate, message", [
    (lambda o: o.__delitem__(slice(39, None)), ">=40 records"),
    (lambda o: o[1].update(record_id=o[0]["record_id"]), "not unique"),
    (lambda o: o[0].update(days_since_created=31), "days_since_created 31"),
    (lambda o: o[0].update(days_since_created=-1), "days_since_created -1"),
    (lambda o: o[0].update(days_since_created=5.0), "wrong type"),
    (lambda o: o[0].update(delayed_shipment=1), "wrong type"),
    (lambda o: o[0].update(order_value_inr=True), "wrong type"),
    (lambda o: o[0].update(order_value_inr=10**7), "outside"),
    (lambda o: o[0].update(status="Lost"), "unknown status"),
    (lambda o: o[0].pop("category"), "fields"),
])
def test_validator_rejects_record_level_violations(mutate, message):
    with pytest.raises(ValueError, match=message):
        validate(_broken(mutate))


def test_validator_rejects_too_few_in_a_category():
    def drop_home(orders):
        home = [i for i, o in enumerate(orders) if o["category"] == "Home"]
        for i in reversed(home[2:]):  # keep only 2 Home records
            orders.pop(i)
    orders = _broken(drop_home)
    orders += copy.deepcopy([o for o in ORDERS if o["category"] != "Home"][:10])  # stay >= 40 records
    for i, o in enumerate(orders):
        o["record_id"] = f"ORD-{2001 + i}"
    with pytest.raises(ValueError, match="category Home has 2 < 3"):
        validate(orders)


def test_validator_rejects_missing_status():
    orders = _broken(lambda o: [r.update(status="Delivered") for r in o if r["status"] == "Placed"])
    with pytest.raises(ValueError, match="status Placed never appears"):
        validate(orders)


@pytest.mark.parametrize("delayed_count", [4, 16])  # 8% and 32% of 50
def test_validator_rejects_delayed_share_outside_band(delayed_count):
    orders = _broken(lambda o: [r.update(delayed_shipment=i < delayed_count) for i, r in enumerate(o)])
    with pytest.raises(ValueError, match="delayed_shipment"):
        validate(orders)


def test_main_prints_report(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset, "OUTPUT_PATH", tmp_path / "orders.json")
    dataset.main()
    out = capsys.readouterr().out
    assert "VALIDATION PASSED" in out and "Seed: 42" in out
    assert json.loads((tmp_path / "orders.json").read_text(encoding="utf-8")) == ORDERS
