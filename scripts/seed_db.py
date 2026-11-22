"""v2: seed the orders table with the capstone's 50 simulated orders (data/dataset.py, same seed). Idempotent:
existing rows are overwritten with the dataset's values, nothing else is touched.

Usage (after `alembic upgrade head`, with DATABASE_URL set):
  .venv\\Scripts\\python.exe -m scripts.seed_db
"""

from __future__ import annotations

import sys

from app.db import database_url, session
from app.db.models import Order
from data.dataset import ORDERS


def seed() -> int:
    with session() as s, s.begin():
        for order in ORDERS:
            s.merge(Order(**order))
    return len(ORDERS)


def main() -> int:
    if not database_url():
        print("DATABASE_URL is not set; nothing to seed.", file=sys.stderr)
        return 1
    print(f"seeded {seed()} orders")
    return 0


if __name__ == "__main__":
    sys.exit(main())
