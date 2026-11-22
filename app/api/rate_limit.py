"""v2 public-demo protection: a per-client request limit (a sliding minute window plus a UTC-day cap).

Off unless configured (RATE_LIMIT_PER_MINUTE / RATE_LIMIT_PER_DAY > 0), so the capstone API behaves exactly as
before; the Docker image turns it on. Only questions are counted (POST /ask, each WebSocket question).
shortcut: in-process memory, so limits are per process and reset on restart; a multi-instance deploy needs a shared
store (Redis or the database).
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from collections.abc import Callable

DAY_S, MINUTE_S = 86_400, 60.0


class RateLimiter:
    def __init__(self, per_minute: int = 0, per_day: int = 0, clock: Callable[[], float] = time.time) -> None:
        self.per_minute, self.per_day, self._clock = per_minute, per_day, clock
        self._lock = threading.Lock()
        self._recent: dict[str, deque[float]] = {}
        self._day = -1  # the current UTC day number
        self._today: dict[str, int] = {}  # questions per client that day

    @classmethod
    def from_env(cls) -> RateLimiter:
        return cls(int(os.environ.get("RATE_LIMIT_PER_MINUTE") or 0), int(os.environ.get("RATE_LIMIT_PER_DAY") or 0))

    @property
    def enabled(self) -> bool:
        return self.per_minute > 0 or self.per_day > 0

    def check(self, client: str) -> float | None:
        """Count one question for `client`. None = allowed; otherwise the seconds to wait (nothing is counted)."""
        if not self.enabled:
            return None
        now = self._clock()
        with self._lock:
            day = int(now // DAY_S)
            if day != self._day:  # a new UTC day: every daily count starts again (and old clients are forgotten)
                self._day, self._today, self._recent = day, {}, {}
            recent = self._recent.setdefault(client, deque())
            while recent and recent[0] <= now - MINUTE_S:
                recent.popleft()
            if self.per_day and self._today.get(client, 0) >= self.per_day:
                return (day + 1) * DAY_S - now
            if self.per_minute and len(recent) >= self.per_minute:
                return recent[0] + MINUTE_S - now
            recent.append(now)
            self._today[client] = self._today.get(client, 0) + 1
            return None
