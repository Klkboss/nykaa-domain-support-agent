"""Charge a real model call against the per-request token budget: reserve before, settle on reported usage.

Before the call: reserve estimate(prompt) + max_tokens (the completion cap), so a call that cannot fit is refused
before anything is sent. After it: settle on the provider's reported prompt + completion tokens (release the
unused part, or charge the excess, which can raise BudgetExceeded and stop the request). A failed call reports no
usage and its reservation is released. If a provider omits usage, the len // 4 estimate is charged instead.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from app.governance.budget import RequestBudget, estimate_tokens


@dataclass
class Spend:
    reserved: int
    actual: int = 0  # set by the caller once the provider has answered


@contextmanager
def metered(budget: RequestBudget | None, prompt: str, max_tokens: int) -> Iterator[Spend]:
    spend = Spend(estimate_tokens(prompt) + max_tokens)
    if budget is not None:
        budget.charge(spend.reserved)  # raises BudgetExceeded before any request is sent
    try:
        yield spend
    finally:
        if budget is not None:
            if spend.actual <= spend.reserved:
                budget.release(spend.reserved - spend.actual)
            else:
                budget.charge(spend.actual - spend.reserved)


def reported_or_estimated(reported: int, prompt: str, completion: str) -> int:
    return reported if reported > 0 else estimate_tokens(prompt) + estimate_tokens(completion)
