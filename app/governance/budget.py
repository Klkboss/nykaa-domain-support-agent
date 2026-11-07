"""Runtime governance layer (spec Task 15): a per-request token budget cap.

One cap, two controls:
  1. Pre-flight: before any model call, the request's total token cost is projected from its input (question +
     session history). A request whose projection exceeds the cap is refused: nothing is sent to a model.
  2. Accounting: every MOCK_LLM call charges a fresh per-request `RequestBudget`: each CrewAI agent call, the
     grounded-generation call inside rag_lookup, and each AutoGen reviewer and editor call (prompt before it is
     sent, completion before it is returned). The review first reserves its upper bound, its calls are metered
     against that reservation, and the unused part is released afterwards. A charge that would pass the cap
     raises `BudgetExceeded`; after that every further charge is refused, so the request stops.
Tokens are ESTIMATED as len(text) // 4 (about 4 characters per token), the estimate the mock clients report. The
cap bounds this accounted estimate; it is not a count from a real tokenizer (under MOCK_LLM no model consumes
real tokens; a real provider's usage report could differ from the estimate).
The budget is per request: every request starts at zero, nothing carries over between requests.
"""

from __future__ import annotations

from crewai.hooks.dispatch import HookAborted

# Measured on real runs with every model call charged (evaluation/transcripts/phase8_runtime_budget.txt): the 15
# evaluation queries spend 3,163-6,396 estimated tokens each; 2,000-character questions (the API maximum) about
# 8,500-8,700. The cap is about 1.9x the largest evaluation request: normal sessions fit, oversized ones are refused.
MAX_REQUEST_TOKENS = 12_000
# Upper-bound projection fitted to measured spend: about 6 tokens per question token and 4 per history token
# (both covered by 6), plus at most 6,692 tokens not proportional to the input. Tests check projection >= spend.
PROJECTION_BASE_TOKENS = 7_000
PROJECTION_INPUT_COPIES = 6


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def project_request_tokens(question: str, history: str = "") -> int:
    return PROJECTION_BASE_TOKENS + PROJECTION_INPUT_COPIES * estimate_tokens(question + history)


class BudgetExceeded(HookAborted):
    """A deliberate stop. Subclassing CrewAI's HookAborted makes the crew re-raise it at once (no task retries)."""

    def __init__(self, needed: int, used: int, cap: int) -> None:
        super().__init__(f"request token budget exceeded: {used} used + {needed} needed > cap {cap}",
                         source="RequestBudget")
        self.needed, self.used, self.cap = needed, used, cap


class RequestBudget:
    def __init__(self, cap: int = MAX_REQUEST_TOKENS) -> None:
        self.cap = cap
        self.used = 0
        self.exhausted = False  # once a charge is refused, every later charge is refused too

    def charge(self, tokens: int) -> None:
        """Spend `tokens`, or raise without spending if that would pass the cap (reaching it exactly is allowed)."""
        if self.exhausted or self.used + tokens > self.cap:
            self.exhausted = True
            raise BudgetExceeded(tokens, self.used, self.cap)
        self.used += tokens

    def release(self, tokens: int) -> None:
        """Return the unused part of a reservation."""
        self.used -= min(tokens, self.used)
