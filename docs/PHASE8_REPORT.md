# Phase 8 report: AutoGen review and AI governance (spec Tasks 14-15)

Status: **VERIFIED: independently approved** (approved checkpoint `29f48df`, after the intermittent-failure
investigation below). The evidence below is kept as it was submitted for review.

- **Baseline:** approved checkpoint `5641cce`. The first Phase 8 checkpoint was `3740ce3`; this report also
  covers the fixes from the owner's review of it.
- **Checklist:** frozen before any code in [PHASE8_CHECKLIST.md](PHASE8_CHECKLIST.md) (`ae39635`).
- **Environment:** everything runs under `MOCK_LLM` with zero API keys. Every transcript was recorded under the
  socket guard with 0 outbound attempts.
- **Not in scope:** Task 16 (caching).

## 1. Pipeline code and execution order
[app/pipeline.py](../app/pipeline.py), `SupportAssistant.ask` → `_answer`:

1. **Input guardrails.** One-line normalisation, then `mask_pii`, then `detect_injection`. A blocked turn runs
   no crew and is never stored.
2. **Budget pre-flight.** `project_request_tokens(question, history) > 12,000` refuses the request with
   `budget_exceeded` before any model call. The turn is not stored.
3. **Memory.** `RunnableWithMessageHistory` wraps steps 4–9.
4. **The crew.** `run_support_crew(..., budget)`:
   - `build_support_crew` calls `enforce_crew_least_autonomy` first.
   - Every `MockCrewLLM` call is charged to the request budget.
   - `rag_lookup` charges its grounded-generation call.
5. **Schema check.** `SupportDraft.model_validate`.
6. **First groundedness check.** `check_draft(draft, rag_output, order_output)`, the Phase 6 guardrail,
   unchanged.
7. **The review.** `review_policy_draft(draft, rag_output, budget)` runs the AutoGen review on the Composer's
   policy draft plus the original retrieved evidence. It reserves its upper bound first, and the reviewer and
   editor calls are metered against that reservation.
8. **Applying the verdict.** If `final_answer != draft`, only the checked claims whose sentence the editor kept
   survive. The editor's own text is never used.
9. **Final groundedness check.** `check_draft(...)` runs **again** on the reviewed draft, with the same tool
   outputs. The two reports are merged.
10. **Rendering.** `render_reply(final checked draft)`, then `SupportResponse`.

## 2. Final groundedness validation (review point 1)
**At `3740ce3`:**
- The reply could only contain claims that had passed step 6. The editor's text was used only as a filter and
  was never rendered.
- But nothing re-validated the reviewed draft, and nothing recorded what the editor tried to add.

**Now:**
- **The final check.** Step 9 runs the same `check_draft` on the reviewed draft against the same evidence, and
  only its output is rendered.
- **Discarded additions.** Sentences the editor wrote that are not in the draft are recorded in
  `ReviewResult.discarded_additions` and never rendered.
- **Order facts.** These come only from the checked draft, which step 9 re-checks against the real
  `check_order_status` output.

Tests ([test_review.py](../tests/test_review.py)). A spy on the pipeline's `check_draft` proves it runs twice and
that the reply equals `render_reply` of the second result. A patched Final_Editor client inside the real AutoGen
team simulates a misbehaving editor.

| # | Case | Test | Result |
|---|---|---|---|
| 1 | Grounded draft; AutoGen approves | `test_case1_and_5_…` | The final check sees the guardrail's draft unchanged; it is clean |
| 2 | Composer adds an unsupported claim; AutoGen removes it | `test_case2_…` | Only the Footwear claim reaches the final check; the reply excludes the claim |
| 3 | The editor adds "All orders ship free worldwide." and "Order ORD-1029 has status Refunded." | `test_case3_…[approved=False/True]` | Both are discarded, the reply excludes them, and the order line keeps the real status, Delivered |
| 4a | The editor appends "[KB-12]" to the sentence | `test_case4_citations_…` | The edited sentence no longer matches the checked claim; the claim is dropped and the reply refuses (fallback). No KB-12 appears |
| 4b | The editor rewrites the claim with "(see KB-12)" | `test_case4_a_claim_rewritten_…` | The same: discarded, the reply refuses, no KB-12 |
| 4c | The Composer cites KB-12 wrongly | `test_case4_composer_wrong_citation_…` | The guardrail removes it, the final check stays clean, and the reply cites only KB-01 |
| 5 | Fully grounded and correctly cited | `test_case1_and_5_…` | Every rendered citation's document text supports its claim |

## 3. Verdict schema and semantics (review point 2)
`Verdict(approved: bool, final_answer: str, reason: str)`. These are exactly the spec's fields (L210-211),
unchanged.

**What `approved` means:** the spec says the stage "either approves it unchanged or revises it". So `approved`
judges **the Composer's draft**, not the final answer.

| Value | Meaning | `final_answer` |
|---|---|---|
| `approved: true` | The draft is approved unchanged | The draft |
| `approved: false` | The draft was revised | The corrected answer; `reason` says why |

**The REVISE demo is consistent with the spec.** Its `approved: false` together with a corrected
`final_answer` is exactly the spec's "revises it" case.

**Inconsistent verdicts:** the pipeline decides on `final_answer != draft`, not on `approved`. So an "approval"
that still changes the text is handled as a revision: removals are honoured and additions discarded.

**Tests:**
- `test_verdict_semantics_approved_judges_the_composer_draft` (both paths).
- `test_an_approval_that_still_changes_the_text_is_treated_as_a_revision`.
- `test_verdict_has_exactly_the_spec_fields`.

Real verdicts ([approve](../evaluation/transcripts/phase8_autogen_approve.txt),
[revise](../evaluation/transcripts/phase8_autogen_revise.txt)):
```json
{"approved": true, "final_answer": "Footwear can be returned within 30 days of delivery if it is unworn and sent back in its original shoebox.", "reason": "All claims are supported by the retrieved context."}
{"approved": false, "final_answer": "Footwear can be returned within 30 days of delivery if it is unworn and sent back in its original shoebox.", "reason": "Removed 1 unsupported claim(s): Premium members get instant cash refunds at any store."}
```

## 4. Model-call budget accounting (review point 3)
**At `3740ce3`, three gaps:**
- AutoGen calls were covered only by an up-front reservation, never charged individually.
- The grounded-generation call was not charged at all.
- After a refused charge, a later smaller charge could still succeed.

**Now:**

| Model call | Charged by | Charged when |
|---|---|---|
| Every CrewAI agent call, including CrewAI retries | `MockCrewLLM.call` | Prompt before it is sent, completion before it is returned |
| Grounded generation in `rag_lookup` | `_MeteredGenerator.generate` | Prompt (question + chunks) before, answer after |
| AutoGen reviewer and editor calls | `MockChatCompletionClient.create` | Against a meter capped at the review's reservation |
| The review reservation | `review_policy_draft` | Charged up front; the unused part released afterwards |

**How a stop behaves:**
- After a refused charge, `RequestBudget.exhausted` stays set, so every later charge is refused.
- A stop inside AutoGen, where AutoGen wraps exceptions, is re-raised as `BudgetExceeded`.
- A stop inside `rag_lookup` is re-raised instead of being swallowed as "policy search unavailable".

**Not counted:** the local MiniLM embedding model, which is local compute with no token cost.

**Evidence from one real request** ([phase8_runtime_budget.txt](../evaluation/transcripts/phase8_runtime_budget.txt),
section 0): 15 charges in total.

| Charged item | Charges | Tokens |
|---|---|---|
| CrewAI calls (4 calls) | 8 | 4,494 |
| Grounded generation | 2 | 209 |
| Review reservation | 1 | 904 |
| Reviewer and editor (2 calls) | 4 | 447 |

457 tokens of the reservation were released. The request used 5,150 of 12,000.

**Tests** ([test_governance.py](../tests/test_governance.py)):
- `test_every_model_call_in_a_request_is_charged`: a spy attributes every charge to its code path; the number
  of charges equals 2 × the model calls; nothing is uncharged.
- `test_grounded_generation_call_is_charged`: an exact token match, and a stop is not swallowed.
- `test_retried_model_calls_are_charged`: a Composer fails once, CrewAI retries, and the charges equal a normal
  run's plus 1.
- `test_a_budget_stop_is_sticky`.
- `test_a_budget_stop_inside_autogen_surfaces_as_budget_exceeded`.
- The earlier boundary tests: below, at and above the cap; exactly at the projection, and one token over;
  repeated requests with fresh budgets; exhaustion mid-crew and at the review; the oversized request; the API.

**Estimated, not actual tokens:**
- All token figures are estimates, `len(text) // 4`.
- The cap is on this accounted estimate. Under MOCK_LLM no model consumes real tokens, and a real provider's
  usage report could differ.
- So the system guarantees that **estimated** spend never passes 12,000. It does **not** guarantee an actual
  12,000-token limit at a provider.

**Formula and cap:** unchanged (`7,000 + 6 × tokens(question + history)`, cap 12,000).
- With every call now charged, the largest part of spend not proportional to the input rose from 6,496 to
  6,692. That is still below 7,000, so the projection remains an upper bound (tested).
- Measured spend: the 15 evaluation queries use 3,163–6,396 tokens; a 2,000-character question uses about
  8,500–8,700.

## 5. Evaluation execution path (review point 4)
**What already ran:** `scripts/run_evaluation.py` calls `SupportAssistant.ask`, so since Phase 8 it has run the
**full** pipeline, CrewAI plus AutoGen.

**What was missing:** the r2 files record replies and scores only. So "byte-identical" was true but showed no
AutoGen evidence.

**What was added:** a separate run, [scripts/run_phase8_evaluation.py](../scripts/run_phase8_evaluation.py),
which writes [evaluation/phase8/](../evaluation/phase8/FULL_PIPELINE_EVALUATION.md). It uses the same 15 frozen
queries through `SupportAssistant.ask`, and records per query:
- whether AutoGen ran, with its message sources and types, verdict, discarded additions and metered spend;
- the final reply, captured after AutoGen;
- an independent re-check of that reply against the same tool outputs;
- scores under rubrics v1 and v3, compared with r2.

The Phase 7 artifacts (the originals, r1 and r2) are not modified, and their hashes are checked.

**Results:**
- **AutoGen ran on 12 of 15 queries** (E01–E12), and every verdict was approved.
- **It did not run on three:** E13 (an out-of-scope fallback, no policy text), E14 (blocked before the crew)
  and E15 (an order-only turn).
- **Every final reply re-checks clean**, and each equals the rendered final draft.
- **Every reply and score equals r2:** rubric v1 4.7333 / 5 / 5 / 5, rubric v3 5 / 5 / 5 / 5. Nothing was
  altered.

**Tests:** [test_phase8_evaluation.py](../tests/test_phase8_evaluation.py) covers the live re-run, these
invariants, and the same 15 ids in order.

**Found while testing:** `run_case` (Phase 7) uses fixed `eval-…` session ids. Running the evaluation twice in
one process leaked history into the second run. The new script empties those sessions first; the approved
runner is unchanged.

## 6. Least autonomy, risk and governance
These are unchanged since `3740ce3`:
- **Tool boundaries:** 14 positive and negative cases plus the Phase 5 test, all passing
  ([phase8_tool_access.txt](../evaluation/transcripts/phase8_tool_access.txt)).
- **Risk classification:** **Medium**.
- **Four layers:** Application and Runtime as named in the spec; Data and Organizational as the owner's
  interpretation, labelled.

All of it is in [GOVERNANCE.md](GOVERNANCE.md), now updated for the final check and the complete accounting.

## 7. Regression and test results
- **Full suite: 632 passed** (631 + the session-leak regression test) on 3 consecutive full runs. This does
  not prove the suite is free of the intermittent Chroma failure above.
  - 582 tests from Phases 0–7, unchanged.
  - 32 from `3740ce3`.
  - 17 new: test_review.py +9, test_governance.py +5, test_phase8_evaluation.py 3.
- **Existing tests:** no Phase 0–7 test file changed (`git diff 5641cce -- tests/` touches only the Phase 8
  files).
- **Corrected test:** one Phase 8 test now measures the crew's spend after the reservation is released.
- **The 8 extra Phase 3/6/7 failures of one run** (investigated on owner request):
  - **Signature (reproduced with full output):** `chromadb.errors.InternalError: Error executing plan: Internal
    error: Error finding id`, raised in Chroma's compiled layer. After it, every query on the in-memory test
    index returns no results, while its ids and documents stay intact. Once broken, the index stays broken.
  - **Not caused by Phase 8 code:** `tests/test_rag.py` alone failed 50 of 145 tests in 1 of 3 runs, and a
    standalone script using only Phase 3 code (no pytest) once built an index whose every read failed.
  - **Ruled out:** client garbage collection, our upsert sequence, test parallelism (none is configured).
  - **Not identified:** the trigger inside Chroma 1.5.9. Similar 1.5.x reports exist for `PersistentClient`.
  - **No code change** for it, because no root cause lies in this project's code. If it happened in the
    deployed path, the reply would be "Policy search is temporarily unavailable" or the refusal: safe, but no
    answer. That path is not verified. Mitigation is an owner decision.
  - **Session leak:** now covered by a deterministic regression test, which fails without the fix with the
    original symptom.
- **Untouched code:** the API, logging, guardrails, retrieval, order tool and memory code.
- **Changed for budget metering only:** `crew_tools.py` (opt-in budget on `rag_lookup`; a budget stop is
  re-raised).
- **Phase 7 artifacts:** none changed. The r2 evaluation is still reproduced live.

## 8. Remaining limitations
- **Review strength under MOCK_LLM.** The reviewer uses the guardrail's lexical rule, so in the live pipeline
  it agrees with the guardrail. Revisions happen only when the Composer or the editor misbehaves.
- **The review can only remove claims.** A correct rewording by the editor is also discarded: the claim is
  dropped, or the reply refuses.
- **Estimated tokens.** All token figures are `len // 4` estimates, and cost is counted in tokens only. The
  projection is empirical.
- **Long sessions.** A long session of near-maximum messages is refused once projected over 12,000.
- **Pinned internals.** The AutoGen guard reads private attributes, pinned to autogen-agentchat 0.7.5.
- **Cosmetic warnings.** CrewAI prints two "Event pairing mismatch" warnings when a budget stop ends a task.

## 9. Commit
Commit hash, working-tree status and push status are given in the hand-over message.
