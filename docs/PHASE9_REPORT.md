# Phase 9 report: response caching (spec Task 16)

- **Scope:** spec Task 16 (assignment.md L226-229) and its acceptance bullet (L271-272). Nothing else.
- **Checklist:** [PHASE9_CHECKLIST.md](PHASE9_CHECKLIST.md), committed before any code (f39d452).
- **Status:** VERIFIED: independently approved (checkpoint `3a502d1`). The evidence below is kept as submitted.

## 1. Requirement mapping

| ID | Spec | Implementation | Evidence |
|---|---|---|---|
| T16.1 | "an in-memory cache keyed by normalized query text for your grounded-generation step" | `app/core/cache.py` `GenerationCache` + `CachedGenerator`; wired in `RagLookupTool`, `SupportAssistant`, `/add-document` | tests/test_cache.py |
| T16.2 | "a repeated identical query producing a cache hit that avoids a redundant LLM/tool call, with before/after evidence (a call counter or timing comparison)" | `MockGroundedGenerator.calls`, `cache_events`, cache stats, request budget | phase9_cache.txt; test_1_2…, test_12… |

The spec asks for nothing more: no expiry, no persistence and no cross-process cache. None was added.

## 2. Architecture and key

```
SupportAssistant.ask -> guardrails -> budget pre-flight -> memory -> crew
  Retrieval Agent -> rag_lookup -> answer_query:
       search() (ALWAYS runs) -> threshold -> CachedGenerator.generate(search_text, supporting chunks)
            eligible?  no  -> generate (event "skipped")
            key = SHA-256(normalised query, [chunk_id, doc_id, text, metadata]..., generator class/max_sentences/margin, embedding model)
            hit        -> stored text, no model call, no budget charge (event "hit")
            miss       -> _MeteredGenerator -> MockGroundedGenerator (charged) -> claim_supported vs the chunks?
                              yes -> store (event "miss")   no -> do not store (event "rejected")
            cache error -> generate normally (event "error"); BudgetExceeded is never caught
  -> claims/evidence -> Composer -> check_draft -> AutoGen review -> check_draft (final) -> render_reply
```

- **Normalisation:** whitespace is collapsed and text lower-cased. This is sound because the embedding model is
  uncased (`do_lower_case: true`) and `query_analysis` is case-insensitive. A test checks that a case/whitespace
  variant generates the same text, and that punctuation is NOT normalised away.
- **No collisions between policy contexts:** the context is part of the key. Fixed vs sentence collections give
  different keys, and an edited KB-01 ("30" → "45 days") gives a miss with the new text.
- **No stale answers:** a changed text, chunk set or metadata changes the key, and `/add-document` also clears the
  cache.

## 3. Eligibility

| Cached | Never cached (and how it is enforced) |
|---|---|
| Generated policy text for a query with supporting chunks, after `claim_supported` passes | Order-id / PII / PII-token queries: `cacheable_query` → neither read nor written |
| | Out-of-scope: the generator is never called (`not_generated`) |
| | Injection-blocked / budget-refused turns: they never reach the crew |
| | Text failing validation: `rejected` |
| | Order lookups, drafts, AutoGen verdicts, replies, history: the cache is only reachable from the generator call |

## 4. Privacy and isolation evidence

These are tests with fabricated data.
- **Keys:** only 64-hex digests. Values are verbatim KB sentences
  (`test_cache_holds_only_digests_and_knowledge_base_sentences`).
- **PII:** a raw phone number, a `[PHONE]` token and card-last-4 queries are not cached. The tool only ever sees the
  masked text.
- **Order data:** order queries leave nothing in the cache, and order facts never reach another session
  (`test_one_customers_order_facts_never_reach_another_session`).
- **Logs:** the cache writes nothing to the JSONL log (`test_11_add_document_clears_the_cache`).
- **Sharing:** one cache per assistant, shared by sessions. This is deliberate: it holds only KB policy sentences.

## 5. Hit / miss and invalidation demo

Source: [phase9_cache.txt](../evaluation/transcripts/phase9_cache.txt), deployed `chroma_db/`, socket guard,
0 network attempts.

| Request | Cache | Generation calls | Budget (est. tokens) | Retrieval / AutoGen / final check |
|---|---|---|---|---|
| Session A, "How long do I have to return shoes?" | miss | 1 | 5,150 | ran / approved / grounded |
| Session B, same question | **hit** | **0** | **4,941** (−209 = the generation charge) | ran / approved / grounded |

- **Same reply:** both replies are identical.
- **Ineligible cases:** order (`skipped`), PII (`skipped`), out-of-scope (`not_generated`) and injection (blocked)
  add no entries.
- **Invalidation:** on a private temporary index, original → miss, repeat → hit, edited KB-01 → miss with
  "45 days", and `clear()` → 0 entries.

## 6. Tests

tests/test_cache.py has **31 tests**, covering all twelve owner cases plus privacy and failure modes:

| Owner case | Covered by |
|---|---|
| 1-2. Miss, then hit | `test_1_2_…` and the budget-difference test |
| 3. Different policy context | `test_3_…` and the key-coverage test |
| 4. Updated policy | `test_4_…`: real re-index of KB-01 in a private index |
| 5. Order queries | `test_5_…` ×4 |
| 6. PII | `test_6_…` ×4 |
| 7. Unsupported query | `test_7_…`, plus the refusal test |
| 8. Groundedness rejection | `test_8_…` ×2 |
| 9. AutoGen revision | `test_9_…`: the cache holds only the tool's text, and the hit is still revised |
| 10. Cache failure | `test_10_…` ×3: broken reads or writes fall back to generation |
| 11. Clearing and bound | `clear()`, LRU bound, `/add-document` through the real API |
| 12. A hit still passes the final check | `test_12_…`: both `check_draft` calls run and the final report is clean |

**Mutation check.** I broke one rule at a time; each mutant was caught:

| Mutation | Tests that failed |
|---|---|
| Context removed from the key | 3 |
| Eligibility check removed | 10 |
| Validation removed | 2 |
| Cache errors propagated | 2 |

**One unconfirmed observation.** The first run of the eligibility mutant failed 19 tests. That run's output was
not saved. The failures matched the known in-memory Chroma failure: every test that uses the shared test index
failed, and the private-index and no-retrieval tests passed. Re-run with saved output, the same mutant failed
exactly the 10 expected tests.

## 7. Persistent Chroma verification

Source: [phase9_persistent_chroma.txt](../evaluation/transcripts/phase9_persistent_chroma.txt), produced by
`scripts/verify_persistent_chroma.py` (read-only).
- **Contents:** both collections equal the KB chunks, in ids, texts and metadata (42 and 55 chunks). The stored
  vectors match freshly computed ones (max difference 1.5e-08).
- **Repeated queries:** 50 rounds × 49 frozen queries × 2 collections = 4,900 searches. Results: 0 exceptions,
  0 empty results, and 0 wrong or changing results.
- **Chroma rewrites its own files.** The HNSW `.bin` files and `chroma.sqlite3` change bytes on open, even with
  reads only, so the content is verified logically. The project writes nothing to `chroma_db/`.
- **Injected failure:** an injected `chromadb.errors.InternalError("… Error finding id")` gives "Policy search is
  temporarily unavailable", with no internal text leaked.
- **Broken index:** an index that returns nothing (what the broken in-memory index did) gives the refusal.
- **No unsupported answer:** neither failure produces an unsupported answer, and neither caches anything.
- **Not changed:** no Chroma version change and no automatic rebuild.
- **Known limitation, unchanged:** the in-memory Chroma 1.5.9 "Error finding id" failure, trigger unknown. 4,900
  clean searches on the persistent index do not prove it cannot happen there.

## 8. Regression and changes to approved code

**Production code.** Only the wiring changed:
- `crew_tools.py`: the `cache` field and the wrapper;
- `pipeline.py`: `generation_cache`;
- `api/main.py`: clear on `/add-document`.

Retrieval, generation, guardrails, review and budget code are unchanged.

**One existing test: setup changed, assertions unchanged.**
- `test_repeated_requests_each_get_a_fresh_budget` now uses `GenerationCache(max_entries=0)`. Its third request
  repeats the question in a fresh session, so it would now be a cache hit with a smaller spend.
- `run_governance_demo.py` section 4 got the same change.

**Frozen evaluation.** It was re-run with the cache on.
- `evaluation/phase8/*` and the r2 files are byte-identical. The averages are unchanged (v1 4.7333 / 5 / 5 / 5,
  v3 5 / 5 / 5 / 5), and the original and r1 hashes are unchanged.
- The Phase 8 APPROVE, REVISE, tool-access and budget transcripts reproduce, apart from the footer and printed
  object ids.

**Full suite:** **663 passed** on 3 consecutive full runs (632 + 31).
