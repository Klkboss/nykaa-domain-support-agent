# Phase 9 checklist: response caching (spec Task 16)

Written before any code. Scope: spec Part 4 Task 16 (assignment.md L226-229) and the acceptance bullet L271-272.
Nothing else is in scope.

## Exact requirements

| ID | Spec text | Implementation | Test / evidence |
|---|---|---|---|
| T16.1 | "an in-memory cache keyed by normalized query text for your grounded-generation step" (L226-227) | `app/core/cache.py` `GenerationCache` (in-process dict, LRU-bounded). Wraps the generator call inside `answer_query`, AFTER retrieval and the threshold, BEFORE the model call. Key = SHA-256 of (normalised query text, fingerprint of the retrieved supporting context, generator config, embedding model) | tests/test_cache.py |
| T16.2 | "a repeated identical query producing a cache hit that avoids a redundant LLM/tool call, with before/after evidence (a call counter or timing comparison)" (L227-229, L271-272) | Hit returns the stored text; `MockGroundedGenerator.calls` stays 0; cache hit/miss counters; per-request budget shows no generation charge | tests/test_cache.py + `scripts/run_cache_demo.py` → evaluation/transcripts/phase9_cache.txt |

Owner Phase 0 decision (CLAUDE.md): cache ONLY grounded RAG generation; never order lookups or full responses;
clear on /add-document.

## Eligible for caching
- The text returned by the grounded generator for a query whose retrieval found supporting chunks above the
  threshold, and only after every sentence passes `claim_supported` against those chunks.

## Never cached
- Complete `SupportResponse`s, replies, drafts, AutoGen verdicts (the cache is never called from the pipeline).
- `check_order_status` output or anything order-specific; a query containing an order id (`ORD-nnnn`) is neither
  read from nor written to the cache.
- A query containing PII or a PII mask token (`[PHONE]`, `[CARD]`, `[CARD-LAST4]`).
- Out-of-scope queries (no chunk above threshold: the fallback is not generated, so nothing is stored).
- Guardrail refusals (injection-blocked and budget-refused turns never reach retrieval).
- Generated text that fails validation against the retrieved context.
- Session history (it is not part of the rag_lookup query).

## The cache does not bypass
Input guardrails, PII masking, injection detection, budget pre-flight, the crew, retrieval (it runs on every
request), the groundedness check, the AutoGen review, the final groundedness check. A hit replaces only the one
generation call.

## Key design
- Normalised query: whitespace collapsed and lower-cased. Sound because the embedding model is uncased
  (`do_lower_case: true`) and `query_analysis` patterns are case-insensitive; a test checks that the generated
  text is identical for case/whitespace variants.
- Context fingerprint: `(chunk_id, text)` of every supporting chunk, in order. A changed policy text, a different
  collection (fixed vs sentence) or a different chunk set gives a different key: no stale answer.
- Generator config: class name, `max_sentences`, `margin`; embedding model name.
- Stored key is the SHA-256 digest only: no raw query text in the cache.

## Clearing / expiry
- The spec requires neither expiry nor clearing. Owner decision: `/add-document` clears the cache. Entries are also
  bounded (LRU, 256). No TTL (not required; the context fingerprint already prevents stale answers).

## Tests (owner brief §5)
1 miss on first query · 2 hit on repeat · 3 different context → miss · 4 updated policy → miss · 5 order query
never uses the cache · 6 PII query not cached · 7 unsupported query not cached · 8 rejected generation not cached ·
9 AutoGen revision cannot put unvalidated text in the cache · 10 cache failure falls back to generation ·
11 clear() and /add-document clearing, LRU bound · 12 a hit still passes the final groundedness check.

## Regression (owner brief §6)
Full suite; frozen 15-query evaluation (phase 8 full-pipeline + r2) re-run and compared; existing transcripts
regenerated only if they change, with every difference explained.

## Persistent Chroma verification (owner brief §7)
`scripts/verify_persistent_chroma.py` (read-only): repeated queries on the deployed `chroma_db/` collection, ids
and texts compared with the KB chunks; injected `chromadb.errors.InternalError` handled safely (test). No Chroma
version change, no automatic rebuild.
