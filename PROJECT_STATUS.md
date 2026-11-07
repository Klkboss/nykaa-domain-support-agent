# PROJECT_STATUS

Status: **all build phases 0-9 complete and independently approved** (spec Tasks 1-16). Final submission-readiness
audit done on 3a502d1; its findings are fixed in the submission-cleanup commit (see below).
Approved checkpoints: Phase 0 893e16d · 1 35cba5a · 2 b5b9ef4 · 3 5d5df26 · 4 1399c5d · 5 388adbb · 6 403a799 ·
7 5641cce · 8 29f48df · 9 3a502d1.
Not done yet: the public GitHub repository (trace G1) and an originality assessment (G12).
Note: Task 9 (structured output), originally planned as Phase 7, was delivered in Phase 6 at the owner's request.
Requirements checklist: [docs/REQUIREMENTS_TRACE.md](docs/REQUIREMENTS_TRACE.md) (spec: docs/assignment.md)
Last updated: 2026-11-07

## Environment
Python 3.12.10 venv · pip 25.0.1 · RTX 4050 6 GB (torch is the CPU build, `cuda available: False`, by design)
· RAM 15.7 GB (2.2 GB free at check) · git on `main`

## Installed stack (verified with `pip check`: no broken requirements; 168 packages, full lock in requirements.lock.txt)
crewai 1.15.33 · autogen-agentchat/core/ext 0.7.5 · chromadb 1.5.9 · sentence-transformers 6.2.0 · torch 2.15.0
· transformers 5.22.1 · langchain-core 1.7.3 · fastapi 0.162.0 · uvicorn 0.54.0 · pydantic 2.12.5 · httpx 0.28.1 · pytest 9.2.0

## Tests
Current full suite (2026-11-07, after the submission cleanup): `.venv\Scripts\python.exe -m pytest -q` → **664 passed**, 0 failed, 0 skipped, with outbound network blocked. Known limit: the intermittent in-memory ChromaDB failure (see Known limits) can fail retrieval tests in a run; a re-run passes.

Phase 0 spike tests (historical, 2026-11-01): `pytest -v` → 9 passed, network blocked, 0 recorded attempts.
| Test | Proves |
|---|---|
| test_network_guard (2) | guard blocks + records outbound DNS/connect; loopback allowed |
| test_telemetry_and_tracing_disabled | `Telemetry._is_telemetry_disabled()` True, `should_enable_tracing()` False |
| test_crew_kickoff_invokes_real_tool_chosen_by_schema | real `crew.kickoff()`, real tool `_run`, schema dispatch beats decoy `order_policy_search`, Observation read from assistant turn only |
| test_no_record_id_means_no_tool_call | no fillable slot → no tool call |
| test_grounded_draft_is_approved_unchanged | RoundRobinGroupChat(max_turns=2), StructuredMessage[Verdict], approved unchanged |
| test_flawed_draft_is_revised | planted unsupported claim removed, approved=False |
| test_verdict_schema_rejects_bad_payload | Pydantic validation rejects bad verdicts |
| test_multi_turn_memory_and_session_isolation | RunnableWithMessageHistory wrapping crew + AutoGen; id recalled in session A, absent in fresh session B |

## Phase 1: dataset (spec Task 1), VERIFIED (35cba5a)
- `data/dataset.py`: `random.Random(42)`, 50 records, module-level `ORDERS`, `validate()`, writes `data/orders.json`
- Seed 42 passed on the first draw, so no seed or weight change was needed. No record was hand-edited.
- Stats: categories Apparel 9 · Electronics 13 · Home 6 · Footwear 13 · Beauty 9. Statuses Placed 3 · Shipped 8 · Delivered 24 · Returned 4 · Refunded 11. Delayed 6/50 = 12.0%.
- Tests: `tests/test_dataset.py` **21 passed**. Full suite **30 passed**, 2 expected LangChain warnings (2026-11-01).
- Transcript: `evaluation/transcripts/phase1_dataset.txt`
- Note: 12.0% delayed is inside the band but near its 10% floor. Realised counts drift from the weights (e.g. Footwear 13 vs weight 0.15) because n=50 is a small sample. All thresholds still hold.

## Phase 2: knowledge base (spec Task 2), VERIFIED (b5b9ef4)
- `data/knowledge_base/policies.json`: 12 simulated documents KB-01…KB-12, one per required topic, 4-5 sentences each, file-level disclaimer and per-document `simulated: true`
- `app/rag/documents.py`: `KBDocument` (Pydantic, extra=forbid, 2-5 sentence validator), `load_knowledge_base()` (duplicate-id check), shared `split_sentences()`, `chroma_metadata()`
- `app/core/mock_autogen.py` now uses the shared splitter (its private copy was removed)
- Tests: `tests/test_knowledge_base.py` **55 passed**, including a real ChromaDB upsert with dummy embeddings under the network guard. Full suite **85 passed**, 2 expected LangChain warnings (2026-11-01).
- Highest pairwise word overlap between documents: 0.186 (KB-06/KB-08); the near-duplicate limit is 0.35
- Transcript: `evaluation/transcripts/phase2_knowledge_base.txt`

## Phase 3: RAG core (spec Tasks 3-5), VERIFIED (5d5df26)
- Model: all-MiniLM-L6-v2@1110a243 (384-d, CPU), downloaded once by scripts/download_model.py into models/ (91.6 MB, gitignored), loaded offline.
- Chunking: fixed 40 words / 10 overlap → 42 chunks; sentence 1/chunk → 55 chunks (54 before the Phase 7 KB-08 clarification). Collections: policies_fixed_overlap, policies_sentence_based (cosine, upsert).
- Calibration (5 in + 5 out): fixed min-in 0.4247 / max-out 0.2360 → **0.33**; sentence 0.4120 / 0.1910 → **0.301**. No overlap.
- P/R at k=3, Q1-Q6 (search() pipeline): fixed P 0.8056 R 1.0000 F1 0.8923; sentence P 0.8333 R 0.9167 F1 0.8730 → **deploy fixed** (F1 rule). First submission: both P 0.7222, recall 1.0000 / 0.9167.
- Labels frozen before retrieval in commit 7d845be. The return-vs-warranty regression set was frozen in aa6de72 before the fix.
- **Review rework (Q3 rejected by owner):** the root cause was that dense similarity is dominated by the product word ("hair dryer" only appears in KB-01), while the warranty intent is implicit. Measured and rejected: query expansion alone, title headers, expanding every query, hybrid keyword scoring. Implemented `app/rag/query_analysis.py` (product → category; warranty routing via Chroma `where={"topic":"warranty"}` + expansion, only for product + warranty/malfunction-after-weeks/months/years) and `search()` as the single retrieval pipeline. Q3 top-1 is now the KB-05 Electronics 12-month sentence/chunk in both collections. Intent set top-1 relevant 4/7 → 7/7 in both. Calibration re-measured via search(): unchanged, unrounded midpoints 0.33034375 → 0.330 and 0.30149090 → 0.301.
- Tests: tests/test_rag.py **110 passed** (real model, network blocked). Full suite **195 passed**, 4 warnings: 2 expected LangChain + 2 chromadb "legacy embedding function config" DeprecationWarnings (2026-11-02).
- conftest network guard no longer clears its log at test start, so attempts during import or a session fixture are caught.
- Development note: the first evaluation run showed weak mock-generator answers (fixed-window fragments, padded unrelated second sentences). I added overlap-window rejoining, a 0.10 relevance margin, and a complete-sentence preference applied only among eligible sentences. Retrieval, thresholds and P/R were not affected.
- Remaining limitations:
  - The Q3 answer has a second, low-value warranty sentence (Apparel/Beauty carry no warranty).
  - Routed similarities come from expanded text, and no calibration query is routed.
  - The routing lexicon covers only the observed return-vs-warranty confusion.
  - The intent set was written knowing the failure mode, so it is not unbiased.

## Phase 4: order lookup + escalation score (spec Task 6), VERIFIED (1399c5d)
- Checklist written before code: docs/PHASE4_CHECKLIST.md
- `app/tools/order_lookup.py`: `check_order_status` (status, order_value_inr, escalation_score, escalation_recommended); `escalation_score = 0.6·delayed + 0.4·days/30`; threshold **0.78** = 0.6 + 0.4 × 13.5/30, where 13.5 = the dataset's median days_since_created
- Result: 4/50 escalated (ORD-1016, -1022, -1029, -1045; all delayed and 14+ days old). Delayed but younger: ORD-1031 (0.6933), ORD-1038 (0.6). Highest non-delayed score 0.3867.
- Safe failure: typed errors; ASCII-only id regex (found and fixed: `\d` accepted Arabic-Indic digits); no input echo; result is a fresh dict
- Tests: tests/test_order_lookup.py **109 passed**. Full suite **304 passed**, 4 warnings (2 expected LangChain, 2 chromadb deprecation) (2026-11-02)
- Transcript: evaluation/transcripts/phase4_order_lookup.txt; evaluation/escalation_analysis.json
- Assumptions (documented, not spec):
  - "recency" = elapsed age
  - `escalation_recommended` is added to the output
  - the score is status-independent
  - KB-12's ₹10,000 rule is left to the Composer
- Not yet integrated: the CrewAI BaseTool wrapper is Phase 5 (tests/spike_support.py still holds the spike tool)

## Phase 5: CrewAI crew (spec Task 7), VERIFIED (388adbb)
- Checklist written before code: docs/PHASE5_CHECKLIST.md
- app/tools/crew_tools.py: `rag_lookup` (args_schema query → Phase 3 `answer_query` on policies_fixed_overlap, threshold 0.33) and `check_order_status` (args_schema record_id → Phase 4 function + threshold). JSON outputs, generic errors, invocation logs.
- app/agents/crew.py: Retrieval / Lookup / Composer, sequential; lookup `context=[]`; composer `context=[retrieval, lookup]`; one MockCrewLLM per agent; real `crew.kickoff()`.
- app/core/mock_llm.py extended: tool-less agents compose from JSON after CrewAI's context header; no-id handling judged from the question line only; record-id extraction is now case-insensitive.
- Found and fixed during testing: a doubled full stop after tool error messages ("unavailable..").
- Demos (real kickoff, socket guard active, 0 outbound attempts):
  - A: rag_lookup only
  - B: check_order_status(ORD-1016) → Returned, ₹8,504, 0.9467, escalate
  - C: both tools
  - edge cases: unknown order, out-of-scope, missing id
- Tests: tests/test_crew.py **27 passed**. Full suite **331 passed**, 4 warnings (2 expected LangChain, 2 chromadb deprecation) (2026-11-02). Shared Chroma fixtures moved to tests/conftest.py.
- Transcript: evaluation/transcripts/phase5_crewai.txt
- Limitations:
  - The policy citation lists every document above the threshold (e.g. KB-01, KB-12 in Demo C), not only the quoted one.
  - The Composer's policy-cue and order-intent rules are mock heuristics.
  - Answers on the fixed collection can be window fragments (Phase 3 behaviour).
  - The Retrieval Agent always searches, even for pure order questions; the Composer drops the result.

## Phase 6: memory + structured output + guardrails (spec Tasks 8-10), VERIFIED (403a799)
- Checklist written before code: docs/PHASE6_CHECKLIST.md
- New modules:
  - app/pipeline.py (`SupportAssistant`)
  - app/schemas.py (`SupportDraft` etc.)
  - app/guardrails/pii.py, injection.py, groundedness.py
- Changed modules:
  - crew.py: history input, Composer `output_pydantic=SupportDraft`, optional composer LLM, richer `CrewRun`
  - crew_tools.py: per-sentence `claims` + `evidence`, `outputs` log
  - mock_llm.py: the Composer emits SupportDraft JSON and reads the LAST context header
  - mock_autogen.py: uses the shared `claim_supported`
- Demonstrated (transcript, socket guard, 0 outbound attempts):
  - A same-session memory (ORD-1016 from turn 1); B fresh session asks for the id
  - SupportDraft validation; invalid composer → safe error
  - phone + card masked before every prompt and in history
  - injection blocked and not stored; benign "ignore my previous message" passes
  - out-of-scope refusal; invented claim rejected → refusal; wrong citation removed; altered order status rejected
  - Demo C cites KB-01 only
- Found and fixed during testing:
  - card-last-4 phrasing "the last four digits of my card are 5678" was not masked
  - "you are now …" was too broad (would block "You are now my favourite store!")
  - a stderr warning interleaved with stdout hid one transcript line (the crewai base_llm logger is quietened in the demo script only)
- Tests: tests/test_guardrails.py **96 passed**, tests/test_pipeline.py **26 passed**; full suite **453 passed**. LangChain deprecation warnings are shown, not silenced (2026-11-02).
- Limitations:
  - Injection detection is pattern-based, not complete.
  - The PII regexes cover Indian mobile formats and common card phrasings only.
  - The groundedness coverage rule (80%) is lexical.
  - A rejected order fact turns the reply into the generic refusal.
  - History is capped at 10 messages.
  - Error turns are stored in history (masked).

## Phase 7: FastAPI + logging + evaluation (spec Tasks 11-13), VERIFIED (5641cce)
- Checklist and frozen 15-query set + rubric committed before any run: 1585558
- app/api/main.py + schemas.py:
  - `/ask`, `/add-document` (KBDocument validation, both collections, 409 on duplicates), `/ws/chat`
  - safe 422 (no pydantic `input` echo) and 500 responses
  - per-process temp index (approved chroma_db/ untouched)
- app/observability/logging.py: JSONL with timestamp, trace_id, method/event, endpoint, status, outcome, duration_ms; `mask_pii` over every string
- app/judge.py + scripts/run_evaluation.py: LLM-as-judge prompt → MockJudgeLLM (deterministic rubric scorer, disclosed)
- Evaluation averages: accuracy 4.33, grounding 5.00, completeness 5.00, safety 5.00
  - E01 = 1: a substring-forbidden rule fired on a padded Beauty sentence
  - E08 = 1: the strict fact match missed "3 times"
  - E12 = 3: a real retrieval miss, KB-10 instead of KB-12
- Real uvicorn demos (socket guard, 0 non-loopback attempts): HTTP A-F and WebSocket G-I. The TCP drop mid-turn was logged as `disconnect` 1006; a new client and HTTP were then served.
- Found and fixed during testing:
  - TestClient cancels the server task when a client leaves mid-turn (`CancelledError`, not caught by `except Exception`) → explicit handler
  - the mock judge parsed a fake marker typed inside the question → it now anchors on a line-start marker
  - a `websockets` DeprecationWarning in the demo client
- Tests: tests/test_api.py **27 passed**, tests/test_evaluation.py **18 passed**; full suite **498 passed** (2026-11-02)
- Limitations:
  - The judge is not a real LLM, and its grounding re-uses the guardrail rule.
  - Expected-fact and forbidden-fact checks are substring matches.
  - `/add-document` changes are lost on restart; sessions are in-process.
  - No authentication or rate limiting.
  - The TestClient vs real-server disconnect paths differ (both handled and tested).

## Phase 7 remediation: Task 13 findings E01, E08, E12 (VERIFIED, part of the Phase 7 checkpoint 5641cce)
- Frozen before code (0984a62): evaluation/rag_escalation_queries.json (11 regression queries) and evaluation/remediation/rubric_v2.json (revised rubric, labelled)
- Originals preserved and hash-locked: evaluation_results.json e5e7f560…, EVALUATION_REPORT.md 716bd15e…, test_queries.json 3699ac04…
- E12 root cause: KB-12 was in fact ranked 1st and 2nd. KB-10 took slot 3 and displaced the KB-12 window that completes the best sentence, and the complete-sentence preference then chose KB-10. Fix: escalation route in app/rag/query_analysis.py (shared `search()`; no expansion, so the calibration is unmoved)
- E01 root cause: the Beauty sentence was within the 0.10 margin. Fix: category-subject filter in `MockGroundedGenerator`. E05 reply changed (scores did not): re-verified as correct. A trimmer is Electronics; "12-month" stays first; the Home "6-month" sentence (wrong category) was dropped; the category-neutral "No warranty covers physical damage…" exclusion took its place. No filter change was needed. Regression test: test_category_filter_regression_e01_and_e05_end_to_end
- E08 (checkpoint d5ffadb, then re-verified on owner request): v2 was WRONG. It equated "After 3 failed attempts" (attempts in total) with KB-08's "retry the payment up to 3 times" (retries after the first attempt). **v2 withdrawn**; rubric **v3** (evaluation/remediation/rubric_v3.json, frozen in a822a91 before use) binds E08's "3 times" to the retry event. E08 accuracy is 1 under v1 and v3. KB-08's two count rules are ambiguous; flagged for the owner, KB unchanged
- Results:
  - original v1: 4.33 / 5 / 5 / 5
  - remediation, rubric v1: 4.73 / 5 / 5 / 5 (like-for-like)
  - remediation, rubric v2 (WITHDRAWN): 5 / 5 / 5 / 5, a false E08 pass
  - remediation, rubric v3 (revised): 4.73 / 5 / 5 / 5, identical to v1
  - only E01, E12 scores (and E08 under withdrawn v2) and E01, E05, E12 replies changed
- Tests: full suite **567 passed** (API and logging unchanged and passing). The verification added 22 tests and removed the 5 v2 "accepts" tests (17 net), on top of 52 from d5ffadb
- Remaining limitations:
  - E12's Level-2 trigger sentence is split by fixed windows
  - sentence collection D1/D2 misses (unrouted, pre-existing)
  - a generic "return window for each category?" question falls below the threshold
  - (resolved by the KB-08 clarification below) E08 reply under KB-08 v1 did not state the retry limit
  - the Phase 3 transcript is historical: the Q3 and Q5 demo answers now differ. Q3 drops the Apparel/Beauty no-warranty sentence; Q5 drops the Electronics restriction and adds the international transit-time sentence. Retrieval, P/R and thresholds are unchanged.

## Phase 7 KB-08 clarification (owner-approved; VERIFIED, 5641cce)
- Interpretation (owner): 3 additional retries after the initial failed attempt, at most 4 attempts in total
- Audit trail:
  - ce967bf preserved wording v1 in data/knowledge_base/history/KB-08_wording_v1.json (reason, evaluations that used it, r1 hashes)
  - 79d87a9 made the evaluation hash locks LF-normalised (portable across OS checkouts; the previous locks hashed this machine's CRLF copies) and froze the r1 results
- KB-08 wording v2:
  - s1 "If the first payment attempt fails … retry the payment up to 3 more times within 30 minutes …"
  - s3 "Once all 3 retries have also failed, … switch …"
  - new s5 "The first attempt plus 3 retries means at most 4 payment attempts in total."
  - s2 and s4 verbatim
- Wording chosen against fixed criteria:
  - approved points stated; unrelated facts verbatim
  - no movement of approved Phase 3 results; complete-sentence E08 answer
  - rejected variants: longer s1 dropped fixed precision to 0.7222, which would flip the recommended strategy; a long s3 was cut by 40-word windows
- Phase 3 re-run (transcripts/phase7_kb08_phase3_regression.txt):
  - unchanged: thresholds 0.33/0.301, P/R/F1 both strategies, chunking_comparison.json, rag_intent_regression.json
  - rag_threshold_calibration.json: two out-of-scope similarities fell; one routed_topic label refreshed
  - chroma_db/ re-indexed by the script (sentence 55 chunks)
- E08 end to end:
  - retrieved KB-08::fixed::00/01/02 (0.5611/0.5567/0.4974)
  - reply states "all 3 retries" and "3 more times"; both claims cite KB-08; grounded
- Evaluation r2 (evaluation/remediation/*_r2_kb08v2_*, EVALUATION_REPORT_R2_KB08_CLARIFIED.md):
  - rubric v1: 4.7333 / 5 / 5 / 5 (E08 still 1: literal "3 times" absent)
  - rubric v3: 5 / 5 / 5 / 5 (E08 1 → 5)
  - only the E08 reply changed vs r1
- Tests: **582 passed**
  - 15 new: KB-08 consistency ×2, E08 retrieval, E08 and exhausted-limit pipeline, r2 consistency, and 9 count cases for three additional retries, four total attempts, three total attempts, two and four retries, and "3 more times"
  - sentence-chunk literal 54 → 55 in tests/test_rag.py (×4) and tests/test_api.py (×1, the approved-chroma_db count check; API code unchanged)

## Phase 8: AutoGen review + AI governance (spec Tasks 14-15), VERIFIED (29f48df)
- Checklist (frozen before code, ae39635): [docs/PHASE8_CHECKLIST.md](docs/PHASE8_CHECKLIST.md). Report: [docs/PHASE8_REPORT.md](docs/PHASE8_REPORT.md). Governance: [docs/GOVERNANCE.md](docs/GOVERNANCE.md)
- AutoGen review (Task 14):
  - app/review/autogen_review.py: RoundRobinGroupChat(Policy_Compliance_Reviewer, Final_Editor), max_turns=2; Final_Editor output_content_type=Verdict (approved, final_answer, reason); team custom_message_types=[StructuredMessage[Verdict]]
  - wired into SupportAssistant: the Composer's policy draft + the retrieved context; a revision removes the flagged claims; the Phase 6 groundedness guardrail still runs unchanged
  - APPROVE and REVISE demonstrated through the real pipeline (transcripts phase8_autogen_approve.txt / _revise.txt)
- Least autonomy (Task 15, Application layer): app/governance/least_autonomy.py per-role tool allow-list enforced in build_support_crew (no delegation; order tool detected by name or schema); AutoGen reviewers hold no tools (AutoGen also refuses tools for their client). Transcript phase8_tool_access.txt
- Risk: **Medium** (docs/GOVERNANCE.md)
- Runtime budget (Task 15, Runtime layer): app/governance/budget.py
  - 12,000 tokens per request; pre-flight projection 7,000 + 6 × tokens(question + history) refuses oversized requests before any model call (not stored in memory)
  - per-call accounting in MockCrewLLM, plus a review reservation; BudgetExceeded subclasses CrewAI HookAborted (no retries)
  - new status `budget_exceeded` with a fixed reply. Transcript phase8_runtime_budget.txt
- Changes to approved code (all required by the Phase 8 integration):
  - crew.py: guard call + budget parameter
  - mock_llm.py: optional budget charging
  - pipeline.py: pre-flight, budget, review stage
  - schemas.py: one new status value
  - autogen_review.py: guard, reservation, pipeline entry point
  - no existing test changed; the r2 evaluation re-run is byte-identical (all 15 replies and scores unchanged)
- Tests at 3740ce3: **614 passed** (32 new: test_review.py 6, test_governance.py 26)
- Owner review of 3740ce3 (four points), fixed and verified:
  1. Final groundedness check after AutoGen: the reviewed draft is re-checked with `check_draft` against the same tool outputs; only that is rendered; editor-added sentences, citations and order facts are discarded (5 explicit test cases)
  2. Verdict semantics: `approved` judges the Composer's draft (spec: "approves it unchanged or revises it"); an approval that still changes the text is treated as a revision
  3. Budget accounting completed: grounded generation and each AutoGen reviewer/editor call are now charged (the review meters against its reservation and releases the rest); retries charged; the stop is sticky; stops inside AutoGen / rag_lookup are not swallowed. The cap is on ESTIMATED tokens (len // 4); formula and cap unchanged (still an upper bound: max non-input part 6,692 < 7,000)
  4. Evaluation path: run_evaluation.py already ran the full pipeline (it calls SupportAssistant.ask) but recorded no AutoGen evidence. New scripts/run_phase8_evaluation.py → evaluation/phase8/: same 15 queries, AutoGen ran on 12, every final reply re-checks clean, all replies and scores equal r2
  - Tests: **631 passed** (17 more: test_review.py +9, test_governance.py +5, test_phase8_evaluation.py 3), stable over 3 consecutive full runs; no Phase 0-7 test changed; one Phase 8 test corrected (review-stage spend after the reservation release)
  - Found while testing: run_case's fixed `eval-…` session ids leak history between two evaluation runs in one process; run_phase8_evaluation.py empties those sessions first. One earlier full run also showed 8 further Phase 3/6/7 failures: see the investigation below
  - Intermittent-failure investigation (owner request): the 8 Phase 3/6/7 failures of one run were not caused by Phase 8 code. Reproduced with full output: tests/test_rag.py alone failed 50/145 in 1 of 3 runs, and a standalone script using only Phase 3 code (EphemeralClient + index_all, no pytest) once built an index whose every read failed. Signature: `chromadb.errors.InternalError: Error executing plan: Internal error: Error finding id` from Chroma's compiled layer (`chromadb/api/rust.py` `bindings.get`); after it, every query on that in-memory index returns no results, while ids and documents are intact (count 42/55, same content hash). Once broken, the index stays broken for the process. Ruled out: client garbage collection (9/9 OK), our upsert sequence vs a plain upsert (20/20 OK each), Phase 8 code (failure reproduced without it), test parallelism (none configured). The trigger inside Chroma 1.5.9 is NOT identified; similar 1.5.x reports exist for PersistentClient. No code change made for it (no established root cause in this project's code). Risk: the deployed path uses the same Chroma code; if it occurs there, rag_lookup reports 'Policy search is temporarily unavailable' or retrieval falls back to the refusal: safe, but no answer. Not verified for PersistentClient. Owner decision needed (accept and document; or a health check that rebuilds the index; or test another Chroma version).
  - Session-leak regression test added: test_history_left_in_an_eval_session_does_not_leak_into_a_new_run (fails without the fix with the original symptom 'No order found with record_id ORD-1234'). Suite now **632 passed** in 3 consecutive full runs
- Behaviour change, documented: a session whose projected cost passes 12,000 tokens (long sessions of near-2,000-character messages) is now refused with `budget_exceeded`; every single question the API accepts still fits

## Phase 9: response caching (spec Task 16), VERIFIED (3a502d1)
- Checklist (written before code, f39d452): [docs/PHASE9_CHECKLIST.md](docs/PHASE9_CHECKLIST.md). Report: [docs/PHASE9_REPORT.md](docs/PHASE9_REPORT.md)
- app/core/cache.py: `GenerationCache` (in-memory, LRU 256, thread-safe) + `CachedGenerator`
  - wraps ONLY the grounded-generation call inside `answer_query`, after retrieval and the threshold; every guardrail, the crew, retrieval, groundedness, AutoGen review and the final check still run on a hit
  - key = SHA-256(normalised query, supporting-context fingerprint (chunk ids, texts, metadata), generator config, embedding model); only the digest is stored
  - never cached: order-id / PII / PII-token queries (neither read nor written), out-of-scope (nothing generated), refusals (never reach retrieval), text failing `claim_supported`; order lookups, drafts, verdicts and replies never touch it
  - one cache per `SupportAssistant` (shared by sessions; holds only KB sentences); `/add-document` clears it; no TTL (not in the spec; the context fingerprint prevents stale answers)
- Changes to approved code (all required to wire the cache): crew_tools.py (`cache` field, wrapper, `cache_events`), pipeline.py (`generation_cache`), api/main.py (clear on `/add-document`)
- One existing test's SETUP changed, assertions unchanged: test_governance.py::test_repeated_requests_each_get_a_fresh_budget now uses `GenerationCache(max_entries=0)`, because its third request (same question, fresh session) is now a legitimate cache hit that skips the generation charge. scripts/run_governance_demo.py section 4 got the same change, so phase8_runtime_budget.txt still reproduces (5,150 / 5,345 / 5,150; only printed object ids differ)
- Evidence: transcript phase9_cache.txt: miss (1 generation call, 5,150 est. tokens) then hit in another session (0 calls, 4,941 = minus exactly the 209-token generation charge), replies identical
- Tests: tests/test_cache.py **31** (owner cases 1-12, privacy/isolation, persistent-index failure modes); mutation-checked (4 mutants, each caught). Suite **663 passed** on 3 consecutive full runs
- Regression: the frozen 15-query evaluation (phase 8 full pipeline and r2) re-run with the cache on: every result file byte-identical, averages unchanged (v1 4.7333/5/5/5, v3 5/5/5/5), original + r1 hashes unchanged. Phase 8 APPROVE / REVISE / tool-access transcripts re-run identical (apart from the footer line)
- Persistent Chroma (owner brief §7): scripts/verify_persistent_chroma.py, transcript phase9_persistent_chroma.txt: both chroma_db/ collections equal the KB chunks (ids, texts, metadata; vectors max diff 1.5e-08); 50 rounds × 49 frozen queries × 2 collections = 4,900 searches, 0 exceptions, 0 empty, 0 wrong. Note: Chroma rewrites its index and sqlite files on open even when nothing is written (byte hashes change); the logical content is what is verified. Injected `InternalError` → "Policy search is temporarily unavailable"; an index returning nothing → refusal; neither caches anything (tests)
- Observed once during this phase: a mutation run of tests/test_cache.py showed 19 failures, the pattern of the known in-memory Chroma failure (every shared-index test failed; private-index and no-retrieval tests passed). Its output was not saved, so this is NOT confirmed; re-run with saved output it showed only the expected failures

## Final submission-readiness audit (on 3a502d1) and cleanup
- Audit: every spec task has implementation, tests and a transcript; a fresh local clone (model copied in for the one-time download) rebuilt chroma_db/ and reproduced every scored evaluation and the Phase 8-9 transcripts byte-identically; 663 passed in the main repo and in the clone; no secrets, binaries or local artifacts tracked
- Defect found and fixed: scripts/run_crew_demo.py crashed (`AttributeError: 'CrewRun' object has no attribute 'draft'`) since Phase 6 renamed the field to `structured`. Two-line fix; new regression test tests/test_crew_demo.py runs the real script (fails with the original error if the bug is restored). The Phase 5 transcript is kept as recorded (its Demo C citation predates the Phase 6 citation fix)
- Documentation cleanup: README rewritten at the top (overview, quick start, headline results, known limitations); stale status lines removed here and in the Phase 8/9 reports, ARCHITECTURE and CLAUDE.md; trace G2, G4, G6, G13, G14, S5, T16.1-T16.2 marked VERIFIED; G1 and G12 left open
- Evaluation files and hashes unchanged

## Compatibility issues found
1. crewai's verbose console made a hidden **pypi.org version check** on every kickoff and swallowed the failure. Fixed with `CREWAI_DISABLE_VERSION_CHECK=true`.
2. AutoGen requires agent names to be Python identifiers, so the spec's hyphenated names become underscored.
3. crewai prints a stray `[Finalize] todos_count=0` debug line during kickoff. Cosmetic.
4. crewai warns "approximate input token count" for custom LLMs. Expected and harmless.
5. langchain-core 1.7.3 marks `InMemoryChatMessageHistory` deprecated (removal in 2.0). It still works, and the version is pinned. The spec names this class, so we keep it.
6. When the budget stops a request inside a CrewAI task, CrewAI's event bus prints two "Event pairing mismatch" warnings (task_failed / crew_kickoff_failed). Cosmetic; the request ends with the safe budget reply.
7. AutoGen wraps an exception raised inside a model client into a RuntimeError string, losing its type. The review therefore reserves its budget before the team starts instead of failing inside it.

## Pending
- Owner: create the public GitHub repository and push (G1); originality assessment (G12)
- tests/spike_support.py and the spike tests remain as Phase 0 evidence; the real memory path is tests/test_pipeline.py
- Governance layers resolved: Application + Runtime (spec) and Data + Organizational (owner interpretation, labelled) (trace T15.4)
- Spike code kept as Phase 0 evidence: `tests/spike_support.py`; tests/test_spike_autogen.py still passes against the Phase 8 review team

## Known limits
- Chroma 1.5.9 in-memory index intermittently fails with `InternalError: ... Error finding id` and then returns no results (4 incidents in about 150 index lifetimes during the Phase 8 investigation; trigger unknown; not caused by project code). Tests that need retrieval then fail; a re-run passes. The persistent deployed index showed no failure in 4,900 searches (Phase 9); that does not prove it cannot happen there. If it does, replies degrade safely (unavailable / refusal) and nothing is cached.
- The network guard covers Python sockets. Native-code networking (Rust/C extensions) is not intercepted.
- The embedding model needs a one-time download (outside the test guard): scripts/download_model.py.
