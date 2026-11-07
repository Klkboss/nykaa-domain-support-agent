# CLAUDE.md — Nykaa Domain Support Agent (capstone)

Permanent instructions for working on this repo. Status lives in PROJECT_STATUS.md.

## Ground rules
- The assignment specification `docs/assignment.md` is the single source of truth. If code and spec disagree, the spec wins. Never edit that file.
- Every requirement has an ID in `docs/REQUIREMENTS_TRACE.md` (G*, S*, T1.1-T16.2). Update its Status column when a phase closes, and cite IDs in phase reports.
- Each task needs a saved transcript in `evaluation/transcripts/` (spec: transcripts demonstrating every task).
- One phase at a time. A phase is done only when its acceptance tests pass. Wait for the owner's approval before starting the next phase.
- Never fabricate test results, scores or transcripts. Only report output that was actually executed.
- No hardcoded outputs pretending to be agent/tool execution. CrewAI and AutoGen must really run.
- Everything runs under `MOCK_LLM=true`: no API keys, no network at runtime (one-time HF model download excepted).
- Commit a git checkpoint after each completed phase.

## Environment
- Windows 11, Python 3.12.10, venv at `.venv` (path contains a space: always quote it).
- Run: `.venv\Scripts\python.exe ...`  Tests: `.venv\Scripts\python.exe -m pytest -q`
- CPU torch by default (PyPI wheel). GPU is optional, never required.

## Architecture (see PROJECT_STATUS.md for build state)
Request flow: input guardrails (PII mask, injection) -> budget check -> CrewAI crew
(Retrieval [rag_lookup: retrieval -> cache-or-generate] -> Lookup -> Composer) -> groundedness check -> AutoGen review (Policy-Compliance-Reviewer -> Final-Editor)
-> Pydantic response -> JSONL log. LangChain `RunnableWithMessageHistory` wraps the pipeline for session memory.

## MOCK_LLM rules (the hard part)
- CrewAI: subclass `crewai.BaseLLM` (`crewai.llms.base_llm.BaseLLM`). `call()` must accept `from_task`, `from_agent`, `response_model`, `**kwargs`. `supports_function_calling() -> False` (ReAct text loop).
- Decide state from the model's own generated turns only — never search the whole conversation for "Observation:" (CrewAI's system prompt contains it).
- Pick tools by their declared `args_schema` fields, never by substring of tool name.
- AutoGen: separate `autogen_core.models.ChatCompletionClient` subclass. CrewAI's BaseLLM does not work with AutoGen.
- `app/core/config.py` forces `CREWAI_DISABLE_TELEMETRY=true`, `CREWAI_TRACING_ENABLED=false`, `OTEL_SDK_DISABLED=true`, `CREWAI_DISABLE_VERSION_CHECK=true`, `ANONYMIZED_TELEMETRY=False`. Import it before crewai/chromadb/autogen.
- `CREWAI_DISABLE_VERSION_CHECK` is required: a `verbose=True` crew queries pypi.org on every kickoff (found by the spike).
- Tests block non-loopback sockets and DNS (`tests/conftest.py`) and fail on any recorded attempt, even a swallowed one.
- AutoGen agent names must be Python identifiers: use `Policy_Compliance_Reviewer`, `Final_Editor`.

## Owner decisions (Phase 0)
- Spec lives in `docs/` (assignment spec, ARCHITECTURE.md, GOVERNANCE.md).
- Cache ONLY grounded RAG generation (normalised query key). Never cache order lookups or full customer responses. Clear the cache on /add-document.
- Risk classification: **Medium**, justified with the assignment's risk framework (Phase 8, docs/GOVERNANCE.md).
- Four governance layers: Application + Runtime are **named in the spec**. Data + Organizational are the **owner's proposed interpretation**. Always label them that way, unless course material gives other names.
- Phase reports go to the owner for independent review before the next phase starts.
- RunnableWithMessageHistory wraps the whole pipeline (spike-verified: tests/test_spike_memory.py). Its LangChainDeprecationWarning is expected; do not silence it.

## Knowledge base rules (Phase 2)
- Documents live in `data/knowledge_base/policies.json` and are validated by `app.rag.documents.KBDocument` (2-5 sentences, `KB-NN` ids, flat Chroma metadata). Never renumber KB-01…KB-12, because precision/recall ground truth depends on them.
- `split_sentences()` in app/rag/documents.py is the ONE sentence splitter: it is used for the 2-5 check, sentence chunking and the review mock. Avoid abbreviations like "e.g." followed by a capital in policy text.
- When editing a policy, keep shared facts consistent (tests/test_knowledge_base.py::test_shared_facts_are_stated_consistently).
- A policy change needs owner approval. Preserve the previous wording in `data/knowledge_base/history/` first. Then re-run `scripts/run_rag_evaluation.py` (it re-indexes chroma_db/), check that P/R, thresholds and the chunking recommendation do not move, and run a NEW versioned evaluation (never rewrite earlier ones).
- KB-08 wording v2 (owner-approved): 3 retries after the first attempt, at most 4 attempts in total. Never state retries and attempts with the same number.
- Trace status levels: OPEN → IMPLEMENTED → TESTED → VERIFIED (VERIFIED only after owner review).

## RAG rules (Phase 3)
- Embeddings: only `app.rag.embeddings.embed()` (real all-MiniLM-L6-v2 from `models/`). Never use dummy vectors for retrieval, calibration or evaluation. The model is fetched once with `scripts/download_model.py`.
- Collections: `policies_fixed_overlap` and `policies_sentence_based`, cosine space, `embedding_function=None`, always `index_documents()` (upsert).
- Deployed collection = `RECOMMENDED_STRATEGY` ("fixed"); threshold = `SIMILARITY_THRESHOLDS[strategy]` (fixed 0.33, sentence 0.301). If the KB, chunking or model changes, re-run `scripts/run_rag_evaluation.py` and update both constants; tests compare them to live measurements.
- Never change `evaluation/rag_queries.json` or `rag_intent_queries.json` labels after seeing results.
- Retrieval entry point is `app.rag.retrieval.search()` (metadata-aware); `retrieve()` is the raw dense primitive. Calibration and P/R must use the same pipeline as answering.
- `query_analysis` encodes only observed confusions (return vs warranty). Add a routing rule only with a held-out regression query committed first, and show that unrouted queries are unaffected.
- Chunking recommendation rule: F1 of average precision and recall (they disagree since the Q3 fix).
- Regenerate `evaluation/transcripts/phase3_rag.txt` with `cmd /c "... 2>NUL"` (PowerShell 5.1 wraps native stderr).

## Order lookup rules (Phase 4)
- `app.tools.order_lookup.check_order_status` is the only order-data access path. It returns exactly `status`, `order_value_inr`, `escalation_score`, `escalation_recommended`. Never add dataset fields to the output.
- Score = 0.6·delayed + 0.4·days/30; threshold 0.78 = median-age derivation (tests re-derive it from ORDERS). If the dataset changes, re-run `scripts/run_order_lookup_demo.py`.
- Errors are typed (`InvalidRecordIdError`, `OrderNotFoundError`) and never echo raw input. Phase 5's CrewAI wrapper must catch `OrderLookupError` and return a safe message.
- Record-id regex is ASCII `[0-9]`, not `\d`.

## Crew rules (Phase 5)
- Entry point: `app.agents.crew.run_support_crew(question, rag_tool=None)`. Build a fresh crew per run. Tests pass `RagLookupTool(collection=collections["fixed"])` from the conftest fixtures; the default uses persisted `chroma_db/`.
- Only the Lookup Agent may hold `check_order_status`. Never give the Composer tools.
- The mock Composer reads facts ONLY from JSON after `CONTEXT_MARKER`. Tool results must stay single-object JSON with a `"tool"` key.
- Order-intent and no-id decisions use the `Customer question:` line only, never the task instructions.
- Regenerate the transcript with the socket guard: see the command header in evaluation/transcripts/phase5_crewai.txt.
- scripts/run_crew_demo.py is run by tests/test_crew_demo.py: renaming a `CrewRun` field must update the script (Phase 6 broke it once).

## Pipeline and guardrail rules (Phase 6)
- The customer entry point is `app.pipeline.SupportAssistant.ask(session_id, question)`. API, evaluation and later phases must call it, not the crew directly.
- Order inside `ask`: normalise → `mask_pii` → `detect_injection` (blocked: no crew, no history) → memory → crew → `SupportDraft` validation → `check_draft` → `render_reply`. Anything logged or stored must be the masked text.
- Customer-facing text comes ONLY from `render_reply(validated, checked draft)`.
- Groundedness evidence comes from `RagLookupTool.outputs` / `CheckOrderStatusTool.outputs` (what the tools really returned), never from agent task outputs.
- `rag_lookup` keeps the Phase 3 `sources` and adds per-sentence `claims` and `evidence`; citations come from `claims`.
- Injection detection is pattern-based; add patterns only with benign-question tests (tests/test_guardrails.py BENIGN).

## API, logging and evaluation rules (Phase 7)
- The API (`app.api.main.create_app`) must only call `SupportAssistant.ask`; never add an endpoint that bypasses the pipeline.
- The API index is a per-process temp Chroma dir. Never point `/add-document` at `chroma_db/` (approved Phase 3 evidence).
- Log through `JsonlLogger` only. Log masked text and exception TYPE names; never raw bodies, prompts or `str(exception)`. Pydantic error `input` fields must be dropped.
- The evaluation set and rubric v1 (evaluation/test_queries.json) are frozen. The ORIGINAL results (evaluation/evaluation_results.json, EVALUATION_REPORT.md, from b7e2fe9) must never be rewritten; their SHA-256 is test-locked. `scripts/run_evaluation.py` writes only evaluation/remediation/*_r2_kb08v2_* and EVALUATION_REPORT_R2_KB08_CLARIFIED.md (rubrics v1 and v3). r1 (KB-08 wording v1) is frozen and hash-locked like the originals. Locks hash LF-normalised content (autocrlf-safe). Re-run it after pipeline changes; `test_remediation_evaluation_is_reproducible_live` compares live results with those files.
- Any rubric change is a NEW version file committed before use, scored alongside v1, and labelled as revised.
- A count is not just a number plus a unit: retries (after the first attempt) and attempts (in total) are different events (KB-08). Count facts are annotated in rubric_v3.json `count_facts` with the event they count; never equate the two. Every changed score or reply needs an entry in `EXPLANATIONS` (scripts/run_evaluation.py); a test enforces this.
- The generator's category-subject filter and the escalation route are shared behaviour. Changing them can move answers in every phase, so re-run the evaluation and check the reply diff.
- Describe the judge only as a deterministic rubric scorer behind an LLM interface (MOCK_LLM), never as a real LLM judgement.
- Real-server demo: `scripts/run_api_demo.py` (loopback only; it rewrites both transcripts and evaluation/phase7_requests_log.jsonl).

## Review and governance rules (Phase 8)
- The AutoGen review (`review_policy_draft`) gets the Composer's policy draft (before the guardrail) plus the rag_lookup evidence. It runs after `check_draft`; a revision only removes claims; then `check_draft` runs AGAIN on the reviewed draft and only that is rendered. Never render `Verdict.final_answer` or let the review add text, citations or order facts.
- `approved` judges the Composer's draft (true = unchanged; false = revised). Decide on `final_answer != draft`, not on `approved`.
- Least autonomy: `ALLOWED_TOOLS` in app/governance/least_autonomy.py is the only tool grant. A new agent or tool needs an entry there and a negative test. Never enable delegation.
- Runtime budget: every request gets a fresh `RequestBudget(MAX_REQUEST_TOKENS)`. Every model call charges it: CrewAI mock LLMs, the grounded generation in rag_lookup, and the AutoGen clients (metered against the review's reservation, unused part released). A new model call path must be charged too (test_every_model_call_in_a_request_is_charged). The cap is on ESTIMATED tokens; never describe it as a real-token guarantee. If prompts, agents or tools change, re-run scripts/run_governance_demo.py budget and keep `test_projection_is_an_upper_bound_of_real_spend` passing. Never raise the cap just to make a test pass.
- `BudgetExceeded` subclasses CrewAI's `HookAborted` on purpose: CrewAI re-raises it instead of retrying the task.
- Regenerate the Phase 8 transcripts with the socket guard (see the header of evaluation/transcripts/phase8_*.txt).

## Caching rules (Phase 9, Task 16)
- `app/core/cache.py` caches ONLY the grounded-generation call, inside `answer_query` after retrieval. Never cache order lookups, drafts, verdicts, replies or session history, and never move the cache in front of the crew or retrieval.
- The key must cover every input of the generation: normalised query, the supporting chunks (ids, texts, metadata), generator config, embedding model. A new generation input needs a key field and a test in tests/test_cache.py (test_cache_key_covers_every_input_of_the_generation).
- Order-id / PII / PII-token queries are neither read from nor written to the cache; only text passing `claim_supported` is stored. A cache error must fall back to generation; `BudgetExceeded` must still propagate.
- A hit skips the generation charge. A test or demo about budget isolation that repeats a question must use `GenerationCache(max_entries=0)`; never fake a charge on a hit.
- `/add-document` clears the cache. Chroma rewrites chroma_db/ files on open: verify its content logically (scripts/verify_persistent_chroma.py), not by file hash.

## Spec-derived implementation notes
- `days_since_created` is an int in 0-30. Dataset exposes an `ORDERS` list.
- The RAG tool is named `rag_lookup` (the spec's pitfall example), to prove schema-based dispatch.
- Spec "`response_format`" maps to `Task(output_pydantic=...)` for crews in crewai 1.15.33 (`response_format` exists only on standalone `Agent.kickoff`).
- Evaluation uses an LLM-as-judge prompt sent through the MOCK LLM, with a documented rubric. Label it a mock judge.

## Conventions
- Type hints everywhere; Pydantic v2 models for all boundaries.
- Deterministic: fixed seeds, no time/random in mock outputs.
- Never log raw phone numbers or card digits; mask before logging.
- Policies in the knowledge base are SIMULATED for this capstone, not official Nykaa policy.
- Pinned versions in requirements.txt; do not upgrade without a reason written in PROJECT_STATUS.md.
