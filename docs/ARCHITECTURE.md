# Architecture

Status: designed and approved in Phase 0; every component below is built (phases 1-9, all approved).

## Request flow

```
client: POST /ask | POST /add-document | WS /ws/chat              [built: Phase 7, app/api/main.py]
  -> HTTP middleware / WS handler: trace_id, timing -> JSONL log (masked, every string re-masked) [built: Phase 7]
  -> app.pipeline.SupportAssistant.ask(session_id, question)               [built: Phase 6]
     -> input guardrails: one-line normalisation, phone / card-last-4 masking, prompt-injection check
        (a blocked turn runs no crew and is never stored)                  [built: Phase 6]
     -> runtime budget pre-flight: 7,000 + 6 x tokens(question + history) <= 12,000, else refused before any
        model call and never stored                                       [built: Phase 8, app/governance/budget.py]
     -> RunnableWithMessageHistory (session_id -> InMemoryChatMessageHistory; masked text only) [built: Phase 6]
     -> CrewAI crew, sequential, crew.kickoff(); every mock LLM call (and grounded generation) charged to a
        fresh RequestBudget
        (least-autonomy allow-list enforced when the crew is built)        [built: Phase 5; budget + guard Phase 8]
          retrieval_task: Retrieval Agent  -- tool rag_lookup (answer_query on policies_fixed_overlap;
                          retrieval always runs, then the grounded-generation call goes through the
                          Task 16 cache: hit -> stored validated text, no model call) [cache: Phase 9]
          lookup_task:    Lookup Agent     -- tool check_order_status (ONLY agent holding it); context=[]
          compose_task:   Response Composer -- no tools; context=[retrieval_task, lookup_task];
                          output_pydantic=SupportDraft                     [built: Phase 6]
     -> SupportDraft re-validated in code; invalid -> safe error reply      [built: Phase 6]
     -> output guardrail: groundedness vs the ACTUAL tool outputs (claims, citations, order facts),
        refusal when nothing supported remains; reply rendered from the checked draft [built: Phase 6]
     -> AutoGen RoundRobinGroupChat(max_turns=2) on the Composer's policy draft + the retrieved context;
          Policy_Compliance_Reviewer -> Final_Editor (output_content_type=Verdict); budget reserved first;
          a revision removes the flagged claims; turns without policy text are not reviewed [built: Phase 8]
     -> final groundedness check: check_draft again on the reviewed draft, same tool outputs; the reply is
        rendered only from this re-checked draft (editor text is never rendered)  [built: Phase 8 review]
     -> Pydantic response model (status adds `budget_exceeded`)
     -> AskResponse (SupportResponse + trace_id + new_session)
```

## Module map

| Path | Responsibility |
|---|---|
| `app/core/config.py` | Offline env switches, forced before any library import |
| `app/core/mock_llm.py` | `MockCrewLLM(BaseLLM)`: deterministic ReAct driver for CrewAI |
| `app/core/mock_autogen.py` | `MockChatCompletionClient(ChatCompletionClient)` for AutoGen |
| `app/core/cache.py` | Task 16: `GenerationCache` (in-memory, LRU 256, thread-safe) and `CachedGenerator`, which wraps only the grounded-generation call; key = SHA-256(normalised query, supporting-context fingerprint, generator config, embedding model) |
| `app/rag/documents.py` | `KBDocument` schema, `load_knowledge_base()`, shared `split_sentences()`, `chroma_metadata()` |
| `app/rag/embeddings.py` | `embed()`: local all-MiniLM-L6-v2 (384-d, normalised, CPU), loaded from `models/`; raises if missing |
| `app/rag/chunking.py` | `fixed_size_chunks` (40 words / 10 overlap), `sentence_chunks` (1 sentence), deterministic chunk ids |
| `app/rag/vector_store.py` | `policies_fixed_overlap`, `policies_sentence_based` (cosine), `index_documents()` via upsert + stale removal |
| `app/rag/query_analysis.py` | `analyse_query()`: product → category; warranty-intent routing → `where={"topic": "warranty"}` + expansion; escalation routing (cue + no other policy-topic cue) → `where={"topic": "escalation_matrix"}`, no expansion (Phase 7 remediation) |
| `app/rag/retrieval.py` | `retrieve(collection, query, k, where)` plain dense; `search()` = analysis + retrieve (the pipeline everyone uses) |
| `app/rag/generation.py` | `answer_query()`, `MockGroundedGenerator` (extractive, `calls` counter), `SIMILARITY_THRESHOLDS`, `RECOMMENDED_STRATEGY="fixed"` |
| `app/rag/evaluation.py` | `choose_threshold()`, `precision_recall()`, `dedupe()` |
| `scripts/download_model.py`, `scripts/run_rag_evaluation.py` | one-time model fetch; Phase 3 evidence run |
| `data/knowledge_base/policies.json` | 12 simulated policy documents, KB-01…KB-12 |
| `app/tools/order_lookup.py` | `check_order_status(record_id) -> dict` over `ORDERS`; `escalation_score()` = 0.6·delayed + 0.4·days/30; threshold 0.78 (dataset median age 13.5); typed `OrderLookupError`s that never echo input |
| `app/tools/crew_tools.py` | CrewAI `BaseTool`s: `rag_lookup` (args_schema `query`) and `check_order_status` (args_schema `record_id`). JSON results, `{"error"}` on failure, `invocations` log |
| `app/agents/crew.py` | `build_support_crew()`, `run_support_crew(question)` → `CrewRun` (draft, task outputs, tool invocations, LLM call counts) |
| `app/review/autogen_review.py` | `Verdict` (approved, final_answer, reason), `build_review_team()` (2 agents, max_turns=2, StructuredMessage[Verdict]), `review()`, `review_policy_draft()` (pipeline entry; budget reservation) |
| `app/memory/session_memory.py` | `with_session_memory(pipeline)` |
| `app/schemas.py` | `SupportDraft` / `PolicyClaim` / `OrderFacts` (crew output schema), `render_reply`, `GuardrailReport`, `SupportResponse` |
| `app/guardrails/pii.py` | `mask_pii()`: `[PHONE]`, `[CARD-LAST4]`, `[CARD]` |
| `app/guardrails/injection.py` | `detect_injection()`: 5 pattern categories, `BLOCKED_REPLY` |
| `app/guardrails/groundedness.py` | `claim_supported`, `evidence_by_doc`, `attribute_claims`, `check_draft` (also used by the AutoGen mock reviewer) |
| `app/pipeline.py` | `SupportAssistant`: masking → injection → memory → crew → validation → groundedness → `SupportResponse` |
| `app/api/main.py`, `app/api/schemas.py` | `create_app()`: `/ask`, `/add-document`, `/ws/chat`; safe 422/409/500 errors; per-process temp index built at startup |
| `app/observability/logging.py` | `JsonlLogger` (one JSON per line, `mask_pii` over every string), `new_trace_id()` |
| `app/judge.py` | `JUDGE_PROMPT`, `MockJudgeLLM(BaseLLM)` deterministic rubric scorer (rubric v1; v2 withdrawn; revised v3 binds annotated count facts to their counted event), `fact_present`, `JudgeScores` |
| `scripts/run_api_demo.py`, `scripts/run_evaluation.py` | real-uvicorn API/WS evidence; the 15-query evaluation |
| `app/governance/least_autonomy.py` | Application layer: per-role tool allow-list (`enforce_crew_least_autonomy`), reviewers without tools (`enforce_reviewers_have_no_tools`) |
| `app/governance/budget.py` | Runtime layer: `MAX_REQUEST_TOKENS`, `project_request_tokens()`, `RequestBudget`, `BudgetExceeded(HookAborted)` |
| `scripts/run_review_demo.py`, `scripts/run_governance_demo.py` | Phase 8 evidence (approve / revise; tool access / budget) |
| `scripts/run_phase8_evaluation.py` | the 15 frozen queries through the full pipeline (AutoGen + final check) → evaluation/phase8/ |

## MOCK_LLM design

**CrewAI (`MockCrewLLM`).** `supports_function_calling()` is False, so CrewAI drives tools with its
ReAct text loop. Verified in crewai 1.15.33: after a tool runs, CrewAI appends one `assistant` message
holding the model's own `Thought/Action/Action Input` text plus `\nObservation: <tool result>`
(`crewai/utilities/agent_utils.py: handle_agent_action_core`). CrewAI's system prompt also contains
`Observation:`, so the mock reads loop state from `assistant` turns only.

Tool dispatch reads `from_agent.tools[*].args_schema` and picks the tool whose required field set is
exactly one value the mock can fill (`record_id` from an `ORD-NNNN` pattern, otherwise `query` from the
`Customer question:` line). Tool names are never matched. The spike proves this with a decoy tool
named `order_policy_search`.

**Composer (Phase 5).** An agent without tools composes from JSON tool results that appear after CrewAI's
context header (`This is the context you're working with:`). The header text was verified against the
prompts CrewAI 1.15.33 actually sends. If a lookup ran and the question has no policy cue, the policy section
is omitted. Without an order id, the Lookup Agent asks for one only when the customer's *question* mentions
an order or status; the task instructions themselves contain the word "order".

**AutoGen (`MockChatCompletionClient`).** It implements all 9 abstract members of
`autogen_core.models.ChatCompletionClient` and declares `structured_output=True`. Each agent gets its
own client (`reviewer` or `editor`). AssistantAgent passes `json_output=Verdict` for the Final-Editor.
The client returns `Verdict` JSON, which AutoGen validates into `StructuredMessage[Verdict]`.

## RAG integration points (Phase 3)
- **Phase 5 `rag_lookup` tool:** calls `answer_query(query, get_collection(client, RECOMMENDED_STRATEGY), SIMILARITY_THRESHOLDS[RECOMMENDED_STRATEGY])`. The collection is a parameter, so either strategy can be selected.
- **Task 16 cache (built in Phase 9):** wraps the generator call INSIDE `answer_query`, after retrieval and the threshold (not `answer_query` itself, so retrieval is never skipped and the retrieved context can be part of the key). `MockGroundedGenerator.calls` shows the avoided generation call.
- **Phase 7 `/add-document`:** validates a `KBDocument`, calls `index_documents(client, [doc], strategy)` for both strategies, then clears the cache.

## Caching decision (owner, Phase 0)
Cache **only grounded RAG generation**, keyed by normalised query text. Never cache order lookup
results or complete customer responses, because they are personalised. Adding a document clears the cache.

As built (Phase 9): the key is the normalised query text plus a fingerprint of the retrieved supporting context, the
generator settings and the embedding model, so changed policy text can never return a stale answer. Queries with an
order id, PII or a PII mask token are neither read from nor written to the cache; only generated text that passes
`claim_supported` against the retrieved context is stored. Order lookups, drafts, AutoGen verdicts and replies never
touch the cache. One cache per `SupportAssistant`, shared by sessions (it holds only KB sentences); no TTL.

## Network isolation
`config.py` forces `CREWAI_DISABLE_TELEMETRY`, `CREWAI_TRACING_ENABLED=false`, `OTEL_SDK_DISABLED`,
`CREWAI_DISABLE_VERSION_CHECK` and `ANONYMIZED_TELEMETRY=False`. `tests/conftest.py` patches
`socket.connect`, `connect_ex` and `getaddrinfo` before any import. Loopback is allowed (the Windows
asyncio self-pipe needs it). Every other attempt is refused and recorded, and the test fails at
teardown even if the library swallowed the error.
