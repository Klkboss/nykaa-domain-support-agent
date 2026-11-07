# Phase 6 checklist: session memory, structured output, guardrails (spec Tasks 8, 9, 10)

Written before implementation. Sources: [assignment.md](assignment.md) L85-89 (PII fields), L150-167 (Tasks
8-10), L251-255 (acceptance). **(assumption)** marks design choices the spec leaves open.

## Request pipeline (new: `app/pipeline.py`)

```
raw question
  -> normalise whitespace (one line: no forged prompt structure)
  -> PII masking (phone, card last-4)                 [input guardrail 1]
  -> prompt-injection detection -> blocked? reply, NOT stored in history   [input guardrail 2]
  -> RunnableWithMessageHistory(session_id)  (stores the MASKED question + the final reply)
       -> crew.kickoff(question, history)   Composer task output_pydantic=SupportDraft
       -> SupportDraft validated in code (invalid -> safe error reply)
       -> groundedness check against the ACTUAL tool outputs              [output guardrail]
       -> reply rendered in code from the validated, checked draft
  -> SupportResponse (Pydantic)
```

| # | Exact requirement | Planned implementation | Files | Acceptance | Tests / evidence |
|---|---|---|---|---|---|
| 8.1 | LangChain session memory: `InMemoryChatMessageHistory` + `RunnableWithMessageHistory`; the deprecation warning is expected, do not silence it (L150-155) | Existing `with_session_memory()` wraps the whole pipeline; one `InMemoryChatMessageHistory` per session id | app/memory/session_memory.py, app/pipeline.py | The real crew runs inside the wrapper; the warning stays visible | integration tests; warning shown in pytest output |
| 8.2 | Maintain history across turns within one process run (L154-155) | History (last 10 messages, **(assumption)** to bound prompt size) is formatted as `customer:` / `assistant:` lines into every task description | app/agents/crew.py | Turn 2 resolves "that order" from turn 1 | test + transcript A |
| 8.3 | Multi-turn transcript with state carried, plus a **separate** fresh-conversation transcript with state absent (L155-157) | `scripts/run_phase6_demo.py`: session A (id given, then "that order"); session B (same follow-up, new session) | evaluation/transcripts/phase6_memory_guardrails.txt | A uses ORD-1016 without repeating it; B asks for the id and calls no tool | tests + transcripts A/B |
| 8.4 | In-process only (L157-158) | Module-level dict of histories | — | — | — |
| 9.1 | A Pydantic `BaseModel` that **every crew response** must conform to (`response_format`) (L159-160) | `SupportDraft` (policy_claims with per-claim doc ids, order facts, order_error, needs_order_id, policy_error; extra=forbid, typed ids/statuses/ranges, cross-field validators). In crewai 1.15.33 the crew mechanism is `Task(output_pydantic=SupportDraft)` on the Composer task (`response_format` exists only on standalone `Agent.kickoff`) | app/schemas.py, app/agents/crew.py | `CrewOutput.pydantic` is a `SupportDraft` for every run | tests over all demo questions |
| 9.2 | Validate each response in code (L160-161) | The pipeline re-validates with `SupportDraft.model_validate`. Invalid or incomplete → `status="error"` with a safe reply and no partial facts | app/pipeline.py | A deliberately broken composer LLM gives the error response, never a crash | tests with malformed JSON / wrong types / extra fields |
| 9.3 | Customer reply derived from validated structure (owner) | `render_reply(draft)` in code; the LLM never writes the final reply text | app/schemas.py | Reply = render(validated draft) | test |
| 10.1 | Input PII masking firing on the fixed-format fields: **phone number** and **payment-card last-4** (L85-86, L162-163) | Regex masks → `[PHONE]`, `[CARD-LAST4]` (full card numbers → `[CARD]`). Order ids and amounts are preserved | app/guardrails/pii.py | The model, history and logs only ever see masked text | unit tests; test that no LLM call contains the raw value; history test; transcript |
| 10.2 | Name / delivery address out of scope; fabricated examples only (L87-89) | Documented; not masked | README | — | — |
| 10.3 | Input prompt-injection detection (L163-164) | Pattern detector by category (instruction override, role hijack, prompt exfiltration, bulk data request, fake role / delimiter tokens). A flagged turn is blocked: no crew run, no history entry | app/guardrails/injection.py | Deliberate injections blocked; all project queries (eval, calibration, intent, demos) pass | tests + transcript |
| 10.4 | Output groundedness check that **refuses when the retrieved context does not support the question** (L164-165) | (a) Below the retrieval threshold there are no claims → fallback refusal. (b) Every policy claim must be supported by the cited doc's retrieved evidence: verbatim, or every number present and ≥ 80% content-word coverage. Unsupported claims are removed; nothing left → refusal. (c) Order facts must equal an actual `check_order_status` output; order data is never policy evidence | app/guardrails/groundedness.py | A deliberately hallucinating Composer gets refused; wrong citations are removed | unit + real-kickoff tests; transcript |
| 10.5 | Each guardrail shown firing on a deliberate test case (L165-167) | Demo script cases: phone + card masking, injection block, groundedness refusal (out-of-scope and injected false claim), citation correction | scripts/run_phase6_demo.py | — | transcript |
| C1 | Carry-forward: Phase 5 Demo C cited KB-01, KB-12 for a KB-01 sentence | `rag_lookup` returns per-sentence `claims` attributed to the docs whose retrieved passage contains the sentence. The guardrail rejects citations that do not support the claim | app/tools/crew_tools.py, groundedness.py | Demo C cites only KB-01 | test |

Out of scope here: FastAPI (Phase 8), JSONL logging (Phase 9; masking is reused there), AutoGen review
(Phase 11), budget (Phase 12), cache (Phase 13).
