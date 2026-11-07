# AI governance (spec Part 4, Task 15)

Source: [assignment.md](assignment.md) L215-225 and the Part 4 acceptance criteria (L266-270).

The spec asks for "the four-layer AI-governance model" and names two layers explicitly:
- **Application layer:** least autonomy.
- **Runtime layer:** a per-request token/cost budget.

It does not name the other two. **Data layer** and **Organizational layer** are the **owner's proposed
interpretation** (Phase 0 decision, trace T15.4), not terms from the spec. They are labelled as such below.

All evidence below was produced under `MOCK_LLM`, with zero API keys and the socket guard active.

## Risk classification: Medium

Under the spec's scheme, Low is summarisation or transcription, Medium is code generation or customer support
tickets, and High is medical data, hiring decisions or financial data. **This system is Medium-risk.**

- **Why not Low:** it is a customer-support agent, which is the spec's own Medium example. It answers
  customers directly with policy statements and order facts that they may act on, such as return windows,
  refunds and escalation, so a wrong answer has real consequences.
- **Why not High:** it makes no medical, hiring or financial decisions. It reads fabricated order records
  (status, order value, an escalation score) but never moves money, never issues a refund and never decides
  credit. It only reports the simulated policy and flags cases for human escalation.
- **Payment data:** the only payment-related personal data, card last-4 digits and phone numbers, is masked
  before any model sees it and never logged raw.
- **Controls that match this risk level:** the groundedness guardrail and the AutoGen review stop invented
  policy, least autonomy limits who can read order records, and the runtime budget caps the cost of each
  request.

## The least-autonomy guard (Application layer)

**Only the Lookup Agent can call `check_order_status`.** `build_support_crew()` calls
`enforce_crew_least_autonomy` ([app/governance/least_autonomy.py](../app/governance/least_autonomy.py)) before
the crew is assembled. The guard compares every agent's tools with an explicit allow-list:

| Agent | Allowed tools |
|---|---|
| Retrieval Agent | `rag_lookup` |
| Lookup Agent | `check_order_status` |
| Response Composer | none |

The guard refuses to build the crew, raising `LeastAutonomyViolation`, when any of these is true:

- An agent holds a tool outside its list, or is missing its tool.
- An agent's role has no grant at all.
- An agent other than the Lookup Agent holds an order-lookup tool. A tool counts as order lookup by its name
  or by its `OrderLookupInput` argument schema, so renaming or disguising the tool does not get past the guard.
- An agent has delegation enabled, because CrewAI's delegation tools would let one agent reach another
  agent's tools.

**The AutoGen reviewers hold no tools.** `build_review_team()` calls `enforce_reviewers_have_no_tools`. Their
model client declares no function calling, so AutoGen itself also refuses to attach tools to them.

The restriction is therefore enforced in two ways: the tool is never wired to another agent, and any attempt to
wire it is refused.

## The four layers

### 1. Application layer (named in the spec)
**Purpose:** limit what each agent may do, and stop ungrounded or unsafe content from reaching the customer.

**Controls and where they are implemented:**

| Control | Implemented in |
|---|---|
| Least-autonomy tool allow-list, including no delegation | [least_autonomy.py](../app/governance/least_autonomy.py), called from [crew.py](../app/agents/crew.py) |
| AutoGen reviewers hold no tools | `enforce_reviewers_have_no_tools`, called from [autogen_review.py](../app/review/autogen_review.py) |
| Input guardrails: PII masking, prompt-injection block | [pii.py](../app/guardrails/pii.py), [injection.py](../app/guardrails/injection.py) |
| Output groundedness guardrail | [groundedness.py](../app/guardrails/groundedness.py) |
| AutoGen review of the Composer's draft (Task 14) | [autogen_review.py](../app/review/autogen_review.py), wired in [pipeline.py](../app/pipeline.py) |
| Final groundedness check after the review (Phase 8 review fix) | `check_draft` again on the reviewed draft, against the same tool outputs, in [pipeline.py](../app/pipeline.py) |
| Typed structured output | `SupportDraft` and `render_reply` in [schemas.py](../app/schemas.py) |

**How it is tested:**
- [test_governance.py](../tests/test_governance.py), positive cases: the real crew's grants match the allow-list, both before and after a real kickoff that used both tools.
- test_governance.py, negative cases (8): the order tool on the Retrieval Agent or the Composer, renamed, disguised, an extra tool, no tool, delegation, an unknown role. The crew builder itself also refuses a mis-wired Composer.
- test_governance.py, AutoGen: the reviewers hold no tools, AutoGen refuses tools for their client, and the guard refuses an armed reviewer.
- [test_review.py](../tests/test_review.py): the review stage, the final check after it (5 cases: approve; Composer claim removed; editor adds claims or order facts; editor-written citations; correctly cited answer) and the verdict semantics.
- test_guardrails.py and test_pipeline.py: the Phase 6 guardrails.

**Evidence:** [phase8_tool_access.txt](../evaluation/transcripts/phase8_tool_access.txt),
[phase8_autogen_approve.txt](../evaluation/transcripts/phase8_autogen_approve.txt),
[phase8_autogen_revise.txt](../evaluation/transcripts/phase8_autogen_revise.txt), and the Phase 6 guardrail
transcripts.

**Limitations:**
- Agents are identified by their `role` string. A crew built outside `build_support_crew()` is not checked.
- The AutoGen guard reads `AssistantAgent` private attributes, which are pinned to autogen-agentchat 0.7.5.
- Under `MOCK_LLM`, the reviewer applies the same lexical support rule as the groundedness guardrail. In the
  live pipeline it therefore agrees with the guardrail, so the review adds a structured verdict, not a
  stronger check.
- The review can only remove claims. A correct rewording by the editor is discarded too (the claim is dropped,
  or the reply refuses), because only the Composer's guardrail-checked claims are ever rendered.

### 2. Runtime layer (named in the spec)
**Purpose:** cap what one request may spend, and refuse an oversized request instead of exceeding the cap.

**Controls and where they are implemented**
([budget.py](../app/governance/budget.py), wired in [pipeline.py](../app/pipeline.py)):
- **The cap:** `MAX_REQUEST_TOKENS = 12,000` **estimated** tokens per request. Tokens are estimated as
  `len(text) // 4`. The cap bounds this accounted estimate; it is not a real-tokenizer count. Under `MOCK_LLM`
  no model consumes real tokens, and a real provider's usage report could differ from the estimate.
- **Pre-flight check:** the request's spend is projected as `7,000 + 6 × tokens(question + history)`. A
  request projected over the cap is refused before any model call, with status `budget_exceeded` and a fixed
  reply. It is never stored in session memory, so it cannot inflate later turns.
- **Accounting:** each request gets a fresh `RequestBudget`. Every model call is charged, prompt before it is
  sent and completion before it is returned:
  - every CrewAI agent call (`MockCrewLLM.call`), including calls CrewAI retries;
  - the grounded-generation call inside `rag_lookup`;
  - every AutoGen reviewer and editor call. The review first reserves its upper bound on the request budget,
    its calls are metered against that reservation, and the unused part is released.
  A charge that would pass the cap raises `BudgetExceeded`, a subclass of CrewAI's `HookAborted`, so the crew
  stops at once. After a refused charge the budget stays exhausted, so no later call in that request can be
  charged. A budget stop inside AutoGen is re-raised as `BudgetExceeded`, and inside `rag_lookup` it is not
  swallowed. The request ends with the same safe reply.
- **Not counted:** the local embedding model (MiniLM) used for retrieval and sentence ranking. It is local
  compute with no token cost, not a language-model call.
- **Order of the limits:** the API's 2,000-character request limit (Phase 7) still applies first.

**How the cap and projection were chosen:**
- **Measured spend** (estimated tokens, every model call charged): the 15 evaluation queries spend
  3,163–6,396 tokens each. A 2,000-character question spends about 8,500–8,700. Spend grows by about 6 tokens
  per question token and about 4 per history token; the part not proportional to the input is at most 6,692.
- **Cap:** about 1.9 times the largest evaluation request.
- **Projection:** an upper bound fitted to that measured spend, and tested to stay one.

**How it is tested** ([test_governance.py](../tests/test_governance.py)):
- Below the cap, exactly at it (allowed) and one token over (refused).
- A request projected exactly at the cap is answered, and one token over is refused before the crew runs.
- Repeated requests get a fresh budget each, with no carry-over.
- The budget is exhausted mid-crew, and separately at the review stage.
- The crew stops at its first over-budget call without retries.
- A roughly 20,000-character oversized request is refused, with no model call and nothing stored.
- The projection is at least the real spend for all 15 evaluation queries and a multi-turn session.
- Every model call in a request is charged (a spy on `RequestBudget.charge` attributes each charge to its
  code path); the grounded-generation call is charged; a retried call is charged; the stop is sticky; a stop
  inside AutoGen surfaces as `BudgetExceeded`.
- The API accepts its largest allowed question, and reports a budget refusal with HTTP 200, status
  `budget_exceeded` and a PII-masked log entry.

**Evidence:** [phase8_runtime_budget.txt](../evaluation/transcripts/phase8_runtime_budget.txt).

**Limitations:**
- Tokens are estimates (4 characters per token), not a real tokenizer, so "12,000" is a cap on estimated
  tokens, not a guaranteed cap on any provider's actual token count.
- Cost is counted in tokens only; there is no currency price under `MOCK_LLM`.
- The projection is an empirical upper bound for this pipeline. If the prompts change, re-measure it (the test
  will fail if it stops being an upper bound).
- A long session of near-maximum-length messages is now refused once its projection passes the cap. This is a
  deliberate behaviour change for such sessions only.
- With a real LLM, the completion would also need a `max_tokens` reservation. No real LLM is wired in.

### 3. Data layer (owner's proposed interpretation, not named in the spec)
**Purpose:** keep personal data out of models, memory and logs, and keep the knowledge and evaluation data
truthful and versioned.

**Controls and where they are implemented:**

| Control | Implemented in |
|---|---|
| Fabricated order dataset only, with no names or addresses | `data/dataset.py`, `data/orders.json` |
| A knowledge base marked as simulated policy | [policies.json](../data/knowledge_base/policies.json), `disclaimer` |
| Phone and card-last-4 masking before the crew, memory and logs | `mask_pii`, `JsonlLogger` |
| Previous policy wording preserved before any change | [history/](../data/knowledge_base/history/) |
| Evaluation sets frozen before runs; original results hash-locked | `evaluation/test_queries.json`, `scripts/run_evaluation.py` |

**How it is tested:**
- test_dataset.py and test_knowledge_base.py, including the KB-08 consistency tests.
- test_guardrails.py and test_api.py, which check that no raw PII reaches the log.
- test_evaluation.py, the hash locks.

**Evidence:** the Phase 1, 2, 6, 7 and KB-08 transcripts.

**Limitations:**
- Masking covers only the fixed-format PII the spec names. Names and addresses are out of scope.
- Session memory is in-process and is not encrypted.

### 4. Organizational layer (owner's proposed interpretation, not named in the spec)
**Purpose:** make people accountable for decisions the system must not make alone.

**Controls and where they are implemented:**
- **Phase gates:** each phase needs owner approval before the next starts.
- **Requirements trace:** [REQUIREMENTS_TRACE.md](REQUIREMENTS_TRACE.md) maps every spec requirement to code,
  tests and evidence.
- **Owner approval for changes:** policy wording changes need the owner's approval (KB-08 v2), and rubric
  changes are versioned and labelled.
- **Human escalation:** the escalation score flags cases for a person (`escalation_recommended`). The agent
  reports and flags; it never resolves disputes or issues refunds itself.
- **The risk classification** above, owned by the project owner.

**How it is tested:**
- The escalation score and threshold are tested in test_order_lookup.py.
- The trace and the approvals are process controls, recorded in git history and PROJECT_STATUS.md, not tested
  by code.

**Limitations:** these are process controls of a capstone project, not an operating organisation's procedures.
