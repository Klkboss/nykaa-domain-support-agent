# Phase 8 checklist: AutoGen review + AI governance (spec Tasks 14-15)

Written and committed before any code (commit `ae39635`, first in PROJECT_STATUS.md; copied here unchanged to match the earlier phases' docs/PHASEn_CHECKLIST.md files).

Scope: spec Part 4 Tasks 14 and 15 (assignment.md L198-225) and the Part 4 acceptance bullets 1-2 (L266-270). Task 16 (caching) is NOT part of Phase 8.

| Req | Spec text (abridged) | Implementation | Test | Evidence |
|---|---|---|---|---|
| T14.1 | 2-agent `RoundRobinGroupChat`, `max_turns=2` | app/review/autogen_review.py `build_review_team` (Policy_Compliance_Reviewer, Final_Editor) | team shape: 2 participants, max_turns 2, message order user → reviewer → editor | transcripts phase8_autogen_approve.txt / _revise.txt |
| T14.2 | Final-Editor `output_content_type=Verdict`; team `custom_message_types=[StructuredMessage[Verdict]]` | same | last message is `StructuredMessage[Verdict]` | same |
| T14.3 | Team input = Composer's draft answer + original retrieved context; stage runs after the crew's draft | app/pipeline.py: review the Composer's policy draft against the rag_lookup evidence on every answered turn; the Phase 6 groundedness guardrail still runs unchanged | pipeline test: reviewer saw the Composer draft + retrieved context | same |
| T14.4 | Verdict fields `approved: bool`, `final_answer: str`, `reason: str` | `Verdict` (exactly these three) | schema test (fields, bad payload rejected) | verdict JSON in both transcripts |
| T14.5 | ≥2 sample queries: one approved unchanged, one actually revised (e.g. injected ungrounded claim) | scripts/run_review_demo.py through `SupportAssistant.ask` | approve: verdict.approved, final_answer == draft, reply unchanged; revise: approved False, injected claim removed, reason names it | both transcripts |
| T15.1 | Application layer: only the Lookup Agent may call `check_order_status`; demonstrate (blocked or never wired) + one-paragraph guard explanation | app/governance/least_autonomy.py: per-role tool allow-list enforced when the crew is built (also no delegation); AutoGen reviewers hold no tools | positive (real crew passes) and negative (tool wired to Retrieval / Composer, renamed order tool, delegation, extra tool, AutoGen agent with a tool) | phase8_governance.txt; docs/GOVERNANCE.md paragraph |
| T15.2 | Risk level Low/Medium/High + one-paragraph justification | docs/GOVERNANCE.md: **Medium** (owner decision, Phase 0) | — (document) | GOVERNANCE.md |
| T15.3 | Runtime layer: per-request token/cost budget cap; oversized request rejected, not silently exceeded | app/governance/budget.py: pre-flight projection (rejects before any model call) + per-call accounting in the mock LLM clients (never exceeds the cap) | below / exactly at / above the limit, repeated requests (no carry-over), exhaustion mid-request, oversized request rejected; projection ≥ measured usage | phase8_governance.txt |
| T15.4 | Four-layer governance model | docs/GOVERNANCE.md: Application + Runtime (named in the spec); Data + Organizational (owner's interpretation, labelled) — purpose, controls, where, tests, evidence, limitations per layer | — | GOVERNANCE.md |

Frozen demo inputs (before any run):
- APPROVE: "How long do I have to return shoes?" with the normal pipeline.
- REVISE: the same question, with a Composer that appends the ungrounded claim "Premium members get instant cash refunds at any store." (cited KB-01) to its otherwise correct draft.
- Oversized request: a question of 20,000 characters (simulated; the API already caps questions at 2,000 characters).

Constraints: no change to approved behaviour (Phases 0-7) unless Phase 8 integration needs it; API request/response schemas unchanged except the new `budget_exceeded` status value; every change to an existing test explained.
