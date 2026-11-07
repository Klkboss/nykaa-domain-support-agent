# Full-pipeline evaluation (Phase 8): CrewAI + AutoGen + final groundedness check

Same 15 frozen queries (evaluation/test_queries.json). Pipeline: SupportAssistant.ask: guardrails -> budget -> CrewAI crew -> groundedness -> AutoGen review -> final groundedness check -> render_reply.
Judge: **MockJudgeLLM (deterministic rubric scorer behind a BaseLLM interface; NOT a real LLM judgement)**. The original Phase 7, r1 and r2 artifacts are not modified; original and r1 SHA-256 checked:

- `evaluation/evaluation_results.json`: `52952c90b5456b1c16af4f9188e65a85c99a5c94aa58c707435d388ef33d48ef`
- `evaluation/EVALUATION_REPORT.md`: `40e60a1f54e37e3c64344813b966d44a593701888f713f288b85b1cdbcd573bd`
- `evaluation/test_queries.json`: `3699ac04647a6b552c8432caae2fe134aa45ad472551ad771dba77c14fdd7565`
- `evaluation/remediation/evaluation_results_r1_rubric_v1.json`: `bdf4535e3a1fceb5e4f0b1cc6a5588fa8156cd92f0114afd8dd3ba908e225592`
- `evaluation/remediation/evaluation_results_r1_rubric_v2.json`: `b2205cfa07630437977176340444be168706ddb0d989cd4c9c8eb4a9999366d6`
- `evaluation/remediation/evaluation_results_r1_rubric_v3.json`: `685c730f1faeead88fc75a9d6379f5d636aeb5b41991b6aa8c804532f001c51b`
- `evaluation/remediation/EVALUATION_REPORT_REMEDIATION.md`: `b0f4720853adaaf7c865262c08f1b772c8235d1ac83fcad8103d2399134cf13b`

| ID | status | AutoGen ran | verdict approved | final re-check clean | v1 A/G/C/S | v3 A/G/C/S | same as r2 |
|---|---|---|---|---|---|---|---|
| E01 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E02 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E03 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E04 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E05 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E06 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E07 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E08 | answered | True | True | True | 1/5/5/5 | 5/5/5/5 | True |
| E09 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E10 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E11 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E12 | answered | True | True | True | 5/5/5/5 | 5/5/5/5 | True |
| E13 | fallback | False | — (no policy text to review (order-only or no-answer turn)) | True | 5/5/5/5 | 5/5/5/5 | True |
| E14 | blocked | False | — (blocked before the crew) | n/a | 5/5/5/5 | 5/5/5/5 | True |
| E15 | answered | False | — (no policy text to review (order-only or no-answer turn)) | True | 5/5/5/5 | 5/5/5/5 | True |

Averages: rubric v1 **4.73 / 5.00 / 5.00 / 5.00**; rubric v3 **5.00 / 5.00 / 5.00 / 5.00**.

AutoGen does not run when there is no policy text to review: E13 (out-of-scope fallback), E14 (injection blocked before the crew) and E15 (order not found, no policy question). Order facts are checked against the real tool output by the groundedness check in every case.

