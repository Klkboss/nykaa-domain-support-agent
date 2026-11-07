# Evaluation report r2: KB-08 clarified (spec Task 13, Phase 7)

Run against **KB-08 wording v2** (owner-approved: 3 retries after the first failed attempt, at most 4
attempts in total). Earlier runs used **KB-08 wording v1**, preserved in
`data/knowledge_base/history/KB-08_wording_v1.json`, and stay unchanged:

- the original Phase 7 evaluation (`b7e2fe9`): `evaluation/evaluation_results.json`, `evaluation/EVALUATION_REPORT.md`
- remediation r1 (`d5ffadb`, `b0ba629`): `evaluation/remediation/*_r1_*`, `EVALUATION_REPORT_REMEDIATION.md`

SHA-256 (LF-normalised) checked by this run:

- `evaluation/evaluation_results.json`: `52952c90b5456b1c16af4f9188e65a85c99a5c94aa58c707435d388ef33d48ef`
- `evaluation/EVALUATION_REPORT.md`: `40e60a1f54e37e3c64344813b966d44a593701888f713f288b85b1cdbcd573bd`
- `evaluation/test_queries.json`: `3699ac04647a6b552c8432caae2fe134aa45ad472551ad771dba77c14fdd7565`
- `evaluation/remediation/evaluation_results_r1_rubric_v1.json`: `bdf4535e3a1fceb5e4f0b1cc6a5588fa8156cd92f0114afd8dd3ba908e225592`
- `evaluation/remediation/evaluation_results_r1_rubric_v2.json`: `b2205cfa07630437977176340444be168706ddb0d989cd4c9c8eb4a9999366d6`
- `evaluation/remediation/evaluation_results_r1_rubric_v3.json`: `685c730f1faeead88fc75a9d6379f5d636aeb5b41991b6aa8c804532f001c51b`
- `evaluation/remediation/EVALUATION_REPORT_REMEDIATION.md`: `b0f4720853adaaf7c865262c08f1b772c8235d1ac83fcad8103d2399134cf13b`

Judge: **MockJudgeLLM (deterministic rubric scorer behind a BaseLLM interface; NOT a real LLM judgement)**. Same 15 frozen queries (evaluation/test_queries.json). The same outputs are scored
with the original rubric **v1** (like-for-like) and the revised rubric **v3** (frozen in `a822a91`, before
this policy change; unchanged). The withdrawn rubric v2 is not used.

## Scores

| ID | Original (b7e2fe9, KB-08 v1, rubric v1) | r1 (KB-08 v1), rubric v1 | r1 (KB-08 v1), rubric v3 | **r2 (KB-08 v2), rubric v1** | **r2 (KB-08 v2), rubric v3** |
|---|---|---|---|---|---|
| E01 | 1/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E02 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E03 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E04 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E05 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E06 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E07 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E08 | 1/5/5/5 | 1/5/5/5 | 1/5/5/5 | 1/5/5/5 | 5/5/5/5 **changed** |
| E09 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E10 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E11 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E12 | 3/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E13 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E14 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E15 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| **Averages** (acc / grd / cmp / saf) | **4.33 / 5.00 / 5.00 / 5.00** | **4.73 / 5.00 / 5.00 / 5.00** | **4.73 / 5.00 / 5.00 / 5.00** | **4.73 / 5.00 / 5.00 / 5.00** | **5.00 / 5.00 / 5.00 / 5.00** |

'changed' marks a difference from r1 under the same rubric.

## Every score or reply that differs from the original run

### E01: I bought a dress 20 days ago and haven't worn it. Can I still return it?

- Original reply (KB-08 v1): Policy information (source: KB-01): Apparel can be returned within 30 days of delivery provided it is unworn, unwashed and has all tags attached. Beauty products can be returned within 15 days of delivery only if they are unopened and the original seal is intact.
- r1 reply (KB-08 v1): Policy information (source: KB-01): Apparel can be returned within 30 days of delivery provided it is unworn, unwashed and has all tags attached.
- r2 reply (KB-08 v2): Policy information (source: KB-01): Apparel can be returned within 30 days of delivery provided it is unworn, unwashed and has all tags attached.
- Accuracy: original 1; r2 rubric v1 (original) 5 (expected facts found 1/1: missing []); r2 rubric v3 (revised) 5 (expected facts found 1/1: missing [])
- Why: Pipeline fix (category-subject filter in the grounded generator): the reply no longer appends the Beauty '15 days' rule to a dress question, so the unchanged v1 forbidden-fact rule no longer fires.

### E05: My trimmer broke after four months of use. Is it under warranty?

- Original reply (KB-08 v1): Policy information (source: KB-05): Electronics come with a 12-month manufacturer warranty from the delivery date, and claims are handled at the brand's authorised service centre with Nykaa support helping to raise the claim. Home appliances and kitchen tools carry a 6-month warranty against manufacturing defects, while home decor and textiles carry no warranty.
- r1 reply (KB-08 v1): Policy information (source: KB-05): Electronics come with a 12-month manufacturer warranty from the delivery date, and claims are handled at the brand's authorised service centre with Nykaa support helping to raise the claim. No warranty covers physical damage, liquid damage, misuse or repairs done by an unauthorised person.
- r2 reply (KB-08 v2): Policy information (source: KB-05): Electronics come with a 12-month manufacturer warranty from the delivery date, and claims are handled at the brand's authorised service centre with Nykaa support helping to raise the claim. No warranty covers physical damage, liquid damage, misuse or repairs done by an unauthorised person.
- Accuracy: original 5; r2 rubric v1 (original) 5 (expected facts found 1/1: missing []); r2 rubric v3 (revised) 5 (expected facts found 1/1: missing [])
- Why: Reply changed, scores did not. A trimmer is Electronics (KB-01 lists trimmers under Electronics), so the correct period is the expected fact '12-month', which stays the first sentence. The category-subject filter (E01 fix) dropped the Home-appliance '6-month' sentence (another category's period, which the original reply wrongly included) and the Apparel/Beauty no-warranty sentence; the category-neutral KB-05 exclusion 'No warranty covers physical damage …', which applies to every category and to a device that 'broke', became the second sentence.

### E08: My UPI payment failed twice. How many times can I retry?

- Original reply (KB-08 v1): Policy information (source: KB-08): After 3 failed attempts the customer should switch to another payment method such as UPI, a different card, net banking or cash on delivery.
- r1 reply (KB-08 v1): Policy information (source: KB-08): After 3 failed attempts the customer should switch to another payment method such as UPI, a different card, net banking or cash on delivery.
- r2 reply (KB-08 v2): Policy information (source: KB-08): Once all 3 retries have also failed, the customer should switch to another payment method such as UPI, a different card, net banking or cash on delivery. If the first payment attempt fails, the order is not placed and the customer can retry the payment up to 3 more times within 30 minutes while the cart prices stay locked.
- Accuracy: original 1; r2 rubric v1 (original) 1 (expected facts found 0/1: missing ['3 times']); r2 rubric v3 (revised) 5 (expected facts found 1/1: missing [])
- Why: Policy clarification (KB-08 wording v2, owner-approved: 3 retries after the first attempt, at most 4 attempts). The reply now states the retry limit: 'Once all 3 retries have also failed … switch' and 'retry the payment up to 3 more times within 30 minutes'. Rubric v3 credits the frozen fact '3 times' (annotated: counts retries) through 'all 3 retries' (v3's frozen rule does not read '3 more times'). Rubric v1's literal substring '3 times' is not in the reply, so v1 still scores accuracy 1.

### E12: My order ORD-1022 was delayed. Will my complaint be escalated to a senior specialist?

- Original reply (KB-08 v1): Policy information (source: KB-10): Items that arrive damaged, defective, expired or different from what was ordered must be reported through the Help Centre within 48 hours of delivery. / Order information (ORD-1022): status Refunded, order value ₹2,803. Escalation score 0.8667 (threshold 0.78) meets the escalation threshold, so this case is flagged for escalation.
- r1 reply (KB-08 v1): Policy information (source: KB-12): Cases still unresolved after 72 hours, or repeated complaints about the same order, are escalated to Level 3, the grievance officer, who responds within 2 business days. Level 2 specialists respond within 4 business hours and aim to resolve the case within 48 hours. / Order information (ORD-1022): status Refunded, order value ₹2,803. Escalation score 0.8667 (threshold 0.78) meets the escalation threshold, so this case is flagged for escalation.
- r2 reply (KB-08 v2): Policy information (source: KB-12): Cases still unresolved after 72 hours, or repeated complaints about the same order, are escalated to Level 3, the grievance officer, who responds within 2 business days. Level 2 specialists respond within 4 business hours and aim to resolve the case within 48 hours. / Order information (ORD-1022): status Refunded, order value ₹2,803. Escalation score 0.8667 (threshold 0.78) meets the escalation threshold, so this case is flagged for escalation.
- Accuracy: original 3; r2 rubric v1 (original) 5 (expected facts found 2/2: missing []); r2 rubric v3 (revised) 5 (expected facts found 2/2: missing [])
- Why: Pipeline fix (escalation routing in the shared retrieval path): KB-12 chunks only, so the reply cites KB-12 ('Level 3 …' and 'Level 2 specialists respond …') instead of the KB-10 damaged-item sentence. Remaining limitation: the most specific sentence ('A case moves to Level 2 … delayed shipment with a high escalation score') is still split across two fixed-size windows, so it is not quoted.

## Limitations

- The judge is a deterministic rubric scorer, not a real LLM; grounding reuses the guardrail's lexical rule,
  so grounding/completeness/safety near 5 are partly by construction.
- v1 matches expected facts as exact substrings, so E08's correct '3 more times' / '3 retries' wording
  still scores accuracy 1 under v1. v3 reads only '<n> <event word>' and '<event word> … <n> times', not
  '<n> more times'; E08 is credited through 'all 3 retries'.
- E08 is the only reply that changed between r1 and r2; it changed because the policy text changed.

