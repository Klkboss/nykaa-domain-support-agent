# Evaluation report: Phase 7 remediation (spec Task 13)

**The original Phase 7 evaluation is preserved unchanged** in `evaluation/evaluation_results.json` and
`evaluation/EVALUATION_REPORT.md` (commit `b7e2fe9`). SHA-256 checked by this run:

- `evaluation/evaluation_results.json`: `e5e7f5601a45b0c32c686c647f1af15bb63aa4bd63fad3f05bfccf78c8bb01e4`
- `evaluation/EVALUATION_REPORT.md`: `716bd15e3cf7ebcd8d99aaa6023bb4c82271fd677965b0252b2479fbd7da9972`
- `evaluation/test_queries.json`: `3699ac04647a6b552c8432caae2fe134aa45ad472551ad771dba77c14fdd7565`

Judge: **MockJudgeLLM (deterministic rubric scorer behind a BaseLLM interface; NOT a real LLM judgement)**. Same 15 frozen queries (evaluation/test_queries.json), run again through
the current pipeline. The same outputs are scored with the original rubric **v1** (the like-for-like
comparison), the **withdrawn** rubric v2 (kept only for audit: it treated 'After 3 failed attempts' as
'retry up to 3 times', an off-by-one change of meaning), and the **revised rubric v3**
(evaluation/remediation/rubric_v3.json, which only binds the E08 count fact to the event it counts).

## Scores

| ID | Original (b7e2fe9, v1) A/G/C/S | Remediation, rubric v1 (original) A/G/C/S | Remediation, rubric v2 (WITHDRAWN) A/G/C/S | Remediation, rubric v3 (revised) A/G/C/S |
|---|---|---|---|---|
| E01 | 1/5/5/5 | 5/5/5/5 **changed** | 5/5/5/5 **changed** | 5/5/5/5 **changed** |
| E02 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E03 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E04 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E05 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E06 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E07 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E08 | 1/5/5/5 | 1/5/5/5 | 5/5/5/5 **changed** | 1/5/5/5 |
| E09 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E10 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E11 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E12 | 3/5/5/5 | 5/5/5/5 **changed** | 5/5/5/5 **changed** | 5/5/5/5 **changed** |
| E13 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E14 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| E15 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 | 5/5/5/5 |
| **Averages** (acc / grd / cmp / saf) | **4.33 / 5.00 / 5.00 / 5.00** | **4.73 / 5.00 / 5.00 / 5.00** | **5.00 / 5.00 / 5.00 / 5.00** | **4.73 / 5.00 / 5.00 / 5.00** |

## Every changed score or reply

### E01: I bought a dress 20 days ago and haven't worn it. Can I still return it?

- Original reply: Policy information (source: KB-01): Apparel can be returned within 30 days of delivery provided it is unworn, unwashed and has all tags attached. Beauty products can be returned within 15 days of delivery only if they are unopened and the original seal is intact.
- Remediation reply: Policy information (source: KB-01): Apparel can be returned within 30 days of delivery provided it is unworn, unwashed and has all tags attached.
- Accuracy: original 1 (forbidden fact present: ['15 days']); rubric v1 (original) 5 (expected facts found 1/1: missing []); rubric v2 (WITHDRAWN) 5 (expected facts found 1/1: missing []); rubric v3 (revised) 5 (expected facts found 1/1: missing [])
- Why: Pipeline fix (category-subject filter in the grounded generator): the reply no longer appends the Beauty '15 days' rule to a dress question, so the unchanged v1 forbidden-fact rule no longer fires.

### E05: My trimmer broke after four months of use. Is it under warranty?

- Original reply: Policy information (source: KB-05): Electronics come with a 12-month manufacturer warranty from the delivery date, and claims are handled at the brand's authorised service centre with Nykaa support helping to raise the claim. Home appliances and kitchen tools carry a 6-month warranty against manufacturing defects, while home decor and textiles carry no warranty.
- Remediation reply: Policy information (source: KB-05): Electronics come with a 12-month manufacturer warranty from the delivery date, and claims are handled at the brand's authorised service centre with Nykaa support helping to raise the claim. No warranty covers physical damage, liquid damage, misuse or repairs done by an unauthorised person.
- Accuracy: original 5 (expected facts found 1/1: missing []); rubric v1 (original) 5 (expected facts found 1/1: missing []); rubric v2 (WITHDRAWN) 5 (expected facts found 1/1: missing []); rubric v3 (revised) 5 (expected facts found 1/1: missing [])
- Why: Reply changed, scores did not. A trimmer is Electronics (KB-01 lists trimmers under Electronics), so the correct period is the expected fact '12-month', which stays the first sentence. The category-subject filter (E01 fix) dropped the Home-appliance '6-month' sentence (another category's period, which the original reply wrongly included) and the Apparel/Beauty no-warranty sentence; the category-neutral KB-05 exclusion 'No warranty covers physical damage …', which applies to every category and to a device that 'broke', became the second sentence.

### E08: My UPI payment failed twice. How many times can I retry?

- Original reply: Policy information (source: KB-08): After 3 failed attempts the customer should switch to another payment method such as UPI, a different card, net banking or cash on delivery.
- Remediation reply: Policy information (source: KB-08): After 3 failed attempts the customer should switch to another payment method such as UPI, a different card, net banking or cash on delivery.
- Accuracy: original 1 (expected facts found 0/1: missing ['3 times']); rubric v1 (original) 1 (expected facts found 0/1: missing ['3 times']); rubric v2 (WITHDRAWN) 5 (expected facts found 1/1: missing []); rubric v3 (revised) 1 (expected facts found 0/1: missing ['3 times'])
- Why: Score changes under the withdrawn v2 only (1 -> 5); v1 and v3 both score accuracy 1 and the pipeline output is unchanged. The reply quotes KB-08's 'After 3 failed attempts … switch' (3 attempts in total), but the question asks how many RETRIES are allowed and the expected fact comes from 'can retry the payment up to 3 times' (3 retries after the first attempt). v2 equated the two counts; v3 does not. The retry-limit sentence scored 0.5405 against the best 0.6451, outside the generator's 0.10 margin, so it was not included in the reply.

### E12: My order ORD-1022 was delayed. Will my complaint be escalated to a senior specialist?

- Original reply: Policy information (source: KB-10): Items that arrive damaged, defective, expired or different from what was ordered must be reported through the Help Centre within 48 hours of delivery. / Order information (ORD-1022): status Refunded, order value ₹2,803. Escalation score 0.8667 (threshold 0.78) meets the escalation threshold, so this case is flagged for escalation.
- Remediation reply: Policy information (source: KB-12): Cases still unresolved after 72 hours, or repeated complaints about the same order, are escalated to Level 3, the grievance officer, who responds within 2 business days. Level 2 specialists respond within 4 business hours and aim to resolve the case within 48 hours. / Order information (ORD-1022): status Refunded, order value ₹2,803. Escalation score 0.8667 (threshold 0.78) meets the escalation threshold, so this case is flagged for escalation.
- Accuracy: original 3 (expected facts found 1/2: missing ['Level 2']); rubric v1 (original) 5 (expected facts found 2/2: missing []); rubric v2 (WITHDRAWN) 5 (expected facts found 2/2: missing []); rubric v3 (revised) 5 (expected facts found 2/2: missing [])
- Why: Pipeline fix (escalation routing in the shared retrieval path): KB-12 chunks only, so the reply cites KB-12 ('Level 3 …' and 'Level 2 specialists respond …') instead of the KB-10 damaged-item sentence. Remaining limitation: the most specific sentence ('A case moves to Level 2 … delayed shipment with a high escalation score') is still split across two fixed-size windows, so it is not quoted.

## Limitations

- The judge is a deterministic rubric scorer, not a real LLM; grounding reuses the guardrail's lexical rule,
  so grounding/completeness/safety near 5 are partly by construction.
- v3 differs from v1 only for annotated count facts (E08's '3 times' counts retries). On these 15 outputs
  v3 gives the same scores as v1. Other facts are still exact substrings, so a correct paraphrase of a
  non-count fact can still be missed.
- KB-08 itself is ambiguous: under the strict reading, 'retry up to 3 times' allows 4 attempts in total,
  while 'After 3 failed attempts … switch' stops at 3. The KB is not edited without the owner's approval.
- E08 remains a pipeline weakness: the reply does not state the retry limit the question asks for.

