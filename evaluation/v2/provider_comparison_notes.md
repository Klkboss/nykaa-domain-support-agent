Each differing query was traced through the real pipeline, stage by stage, on qwen3:4b at temperature 0. Every
trace was run twice, with the same result both times.

- **E07 (loyalty points): the Final Editor returned a partial copy.** Retrieval (KB-07, similarity 0.62), the
  composer's draft (two KB-07 claims) and the groundedness check were all correct. The AutoGen Final Editor then
  returned `approved: true` with `final_answer` cut off mid-sentence ("Each point is worth ₹1 at checkout, a minimum
  of 10"). Its token limit (1,024) was not the cause; the model wrote it that way. Phase 8's rule decides on the text,
  not the `approved` flag, and a review may only remove claims. So no claim survived, and only the deterministic
  "please share your order ID" line remained. The question mentions "my order", which counts as an order question.
- **E08 (UPI retries): the Final Editor rewrote instead of removing.** The grounded generator accepted the model's
  own sentence ("You can retry the payment up to 3 more times after the first failed attempt."), which passes the
  lexical grounding check against KB-08. The editor replaced it with KB-08's original sentences (`approved: false`).
  The review may not add text, so every claim was dropped and the reply became the safe refusal.
- **E12 (escalation): the generator chose one quote.** The grounded generator (`quotes` path) selected one KB-12
  sentence, so the expected "Level 2" fact is missing (accuracy 3). The editor also tried to *add* a sentence, and
  the pipeline correctly discarded it. Order facts were copied from the order tool in code, because the composer
  left them out.

In all three, nothing false reached the customer: the pipeline failed safe. But in E07 and E08 the review stage
removed correct, grounded content, because this 4B model doesn't reliably copy the draft verbatim as the editor.
