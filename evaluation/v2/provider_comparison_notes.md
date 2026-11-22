The "before fixes" row is the first qwen3:4b run, which had two problems; both are now fixed.

- **The review used the editor's retyped text.** Each differing query was traced stage by stage, at temperature 0,
  twice, with the same result both times:
  - **E07:** the Final Editor returned a cut-off copy ("…a minimum of 10") marked `approved`. The remove-only rule
    then dropped every correct claim.
  - **E08:** the editor rewrote the draft, and every claim was dropped.

  The owner's decision (option B): for real models, code removes exactly the sentences the Reviewer flags
  (`UNSUPPORTED: <sentence>`) from the original draft, and never uses the editor's text. Tests cover both measured
  cases.
- **Latency included a Windows `localhost` delay.** The default Ollama URL was `http://localhost:11434`, and Ollama
  listens on IPv4 only. Each new synchronous connection to `localhost` waited about 2.2 s (measured: 2.27 / 2.23 s,
  against 0.17 s for `127.0.0.1`) while IPv6 `::1` was tried first. CrewAI opens one connection per agent call, three
  per request. In one request, the three CrewAI calls took 2.57 / 2.71 / 3.36 s for 28 / 37 / 70 generated tokens;
  the AutoGen calls, on an async client, had no such overhead. The default is now `127.0.0.1`. The model itself was
  never the bottleneck: 100% on GPU, no other model loaded, power draw 28–53 W against a 92–105 W limit, P0 clocks,
  and 57–62 tokens/s generation.

The current run, after both fixes, still differs from the mock on two queries:

- **E08 (UPI retries):** the review worked as designed. The grounded generator accepted the model's own sentence
  ("You can retry the payment up to 3 more times after the first failed attempt."). It passes the lexical grounding
  check, but it drops KB-08's "within 30 minutes" condition. The Reviewer flagged it as `UNSUPPORTED`, code removed
  it, nothing was left, and the reply is the safe refusal. Using only verified quotes, never the model's wording,
  would avoid this; that option was not chosen.
- **E12 (escalation):** the generator chose one of the two expected KB-12 sentences (accuracy 3).

**Thermal control in the current run:** every request started at ≤ 52 °C, and every one of the 15 measured requests
had 0 ms of thermal slowdown on its first attempt. One request heats this GPU by about 25–30 °C. In an earlier
attempt that started requests at ≤ 60 °C, most requests reached 86–88 °C and throttled, so that attempt was
discarded.

**Found after this run, outside the 15 queries:** on the chat UI's multi-part demo question ("My order ORD-1029
arrived late. Can I still return the shoes, and is my case being escalated?"), qwen3:4b's Reviewer **over-flags**.
- It marked both draft sentences `UNSUPPORTED`, although both appear word for word in the retrieved KB-01 and KB-12
  context.
- It also listed four fragments of the context itself as `UNSUPPORTED`.
- Option B trusts those flags, so the reply keeps only the order facts. That is safe, but the correct policy part is
  lost.

This is deterministic: two runs at temperature 0 gave the same result. A deterministic guard (ignore a flag on a
sentence that appears verbatim in the retrieved context) is proposed, and awaits the owner's decision.
