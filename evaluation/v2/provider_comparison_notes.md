The "before fixes" row is the first qwen3:4b run, which had two problems; both are now fixed.

- **The review used the editor's retyped text.** Each differing query was traced stage by stage, at temperature 0,
  twice, with the same result both times:
  - **E07:** the Final Editor returned a cut-off copy ("…a minimum of 10") marked `approved`. The remove-only rule
    then dropped every correct claim.
  - **E08:** the editor rewrote the draft, and every claim was dropped.

  The decision (option B): for real models, code removes exactly the sentences the Reviewer flags
  (`UNSUPPORTED: <sentence>`) from the original draft, and never uses the editor's text. Tests cover both measured
  cases.
- **Latency included a Windows `localhost` delay.** The default Ollama URL was `http://localhost:11434`, and Ollama
  listens on IPv4 only. Each new synchronous connection to `localhost` waited about 2.2 s (measured: 2.27 / 2.23 s,
  against 0.17 s for `127.0.0.1`) while IPv6 `::1` was tried first. CrewAI opens one connection per agent call, three
  per request. In one request, the three CrewAI calls took 2.57 / 2.71 / 3.36 s for 28 / 37 / 70 generated tokens;
  the AutoGen calls, on an async client, had no such overhead. The default is now `127.0.0.1`. The model itself was
  never the bottleneck: 100% on GPU, no other model loaded, power draw 28–53 W against a 92–105 W limit, P0 clocks,
  and 57–62 tokens/s generation.

**After options B+ and C, the current run:**

- **What changed.** A flag on a sentence that appears word for word in the retrieved policy text is overruled, and
  the policy part of every answer is exact, code-verified quotes: the model only selects which sentences answer the
  question. On real models the reviewer can therefore no longer change policy text. This is a
  deliberate design choice.
- **E08 (UPI retries) is answered with the exact KB-08 sentence**, including "within 30 minutes". Rubric v1 still
  scores its accuracy 1, like the mock's, because the literal expected fact is "3 times". Rubric v3 also gives 1,
  because it counts the retry fact as "3 retries". The mock's reply happens to include KB-08's *other* sentence ("Once
  all 3 retries have also failed…"); qwen3:4b selected only the "3 more times within 30 minutes" sentence. Both
  rubrics are frozen, so this is a scoring limit, not a wrong answer.
- **E12 (escalation):** the model selected one of the two expected KB-12 sentences (accuracy 3).
- **Demo D4 (ORD-1029 multi-part) fails its KB-12 check.**
  - The model selected only the KB-01 return sentence.
  - The escalation part is still answered, by the order facts ("…flagged for escalation").
  - The policy part is no longer lost, which was the regression this check was added for.
  - The expectation was set before this run and is not changed after seeing the result.

**Thermal rule in this run:**
- Each request starts at ≤ 65 °C, and a request is discarded and re-run only if NVIDIA's thermal or power-cap counters
  grew during it, up to 4 attempts.
- With Cooler Boost on and background apps closed, requests started at 52–57 °C, but most reached 85–87 °C and had
  10–1,000 ms of thermal slowdown on their first attempt, so they were re-run.
- **E10, E12 and D4 were still throttled after 4 attempts.** They are recorded with their throttle time: 838, 119 and
  401 ms. Power-cap slowdown was 0 ms throughout.

**A one-off mock retrieval difference:** one mock run in this session (committed in `67490d9`, since replaced) gave E07
a different retrieved passage. It quoted the fragment "100 points must be redeemed…" instead of "Each point is worth
₹1 at checkout, a minimum of 100 points…", so it did not match the graded r2. Nothing else was running at the time. It
did not recur in 3 full mock re-runs or 3 single E07 runs, all identical to r2. The cause is not identified; Chroma
retrieval is suspected, like the intermittent `test_11` failure.
