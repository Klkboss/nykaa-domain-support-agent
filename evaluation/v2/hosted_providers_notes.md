## GitHub Models: retired, not measurable

GitHub's documentation (docs.github.com/en/rest/models/inference, read 2026-11-23) says: "As of July 30, 2026, GitHub
Models has been fully retired. The playground, model catalog, inference API, and bring your own key (BYOK) are no
longer available to any customer." The probe confirmed it with the token configured at the time (since revoked): both
`https://models.github.ai/catalog/models` and `https://models.github.ai/inference/chat/completions` answer
`HTTP 200` with the plain-text body `OK` and no model output (`hosted_probe.json`). It is therefore excluded; no
quality, latency or limit numbers exist for it.

## Free-tier terms that matter for a public portfolio demo

Read on 2026-11-23 from each provider's official pages (quoted where it matters). This is a summary for planning,
not legal advice.

| Provider | Free allowance | Data use | Public-demo points |
|---|---|---|---|
| **Groq** (Free plan) | Per model: 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD (console.groq.com/docs/rate-limits; RPD and TPM confirmed by the response headers on this key). 429 with `retry-after`. | Services Agreement §4.2: Groq is "not permitted to use Inputs or Outputs for training or fine-tuning". Your-data page: "By default, Groq does not retain customer data for inference requests", except reliability/abuse logs kept "up to 30 days"; Zero Data Retention can be switched on in Data Controls. | The agreement lets customers make the service available to end users through their applications (§3.1). It bars use "in a manner intended to avoid incurring Fees" (§6.3(d)), so the demo must stay a low-volume showcase. The services are "not for consumer use". The AUP was not read. |
| **Cloudflare Workers AI** (Workers Free) | 10,000 Neurons per day, reset 00:00 UTC; beyond that "further operations will fail with an error" (no charge on the Free plan). Text generation: 300 requests per minute. gpt-oss-120b: 31,818 Neurons per M input and 68,182 per M output tokens; gpt-oss-20b: 18,182 and 27,273. | "Cloudflare does not use Customer Content processed by Workers AI to train AI models." For standard inference, inputs and outputs are not retained after the response. The gpt-oss model pages are tagged "Zero data retention". | No free-tier restriction on public use was found on the pages read. Exceeding the daily Neurons fails closed instead of billing, which suits a no-billing demo. |
| **Google Gemini API** (free tier) | Per-model limits are shown only in AI Studio for the project; "Rate limits are more restricted for experimental and preview models"; daily quotas "reset at midnight Pacific time". | Unpaid Services: Google uses submitted content and responses "to provide, improve, and develop" its products; "human reviewers may read, annotate, and process your API input and output"; "Do not submit sensitive, confidential, or personal information to the Unpaid Services." | "You may use only Paid Services when making API Clients available to users in the European Economic Area, Switzerland, or the United Kingdom." A worldwide public demo on the free tier would break this unless those regions are blocked. Users must be 18+. |
| **GitHub Models** | Retired on 2026-07-30 (above). | n/a | n/a |

Sources: console.groq.com/docs/rate-limits, /docs/your-data, /docs/legal/services-agreement;
developers.cloudflare.com/workers-ai/platform/pricing/, /limits/, /data-usage/, /models/gpt-oss-120b/;
ai.google.dev/gemini-api/docs/rate-limits, /terms, /docs/openai (redirected to aistudio.google.com/docs/...);
docs.github.com/en/rest/models/inference.

## What the scores mean (traced from the replies)

Every candidate kept **Grounding 5.00 and Safety 5.00**, so none is rejected at step (a). Every policy reply is an
exact, code-verified quote (option C), so differences are about which sentence a model picked, or a failure.

- **Cloudflare gpt-oss-120b:** 14 of 15 correct; **E08 ended with the polite error reply**. The AutoGen editor's
  structured output did not validate against the `Verdict` schema (OpenAI SDK parse error). The same fault ended D1
  and D4 on Cloudflare gpt-oss-20b in phase 1. On real models the editor's verdict is never used (option B), but
  its failure still fails the request, by design (`test_a_failing_reviewer_gives_the_polite_reply...`).
- **Groq gpt-oss-120b:** **E07 is a real miss**: it quoted the KB-07 expiry sentence instead of the 20% cap, and the
  Composer asked for an order ID on a policy question ("my order value"). E08 is the known rubric artefact: the
  exact KB-08 sentence ("up to 3 more times within 30 minutes") scores 1 because the frozen rubric looks for
  "3 times". E12 quoted one of the two expected KB-12 sentences.
- **Groq gpt-oss-20b:** **E08 is a real miss** (it quoted the automatic-reversal sentence, not the retry limit),
  E12 as above, and demo D4 quoted only KB-12, so KB-01 is missing.
- **qwen3:4b on Ollama (local baseline):** E08 rubric artefact and E12 as above; D4 quoted only KB-01.

One measured run per candidate: each gap is one or two queries, so a re-run can move them.

## Selection order

| Step | Cloudflare gpt-oss-120b | Groq gpt-oss-120b | Groq gpt-oss-20b | Gemini 3 Flash | Cloudflare gpt-oss-20b |
|---|---|---|---|---|---|
| (a) Grounding and Safety 5.00 | yes | yes | yes | not run in phase 2 | not run in phase 2 |
| (b) Accuracy / completeness / demo | **4.73** / 4.73 / **6/6** | 4.33 / 5.00 / **6/6** | 4.60 / 5.00 / 5/6 | screening 4/6 | screening 4/6 (2 errors) |
| (c) Median latency | 16.80 s | **7.02 s** | 6.02 s | 25.82 s (screening) | 11.30 s (screening) |
| (d) Free-tier questions per day / minute | **99** (74 at the worst 134 Neurons) / 56 | 62 / 2.5 | 65 / 2.6 | **4** / 1.1 | 237 / 65 |

Gemini 3 Flash is out at screening on capacity: Google's own 429 bodies on this key name a free-tier quota of
**20 requests per day and 5 per minute per model**, about 4 questions a day, and it could not complete the six demo
questions. Gemini 3.8 Flash (the newest Flash in the model list) answered 503 "high demand" and then timed out, so it
was not measured. Cloudflare gpt-oss-20b is out at screening on its two failed requests.

## Recommendation (as first reported)

**Primary: Cloudflare Workers AI, `@cf/openai/gpt-oss-120b`** (`LLM_REASONING_EFFORT=low`). It ranks first at step
(b): the best accuracy (4.73) and 6/6 demo checks. Its weakness is step (c): about 17 s per answer, 2.4 times
Groq's. It also has the most free capacity on this account (about 99 questions a day), a Free plan that fails
closed instead of billing, and no training on customer content.

**Fallback order when it is rate-limited or down:**
1. **Groq `openai/gpt-oss-120b`:** 6/6 demo checks and 7 s per answer, but only about 2.5 questions a minute
   (8,000 tokens per minute) and about 62 a day.
2. **Groq `openai/gpt-oss-20b`:** a separate per-model quota of the same size, 5/6 demo checks.
3. **The deterministic mock (`MOCK_LLM=true`):** never rate-limited, needs no network, and is the graded behaviour.
   Label it as the offline mode in the UI.

## Decision: both providers, with automatic failover

**Groq `openai/gpt-oss-120b` first** (about 7 s per answer, for a faster first impression), **Cloudflare
`@cf/openai/gpt-oss-120b` second**. Built and measured (the failover tables above):
- **Failover:** a request that fails with any provider error (rate limit, outage, timeout, HTTP error, bad key) is
  run again **from the start** on the fallback (`LLM_FALLBACK_*` in `.env`; the order is which provider is set as
  primary). Each attempt gets its own token budget with the same cap.
- **No waiting on the primary:** with a fallback configured, the primary runs with no retries at all: the OpenAI
  SDK's are set to 0 and CrewAI's own 429 retry is skipped. Its first rate limit or outage therefore starts the
  failover at once, and its `Retry-After` is never waited on. The fallback keeps the normal retries. A failure that is not the provider's (an
  invalid draft, for example) is not retried elsewhere and keeps the generic reply.
- **Busy reply:** if no provider could answer, the visitor gets "The assistant is busy right now. Please try again
  in a minute." instead of the generic error.
- **The provider that answered** is shown in "How this answer was made" (its host, e.g. `api.cloudflare.com`, never
  the URL path with the account ID), marked "fallback" after a failover.
- **The editor's reply is read tolerantly:** strict JSON, then JSON inside text, then a safe "unreadable" verdict.
  That verdict can remove nothing, because on real models the reply is decided in code from the reviewer's flags
  and the verified quotes (options B, B+ and C). This fixes the failures measured on Cloudflare (E08, and D1 and
  D4 on gpt-oss-20b). The phase 1 and 2 numbers above were measured before this fix.
- **Measured:**
  - With both providers healthy, Groq answered all six demo questions (6/6, 5.7–9.1 s).
  - With every Groq request answered by a simulated 429 asking for a 30 s `Retry-After`, each question made exactly
    one Groq request, and Cloudflare took over at once (6/6, 12.9–22.4 s in total, none of it spent waiting on
    Groq).

## Demo mode: settings to stay inside the free tier

In the Docker image and `.env.example`:
- `RATE_LIMIT_PER_MINUTE=3`, `RATE_LIMIT_PER_DAY=15` per visitor.
- `RATE_LIMIT_SITE_PER_DAY=70` for all visitors together: 10,000 free Cloudflare Neurons / 134, the most expensive
  measured question, so the whole day fits the fallback's quota even if Groq is out. Groq's own free tier holds
  about 62 questions a day. When the cap is reached, visitors are told the demo has reached its daily limit.
- `LLM_REASONING_EFFORT=low`. Leave `LLM_MAX_RETRIES` at its default (2): it now applies to the fallback only,
  because the primary never retries when a fallback is configured.
- Keep the generation cache on (the default). It saves the grounded-generation call for a repeated question.
  Whole replies are never cached (governance rule: no full customer responses).
- Hosted keys stay on the server side; nothing is sent to the browser.
- Not covered: the per-minute rate of the whole site. Groq's 8,000 tokens per minute (about 2.5 questions a minute)
  can still be exceeded by simultaneous visitors; failover to Cloudflare (300 requests per minute) absorbs that.

## Azure for Students as a hosting input

Read on 2026-11-23 (azure.microsoft.com/free/students, learn.microsoft.com/azure/container-apps/billing,
azure.microsoft.com/pricing/details/app-service):
- **The credit:** USD 100 to use within 12 months, no credit card, renewable each year while a student. When it
  runs out, Azure emails; without a move to pay-as-you-go "your subscription and products will be disabled", so
  there is no surprise bill.
- **Azure Container Apps (Consumption):** free every month: 180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million
  requests. The image carries CPU torch and the embedding model, so give it at least 2 GiB (its real memory use was
  not measured here). At 1 vCPU / 2 GiB always on, a month needs about 2.6 million vCPU-seconds, far above the grant.
  With **scale to zero**, only active time counts, at the cost of a cold start that loads the model.
- **App Service F1 (Free):** 1 GB RAM and 60 CPU minutes a day, "not supported" for production. Too small for this
  image. **B1** (1 core, 1.75 GB) is the smallest paid tier that fits; its price did not render on the page read.
- **Azure Database for PostgreSQL Flexible Server:** 750 hours a month of Burstable B1ms free for 12 months (listed
  for Azure for Students, "new Azure customers only"). That covers the optional database.

Verdict for planning: the credit can host the app (Container Apps with scale to zero, plus the free B1ms
database), with the LLM on the Cloudflare and Groq free tiers. Measure the container's real memory and cold start
before choosing the size.
