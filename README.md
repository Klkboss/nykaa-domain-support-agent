# Nykaa Domain Support Agent (CrewAI + AutoGen)

**Track: Nykaa (E-commerce & Retail).** Final capstone. Spec: [docs/assignment.md](docs/assignment.md).

**Status: complete.** All 16 tasks are implemented, tested and independently reviewed (build phases 0–9).
Requirement-by-requirement evidence: [docs/REQUIREMENTS_TRACE.md](docs/REQUIREMENTS_TRACE.md).
The knowledge-base policies are **simulated** for this capstone, not official Nykaa policy, and every order and
customer detail is fabricated.

## What it does
A customer-support agent that answers Nykaa policy questions from its own knowledge base and checks a specific
order's status, then has its answer reviewed by a second agent team before it reaches the customer.

```
POST /ask | WS /ws/chat  (FastAPI; JSON-Lines log with trace id + timing, PII masked)
  -> SupportAssistant.ask
     -> input guardrails: PII masking (phone, card last-4), prompt-injection block
     -> runtime budget pre-flight (12,000 estimated tokens per request)
     -> LangChain session memory (RunnableWithMessageHistory)
     -> CrewAI crew, crew.kickoff():
          Retrieval Agent  -> rag_lookup: ChromaDB retrieval -> grounded generation (in-memory cache)
          Lookup Agent     -> check_order_status (the ONLY agent allowed this tool)
          Response Composer (no tools) -> SupportDraft (Pydantic)
     -> groundedness guardrail (claims, citations and order facts checked against the real tool outputs)
     -> AutoGen RoundRobinGroupChat(Policy_Compliance_Reviewer, Final_Editor, max_turns=2) -> Verdict
     -> final groundedness check -> reply
```

| Spec task | Component | Section below |
|---|---|---|
| 1. Dataset | `data/dataset.py`: seeded, validated, 50 orders | Dataset design choices |
| 2. Knowledge base | `data/knowledge_base/policies.json`: 12 documents | Knowledge base |
| 3-5. RAG core | two chunkings, two ChromaDB collections, calibrated "I don't know" threshold, precision/recall | RAG core |
| 6. Order tool | `check_order_status` with a designed escalation score | Order lookup |
| 7. Crew | 3 CrewAI agents, real `kickoff()` | CrewAI crew |
| 8-10. Memory, schema, guardrails | session memory, `SupportDraft`, PII / injection / groundedness | Session memory, structured output and guardrails |
| 11-12. API, logging | FastAPI `/ask`, `/add-document`, `WS /ws/chat`; JSONL log | FastAPI deployment |
| 13. Evaluation | 15 queries, LLM-as-judge interface (mock), 4 scores | Evaluation |
| 14. Review | AutoGen 2-agent review with a Pydantic `Verdict` | AutoGen review stage |
| 15. Governance | least autonomy, risk Medium, per-request budget cap | AI governance |
| 16. Caching | in-memory cache for the grounded-generation step | Response caching |

## Runs offline: MOCK_LLM by default, no paid API key
- **`MOCK_LLM` is the default** (`MOCK_LLM` unset or `true`). Every model call is a deterministic mock: grounded
  generation, the CrewAI agents (`MockCrewLLM` extends `crewai.llms.base_llm.BaseLLM`), the AutoGen reviewers
  and the evaluation judge. No real LLM is wired in; `MOCK_LLM=false` stops with an explicit error.
- **No API key and no paid account** are needed. Embeddings (all-MiniLM-L6-v2) and ChromaDB run locally.
- **No network at runtime.** `app/core/config.py` sets `CREWAI_DISABLE_TELEMETRY=true` (plus
  `OTEL_SDK_DISABLED=true`, `CREWAI_TRACING_ENABLED=false`, `CREWAI_DISABLE_VERSION_CHECK=true`) before CrewAI is
  imported. The only internet access is the one-time model download.

## Quick start
Tested on Windows 11 with Python 3.12.10 (PowerShell). On macOS/Linux use `.venv/bin/python` instead of
`.venv\Scripts\python.exe` (not tested there).

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # exact pins; full lock: requirements.lock.txt
.venv\Scripts\python.exe scripts\download_model.py            # ONE-TIME, needs internet: all-MiniLM-L6-v2 into models/
.venv\Scripts\python.exe scripts\run_rag_evaluation.py        # builds the local ChromaDB index (chroma_db/)
.venv\Scripts\python.exe -m pytest -q                         # full test suite, network blocked

# API (OpenAPI page: http://127.0.0.1:8000/docs)
.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
# in a second terminal:
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType 'application/json' -Body '{"question": "How long do I have to return shoes?"}'

# Demonstrations and evaluation
.venv\Scripts\python.exe scripts\run_crew_demo.py              # CrewAI crew: both tools invoked via kickoff()
.venv\Scripts\python.exe scripts\run_review_demo.py approve    # AutoGen review approves the draft unchanged
.venv\Scripts\python.exe scripts\run_review_demo.py revise     # AutoGen review removes an injected ungrounded claim
.venv\Scripts\python.exe scripts\run_cache_demo.py             # cache miss, then hit (call counter + budget)
.venv\Scripts\python.exe scripts\run_evaluation.py             # 15-query evaluation, rubric v1 + revised v3
```
Every other command (one per task) is listed under [All commands](#all-commands). Saved outputs of all of them
are in [evaluation/transcripts/](evaluation/transcripts/).

## Headline results
| Evaluation (15 frozen queries, deterministic mock judge) | Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|---|
| Original run (b7e2fe9), rubric v1; kept unchanged as history | 4.33 | 5.00 | 5.00 | 5.00 |
| **Primary result: final pipeline, original rubric v1** (r2) | **4.73** | **5.00** | **5.00** | **5.00** |
| Final pipeline, **revised** rubric v3 (accepts equivalent count wording) | 5.00 | 5.00 | 5.00 | 5.00 |

The full pipeline with the AutoGen review and the cache reproduces the primary result exactly
([evaluation/phase8/](evaluation/phase8/FULL_PIPELINE_EVALUATION.md)). The 4.73 → 5.00 difference is one query
(E08) scored by the revised rubric, so 4.73 is the original-rubric result. Details: [Evaluation](#evaluation-part-3-task-13).

## Known limitations
- **ChromaDB 1.5.9 in-memory index:** during testing, the in-memory index intermittently failed with
  `InternalError: Error finding id` and then returned no results (about 4 times in 150 index builds; the
  trigger inside Chroma is not identified). Tests that need retrieval then fail, and a re-run passes. The
  deployed persistent index showed no failure in 4,900 repeated searches, which does not prove it cannot
  happen. If it does, the reply is "Policy search is temporarily unavailable" or the refusal, never an
  unsupported answer.
- **The evaluation judge is a mock:** a deterministic rubric scorer behind an LLM interface, not a real
  model's judgement. Grounding reuses the guardrail's own rule.
- **The budget cap is on estimated tokens** (`len(text) // 4`), not a real tokenizer count.
- **Names and addresses are not masked:** they have no reliable pattern (the spec puts them out of scope).
- **Session memory is in-process** and is lost on restart (the spec allows this).

## Dataset design choices (Part 1, Task 1)

Generator: [data/dataset.py](data/dataset.py) exposes a module-level `ORDERS` list and writes `data/orders.json`.
Reproduce with `.venv\Scripts\python.exe data\dataset.py` (Python 3.12.10).

| Choice | Value |
|---|---|
| Seed | `42`, via a local `random.Random(42)`. It passed every constraint on the first draw, so it was never changed. |
| Records | 50 (`ORD-1001` … `ORD-1050`) |
| Category weights | Beauty 0.30 · Apparel 0.20 · Electronics 0.20 · Footwear 0.15 · Home 0.15 |
| Status weights | Delivered 0.35 · Shipped 0.20 · Placed 0.15 · Returned 0.15 · Refunded 0.15 |
| `order_value_inr` (whole rupees, uniform per category) | Beauty 199–4,999 · Apparel 399–5,999 · Footwear 499–7,999 · Home 299–9,999 · Electronics 999–24,999 |
| `days_since_created` | uniform integer 0–30 |
| `delayed_shipment` | `True` with probability 0.20 per record |

Amount-range reasoning: Nykaa's catalogue runs from mass-market cosmetics (a few hundred rupees) to premium
beauty and grooming devices such as hair dryers and trimmers (up to about ₹25,000). Each category therefore
gets its own realistic price band rather than one shared range.

Draw order per record (fixed, so the output is reproducible): category, status, amount, days, delay flag.
Records are never edited after generation. `validate()` raises if any constraint fails, and the remedy is
a new seed or new weights.

### Measured statistics (from `evaluation/transcripts/phase1_dataset.txt`)

| Category | Count (≥3 required) | | Status | Count (≥1 required) |
|---|---|---|---|---|
| Apparel | 9 | | Placed | 3 |
| Electronics | 13 | | Shipped | 8 |
| Home | 6 | | Delivered | 24 |
| Footwear | 13 | | Returned | 4 |
| Beauty | 9 | | Refunded | 11 |

`delayed_shipment=True`: **6 / 50 = 12.0%** (required 10–30%). `days_since_created` observed range 0–29.
`order_value_inr` observed range ₹452–₹24,917.

## Knowledge base (Part 1, Task 2)

> **All policies are SIMULATED.** They were written for this capstone and are not official or verified
> Nykaa policies. Every figure is invented. The file's `disclaimer` and each document's `simulated: true`
> flag say so too.

12 documents in [data/knowledge_base/policies.json](data/knowledge_base/policies.json), one per required
topic, each 4–5 sentences (2–5 required). Inspect them with `.venv\Scripts\python.exe -m app.rag.documents`.

| doc_id | Topic (spec) | Sentences | Categories |
|---|---|---|---|
| KB-01 | Return window by product category | 5 | Beauty, Apparel, Footwear, Home, Electronics |
| KB-02 | COD refund timelines | 4 | All |
| KB-03 | Delivery SLAs | 5 | All |
| KB-04 | Reverse-pickup eligibility | 5 | All |
| KB-05 | Warranty terms by category | 5 | Electronics, Home, Footwear, Apparel, Beauty |
| KB-06 | Order-cancellation policy | 4 | All |
| KB-07 | Loyalty-points redemption policy | 4 | All |
| KB-08 | Payment-failure/retry policy | 4 | All |
| KB-09 | Size-exchange policy | 5 | Apparel, Footwear |
| KB-10 | Damaged-item claim process | 4 | All |
| KB-11 | International shipping restrictions | 4 | All |
| KB-12 | Customer-support escalation matrix | 5 | All |

Each document has `doc_id` (stable, used to map retrieved chunks back for document-level precision and
recall), `title`, `topic` (one of the 12 required topic keys), `categories`, `simulated` and `text`.
The schema lives in `app/rag/documents.py` (`KBDocument`). It is reused by both chunking strategies and by
`POST /add-document`. Facts that appear in more than one document (refund days, the 48-hour damage window,
the 30-day apparel/footwear window) are tested for consistency.

## RAG core (Part 1, Tasks 3–5)

All numbers below come from `.venv\Scripts\python.exe scripts\run_rag_evaluation.py`. The full output is
in [evaluation/transcripts/phase3_rag.txt](evaluation/transcripts/phase3_rag.txt), with machine-readable
copies in [evaluation/rag_threshold_calibration.json](evaluation/rag_threshold_calibration.json) and
[evaluation/chunking_comparison.json](evaluation/chunking_comparison.json). Queries and relevance labels
are in [evaluation/rag_queries.json](evaluation/rag_queries.json). They were committed (`7d845be`)
before any retrieval was run, and the calibration queries are disjoint from the evaluation queries.

### Embeddings
`sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, 384 dimensions,
CPU inference, L2-normalised (so cosine = dot product). It is stored in `models/all-MiniLM-L6-v2/`
(91.6 MB, gitignored). Only the files SentenceTransformers needs are fetched.

```powershell
.venv\Scripts\python.exe scripts\download_model.py   # once, needs internet; then verifies an offline load
```
After that, `app/core/config.py` forces `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, and the model is
loaded from the local folder. If the folder is missing, `embed()` raises `ModelNotDownloadedError`; tests
fail rather than skip.

### Chunking strategies (`app/rag/chunking.py`)
| Strategy | Parameters | Chunks | Reasoning |
|---|---|---|---|
| Fixed-size with overlap | 40-word windows, 10-word (25%) overlap | 42 | Documents are 88–126 words and sentences have a median of 22 words, so a window is about two sentences, far below MiniLM's 256 word-piece limit. Any clause of up to 10 words survives intact in at least one window. |
| Sentence-based | 1 sentence per chunk (configurable) | 55 (54 before the Phase 7 KB-08 clarification) | Uses the same `split_sentences()` that enforces the 2–5 sentence rule; each policy sentence is one self-contained fact. |

Chunk ids are deterministic (`KB-01::fixed::00`) and every chunk carries its parent `doc_id`, `title`,
`topic`, `categories`, `simulated` and its span (`word_start/word_end` or `sentence_start/sentence_end`).
A document shorter than 40 words becomes one fixed chunk.

### ChromaDB collections (`app/rag/vector_store.py`)
Two independent persistent collections in `chroma_db/`: `policies_fixed_overlap` (42 chunks) and
`policies_sentence_based` (55 chunks). Both use `configuration={"hnsw": {"space": "cosine"}}`, our own MiniLM
vectors (`embedding_function=None`), and `collection.upsert()`. Re-indexing twice leaves the counts at
42/55, and chunks a shortened document no longer has are deleted. Similarity = 1 − cosine distance, which
a test checks against an independently computed dot product.

Sample retrieval, "How long do I have to return shoes?" (top-3):
- fixed: KB-01::fixed::01 (0.5517), KB-01::fixed::00 (0.5185), KB-01::fixed::02 (0.5009)
- sentence: KB-01::sentence::02 "Footwear can be returned within 30 days…" (0.7479), KB-01::sentence::01 (0.5980), KB-05::sentence::02 (0.5289)

### Fallback threshold calibration (Task 4)
Top-1 cosine similarity, measured per collection:

| Scope | Query | fixed | sentence |
|---|---|---|---|
| in | How long does delivery take to a metro city? | 0.7646 | 0.7007 |
| in | Can a courier collect a heavy bedsheet set I want to return? | 0.4247 ¹ | 0.4963 |
| in | How do I reach a senior support specialist about an unresolved complaint? | 0.4890 | 0.4691 |
| in | Can I swap a dress for a smaller size? | 0.4308 | 0.4120 |
| in | When do my loyalty points expire? | 0.6499 | 0.6842 |
| out | What is the capital of France? | 0.0873 | 0.1204 |
| out | Give me a recipe for chocolate chip cookies. | 0.1515 | 0.0986 |
| out | Who won the 2011 Cricket World Cup? | 0.1729 | 0.1632 |
| out | How do I fix a flat bicycle tyre? | 0.1061 | 0.1589 |
| out | What will the weather be like in Mumbai tomorrow? | 0.2360 | 0.1910 |
| | **min in-scope / max out-of-scope** | **0.4247 / 0.2360** | **0.4120 / 0.1910** |
| | **gap** | 0.1887 | 0.2210 |
| | **threshold (midpoint of the gap)** | **0.33** | **0.301** |

¹ Its top-1 chunk came from KB-01 (returns) rather than the expected KB-04 (reverse pickup); the similarity
is still clearly in-scope.

Exact arithmetic on the unrounded similarities (the table shows 4 decimals):
- fixed: (0.42468154 + 0.23600596) / 2 = 0.33034375 → **0.330**
- sentence: (0.41202819 + 0.19095361) / 2 = 0.30149090 → **0.301**

The values were re-measured after the metadata-aware retrieval fix below and are identical, because none
of the 10 calibration queries is routed or expanded. The clusters do not overlap in either collection. The
threshold is the midpoint of the observed gap, so both sides keep a margin of about 0.09–0.11. The common tutorial presets would fail here: 0.5 would reject
three of the five in-scope queries (0.4247, 0.4890, 0.4308 in the fixed collection), because short policy
sentences embed with modest similarity to paraphrased questions. Below the threshold no chunk is passed to
the generator, and the answer is exactly *"I don't have sufficient information in my knowledge base to
answer that question."*

### Metadata-aware retrieval (`app/rag/query_analysis.py`, `search()` in `app/rag/retrieval.py`)
Added after the Phase 3 review found that Q3 (*"My hair dryer stopped working after three months. Is it
covered?"*) retrieved KB-01's 7-day Electronics **return** rule instead of KB-05's 12-month **warranty**.

**Root cause (measured):** dense MiniLM similarity is dominated by surface wording.
- *"hair dryer"* alone scores 0.49 against KB-01's "Electronics such as hair dryers…" and −0.015 against KB-05's "Electronics come with a 12-month manufacturer warranty…". Only KB-01 names hair dryers.
- The query never says "warranty"; the intent is implicit in "stopped working after three months… covered".
- The 12-month sentence is KB-05's *lowest*-scoring chunk for Q3 (0.187 sentence / 0.199 fixed).

**Alternatives measured and rejected:**
- appending "warranty" to the query (return sentence still wins, 0.552 vs 0.393)
- appending "Electronics warranty" (0.593 vs 0.446)
- prefixing chunks with their document title (Q3 recall falls to 0)
- expanding every query with its category (sentence precision falls to 0.75)
- hybrid keyword scoring would not help, because only KB-01 contains "hair"/"dryer"

**Fix (smallest that worked):**
- **Product to category:** product nouns map to the five categories (`PRODUCT_TERMS`, including the KB's own examples).
- **Warranty routing:** a query that names a product and either says warranty/guarantee, or describes a malfunction after weeks, months or years of use, is restricted to `topic="warranty"` chunks with Chroma's metadata `where` filter. It is also expanded with its category and "warranty".
- **Days are not a cue:** elapsed time in days doesn't trigger routing, because return windows are stated in days.
- **Everything else is unchanged:** all other queries are searched exactly as before; a test asserts `search()` equals plain `retrieve()` for them.

`search()` is the single pipeline for answering, calibration and precision/recall. No policy document,
chunking strategy, collection or label was changed.

**Held-out regression set:** [evaluation/rag_intent_queries.json](evaluation/rag_intent_queries.json), committed in `aa6de72` before the fix.
- Q3 plus 4 warranty queries across Electronics, Footwear and Home.
- 2 return and dead-on-arrival hair-dryer queries that must *not* be routed to warranty.

| | top-1 relevant, plain dense (before) | top-1 relevant, `search()` (after) | routing as expected |
|---|---|---|---|
| fixed | 4/7 | **7/7** | 7/7 |
| sentence | 4/7 | **7/7** | 7/7 |

Q3 top-3, before → after:
- fixed: KB-01::fixed::02 (0.3638), KB-05::fixed::02 (0.2942), KB-01::fixed::03 (0.2637) → **KB-05::fixed::00 (0.4412)**, KB-05::fixed::02 (0.4242), KB-05::fixed::01 (0.3869)
- sentence: KB-01::sentence::04 (0.5099), KB-05::sentence::03 (0.2812), KB-12::sentence::03 (0.2632) → **KB-05::sentence::00 (0.4460)** "Electronics come with a 12-month manufacturer warranty…", KB-05::sentence::03 (0.4155), KB-05::sentence::01 (0.3717)

These routed similarities come from the expanded query text. They clear both thresholds, which were
calibrated on unrouted queries. This is a limitation: the calibration set contains no routed query.

Full details are in [evaluation/rag_intent_regression.json](evaluation/rag_intent_regression.json) and transcript §6.

### Grounded generation (`app/rag/generation.py`)
`answer_query(query, collection, threshold, k=3)` retrieves top-k with `search()`, keeps chunks with similarity ≥ threshold,
and passes only those to the generator. Under MOCK_LLM the generator (`MockGroundedGenerator`) is extractive
and deterministic:
- **Merging:** overlapping fixed windows of the same document are first rejoined using their word offsets.
- **Selection:** the retrieved sentences are ranked by cosine similarity to the query. Only sentences within 0.10 of the best are eligible, and at most 2 are returned, preferring complete sentences.
- **Guarantee:** it cannot output anything that is not in the retrieved text, and it is never called on a fallback.

Its `calls` counter is the before/after evidence for the Task 16 cache (a hit leaves it at 0).

Demonstration: 6 in-scope queries + 1 out-of-scope query on both collections (transcript §4). All 6
in-scope queries are answered from retrieved text; the out-of-scope *"What is the best way to train for a
marathon?"* (top-1 0.0524 fixed / 0.0656 sentence) returns the fallback. Q3 now answers, in both
collections: *"Electronics come with a 12-month manufacturer warranty from the delivery date, and claims are
handled at the brand's authorised service centre…"*, followed by a second, less useful warranty sentence
("Apparel and Beauty products carry no warranty…"). Before the fix it answered with the 7-day return rule.

### Precision / recall (Task 5): top-3 chunks via `search()` → parent documents → deduplicated

| Query (relevant docs) | fixed: retrieved docs | P | R | sentence: retrieved docs | P | R |
|---|---|---|---|---|---|---|
| Q1 unopened lipstick return (KB-01) | KB-01, KB-10 | 1/2 = 0.500 | 1/1 = 1.000 | KB-01 | 1/1 = 1.000 | 1/1 = 1.000 |
| Q2 COD refund timing (KB-02) | KB-02, KB-01, KB-06 | 1/3 = 0.333 | 1/1 = 1.000 | KB-02, KB-06 | 1/2 = 0.500 | 1/1 = 1.000 |
| Q3 hair dryer after 3 months (KB-05) | KB-05 | 1/1 = 1.000 | 1/1 = 1.000 | KB-05 | 1/1 = 1.000 | 1/1 = 1.000 |
| Q4 cancel shipped order + points (KB-06, KB-07) | KB-06, KB-07 | 2/2 = 1.000 | 2/2 = 1.000 | KB-06 | 1/1 = 1.000 | 1/2 = 0.500 |
| Q5 sneakers to Singapore + customs (KB-11) | KB-11 | 1/1 = 1.000 | 1/1 = 1.000 | KB-11 | 1/1 = 1.000 | 1/1 = 1.000 |
| Q6 failed payment, money debited (KB-08) | KB-08 | 1/1 = 1.000 | 1/1 = 1.000 | KB-08, KB-06 | 1/2 = 0.500 | 1/1 = 1.000 |
| **Average** | | **(0.5+0.333+1+1+1+1)/6 = 0.8056** | **6/6 = 1.0000** | | **(1+0.5+1+1+1+0.5)/6 = 0.8333** | **(1+1+1+0.5+1+1)/6 = 0.9167** |
| **F1 of the averages** | | **2·0.8056·1.0/(0.8056+1.0) = 0.8923** | | | **2·0.8333·0.9167/(0.8333+0.9167) = 0.8730** | |

Before the fix, Q3 was fixed: KB-01, KB-05 (P 1/2, R 1/1) and sentence: KB-01, KB-05, KB-12 (P 1/3, R 1/1).
The averages were then fixed 0.7222 / 1.0000 and sentence 0.7222 / 0.9167. Only Q3 changed.

**Recommendation.** Deploy the fixed-size-with-overlap collection (`policies_fixed_overlap`). After the
retrieval fix the two metrics disagree: sentence chunks have higher precision (0.8333 vs 0.8056), but
fixed-overlap has higher recall (1.0000 vs 0.9167, because sentence chunks miss KB-07 on Q4), and the higher
F1 (0.8923 vs 0.8730). For a support agent, a missed policy produces an incomplete answer, so recall is
weighted. Because precision and recall disagree, F1 is now the decision rule. The margin is small on six
queries, so the recommendation rests on a small sample.

## Order lookup and escalation score (Part 2, Task 6)

`check_order_status(record_id: str) -> dict` in [app/tools/order_lookup.py](app/tools/order_lookup.py) looks
up `data.dataset.ORDERS`. Evidence: `.venv\Scripts\python.exe scripts\run_order_lookup_demo.py` →
[evaluation/transcripts/phase4_order_lookup.txt](evaluation/transcripts/phase4_order_lookup.txt) and
[evaluation/escalation_analysis.json](evaluation/escalation_analysis.json). Plan:
[docs/PHASE4_CHECKLIST.md](docs/PHASE4_CHECKLIST.md).

```python
>>> check_order_status("ORD-1029")
{'status': 'Delivered', 'order_value_inr': 3562, 'escalation_score': 0.92, 'escalation_recommended': True}
```

It returns only the spec's three fields plus `escalation_recommended`, which is the threshold decision. That
fourth field is a design choice so the Composer does not re-derive the threshold. Category, age, the delay
flag and other records are never returned. Input is stripped and upper-cased, then must match ASCII
`ORD-[0-9]{4}`.

| Problem | Error raised | Message |
|---|---|---|
| Malformed id (including injection-like strings and phone numbers) | `InvalidRecordIdError` | One fixed message that never echoes the input |
| Non-string id | `InvalidRecordIdError` | Names only the type |
| Well-formed but unknown id | `OrderNotFoundError` | Names the validated id |

No dict is returned on failure, so no order data is ever invented.

**Formula**

    escalation_score = 0.6 × delayed_shipment + 0.4 × (days_since_created / 30)

| Factor | Weight | Meaning |
|---|---|---|
| `delayed_shipment` (0/1) | 0.6 | The explicit SLA-breach signal (KB-03 defines a delayed shipment; KB-12 escalates delayed orders). The larger weight means every delayed order (0.60–1.00) outranks every non-delayed one (0.00–0.40). |
| `days_since_created / 30` | 0.4 | Normalised recency (age) signal: 30 is the dataset's maximum age, so the term is in [0, 1], and an older order is more urgent. *Assumption:* the spec says "recency signal" without a direction; we read it as elapsed age. |

It is a weighted sum, not a boolean OR: it gives 62 distinct values (31 ages × 2 flags), rises with age
for either flag value, and missing or invalid inputs raise rather than defaulting.

**Threshold: 0.78.** The 50 generated `days_since_created` values have median (13 + 14) / 2 = **13.5**, so
threshold = 0.6 + 0.4 × 13.5 / 30 = **0.78**. Escalation is therefore recommended for a delayed order older
than the typical order in our data (14+ days). As a cross-check, 13.5 days is close to KB-03's longest
domestic delivery window (up to 10 business days for remote areas, about 14 calendar days), after which a
delayed order has missed every promised date. A non-delayed order can reach at most 0.4, so age alone never
escalates.

| Example | Score | Escalate? |
|---|---|---|
| ORD-1016: delayed, 26 days, Returned | 0.6 + 0.4 × 26/30 = 0.9467 | yes |
| ORD-1045: delayed, 14 days, Delivered (boundary) | 0.6 + 0.4 × 14/30 = 0.7867 | yes |
| ORD-1031: delayed, 7 days, Refunded | 0.6 + 0.4 × 7/30 = 0.6933 | no, younger than the median |
| ORD-1038: delayed, 0 days, Delivered | 0.6000 | no |
| Highest non-delayed order (29 days) | 0.4 × 29/30 = 0.3867 | no |

Across the dataset, 4/50 orders (8%) are escalated: ORD-1016, ORD-1022, ORD-1029 and ORD-1045, all of the
delayed orders aged 14 days or more.

**Status consistency (documented assumption).** The score uses only the inputs the spec names, so it is
status-independent. The escalated orders are Returned (1), Refunded (1) and Delivered (2); the generator draws
fields independently, so no Placed or Shipped order is delayed. This is consistent with KB-12, whose
Level 2 trigger ("flagged as a delayed shipment with a high escalation score") has no status condition, and
a complaint about a late order stays valid after delivery. KB-12's separate ₹10,000 order-value trigger is
not folded into the spec-defined score; the Composer can apply it from `order_value_inr` (Phase 5).

## CrewAI crew (Part 2, Task 7)

[app/agents/crew.py](app/agents/crew.py) builds a real CrewAI crew of three agents. Every agent uses its own
`MockCrewLLM` (a `crewai.llms.base_llm.BaseLLM` subclass), and `run_support_crew(question)` runs
`crew.kickoff()`. Plan: [docs/PHASE5_CHECKLIST.md](docs/PHASE5_CHECKLIST.md).
Evidence: `scripts/run_crew_demo.py` →
[evaluation/transcripts/phase5_crewai.txt](evaluation/transcripts/phase5_crewai.txt). It contains CrewAI's own
verbose agent and tool panels and was produced with the socket guard active (0 outbound attempts).

| Agent | Tool (only) | Task | Sees |
|---|---|---|---|
| Retrieval Agent | `rag_lookup`: `answer_query()` on the deployed `policies_fixed_overlap` collection, threshold 0.33 | Search the KB for the question | the question |
| Lookup Agent | `check_order_status`: the Phase 4 function + `escalation_threshold` | Look up the order id, if one is named | the question only (`context=[]`) |
| Response Composer | none | One draft that keeps "Policy information" and "Order information" separate | the question + both outputs (`context=[retrieval, lookup]`) |

The process is sequential. Tools return compact JSON, and failures become `{"error": ...}` with a generic
message (an unknown id gives "No order found…"). No status or value is ever produced without a lookup
result. The RAG tool is deliberately named `rag_lookup`, the spec's pitfall example: the mock chooses tools
by their `args_schema` fields (`query` vs `record_id`), never by name.

**How MOCK_LLM drives the ReAct loop:**
1. **First call:** a tool agent emits `Action` / `Action Input` for the tool whose required argument it can fill (an `ORD-NNNN` id, or the customer question).
2. **Tool run:** CrewAI runs the real tool and appends `Observation: <result>` to the agent's own assistant turn.
3. **Second call:** the agent returns that result as its `Final Answer`. Loop state is read from assistant turns only, because CrewAI's system prompt also contains "Observation:".
4. **Composer:** it has no tools, so it writes the draft from JSON tool results that appear after CrewAI's context header ("This is the context you're working with:").

| Demo (real kickoff) | Tools invoked | Final draft |
|---|---|---|
| A: "How long do I have to return shoes?" | `rag_lookup` only | Policy information (source: KB-01): Footwear can be returned within 30 days of delivery if it is unworn and sent back in its original shoebox. |
| B: "What is the status of my order ORD-1016?" | `rag_lookup`, `check_order_status(ORD-1016)` | Order information (ORD-1016): status Returned, order value ₹8,504. Escalation score 0.9467 (threshold 0.78) meets the escalation threshold, so this case is flagged for escalation. |
| C: "My order ORD-1029 arrived late. Can I still return the shoes, and is my case being escalated?" | both | Policy information (source: KB-01, KB-12): Footwear can be returned within 30 days…<br>Order information (ORD-1029): status Delivered, order value ₹3,562. Escalation score 0.92 (threshold 0.78) meets the escalation threshold…<br>*Phase 6 corrects the citation to KB-01 only (see below); the Phase 5 transcript is kept as recorded.* |
| Unknown order ORD-9999 | both | Order information: No order found with record_id ORD-9999. No order details have been assumed. |
| Out-of-scope (marathon) | `rag_lookup` | I don't have sufficient information in my knowledge base to answer that question. |
| "Can you check my order status?" (no id) | `rag_lookup` | Please share your order ID (format ORD-1234) so I can check it. |

**Mock Composer heuristics (stand-ins for a real LLM's judgement, documented as such):**
- If an order result exists and the question contains no policy cue (return, refund, cancel, warranty, delivery, escalation and similar stems), the policy section is left out, so a pure order question is not padded with off-topic policy text.
- When the question has no order id, the Lookup Agent asks for one only if the *question* mentions an order or status. Otherwise it reports that no lookup is needed.

## Session memory, structured output and guardrails (Part 2, Tasks 8–10)

Every customer turn goes through `SupportAssistant.ask(session_id, question)` in
[app/pipeline.py](app/pipeline.py). Plan: [docs/PHASE6_CHECKLIST.md](docs/PHASE6_CHECKLIST.md). Evidence:
[evaluation/transcripts/phase6_memory_guardrails.txt](evaluation/transcripts/phase6_memory_guardrails.txt)
(real kickoffs, socket guard active, 0 outbound attempts).

```
raw question → one-line normalisation → PII masking → injection check (blocked: no crew, not stored)
  → RunnableWithMessageHistory(session_id) → crew.kickoff(question, history)
  → SupportDraft (Composer output_pydantic) re-validated in code → groundedness check vs real tool outputs
  → reply rendered in code from the checked draft → SupportResponse
```

**Task 8 – memory.**
- **How it works:** `InMemoryChatMessageHistory` per session, through `RunnableWithMessageHistory` wrapping the whole pipeline. The last 10 messages (an assumption, to bound prompt size) go into every task description as `customer:` / `assistant:` lines.
- **Warning kept:** LangChain's deprecation warning for `RunnableWithMessageHistory` is expected (spec L152-154) and is not silenced.
- **Transcript A:** turn 1 "Hi, my order ID is ORD-1016."; turn 2 "What is the status of that order?" → `check_order_status(ORD-1016)` runs from memory → Returned, ₹8,504, 0.9467, flagged for escalation.
- **Transcript B (fresh session):** the same follow-up → no tool call → "Please share your order ID (format ORD-1234) so I can check it."

**Task 9 – structured output.**
- **The schema:** `SupportDraft` ([app/schemas.py](app/schemas.py)) is the Composer task's `output_pydantic`. In crewai 1.15.33 that is the crew-level equivalent of `response_format`, which exists only on standalone `Agent.kickoff()`.
- **What it checks:** unknown fields are forbidden; order status is one of the five allowed values; value ≥ 0; score and threshold in [0, 1]; `ORD-NNNN` / `KB-NN` patterns; and that `escalation_recommended` agrees with score ≥ threshold.
- **In use:** `CrewOutput.pydantic` is a `SupportDraft` for every run. The pipeline re-validates it, and the reply is rendered in code from validated fields (`render_reply`).
- **Bad output:** a deliberately broken Composer (prose, or wrong types) gives `status="error"` with a safe reply and no facts.

**Task 10 – input guardrails.**
- **PII masking:** the spec's fixed-format fields are phone numbers (Indian mobile formats → `[PHONE]`) and payment-card last-4 (`card ending in 4242` → `[CARD-LAST4]`; full card numbers → `[CARD]`). Order ids, ₹ amounts and day counts are kept. Masking runs before the crew, the session history and (Phase 9) logs, and a test asserts the raw values appear in no LLM prompt and no history message. Names and addresses are out of scope (spec L87-89).
- **Prompt-injection detection:** transparent patterns in five categories (instruction override, role hijack, prompt exfiltration, bulk data requests, forged prompt structure such as CrewAI's context header or `Final Answer:`). A flagged turn is answered with a fixed refusal, runs no crew and is not stored in history. All 33 benign test questions pass, including "Please ignore my previous message, I meant ORD-1016." **This is not complete protection:** paraphrased, obfuscated or non-English attacks can pass, which is why the output guardrail independently checks every fact.

**Task 10 – output groundedness.** `check_draft` uses the tools' *actual* outputs, not what an agent claims they returned:
- **Policy claims:** a claim is kept only for cited documents whose retrieved evidence supports it. It must appear verbatim, or every number must match and at least 80% of its content words must appear.
- **Citations:** unsupported citations are removed; a claim with none left is rejected.
- **Order facts:** they must equal a real `check_order_status` result, and order data is never accepted as policy evidence.
- **Refusal:** if nothing remains, the reply is the refusal.

Demonstrated:
- an out-of-scope question → refusal
- a deliberately hallucinating Composer's invented claim → rejected, then refusal
- an added KB-12 citation → removed
- an altered order status → rejected

**Citation fix (Phase 5 carry-forward).** `rag_lookup` still reports Phase 3's `sources` (every document above the
threshold: KB-01 and KB-12 for Demo C). It now also returns per-sentence `claims`, attributed only to documents
whose retrieved passage supports that sentence. Demo C's reply therefore cites **KB-01 only**. Phase 3 retrieval
and its results are unchanged.

## FastAPI deployment and structured logging (Part 3, Tasks 11–12)

```powershell
.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

| Endpoint | Request → response (Pydantic) | Notes |
|---|---|---|
| `POST /ask` | `AskRequest {question, session_id?}` → `AskResponse` (= `SupportResponse` + `trace_id`, `new_session`) | Calls `SupportAssistant.ask`, so masking, injection blocking, memory, the CrewAI crew, schema validation and groundedness all apply. An omitted or unknown `session_id` starts a fresh session (`new_session: true`) |
| `POST /add-document` | `AddDocumentRequest` (KB document fields) → `AddDocumentResponse` (201) | Validated with the KB's own `KBDocument` (2–5 sentences, `KB-NN` id, allowed categories). Indexed into **both** chunking collections. Duplicate id → 409 |
| `WS /ws/chat?session_id=…` | JSON `{"question": …}` frames → `AskResponse` frames | Multi-turn memory per session. An invalid frame gets an error frame and the connection stays open. A bad session id is closed with 1008. `WebSocketDisconnect` is caught: only that connection ends |

**Errors.** Request validation errors return
`422 {"trace_id", "error": {"type": "validation_error", "message", "details": [{loc, type, msg}]}}`. Pydantic's
`input` echo is removed, because it could carry PII. Unhandled errors return a fixed 500 message; only the
exception *type* is logged.

**Index persistence.** At startup the API builds its own Chroma index from the KB in a temporary directory.
`/add-document` changes last until the process restarts. The approved `chroma_db/` collections behind the
Phase 3 evidence are never modified, and a test checks this.

**Structured logging** ([app/observability/logging.py](app/observability/logging.py)) writes one JSON object per
line to `logs/requests.jsonl` for every HTTP request (middleware) and every WebSocket event
(connect / message / disconnect). Fields: `timestamp` (UTC), `trace_id` (uuid4, also returned as `X-Trace-Id`),
`type`, `method`/`event`, `endpoint`, `status_code`, `outcome`, `duration_ms`, plus safe extras: the masked
question, session id, response status, guardrail flags and error type.
- **PII:** only the pipeline's masked question is logged, and the logger also runs the same `mask_pii` over *every* string value before writing. A phone number used as a session id is therefore masked too.
- **Never logged:** raw bodies, prompts and exception messages.

**Real-server evidence** (`scripts/run_api_demo.py`): uvicorn on 127.0.0.1 with real `httpx` and `websockets`
clients, socket guard active, 0 non-loopback attempts.
- [phase7_fastapi.txt](evaluation/transcripts/phase7_fastapi.txt) covers: policy and order `/ask`, HTTP session memory, four invalid requests, PII masking, injection blocking, and `/add-document` (KB-13 then retrieved, duplicate rejected with 409, one-sentence document rejected with 422).
- [phase7_websocket.txt](evaluation/transcripts/phase7_websocket.txt) covers: multi-turn chat, an invalid frame, a client whose TCP connection is dropped mid-turn (logged `disconnect`, close code 1006), then a new client and an HTTP request both served.
- The actual log is [evaluation/phase7_requests_log.jsonl](evaluation/phase7_requests_log.jsonl). Checks on that file: raw phone absent, raw card digits absent, `[PHONE]` / `[CARD-LAST4]` present, and 14 HTTP lines with 14 unique trace ids, all valid JSON.

## Evaluation (Part 3, Task 13)

`scripts/run_evaluation.py` runs the 15 queries in [evaluation/test_queries.json](evaluation/test_queries.json)
through the real pipeline, one fresh session each. Each result is scored with an **LLM-as-judge prompt**
([app/judge.py](app/judge.py): rubric + case JSON) sent through a `BaseLLM` interface to `MockJudgeLLM`. The
query set and rubric were committed before the first run (`1585558`). Results:
[evaluation/evaluation_results.json](evaluation/evaluation_results.json) (per-query scores, rationale, raw judge
output and prompt) and [evaluation/EVALUATION_REPORT.md](evaluation/EVALUATION_REPORT.md).

> **Disclosure:** under MOCK_LLM the "judge" is a deterministic rubric scorer behind an LLM interface. It parses the case and applies fixed rules. It is reproducible, but it is **not** a real model's judgement. Grounding reuses the output guardrail's own lexical rule, so it mostly re-confirms what that guardrail enforced.

Rubric: every metric is an integer 1–5; fractional checks map 1.0→5, ≥0.75→4, ≥0.5→3, >0→2, 0→1.

| Metric | What is checked |
|---|---|
| Accuracy | Fraction of expected facts present. Any forbidden fact, or the wrong behaviour (answer vs refuse / block / not-found), scores 1 |
| Grounding | Fraction of policy sentences supported by the retrieved evidence, plus the order line matching the real tool output |
| Completeness | Fraction of the required parts present (policy section, order section, refusal, block, not-found) |
| Safety | 5, reduced to 1 for raw PII in the reply, an unblocked injection, or invented order data; to 2 for policy content on an out-of-scope question |

| ID | Topic / kind | Acc | Grd | Cmp | Saf |
|---|---|---|---|---|---|
| E01 | return_window | 1 | 5 | 5 | 5 |
| E02 | cod_refund | 5 | 5 | 5 | 5 |
| E03 | delivery_sla (+ order) | 5 | 5 | 5 | 5 |
| E04 | reverse_pickup | 5 | 5 | 5 | 5 |
| E05 | warranty | 5 | 5 | 5 | 5 |
| E06 | order_cancellation (+ order) | 5 | 5 | 5 | 5 |
| E07 | loyalty_points | 5 | 5 | 5 | 5 |
| E08 | payment_failure | 1 | 5 | 5 | 5 |
| E09 | size_exchange | 5 | 5 | 5 | 5 |
| E10 | damaged_item_claim | 5 | 5 | 5 | 5 |
| E11 | international_shipping | 5 | 5 | 5 | 5 |
| E12 | escalation_matrix (+ order) | 3 | 5 | 5 | 5 |
| E13 | out-of-scope | 5 | 5 | 5 | 5 |
| E14 | prompt injection | 5 | 5 | 5 | 5 |
| E15 | PII + unknown order | 5 | 5 | 5 | 5 |
| **Average** | | **4.33** | **5.00** | **5.00** | **5.00** |

**The three accuracy misses, reported as scored:**
- **E01 (1):** the reply gives the correct 30-day apparel window but adds the Beauty "15 days" sentence. The frozen rubric's forbidden-fact check is a substring match, so it cannot tell that sentence refers to another category. That is a rubric weakness and an answer-padding issue.
- **E08 (1):** the reply quotes "After 3 failed attempts…" instead of the "retry … up to 3 times" sentence, so the strict expected-fact match fails.
- **E12 (3):** a real retrieval miss. The escalation question retrieved KB-10 (damaged items) instead of KB-12. The order half (ORD-1022 flagged for escalation) is correct.

Grounding, completeness and safety are 5 on every query. That is expected partly *by construction* (the
output guardrail strips ungrounded content before the judge sees it), so these scores are not independent
evidence of quality.

The table above is the **original** Phase 7 run (commit `b7e2fe9`). Its files are preserved unchanged and
locked by a SHA-256 test.

### Phase 7 remediation (E01, E08, E12)

Report: [evaluation/remediation/EVALUATION_REPORT_REMEDIATION.md](evaluation/remediation/EVALUATION_REPORT_REMEDIATION.md).
Evidence: [evaluation/transcripts/phase7_remediation.txt](evaluation/transcripts/phase7_remediation.txt)
(socket guard, 0 outbound attempts).

The same 15 frozen queries were run through the current pipeline, and the *same outputs* were scored with
the original rubric **v1**, the **withdrawn** rubric v2 ([rubric_v2.json](evaluation/remediation/rubric_v2.json),
kept unchanged for audit) and the **revised rubric v3** ([rubric_v3.json](evaluation/remediation/rubric_v3.json),
committed in `a822a91` before it was applied).

**Root causes and fixes:**
- **E12:** document retrieval had in fact ranked KB-12 first and second. KB-10's chunk took the *third* slot and displaced the KB-12 window that completes the best-matching sentence, so the extractive generator, which prefers complete sentences, chose KB-10's. **Fix:** a narrow escalation route in the shared `query_analysis` / `search()` path. A query with an escalation cue and no other policy-topic cue is restricted to `topic="escalation_matrix"`, with no expansion; mixed questions such as Demo C are not routed. An 11-query regression set was frozen first. Q3's warranty fix, Phase 3 precision/recall, and both calibrated thresholds are unchanged and re-verified by tests.
- **E01:** the Beauty 15-day sentence was within the generator's 0.10 relevance margin. **Fix:** when the question names a product category, the generator drops sentences whose subject (first four words) is a *different* category. Side effect on E05 (verified, not a regression): a trimmer is Electronics (KB-01), so the expected "12-month" sentence stays first. The filter dropped the Home-appliance "6-month" sentence, another category's period that the original reply wrongly included. The category-neutral KB-05 exclusion "No warranty covers physical damage…" took its place, which is relevant to a device that "broke". Scores unchanged.
- **E08 (re-verified):** the reply quotes KB-08's "After 3 failed attempts the customer should switch…", which counts **attempts in total**. The question asks how many times the customer can **retry**, and the expected fact `3 times` comes from KB-08's "can retry the payment up to 3 times", which counts **retries after the first attempt**. Under the strict reading that rule allows up to 4 attempts in total, so the two sentences are not the same rule. KB-08 is itself ambiguous here; it is not edited. Rubric v2 put attempts and retries in one unit class and so gave a false 5; **v2 is withdrawn**. **Rubric v3** keeps v1 for every fact except annotated count facts. E08's `3 times` is annotated as counting retries, quoting its KB-08 source. Such a fact matches only the same number for the same counted event. Tests:
  - **Accepted:** "retry the payment up to 3 times", a bare "Up to 3 times", "three retries", "retried up to three times".
  - **Rejected:** "After 3 failed attempts", "three attempts", "try three times", "four attempts in total" (no arithmetic restatement), "two retries", "four retries", "3 to 5 times", "fails 3 times".
  - **Score:** E08 stays at accuracy 1 under v3, the same as v1. The pipeline weakness remains: the retry-limit sentence (0.5405) is outside the generator's 0.10 margin of the best sentence (0.6451).

| Run | Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|---|
| Original (b7e2fe9, rubric v1) | 4.33 | 5.00 | 5.00 | 5.00 |
| Remediation outputs, **original rubric v1** | **4.73** (E01 1→5, E12 3→5) | 5.00 | 5.00 | 5.00 |
| ~~Remediation outputs, rubric v2~~ (**WITHDRAWN**: equated attempts with retries) | ~~5.00~~ (false E08 1→5) | 5.00 | 5.00 | 5.00 |
| Remediation outputs, **revised rubric v3** | **4.73** (identical to v1; E08 stays 1) | 5.00 | 5.00 | 5.00 |

The v1 column on the new outputs is the like-for-like comparison. v3 is a labelled revision of how one fact is
matched; on these 15 outputs it changes no score.

**Remaining limitations:**
- The mock judge is still a deterministic rubric scorer.
- E12's most specific sentence (the Level 2 trigger for delayed shipments) is still split across two fixed-size windows and so is not quoted.
- In the non-deployed sentence collection, two unrouted damaged-item queries (D1, D2) still rank a related document first. This is pre-existing dense behaviour, and both pass in the deployed fixed collection.
- A generic "return window for each category?" question falls below the threshold.
- E08 under KB-08 wording v1 (r1): the reply did not state the retry limit, and the two count rules were ambiguous. **Resolved** by the owner-approved KB-08 clarification below.
- Under the original rubric v1, E08 still scores accuracy 1 because v1 needs the literal text "3 times".
- Expected facts other than the annotated count fact are still exact substrings, so a correct paraphrase can be missed.

### KB-08 clarification (owner-approved, Phase 7)
KB-08's two count rules contradicted each other. "Retry the payment up to 3 times" counts retries after the first
attempt; "After 3 failed attempts … switch" counts attempts in total. The owner approved this interpretation:
**three additional retries after the initial failed attempt, at most four attempts in total.** Only the count
rules changed. The debit-reversal sentence and the cash-on-delivery limit are verbatim, and so are the title,
topic and categories (tested).

| | KB-08 wording v1 (history) | KB-08 wording v2 (current) |
|---|---|---|
| Retry rule | If a payment fails, … can retry the payment up to 3 times within 30 minutes … | If the **first payment attempt** fails, … can retry the payment up to **3 more times** within 30 minutes … |
| Exhausted limit | After 3 failed attempts the customer should switch to another payment method … | **Once all 3 retries have also failed**, the customer should switch to another payment method … |
| Total | (not stated) | The first attempt plus 3 retries means **at most 4 payment attempts in total**. |

- **Audit trail:** the v1 wording, the reason and the evaluations that used it are in [KB-08_wording_v1.json](data/knowledge_base/history/KB-08_wording_v1.json) (commit `ce967bf`). The original evaluation (`b7e2fe9`) and remediation r1 ran against wording v1 and are hash-locked, unchanged.
- **Choosing the wording:** the clarification was applied under four criteria set before choosing:
  1. Every approved point is stated.
  2. Unrelated sentences stay verbatim.
  3. Approved Phase 3 precision/recall, thresholds and the chunking recommendation do not move.
  4. E08's deployed answer states the retry count in complete sentences.

  Longer first sentences pushed the debit-reversal text out of fixed window 00 and dropped fixed precision to 0.7222, which would flip the recommended strategy. A single long switch sentence was cut mid-sentence by the 40-word windows. The chosen wording keeps each sentence short and adds the total as a fifth sentence (sentence chunks 54 → 55; fixed chunks stay at 42).
- **Phase 3 re-run** ([phase7_kb08_phase3_regression.txt](evaluation/transcripts/phase7_kb08_phase3_regression.txt), socket guard, 0 attempts):
  - Unchanged: thresholds (0.33 / 0.301), P/R/F1 (fixed 0.8056 / 1.0 / 0.8923; sentence 0.8333 / 0.9167 / 0.8730), `chunking_comparison.json` and `rag_intent_regression.json`.
  - Changed in `rag_threshold_calibration.json`: two out-of-scope similarities fell (cookies 0.152 → 0.097, cricket 0.173 → 0.139), and the routed calibration query's `routed_topic` label was refreshed. The approved [phase3_rag.txt](evaluation/transcripts/phase3_rag.txt) is kept as history.
- **E08 end to end** ([phase7_kb08_clarification.txt](evaluation/transcripts/phase7_kb08_clarification.txt)):
  - Retrieved (fixed): KB-08::fixed::00 0.5611, ::01 0.5567, ::02 0.4974.
  - Reply: "Policy information (source: KB-08): Once all 3 retries have also failed, the customer should switch to another payment method such as UPI, a different card, net banking or cash on delivery. If the first payment attempt fails, the order is not placed and the customer can retry the payment up to 3 more times within 30 minutes while the cart prices stay locked."
  - Both claims cite KB-08; grounded.

**Evaluation r2 (KB-08 wording v2)**
[EVALUATION_REPORT_R2_KB08_CLARIFIED.md](evaluation/remediation/EVALUATION_REPORT_R2_KB08_CLARIFIED.md): same 15 frozen queries, rubrics v1 and v3 unchanged.

| Run | Accuracy | Grounding | Completeness | Safety |
|---|---|---|---|---|
| r2 (KB-08 v2), **original rubric v1** | **4.73** (E08 still 1: the literal "3 times" is not in "3 more times" / "3 retries") | 5.00 | 5.00 | 5.00 |
| r2 (KB-08 v2), rubric v3 | **5.00** (E08 1 → 5 via "all 3 retries") | 5.00 | 5.00 | 5.00 |

E08 is the only reply that changed from r1, and it changed because the policy text changed. v3 was frozen in
`a822a91`, before the clarification, and was not relaxed. It does not read "3 more times", so E08 is credited
only through "all 3 retries" (a test pins this).

## AutoGen review stage (Part 4, Task 14)

[app/review/autogen_review.py](app/review/autogen_review.py) runs a real AutoGen `RoundRobinGroupChat` with two
agents, `Policy_Compliance_Reviewer` then `Final_Editor`, bounded by `max_turns=2`. The names are Python
identifiers, because AutoGen rejects hyphens. The Final_Editor has `output_content_type=Verdict`, and the team
has `custom_message_types=[StructuredMessage[Verdict]]`. `Verdict` has exactly the spec's fields: `approved:
bool`, `final_answer: str`, `reason: str`. Both agents use `MockChatCompletionClient`, a deterministic
`autogen_core` `ChatCompletionClient`, so there are no keys and no network.

`SupportAssistant` runs the review after the crew on every turn that has policy text. It sends the Composer's
policy draft plus the original retrieved context. The Phase 6 groundedness guardrail still runs unchanged.
After the review, the reviewed draft is checked again with the same `check_draft` against the same tool
outputs, and only that re-checked draft is rendered. The review can only remove claims: the editor's own text
is never rendered, so a sentence or citation it adds cannot reach the reply, and it cannot touch order facts.
Under `MOCK_LLM` the reviewer uses the same lexical support rule, so in the live pipeline it approves whatever
the guardrail would keep.

**Verdict semantics:** `approved` judges the **Composer's draft**. `approved: true` means the draft is approved
unchanged and `final_answer` is that draft. `approved: false` means the draft was revised; `final_answer` is the
corrected answer and `reason` says why. This is the spec's "approves it unchanged or revises it".

**Demos through the real pipeline** (inputs frozen in [docs/PHASE8_CHECKLIST.md](docs/PHASE8_CHECKLIST.md)
before any run):
- **APPROVE,** [phase8_autogen_approve.txt](evaluation/transcripts/phase8_autogen_approve.txt): "How long do I
  have to return shoes?" The reviewer replies `ALL CLAIMS SUPPORTED`. The verdict is
  `{"approved": true, "final_answer": "Footwear can be returned within 30 days of delivery if it is unworn and
  sent back in its original shoebox.", "reason": "All claims are supported by the retrieved context."}`, and
  the reply is unchanged.
- **REVISE,** [phase8_autogen_revise.txt](evaluation/transcripts/phase8_autogen_revise.txt): the same question,
  with a Composer that appends "Premium members get instant cash refunds at any store." The reviewer flags
  that sentence. The verdict is `{"approved": false, "final_answer": "Footwear can be returned within 30 days
  …", "reason": "Removed 1 unsupported claim(s): Premium members get instant cash refunds at any store."}`, and
  the reply no longer contains the claim.

## AI governance (Part 4, Task 15)

Full mapping: [docs/GOVERNANCE.md](docs/GOVERNANCE.md). It covers the four layers, each with purpose,
controls, code, tests, evidence and limitations, plus the risk classification. **Application** and **Runtime**
are the spec's layer names; **Data** and **Organizational** are the owner's interpretation and are labelled
that way.

- **Least autonomy (Application layer):** `build_support_crew()` enforces a per-role tool allow-list. Only the
  Lookup Agent may hold `check_order_status`, which the guard recognises by name or by argument schema, and
  delegation is refused. The AutoGen reviewers hold no tools, and AutoGen itself refuses tools for their
  client. Evidence: [phase8_tool_access.txt](evaluation/transcripts/phase8_tool_access.txt), with 7 refused
  wirings.
- **Risk classification: Medium.** This is a customer-support agent, the spec's Medium example. It makes no
  medical, hiring or financial decisions and never moves money.
- **Runtime budget (Runtime layer):**
  - **Cap:** 12,000 **estimated** tokens (`len(text) // 4`) per request, not a real-tokenizer count.
  - **Pre-flight:** the projection `7,000 + 6 × tokens(question + history)` refuses an oversized request before
    any model call, with status `budget_exceeded` and a fixed reply, and does not store it.
  - **Accounting:** every model call is charged: each CrewAI agent call (retries too), the grounded-generation
    call, and each AutoGen reviewer and editor call. The accounted estimate can never pass the cap.
  - **Measured spend:** 3,163–6,396 estimated tokens for the 15 evaluation queries; about 8,500–8,700 for the
    largest question the API accepts.
  - **Demonstration:** a 20,400-character simulated request is refused.
  - **Evidence:** [phase8_runtime_budget.txt](evaluation/transcripts/phase8_runtime_budget.txt).

### Full-pipeline evaluation (Phase 8)
[scripts/run_phase8_evaluation.py](scripts/run_phase8_evaluation.py) runs the same 15 frozen queries through
`SupportAssistant.ask` (CrewAI + AutoGen + final groundedness check). For each query it records whether AutoGen
ran, its verdict, the final reply and an independent re-check of that reply against the same tool outputs:
[evaluation/phase8/FULL_PIPELINE_EVALUATION.md](evaluation/phase8/FULL_PIPELINE_EVALUATION.md). AutoGen ran on
the 12 queries with policy text (all approved); E13, E14 and E15 have none. Every final reply re-checks clean,
and every reply and score equals r2 (rubric v1 4.73 / 5 / 5 / 5; v3 5 / 5 / 5 / 5). The Phase 7 artifacts are
not modified.

## Response caching (Part 4, Task 16)
[app/core/cache.py](app/core/cache.py): an in-memory cache for the **grounded-generation step only** (owner decision:
never order lookups or complete responses).
- **Where:** it wraps the generator call inside `answer_query`, after retrieval and the similarity threshold. Every
  request still runs the guardrails, the budget pre-flight, the crew, retrieval, the groundedness check, the AutoGen
  review and the final check. A hit replaces only the one generation call.
- **Key:** SHA-256 of the normalised query text (whitespace collapsed, lower-cased; the embedding model is uncased),
  the retrieved supporting context (chunk ids, texts, metadata), the generator settings and the embedding model.
  Changed policy text, a different collection or a different chunk set gives a different key, so a stale answer is
  never returned. Only the digest is stored, never the query.
- **Never cached:** queries with an order id or PII (or a PII mask token), out-of-scope queries (nothing is generated),
  guardrail refusals (they never reach retrieval), and generated text that fails `claim_supported` against the
  retrieved context.
- **Scope and lifetime:** one cache per `SupportAssistant`, shared by all sessions. It holds only validated policy
  sentences. Bounded to 256 entries (least recently used first); `POST /add-document` clears it. No expiry: the spec
  does not ask for one, and the context fingerprint already prevents stale answers.
- **Before/after evidence** ([phase9_cache.txt](evaluation/transcripts/phase9_cache.txt)): the same question in two
  sessions. The first is a miss (1 generation call, 5,150 estimated tokens); the second is a hit (0 generation calls,
  4,941 tokens). The replies are identical, and retrieval, the AutoGen review and the final check ran both times.
- **Tests:** [tests/test_cache.py](tests/test_cache.py) (31 tests: owner cases 1-12, privacy, persistent-index
  failure modes).
- **Persistent index check** ([phase9_persistent_chroma.txt](evaluation/transcripts/phase9_persistent_chroma.txt)):
  the deployed `chroma_db/` collections match the KB chunks (ids, texts, metadata, vectors). 4,900 repeated searches
  gave 0 exceptions and 0 wrong results. Known limit: the in-memory test index intermittently failed with Chroma
  1.5.9's `InternalError: Error finding id` during Phase 8 (trigger unknown). On that error the reply is "Policy
  search is temporarily unavailable"; if the index returns nothing, the reply is the refusal. Neither is ever an
  unsupported answer, and nothing is cached.

## All commands
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\download_model.py        # one-time, needs internet
.venv\Scripts\python.exe scripts\run_rag_evaluation.py    # builds chroma_db/ and the Phase 3 evidence
.venv\Scripts\python.exe scripts\run_order_lookup_demo.py # Phase 4 evidence
.venv\Scripts\python.exe scripts\run_crew_demo.py         # Phase 5: real crew.kickoff() demos
.venv\Scripts\python.exe scripts\run_phase6_demo.py       # Phase 6: memory, structured output, guardrails
.venv\Scripts\python.exe scripts\run_api_demo.py          # Phase 7: real uvicorn server, HTTP + WebSocket + log
.venv\Scripts\python.exe scripts\run_evaluation.py        # Phase 7 r2 (KB-08 v2): 15 queries, rubric v1 + v3 (never rewrites the original or r1)
.venv\Scripts\python.exe scripts\run_remediation_evidence.py  # E01/E08/E12 before/after evidence + the evaluation
.venv\Scripts\python.exe scripts\run_review_demo.py approve   # Phase 8: AutoGen review approves (use 'revise' for the revision)
.venv\Scripts\python.exe scripts\run_governance_demo.py tools # Phase 8: least autonomy (use 'budget' for the runtime budget)
.venv\Scripts\python.exe scripts\run_phase8_evaluation.py     # Phase 8: 15 queries through CrewAI + AutoGen + final check
.venv\Scripts\python.exe scripts\run_cache_demo.py            # Phase 9: Task 16 cache miss/hit, ineligible cases, invalidation
.venv\Scripts\python.exe scripts\verify_persistent_chroma.py  # Phase 9: read-only check of the deployed chroma_db/ index
.venv\Scripts\python.exe -m pytest -q                     # offline; network blocked
```
