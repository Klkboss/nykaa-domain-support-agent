"""v2 Stage B measurement: the full pipeline (SupportAssistant.ask) on local Ollama models.

Per request it records: status, latency, model calls by role, provider-reported tokens (the request budget's
settled spend), tool routing (did each agent call the right tool with the right id), JSON reliability (grounded
generation outcome, Composer attempts) and, per model, Ollama's GPU/CPU split from /api/ps.

Usage (Ollama running on loopback; nothing else is contacted):
  .venv\\Scripts\\python.exe scripts\\measure_providers.py qwen2.5:7b-instruct qwen2.5:3b-instruct
Writes evaluation/v2/ollama_measurements.json (raw rows + per-model summary) and prints a table.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OLLAMA = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
os.environ.update({"MOCK_LLM": "false", "LLM_PROVIDER": "ollama", "OLLAMA_BASE_URL": OLLAMA})

import app.core.config  # noqa: E402,F401  (offline env first)
from app.governance.budget import MAX_REQUEST_TOKENS_REAL  # noqa: E402
from app.pipeline import SupportAssistant  # noqa: E402
from app.tools.crew_tools import RagLookupTool  # noqa: E402

# question, expected order lookup (None = must not look up), expected status set
QUESTIONS = [
    ("How long do I have to return shoes?", None, {"answered"}),
    ("What is the status of my order ORD-1016?", "ORD-1016", {"answered"}),
    ("My order ORD-1029 arrived late. Can I still return the shoes, and is my case being escalated?", "ORD-1029",
     {"answered"}),
    ("Where is my order ORD-9999?", "ORD-9999", {"answered"}),
    ("What is the best way to train for a marathon?", None, {"fallback"}),
    ("Can you check my order status?", None, {"needs_order_id"}),
]
OUT = ROOT / "evaluation" / "v2" / "ollama_measurements.json"
ROUNDS = int(os.environ.get("MEASURE_ROUNDS", "2"))  # each question is asked ROUNDS times (fresh session)


def ollama_ps(model: str) -> dict:
    with urllib.request.urlopen(f"{OLLAMA}/api/ps", timeout=10) as r:  # loopback only
        for m in json.load(r)["models"]:
            if m["name"] == model:
                return {"size_bytes": m["size"], "size_vram_bytes": m["size_vram"],
                        "gpu_fraction": round(m["size_vram"] / m["size"], 3), "context": m.get("context_length")}
    return {}


def measure(model: str) -> dict:
    os.environ["LLM_MODEL"] = model
    tools: list[RagLookupTool] = []

    def tool_factory() -> RagLookupTool:
        tools.append(RagLookupTool())
        return tools[-1]

    assistant = SupportAssistant(rag_tool_factory=tool_factory)
    assistant.generation_cache.max_entries = 0  # every request generates: no cache hits in the measurement
    t = time.perf_counter()
    assistant.ask("warmup", "How long do I have to return shoes?")  # loads the model; not counted
    load_s = round(time.perf_counter() - t, 1)
    rows = []
    for i, (question, order_id, expected) in enumerate(QUESTIONS * ROUNDS):
        assistant.last_crew_run = None  # set only on success: never score a failed request from the previous one
        t = time.perf_counter()
        r = assistant.ask(f"m{i}", question)
        latency = round(time.perf_counter() - t, 2)
        run, review, tool = assistant.last_crew_run, assistant.last_review, tools[-1]
        crew_calls = dict(run.llm_calls) if run and r.status != "error" else {}
        looked_up = [x["record_id"] for x in run.order_invocations] if run else []
        rows.append({
            "question": question, "status": r.status, "expected_status": sorted(expected),
            "status_ok": r.status in expected, "latency_s": latency,
            "tokens": assistant.last_budget.used if assistant.last_budget else 0,
            "crew_calls": crew_calls, "generation": getattr(tool.generator, "outcomes", []),
            "review_calls": 2 if review else 0,
            "rag_called": bool(run and run.rag_invocations),
            "order_lookups": looked_up,
            "routing_ok": bool(run and run.rag_invocations) and looked_up == ([order_id] if order_id else []),
            "composer_attempts": crew_calls.get("composer"),
            "draft_repairs": run.draft_repairs if run and r.status != "error" else [],
            "grounded": r.guardrails.grounded, "rejected_claims": r.guardrails.rejected_claims,
            "reply": r.reply,
        })
        print(f"  [{model}] {r.status:<15} {latency:>6}s {rows[-1]['tokens']:>6} tok  {question[:60]}", flush=True)
    return {"model": model, "load_s": load_s, "ollama_ps": ollama_ps(model), "rows": rows,
            "stress": stress(assistant)}


def stress(assistant: SupportAssistant) -> dict:
    """Worst case for the real-provider cap: 5 prior exchanges, then a question near the API's 2,000-char limit."""
    for q, _, _ in QUESTIONS[:5]:
        assistant.ask("stress", q)
    filler = ("I bought a pair of running shoes and a lipstick last month, the shoes were a gift for my brother and "
              "he has not worn them yet, and I still have the original shoebox and the receipt with me today. ")
    long_q = (filler * 10)[:1900] + " How long do I have to return the shoes?"
    t = time.perf_counter()
    r = assistant.ask("stress", long_q)
    return {"question_chars": len(long_q), "history_messages": 10, "status": r.status,
            "latency_s": round(time.perf_counter() - t, 2),
            "tokens": assistant.last_budget.used if assistant.last_budget else 0}


def summary(result: dict) -> dict:
    rows = result["rows"]
    lat, tok = [r["latency_s"] for r in rows], [r["tokens"] for r in rows]
    gens = [o for r in rows for o in r["generation"]]
    composer = [r["composer_attempts"] for r in rows if r["composer_attempts"]]
    calls = [sum(r["crew_calls"].values()) + len(r["generation"]) + r["review_calls"] for r in rows]
    return {
        "model": result["model"], "gpu_fraction": result["ollama_ps"].get("gpu_fraction"),
        "status_ok": f"{sum(r['status_ok'] for r in rows)}/{len(rows)}",
        "routing_ok": f"{sum(r['routing_ok'] for r in rows)}/{len(rows)}",
        "errors": sum(r["status"] == "error" for r in rows),
        "latency_s_median": statistics.median(lat), "latency_s_max": max(lat),
        "tokens_median": statistics.median(tok), "tokens_max": max(tok),
        "model_calls_median": statistics.median(calls),
        "generation_verified": f"{sum(not g.startswith('fallback') for g in gens)}/{len(gens)}",
        "composer_first_try": f"{sum(c == 1 for c in composer)}/{len(composer)}",
        "draft_repairs": sum(len(r["draft_repairs"]) for r in rows),
        "stress_tokens": result.get("stress", {}).get("tokens"),
        "stress_status": result.get("stress", {}).get("status"),
        "failure_reasons": sorted({c for r in rows if r["status"] == "error" for c in r["rejected_claims"]}),
    }


def main() -> None:
    models = sys.argv[1:] or ["qwen2.5:7b-instruct"]
    results = []
    for model in models:
        print(f"\n== {model} ==", flush=True)
        results.append(measure(model))
    data = {"ollama": OLLAMA, "cap_tokens": MAX_REQUEST_TOKENS_REAL, "results": results,
            "summary": [summary(r) for r in results]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"results": []}
    kept = [r for r in previous["results"] if r["model"] not in models]  # re-measured models are replaced
    data["results"] = kept + results
    data["summary"] = [summary(r) for r in data["results"]]
    OUT.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n" + json.dumps(data["summary"], indent=2))


if __name__ == "__main__":
    main()
