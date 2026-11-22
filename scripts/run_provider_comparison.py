"""v2 Stage G: the frozen 15-query evaluation (evaluation/test_queries.json) on the mock and on a real provider.

Same queries, same pipeline entry point (SupportAssistant.ask), same deterministic rubric judge (rubric v1, the
original, and v3, the revised one) as scripts/run_evaluation.py, so the two providers are scored with one
yardstick. Per query it also records latency, tokens charged to the request budget, model calls (crew agents,
grounded generation, AutoGen review) and, on the real provider, the GPU temperature and NVIDIA's cumulative
thermal-slowdown counters before and after the request. A request during which any thermal slowdown happened is
discarded and re-run after the GPU cools (up to MAX_ATTEMPTS). Every query gets a fresh session and a generation cache
of size 0, so each is measured from scratch. The original graded files are hash-checked before and after and are
never written.

Usage (one provider per process: MOCK_LLM is read at import):
  python scripts/run_provider_comparison.py mock
  $env:MOCK_LLM="false"; $env:LLM_PROVIDER="ollama"; python scripts/run_provider_comparison.py ollama
  python scripts/run_provider_comparison.py report      # writes evaluation/V2_PROVIDER_COMPARISON.md
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)  # per-call token-estimate notice only

import app.core.config  # noqa: E402,F401
from app.core.config import ROOT  # noqa: E402
from scripts.run_evaluation import METRICS, QUERIES_PATH, run_case, score, verify_originals  # noqa: E402

OUT_DIR = ROOT / "evaluation" / "v2"
REPORT = ROOT / "evaluation" / "V2_PROVIDER_COMPARISON.md"
R2_V1 = ROOT / "evaluation" / "remediation" / "evaluation_results_r2_kb08v2_rubric_v1.json"
COOL_C = 65.0  # start a real-model request only at or below this GPU temperature
MAX_ATTEMPTS = 3
GPU_FIELDS = ("temperature.gpu", "clocks_event_reasons_counters.sw_thermal_slowdown",
              "clocks_event_reasons_counters.hw_thermal_slowdown", "clocks_event_reasons_counters.sw_power_cap")


def gpu() -> dict[str, float] | None:
    try:
        line = subprocess.run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=20, check=True).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None
    temp, sw, hw, cap = (float(x) for x in line.split(","))
    return {"temp_c": temp, "sw_thermal_us": sw, "hw_thermal_us": hw, "power_cap_us": cap}


def wait_until_cool(limit_s: float = 900) -> float:
    start = time.monotonic()
    while (g := gpu()) and g["temp_c"] > COOL_C and time.monotonic() - start < limit_s:
        time.sleep(10)
    return round(time.monotonic() - start, 1)


def environment(provider: str, settings: Any) -> dict[str, Any]:
    env: dict[str, Any] = {"provider": provider, "model": settings.model or "deterministic mock",
                           "python": platform.python_version(), "os": platform.platform()}
    if provider != "mock":
        with urllib.request.urlopen(settings.base_url.removesuffix("/v1") + "/api/version", timeout=10) as r:
            env["ollama_version"] = json.load(r)["version"]
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        env["gpu"] = q.stdout.strip()
        scheme = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=20)
        env["power_scheme"] = scheme.stdout.strip()
    return env


def model_calls(assistant: Any, tools: list[Any]) -> dict[str, int]:
    run, review = assistant.last_crew_run, assistant.last_review
    crew = dict(run.llm_calls) if run else {}
    generation = sum(1 for t in tools for e in t.cache_events if e not in ("hit", "not_generated"))
    review_calls = sum(1 for m in review.messages if getattr(m, "source", "user") != "user") if review else 0
    return {**{f"crew_{k}": v for k, v in crew.items()}, "generation": generation, "review": review_calls,
            "total": sum(crew.values()) + generation + review_calls}


def measure(provider: str) -> None:
    from app.core.cache import GenerationCache
    from app.db import database_url
    from app.pipeline import SupportAssistant
    from app.tools.crew_tools import RagLookupTool

    assert database_url() is None, "unset DATABASE_URL: every evaluation session must start empty, in memory"
    hashes_before = verify_originals()
    tools: list[RagLookupTool] = []

    def tool_factory() -> RagLookupTool:
        tools.append(RagLookupTool())
        return tools[-1]

    assistant = SupportAssistant(rag_tool_factory=tool_factory, generation_cache=GenerationCache(max_entries=0))
    settings = assistant.llm_settings
    assert settings.provider == provider, (
        f"settings say {settings.provider!r}; set MOCK_LLM/LLM_PROVIDER for {provider}")
    real = settings.is_real
    assistant.ask("warm-up", "What is the refund policy for cancelled orders?")  # model, embeddings, index: loaded
    spec = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    cases, extra = [], {}
    for q in spec["queries"]:
        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            waited = wait_until_cool() if real else 0.0
            tools.clear()
            assistant.last_crew_run = None  # a blocked request runs no crew: never count the previous request's
            before = gpu()
            t = time.perf_counter()
            case = run_case(assistant, {**q, "id": f"{q['id']}-a{attempt}"} if attempt > 1 else q)
            latency = round(time.perf_counter() - t, 2)
            after = gpu()
            thermal_ms = power_ms = None
            if before and after:
                thermal_ms = round((after["sw_thermal_us"] - before["sw_thermal_us"]
                                    + after["hw_thermal_us"] - before["hw_thermal_us"]) / 1000, 1)
                power_ms = round((after["power_cap_us"] - before["power_cap_us"]) / 1000, 1)
            attempts.append({
                "attempt": attempt, "latency_s": latency, "cooled_for_s": waited,
                "tokens": assistant.last_budget.used if assistant.last_budget else 0,
                "model_calls": model_calls(assistant, tools),
                "gpu_temp_before_c": before and before["temp_c"], "gpu_temp_after_c": after and after["temp_c"],
                "thermal_slowdown_ms": thermal_ms, "power_cap_ms": power_ms,
            })
            print(f"{q['id']} a{attempt}: {case['status']:15} {latency:6.2f}s tokens={attempts[-1]['tokens']} "
                  f"calls={attempts[-1]['model_calls']['total']} gpu={attempts[-1]['gpu_temp_before_c']}"
                  f"->{attempts[-1]['gpu_temp_after_c']}C thermal={thermal_ms}ms", flush=True)
            if not real or not thermal_ms:
                break
        cases.append((q, case))
        extra[q["id"]] = {"measured": attempts[-1], "throttled_attempts": attempts[:-1],
                          "still_throttled": bool(real and attempts[-1]["thermal_slowdown_ms"])}
    scored = {v: score(cases, v) for v in ("v1", "v3")}
    rows = []
    for r1, r3 in zip(scored["v1"]["results"], scored["v3"]["results"], strict=True):
        rows.append({"id": r1["id"], "kind": r1["kind"], "question": r1["question"], "status": r1["status"],
                     "reply": r1["reply"], "scores_v1": r1["scores"], "scores_v3": r3["scores"],
                     "rationale_v1": r1["rationale"], **extra[r1["id"]]})
    assert verify_originals() == hashes_before  # the graded files were not touched
    out = {"environment": environment(provider, settings), "judge": scored["v1"]["judge"],
           "averages": {v: scored[v]["averages"] for v in scored}, "original_file_sha256": hashes_before, "rows": rows}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"provider_comparison_{provider}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("averages v1:", out["averages"]["v1"], "| v3:", out["averages"]["v3"], "->", path.relative_to(ROOT))


# --- report -------------------------------------------------------------------------------------------------------

def _stats(values: list[float]) -> str:
    return f"{statistics.median(values):.2f} / {statistics.mean(values):.2f} / {max(values):.2f}"


def _scores(s: dict[str, int]) -> str:
    return " ".join(str(s[m]) for m in METRICS)


def report() -> None:
    runs = {p: json.loads((OUT_DIR / f"provider_comparison_{p}.json").read_text(encoding="utf-8"))
            for p in ("mock", "ollama")}
    mock, real = runs["mock"], runs["ollama"]
    r2 = {r["id"]: r for r in json.loads(R2_V1.read_text(encoding="utf-8"))["results"]}
    mock_matches_r2 = all(r["scores_v1"] == r2[r["id"]]["scores"] and r["reply"] == r2[r["id"]]["reply"]
                          for r in mock["rows"])
    env = real["environment"]
    lines = [
        "# v2 provider comparison: mock vs Ollama qwen3:4b", "",
        "Generated by `scripts/run_provider_comparison.py` from",
        "`evaluation/v2/provider_comparison_{mock,ollama}.json`.",
        "Every number below was measured in those runs; nothing is estimated or copied from elsewhere.", "",
        "## Method", "",
        "- **Queries:** the frozen 15-query set (`evaluation/test_queries.json`), one fresh session per query.",
        "- **Pipeline:** the same `SupportAssistant.ask` path as production (guardrails, CrewAI crew, groundedness",
        "  check, AutoGen review). The generation cache is size 0, so no query reuses another's work.",
        f"- **Judge:** {mock['judge']}. It is the **same** judge for both providers, with the",
        "  original rubric **v1** and the revised rubric **v3**. These are rubric scores, not a human or LLM",
        "  judgement.",
        "- **Model calls** count every model request: the three CrewAI agents, the grounded generation and the two",
        "  AutoGen reviewers. **Tokens** are those charged to the request budget: provider-reported usage for the",
        "  real model, the estimator for the mock.",
        f"- **Real model:** `{env['model']}` on Ollama {env.get('ollama_version', '?')}, temperature 0; GPU",
        f"  `{env.get('gpu', '?')}`; {env.get('power_scheme', '?')}.",
        f"- **Thermal control:** a request starts only at ≤ {COOL_C:.0f} °C. Any request during which NVIDIA's",
        "  thermal-slowdown counters grew was discarded and re-run after cooling. The GPU temperature before and",
        "  after every measured request is in the tables.",
        "- **Graded files untouched:** the SHA-256 lock of the original evaluation was checked before and after",
        "  both runs.",
        f"  The mock run reproduces the committed r2 (rubric v1) scores and replies exactly: "
        f"**{'yes' if mock_matches_r2 else 'NO'}**.", "",
        "## Averages", "",
        "| Provider | Rubric | Accuracy | Grounding | Completeness | Safety |", "|---|---|---|---|---|---|",
    ]
    for name, run in (("mock", mock), ("qwen3:4b", real)):
        for v in ("v1", "v3"):
            a = run["averages"][v]
            lines.append(f"| {name} | {v} | " + " | ".join(f"{a[m]:.2f}" for m in METRICS) + " |")
    lines += ["", "| Provider | Statuses | Latency s (median / mean / max) | Tokens (median / max) | "
                  "Model calls (median / total) | GPU °C before requests (min–max) | Throttled re-runs |",
              "|---|---|---|---|---|---|---|"]
    for name, run in (("mock", mock), ("qwen3:4b", real)):
        m = [r["measured"] for r in run["rows"]]
        statuses = {}
        for r in run["rows"]:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        temps = [x["gpu_temp_before_c"] for x in m if x["gpu_temp_before_c"] is not None]
        tokens = [x["tokens"] for x in m]
        calls = [x["model_calls"]["total"] for x in m]
        lines.append(
            f"| {name} | {', '.join(f'{k} {v}' for k, v in sorted(statuses.items()))} | "
            f"{_stats([x['latency_s'] for x in m])} | {statistics.median(tokens):.0f} / {max(tokens)} | "
            f"{statistics.median(calls):.0f} / {sum(calls)} | "
            f"{(f'{min(temps):.0f}–{max(temps):.0f}' if temps and name != 'mock' else 'n/a (CPU)')} | "
            f"{sum(len(r['throttled_attempts']) for r in run['rows'])} |")
    for name, run in (("mock", mock), ("qwen3:4b", real)):
        lines += ["", f"## Per query: {name}", "",
                  "Scores are accuracy, grounding, completeness, safety (1–5). Rubric v3 is shown only where it",
                  "differs.",
                  "", "| ID | Kind | Status | v1 scores | v3 | Latency s | Tokens | Calls | GPU °C before → after |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in run["rows"]:
            x = r["measured"]
            v3 = _scores(r["scores_v3"]) if r["scores_v3"] != r["scores_v1"] else ""
            gpu_c = (f"{x['gpu_temp_before_c']:.0f} → {x['gpu_temp_after_c']:.0f}"
                     if x["gpu_temp_before_c"] is not None else "n/a")
            lines.append(f"| {r['id']} | {r['kind']} | {r['status']} | {_scores(r['scores_v1'])} | {v3} | "
                         f"{x['latency_s']:.2f} | {x['tokens']} | {x['model_calls']['total']} | {gpu_c} |")
    diffs = [(a, b) for a, b in zip(mock["rows"], real["rows"], strict=True)
             if a["scores_v1"] != b["scores_v1"] or a["status"] != b["status"]]
    lines += ["", "## Where the real model differs from the mock (rubric v1)", ""]
    if not diffs:
        lines.append("No query differs in status or rubric v1 scores.")
    for a, b in diffs:
        lines += [f"### {a['id']}: {a['question']}", "",
                  f"- **mock** ({a['status']}, {_scores(a['scores_v1'])}): {a['reply']}",
                  f"- **qwen3:4b** ({b['status']}, {_scores(b['scores_v1'])}): {b['reply']}",
                  f"- Judge rationale (qwen3:4b): {b['rationale_v1']}", ""]
    notes = OUT_DIR / "provider_comparison_notes.md"
    if notes.exists():  # hand-written trace analysis of the differences, kept apart from the measured numbers
        lines += ["## Traced causes of the differences", "", notes.read_text(encoding="utf-8").strip(), ""]
    throttled = [(r["id"], t) for r in real["rows"] for t in r["throttled_attempts"]]
    lines += ["", "## Discarded (throttled) requests", ""]
    lines += [f"- {i} attempt {t['attempt']}: {t['latency_s']:.2f} s, {t['thermal_slowdown_ms']} ms thermal slowdown, "
              f"GPU {t['gpu_temp_before_c']:.0f} → {t['gpu_temp_after_c']:.0f} °C" for i, t in throttled] or ["None."]
    still = [r["id"] for r in real["rows"] if r["still_throttled"]]
    if still:
        lines.append(f"- Still throttled after {MAX_ATTEMPTS} attempts (reported as measured): {', '.join(still)}")
    lines += ["", "## Caveats", "",
              "- One measured request per query and provider: latency varies run to run, and these are not averages",
              "  over repeats.",
              "- The judge is the deterministic rubric scorer (lexical facts and the guardrail's grounding rule), so a",
              "  correct answer phrased differently from the expected facts can score lower than a human would give.",
              "- Latency is on one laptop GPU; a hosted or larger GPU would differ."]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", REPORT.relative_to(ROOT), "| mock reproduces r2:", mock_matches_r2)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "report":
        report()
    elif mode in ("mock", "ollama"):
        measure(mode)
    else:
        sys.exit(__doc__)
