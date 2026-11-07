"""Phase 8 full-pipeline evaluation: the same 15 frozen queries through SupportAssistant.ask (CrewAI crew + AutoGen
review + final groundedness check), recording per query whether AutoGen ran, its verdict, the final reply (after
AutoGen), an independent re-check of that final reply against the same tool outputs, and the rubric v1 / v3 scores.

It writes only evaluation/phase8/full_pipeline_evaluation.json and FULL_PIPELINE_EVALUATION.md. It never writes the
original Phase 7 results, r1 or r2 (scripts/run_evaluation.py); it checks the original and r1 hashes and compares
every reply and score with r2.

Usage: .venv\\Scripts\\python.exe scripts\\run_phase8_evaluation.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)

import app.core.config  # noqa: E402,F401
import scripts.run_evaluation as runner  # noqa: E402
from app.core.config import ROOT  # noqa: E402
from app.guardrails.groundedness import check_draft  # noqa: E402
from app.memory.session_memory import get_session_history, has_session  # noqa: E402
from app.pipeline import SupportAssistant  # noqa: E402
from app.schemas import render_reply  # noqa: E402

OUT_DIR = ROOT / "evaluation" / "phase8"
OUT = OUT_DIR / "full_pipeline_evaluation.json"
REPORT = OUT_DIR / "FULL_PIPELINE_EVALUATION.md"
METRICS = runner.METRICS


def evaluate(assistant: SupportAssistant | None = None) -> dict[str, Any]:
    assistant = assistant or SupportAssistant()
    real_ask, responses = assistant.ask, []

    def ask(session_id, question):  # keep the full SupportResponse (run_case only keeps the reply fields)
        responses.append(real_ask(session_id, question))
        return responses[-1]

    assistant.ask = ask
    spec = json.loads(runner.QUERIES_PATH.read_text(encoding="utf-8"))
    cases, extra = [], []
    for q in spec["queries"]:
        if has_session(f"eval-{q['id']}"):  # run_case's fixed session ids: start each query fresh in this process
            get_session_history(f"eval-{q['id']}").clear()
        before = assistant.last_crew_run
        case = runner.run_case(assistant, q)
        response, review = responses[-1], assistant.last_review
        run = assistant.last_crew_run if assistant.last_crew_run is not before else None
        final_check = None
        if response.draft is not None and run is not None:
            rechecked, rep = check_draft(response.draft, (run.rag_outputs or [None])[-1],
                                         (run.order_outputs or [None])[-1])
            final_check = {"unchanged_by_recheck": rechecked == response.draft, "rejected_claims": rep.rejected_claims,
                           "removed_citations": rep.removed_citations, "rejected_order": rep.rejected_order,
                           "reply_is_rendered_final_draft": render_reply(response.draft) == response.reply}
        autogen = ({"ran": True, "message_sources": [m.source for m in review.messages],
                    "message_types": [type(m).__name__ for m in review.messages],
                    "verdict": review.verdict.model_dump(), "discarded_additions": review.discarded_additions,
                    "reserved_tokens": review.reserved_tokens, "spent_tokens": review.spent_tokens}
                   if review else {"ran": False, "why": "blocked before the crew" if run is None
                                   else "no policy text to review (order-only or no-answer turn)"})
        cases.append((q, case))
        extra.append({"autogen": autogen, "final_check": final_check,
                      "budget_used": assistant.last_budget.used if assistant.last_budget else None})
    scored = {v: runner.score(cases, v) for v in ("v1", "v3")}
    r2 = {v: {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))["results"]}
          for v, p in runner.OUTPUTS.items()}
    rows = []
    for i, (q, case) in enumerate(cases):
        s1, s3 = scored["v1"]["results"][i]["scores"], scored["v3"]["results"][i]["scores"]
        rows.append({"id": q["id"], "question": q["query"], "status": case["status"], "reply": case["reply"],
                     **extra[i], "scores_rubric_v1": s1, "scores_rubric_v3": s3,
                     "same_as_r2": case["reply"] == r2["v1"][q["id"]]["reply"]
                     and s1 == r2["v1"][q["id"]]["scores"] and s3 == r2["v3"][q["id"]]["scores"]})
    return {"pipeline": "SupportAssistant.ask: guardrails -> budget -> CrewAI crew -> groundedness -> AutoGen review "
                        "-> final groundedness check -> render_reply",
            "judge": runner.JUDGE_LABEL, "queries": len(rows),
            "averages": {v: scored[v]["averages"] for v in scored}, "results": rows}


def write_report(result: dict[str, Any], hashes: dict[str, str]) -> None:
    lines = ["# Full-pipeline evaluation (Phase 8): CrewAI + AutoGen + final groundedness check", "",
             f"Same 15 frozen queries (evaluation/test_queries.json). Pipeline: {result['pipeline']}.",
             f"Judge: **{result['judge']}**. The original Phase 7, r1 and r2 artifacts are not modified; "
             "original and r1 SHA-256 checked:", "", *[f"- `{p}`: `{h}`" for p, h in hashes.items()], "",
             "| ID | status | AutoGen ran | verdict approved | final re-check clean | v1 A/G/C/S | v3 A/G/C/S | same as r2 |",
             "|---|---|---|---|---|---|---|---|"]
    fmt = lambda s: "/".join(str(s[m]) for m in METRICS)  # noqa: E731
    for r in result["results"]:
        fc = r["final_check"]
        clean = "n/a" if fc is None else str(fc["unchanged_by_recheck"] and not fc["rejected_claims"]
                                            and not fc["removed_citations"] and not fc["rejected_order"]
                                            and fc["reply_is_rendered_final_draft"])
        approved = r["autogen"]["verdict"]["approved"] if r["autogen"]["ran"] else f"— ({r['autogen']['why']})"
        lines.append(f"| {r['id']} | {r['status']} | {r['autogen']['ran']} | {approved} | {clean} | "
                     f"{fmt(r['scores_rubric_v1'])} | {fmt(r['scores_rubric_v3'])} | {r['same_as_r2']} |")
    avg = lambda d: " / ".join(f"{d[m]:.2f}" for m in METRICS)  # noqa: E731
    lines += ["", f"Averages: rubric v1 **{avg(result['averages']['v1'])}**; rubric v3 **{avg(result['averages']['v3'])}**.",
              "", "AutoGen does not run when there is no policy text to review: E13 (out-of-scope fallback), E14 "
              "(injection blocked before the crew) and E15 (order not found, no policy question). Order facts are "
              "checked against the real tool output by the groundedness check in every case.", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    hashes = runner.verify_originals()
    result = evaluate()
    OUT_DIR.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({**result, "original_and_r1_sha256": hashes}, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    write_report(result, hashes)
    runner.verify_originals()
    for r in result["results"]:
        print(f"{r['id']:<4} {r['status']:<15} autogen={str(r['autogen']['ran']):<5} "
              f"approved={r['autogen'].get('verdict', {}).get('approved', '-')!s:<5} "
              f"final_check={'n/a' if r['final_check'] is None else r['final_check']['reply_is_rendered_final_draft']} "
              f"same_as_r2={r['same_as_r2']}  {r['reply'][:60]}")
    print("AVERAGES:", result["averages"])
    print("ORIGINAL AND r1 FILES SHA-256 unchanged:", hashes)


if __name__ == "__main__":
    main()
