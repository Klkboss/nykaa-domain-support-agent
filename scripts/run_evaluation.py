"""Spec Task 13: run the frozen 15-query set through the real pipeline and score it with the LLM-as-judge
prompt under MOCK_LLM.

This script NEVER writes the original Phase 7 results (evaluation/evaluation_results.json,
evaluation/EVALUATION_REPORT.md, commit b7e2fe9) or the frozen remediation r1 results (evaluation/remediation/
*_r1_*, EVALUATION_REPORT_REMEDIATION.md: both run against KB-08 wording v1, preserved in
data/knowledge_base/history/KB-08_wording_v1.json). It checks both sets' SHA-256, runs the pipeline once on the
current KB (KB-08 wording v2, owner-approved clarification) and scores the same outputs twice:
  - with the ORIGINAL rubric v1 (evaluation/test_queries.json) -> remediation/evaluation_results_r2_kb08v2_rubric_v1.json
  - with the REVISED rubric v3 (remediation/rubric_v3.json)     -> remediation/evaluation_results_r2_kb08v2_rubric_v3.json
and writes evaluation/remediation/EVALUATION_REPORT_R2_KB08_CLARIFIED.md. The withdrawn rubric v2 is no longer run.
To reproduce the original run exactly: check out commit b7e2fe9 (e.g. in a git worktree) and run this script there.

Usage: .venv\\Scripts\\python.exe scripts\\run_evaluation.py
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)  # per-call token-estimate notice only

import app.core.config  # noqa: E402,F401
from app.core.config import ROOT  # noqa: E402
from app.judge import JUDGE_PROMPT, judge  # noqa: E402
from app.pipeline import SupportAssistant  # noqa: E402
from app.rag.documents import REQUIRED_TOPICS  # noqa: E402

METRICS = ("accuracy", "grounding", "completeness", "safety")
EVAL = ROOT / "evaluation"
QUERIES_PATH = EVAL / "test_queries.json"
RUBRIC_PATHS = {v: EVAL / "remediation" / f"rubric_{v}.json" for v in ("v2", "v3")}
ORIGINAL_RESULTS = EVAL / "evaluation_results.json"
ORIGINAL_REPORT = EVAL / "EVALUATION_REPORT.md"
# SHA-256 of LF-normalised content (= the git blob), so the locks hold on every OS and checkout.
ORIGINAL_SHA256 = {  # the original Phase 7 evaluation (b7e2fe9); must never change
    "evaluation/evaluation_results.json": "52952c90b5456b1c16af4f9188e65a85c99a5c94aa58c707435d388ef33d48ef",
    "evaluation/EVALUATION_REPORT.md": "40e60a1f54e37e3c64344813b966d44a593701888f713f288b85b1cdbcd573bd",
    "evaluation/test_queries.json": "3699ac04647a6b552c8432caae2fe134aa45ad472551ad771dba77c14fdd7565",
}
R1_SHA256 = {  # remediation r1, run against KB-08 wording v1 (d5ffadb, b0ba629); frozen
    "evaluation/remediation/evaluation_results_r1_rubric_v1.json": "bdf4535e3a1fceb5e4f0b1cc6a5588fa8156cd92f0114afd8dd3ba908e225592",
    "evaluation/remediation/evaluation_results_r1_rubric_v2.json": "b2205cfa07630437977176340444be168706ddb0d989cd4c9c8eb4a9999366d6",
    "evaluation/remediation/evaluation_results_r1_rubric_v3.json": "685c730f1faeead88fc75a9d6379f5d636aeb5b41991b6aa8c804532f001c51b",
    "evaluation/remediation/EVALUATION_REPORT_REMEDIATION.md": "b0f4720853adaaf7c865262c08f1b772c8235d1ac83fcad8103d2399134cf13b",
}
REMEDIATION = EVAL / "remediation"
R1_RESULTS = {v: REMEDIATION / f"evaluation_results_r1_rubric_{v}.json" for v in ("v1", "v3")}  # frozen, read only
OUTPUTS = {v: REMEDIATION / f"evaluation_results_r2_kb08v2_rubric_{v}.json" for v in ("v1", "v3")}
REPORT = REMEDIATION / "EVALUATION_REPORT_R2_KB08_CLARIFIED.md"
JUDGE_LABEL = "MockJudgeLLM (deterministic rubric scorer behind a BaseLLM interface; NOT a real LLM judgement)"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def verify_originals() -> dict[str, str]:
    actual = {rel: sha256(ROOT / rel) for rel in {**ORIGINAL_SHA256, **R1_SHA256}}
    if actual != {**ORIGINAL_SHA256, **R1_SHA256}:
        raise RuntimeError(f"original or r1 evaluation files changed: {actual}")
    return actual


def load_rubric(version: str) -> dict[str, str]:
    if version == "v1":
        return json.loads(QUERIES_PATH.read_text(encoding="utf-8"))["rubric"]
    return json.loads(RUBRIC_PATHS[version].read_text(encoding="utf-8"))["rubric"]


def count_facts() -> dict[str, dict[str, str]]:
    """v3 annotations: case id -> {expected fact: counted event}, frozen in rubric_v3.json."""
    spec = json.loads(RUBRIC_PATHS["v3"].read_text(encoding="utf-8"))["count_facts"]
    return {qid: {fact: a["counts"] for fact, a in facts.items()} for qid, facts in spec.items()}


def run_case(assistant: SupportAssistant, q: dict[str, Any]) -> dict[str, Any]:
    before = assistant.last_crew_run
    response = assistant.ask(f"eval-{q['id']}", q["query"])  # one fresh session per query
    run = assistant.last_crew_run if assistant.last_crew_run is not before else None
    rag = (run.rag_outputs or [None])[-1] if run else None
    evidence = [p for e in rag.get("evidence", []) for p in e["passages"]] if rag and rag.get("grounded") else []
    return {
        "question": q["query"], "masked_question": response.masked_question, "status": response.status,
        "reply": response.reply, "expected": q["expected"], "pii": q.get("pii", []),
        "evidence": evidence, "order_output": (run.order_outputs or [None])[-1] if run else None,
        "tools_invoked": {"rag_lookup": run.rag_invocations, "check_order_status": run.order_invocations} if run else {},
        "guardrails": response.guardrails.model_dump(),
    }


def run_cases(assistant: SupportAssistant | None = None) -> list[tuple[dict, dict]]:
    spec = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    assistant = assistant or SupportAssistant()
    return [(q, run_case(assistant, q)) for q in spec["queries"]]


def score(cases: list[tuple[dict, dict]], version: str) -> dict[str, Any]:
    rubric, rows = load_rubric(version), []
    annotations = count_facts() if version == "v3" else {}
    for q, case in cases:
        judged_case = {k: v for k, v in case.items() if k not in ("tools_invoked", "guardrails")}
        if q["id"] in annotations:
            judged_case["count_facts"] = annotations[q["id"]]
        scores, raw, prompt = judge(rubric, {**judged_case, "rubric_version": version})
        rows.append({"id": q["id"], "topics": q["topics"], "kind": q.get("kind", "topic"), **case,
                     "scores": {m: getattr(scores, m) for m in METRICS}, "rationale": scores.rationale,
                     "judge_raw_output": raw, "judge_prompt": prompt})
    covered = sorted({t for r in rows for t in r["topics"]})
    return {
        "judge": JUDGE_LABEL, "rubric_version": version,
        "rubric_source": {"v1": "evaluation/test_queries.json (original v1)",
                          "v2": "evaluation/remediation/rubric_v2.json (REVISED, WITHDRAWN: equated attempts with retries)",
                          "v3": "evaluation/remediation/rubric_v3.json (REVISED, supersedes v2)"}[version],
        "rubric": rubric, "judge_prompt_template": JUDGE_PROMPT,
        "coverage": {"queries": len(rows), "topics_covered": covered,
                     "topics_missing": sorted(set(REQUIRED_TOPICS) - set(covered)),
                     "edge_or_out_of_scope": [r["id"] for r in rows if r["kind"] != "topic"]},
        "averages": {m: round(mean(r["scores"][m] for r in rows), 4) for m in METRICS},
        "results": rows,
    }


def evaluate(assistant: SupportAssistant | None = None, rubric_version: str = "v1") -> dict[str, Any]:
    return score(run_cases(assistant), rubric_version)


# Explanations of every score or reply that differs from the original run, written after inspecting the outputs.
EXPLANATIONS = {
    "E01": "Pipeline fix (category-subject filter in the grounded generator): the reply no longer appends the "
           "Beauty '15 days' rule to a dress question, so the unchanged v1 forbidden-fact rule no longer fires.",
    "E05": "Reply changed, scores did not. A trimmer is Electronics (KB-01 lists trimmers under Electronics), so "
           "the correct period is the expected fact '12-month', which stays the first sentence. The category-subject "
           "filter (E01 fix) dropped the Home-appliance '6-month' sentence (another category's period, which the "
           "original reply wrongly included) and the Apparel/Beauty no-warranty sentence; the category-neutral "
           "KB-05 exclusion 'No warranty covers physical damage …', which applies to every category and to a "
           "device that 'broke', became the second sentence.",
    "E08": "Policy clarification (KB-08 wording v2, owner-approved: 3 retries after the first attempt, at most 4 "
           "attempts). The reply now states the retry limit: 'Once all 3 retries have also failed … switch' and "
           "'retry the payment up to 3 more times within 30 minutes'. Rubric v3 credits the frozen fact '3 times' "
           "(annotated: counts retries) through 'all 3 retries' (v3's frozen rule does not read '3 more times'). "
           "Rubric v1's literal substring '3 times' is not in the reply, so v1 still scores accuracy 1.",
    "E12": "Pipeline fix (escalation routing in the shared retrieval path): KB-12 chunks only, so the reply cites "
           "KB-12 ('Level 3 …' and 'Level 2 specialists respond …') instead of the KB-10 damaged-item sentence. "
           "Remaining limitation: the most specific sentence ('A case moves to Level 2 … delayed shipment with a "
           "high escalation score') is still split across two fixed-size windows, so it is not quoted.",
}

COLUMNS = ("v1", "v3")
LABELS = {"v1": "rubric v1 (original)", "v3": "rubric v3 (revised)"}


def write_report(original: dict, r1: dict[str, dict], runs: dict[str, dict], hashes: dict[str, str]) -> None:
    orig = {r["id"]: r for r in original["results"]}
    old = {v: {r["id"]: r for r in r1[v]["results"]} for v in COLUMNS}
    fmt = lambda s: "/".join(str(s[m]) for m in METRICS)  # noqa: E731
    avg = lambda d: " / ".join(f"{d['averages'][m]:.2f}" for m in METRICS)  # noqa: E731
    lines = [
        "# Evaluation report r2: KB-08 clarified (spec Task 13, Phase 7)", "",
        "Run against **KB-08 wording v2** (owner-approved: 3 retries after the first failed attempt, at most 4",
        "attempts in total). Earlier runs used **KB-08 wording v1**, preserved in",
        "`data/knowledge_base/history/KB-08_wording_v1.json`, and stay unchanged:", "",
        "- the original Phase 7 evaluation (`b7e2fe9`): `evaluation/evaluation_results.json`, `evaluation/EVALUATION_REPORT.md`",
        "- remediation r1 (`d5ffadb`, `b0ba629`): `evaluation/remediation/*_r1_*`, `EVALUATION_REPORT_REMEDIATION.md`", "",
        "SHA-256 (LF-normalised) checked by this run:", "", *[f"- `{p}`: `{h}`" for p, h in hashes.items()], "",
        f"Judge: **{JUDGE_LABEL}**. Same 15 frozen queries (evaluation/test_queries.json). The same outputs are scored",
        "with the original rubric **v1** (like-for-like) and the revised rubric **v3** (frozen in `a822a91`, before",
        "this policy change; unchanged). The withdrawn rubric v2 is not used.", "", "## Scores", "",
        "| ID | Original (b7e2fe9, KB-08 v1, rubric v1) | r1 (KB-08 v1), rubric v1 | r1 (KB-08 v1), rubric v3 | "
        "**r2 (KB-08 v2), rubric v1** | **r2 (KB-08 v2), rubric v3** |",
        "|---|---|---|---|---|---|",
    ]
    for rows in zip(*(runs[v]["results"] for v in COLUMNS)):
        qid = rows[0]["id"]
        cells = [fmt(old[v][qid]["scores"]) for v in COLUMNS] + [
            fmt(r["scores"]) + (" **changed**" if r["scores"] != old[v][qid]["scores"] else "") for v, r in zip(COLUMNS, rows)]
        lines.append(f"| {qid} | {fmt(orig[qid]['scores'])} | " + " | ".join(cells) + " |")
    lines += ["| **Averages** (acc / grd / cmp / saf) | **" + avg(original) + "** | "
              + " | ".join(f"**{avg(r1[v])}**" for v in COLUMNS) + " | "
              + " | ".join(f"**{avg(runs[v])}**" for v in COLUMNS) + " |", "",
              "'changed' marks a difference from r1 under the same rubric.", "",
              "## Every score or reply that differs from the original run", ""]
    for rows in zip(*(runs[v]["results"] for v in COLUMNS)):
        r1_, o = rows[0], orig[rows[0]["id"]]
        if any(r["scores"] != o["scores"] for r in rows) or r1_["reply"] != o["reply"]:
            lines += [f"### {r1_['id']}: {r1_['question']}", "",
                      f"- Original reply (KB-08 v1): {o['reply'].replace(chr(10), ' / ')}",
                      f"- r1 reply (KB-08 v1): {old['v1'][r1_['id']]['reply'].replace(chr(10), ' / ')}",
                      f"- r2 reply (KB-08 v2): {r1_['reply'].replace(chr(10), ' / ')}",
                      f"- Accuracy: original {o['scores']['accuracy']}; "
                      + "; ".join(f"r2 {LABELS[v]} {r['scores']['accuracy']} ({r['rationale']['accuracy']})"
                                  for v, r in zip(COLUMNS, rows)),
                      f"- Why: {EXPLANATIONS.get(r1_['id'], 'see rationale')}", ""]
    lines += ["## Limitations", "",
              "- The judge is a deterministic rubric scorer, not a real LLM; grounding reuses the guardrail's lexical rule,",
              "  so grounding/completeness/safety near 5 are partly by construction.",
              "- v1 matches expected facts as exact substrings, so E08's correct '3 more times' / '3 retries' wording",
              "  still scores accuracy 1 under v1. v3 reads only '<n> <event word>' and '<event word> … <n> times', not",
              "  '<n> more times'; E08 is credited through 'all 3 retries'.",
              "- E08 is the only reply that changed between r1 and r2; it changed because the policy text changed.", ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    hashes = verify_originals()
    original = json.loads(ORIGINAL_RESULTS.read_text(encoding="utf-8"))
    r1 = {v: json.loads(path.read_text(encoding="utf-8")) for v, path in R1_RESULTS.items()}
    cases = run_cases()
    runs = {v: score(cases, v) for v in COLUMNS}
    for v, path in OUTPUTS.items():
        path.write_text(json.dumps({**runs[v], "kb08_wording_version": 2, "original_files_sha256": hashes}, indent=2,
                                   ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(original, r1, runs, hashes)
    verify_originals()  # still unchanged after writing
    print(f"{'ID':<4} {'status':<15} orig(v1)  r1(v1)  r1(v3)  r2(v1)  r2(v3)  reply")
    f = lambda s: "".join(str(s[m]) for m in METRICS)  # noqa: E731
    for o, a, b, c, d in zip(original["results"], r1["v1"]["results"], r1["v3"]["results"],
                             *(runs[v]["results"] for v in COLUMNS)):
        print(f"{o['id']:<4} {c['status']:<15} {f(o['scores']):<9} {f(a['scores']):<7} {f(b['scores']):<7} "
              f"{f(c['scores']):<7} {f(d['scores']):<7} {c['reply'][:60].replace(chr(10), ' / ')}")
    print("AVERAGES original (KB-08 v1, rubric v1):", original["averages"])
    for v in COLUMNS:
        print(f"AVERAGES r1 (KB-08 v1), {LABELS[v]}:", r1[v]["averages"])
    for v in COLUMNS:
        print(f"AVERAGES r2 (KB-08 v2), {LABELS[v]}:", runs[v]["averages"])
    print("ORIGINAL AND r1 FILES SHA-256 unchanged:", hashes)


if __name__ == "__main__":
    main()
