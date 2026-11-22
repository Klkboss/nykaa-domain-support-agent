"""Probe the free hosted LLM candidates: list their models and make ONE tiny request each.

For each candidate it records (in evaluation/v2/hosted_probe.json):
  - whether the model ID exists in the provider's own model list;
  - one small chat request with a json_schema response_format: HTTP status, latency, token usage, which message
    fields carry reasoning (`reasoning`, `reasoning_content`) and whether reasoning leaks into `content`;
  - the provider's rate-limit response headers (x-ratelimit-*, retry-after), which are the limits on THESE keys.
Keys are read from .env (git-ignored) by NAME and are never printed or written: error bodies are cut to 200
characters with every key value replaced by "***". The Cloudflare account ID is only ever written as
${CLOUDFLARE_ACCOUNT_ID}.

GitHub Models was probed on 2026-11-23 and found retired (see hosted_probe.json); it is no longer a candidate.

Usage: python scripts/probe_hosted_providers.py [candidate ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evaluation" / "v2" / "hosted_probe.json"
CF = "https://api.cloudflare.com/client/v4/accounts/${CLOUDFLARE_ACCOUNT_ID}/ai/v1"
# name: (base URL template, model, key variable, reasoning_effort sent in the probe)
CANDIDATES: dict[str, tuple[str, str, str, str | None]] = {
    "groq-gpt-oss-120b": ("https://api.groq.com/openai/v1", "openai/gpt-oss-120b", "GROQ_API_KEY", "low"),
    "groq-gpt-oss-20b": ("https://api.groq.com/openai/v1", "openai/gpt-oss-20b", "GROQ_API_KEY", "low"),
    "gemini-3-flash": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3-flash-preview",
                       "GEMINI_API_KEY", "low"),
    # the newest Flash model in the /models list on 2026-11-23, free of charge per Google's pricing page
    "gemini-3.8-flash": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-3.8-flash",
                         "GEMINI_API_KEY", "low"),
    "cloudflare-gpt-oss-120b": (CF, "@cf/openai/gpt-oss-120b", "CLOUDFLARE_API_TOKEN", "low"),
    "cloudflare-gpt-oss-20b": (CF, "@cf/openai/gpt-oss-20b", "CLOUDFLARE_API_TOKEN", "low"),
}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {
    "type": "object", "properties": {"quotes": {"type": "array", "items": {"type": "string"}}},
    "required": ["quotes"]}}}
PROBE = [{"role": "system", "content": "Answer with JSON {\"quotes\": [...]}, copying sentences exactly."},
         {"role": "user", "content": "Policy sentences:\n1. Returns are accepted within 15 days of delivery.\n"
                                     "2. Gift cards cannot be refunded.\n\nCustomer question: Can I return shoes?"}]


def expand(template: str, env: dict[str, str | None]) -> str:
    out = template
    for name, value in env.items():
        out = out.replace("${" + name + "}", value or "")
    return out


def scrub(text: str, secrets: list[str]) -> str:
    for s in secrets:
        text = text.replace(s, "***")
    return text[:200]


def model_ids(client: httpx.Client, base: str, provider: str) -> list[str] | str:
    r = client.get(base + "/models")
    if r.status_code == 200:
        body = r.json()
        return [m["id"].removeprefix("models/") for m in body.get("data", body.get("result", []))]
    if provider.startswith("cloudflare"):  # Cloudflare's own catalog API
        r = client.get(base.removesuffix("/v1") + "/models/search", params={"search": "gpt-oss"})
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("result", [])]
    return f"HTTP {r.status_code}"


def probe(name: str, env: dict[str, str | None], secrets: list[str]) -> dict[str, Any]:
    template, model, key_var, effort = CANDIDATES[name]
    key = (env.get(key_var) or "").strip()
    row: dict[str, Any] = {"base_url": template, "model": model, "key_variable": key_var, "reasoning_effort": effort}
    if not key:
        return {**row, "error": f"{key_var} is not set"}
    base = expand(template, env)
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=60) as client:
        try:
            ids = model_ids(client, base, name)
        except httpx.HTTPError as exc:
            ids = type(exc).__name__
        row["model_listed"] = model in ids if isinstance(ids, list) else ids
        row["related_models"] = sorted(i for i in ids if any(t in i for t in ("gpt-oss", "flash", "gpt-4", "gpt-5")))[
            :40] if isinstance(ids, list) else []
        body: dict[str, Any] = {"model": model, "messages": PROBE, "temperature": 0, "max_tokens": 1024,
                                "response_format": SCHEMA}
        if effort:
            body["reasoning_effort"] = effort
        t = time.perf_counter()
        try:
            r = client.post(base + "/chat/completions", json=body)
        except httpx.HTTPError as exc:
            return {**row, "error": type(exc).__name__}
        row["latency_s"] = round(time.perf_counter() - t, 2)
        row["status"] = r.status_code
        row["content_type"] = r.headers.get("content-type")
        row["rate_limit_headers"] = {k: v for k, v in r.headers.items()
                                     if k.lower().startswith(("x-ratelimit", "retry-after", "x-ms-ratelimit"))}
        if r.status_code != 200 or "json" not in (row["content_type"] or ""):
            row["error_body"] = scrub(r.text, secrets)
            return row
        data = r.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        row["message_fields"] = sorted(k for k, v in msg.items() if v not in (None, "", []))
        row["reasoning_field_chars"] = {k: len(str(msg[k])) for k in ("reasoning", "reasoning_content") if msg.get(k)}
        row["content_starts_with_think_tag"] = content.lstrip().startswith(("<think", "<thought"))
        row["content"] = content[:300]
        row["finish_reason"] = data["choices"][0].get("finish_reason")
        row["usage"] = data.get("usage")
    return row


def main(names: list[str]) -> None:
    env = dotenv_values(ROOT / ".env")
    secrets = [v for k, v in env.items() if v and ("KEY" in k or "TOKEN" in k or k == "CLOUDFLARE_ACCOUNT_ID")]
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for name in names or list(CANDIDATES):
        row = probe(name, env, secrets)
        row["probed_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        text = json.dumps(row, ensure_ascii=False)
        assert not any(s in text for s in secrets), "a secret reached the probe output"
        results[name] = row
        print(name, json.dumps({k: row.get(k) for k in ("model_listed", "status", "latency_s", "message_fields",
                                                        "reasoning_field_chars", "content_starts_with_think_tag",
                                                        "finish_reason", "usage", "rate_limit_headers", "error",
                                                        "error_body", "content")}, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unknown = [n for n in sys.argv[1:] if n not in CANDIDATES]
    if unknown:
        sys.exit(f"unknown candidate(s) {unknown}; choose from {list(CANDIDATES)}")
    main(sys.argv[1:])
