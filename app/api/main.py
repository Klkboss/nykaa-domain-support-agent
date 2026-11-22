"""FastAPI deployment (spec Task 11) with JSONL request logging (spec Task 12).

Every entry point calls `SupportAssistant.ask`, so PII masking, injection blocking, session memory, the CrewAI
crew, schema validation and the groundedness guardrail all apply over HTTP and WebSocket alike.

Index: at startup the app builds its own Chroma index in a temporary directory from the KB (both strategies).
`POST /add-document` adds to that index only: additions last until the process restarts, and the approved
`chroma_db/` collections used for the Phase 3 evidence are never modified.

Run: .venv\\Scripts\\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

v2: GET / serves the chat page (app/web), GET /health reports liveness and the active provider, and a per-client
rate limit (app/api/rate_limit.py, off unless configured) protects a public demo. /ws/chat gains two OPT-IN query
parameters for the chat page: progress=1 sends {"type": "progress", "step": ...} events while a question runs, and
explain=1 adds an "explain" object to each reply. Without them every message is exactly as before.
"""

from __future__ import annotations

import asyncio
import math
import re
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from app.api.schemas import (
    AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse, ErrorBody, ErrorResponse, WsQuestion,
)
from app.api.rate_limit import RateLimiter
from app.api.schemas import SESSION_ID_PATTERN, Explanation
from app.db import database_url
from app.llm.settings import LLMSettings
from app.memory.session_memory import has_session
from app.observability.logging import DEFAULT_LOG_PATH, JsonlLogger, new_trace_id
from app.pipeline import RequestTrace, SupportAssistant
from app.rag.chunking import STRATEGIES
from app.rag.documents import KBDocument, load_knowledge_base
from app.rag.generation import RECOMMENDED_STRATEGY
from app.rag.vector_store import get_collection, index_all, index_documents
from app.core.config import ROOT
from app.schemas import SupportResponse
from app.tools.crew_tools import RagLookupTool

CONFLICT, INTERNAL, RATE_LIMITED = "conflict", "internal_error", "rate_limited"
WEB_DIR = ROOT / "app" / "web"


def _safe_validation_details(errors: list[dict]) -> list[dict]:
    # pydantic's default error includes the raw `input` value, which could carry PII: drop it.
    return [{"loc": [str(p) for p in e.get("loc", ())], "type": e.get("type"), "msg": e.get("msg")} for e in errors]


def _error(trace_id: str, code: int, kind: str, message: str, details: list[dict] | None = None) -> JSONResponse:
    body = ErrorResponse(trace_id=trace_id, error=ErrorBody(type=kind, message=message, details=details or []))
    return JSONResponse(body.model_dump(), status_code=code, headers={"X-Trace-Id": trace_id})


def _rate_limited(trace_id: str, retry_after: float) -> JSONResponse:
    seconds = math.ceil(retry_after)
    response = _error(trace_id, 429, RATE_LIMITED, f"Too many questions. Please try again in {seconds} s.")
    response.headers["Retry-After"] = str(seconds)
    return response


def _explain(result: SupportResponse, trace: RequestTrace, settings: LLMSettings, latency_ms: float) -> Explanation:
    """The chat UI's "How this answer was made", from this request's own evidence (never re-computed)."""
    run, review, draft = trace.crew_run, trace.review, result.draft
    rag = (run.rag_outputs or [{}])[-1] if run else {}
    return Explanation(
        cited_sources=sorted({d for c in (draft.policy_claims if draft else []) for d in c.source_doc_ids}),
        retrieved_sources=list(rag.get("sources", [])),
        tools=[*(["rag_lookup"] * len(run.rag_invocations)), *(["check_order_status"] * len(run.order_invocations))]
        if run else [],
        grounded=result.guardrails.grounded,
        rejected_claims=len(result.guardrails.rejected_claims),
        # Phase 8 rule: decide on whether the text changed, not on the reviewer's `approved` flag.
        review="not_run" if review is None else "revised" if review.verdict.final_answer != review.draft
        else "approved",
        escalation_recommended=draft.order.escalation_recommended if draft and draft.order else None,
        injection_categories=result.guardrails.injection_categories,
        provider=settings.provider,
        model=settings.model if settings.is_real else "deterministic mock",
        latency_ms=latency_ms,
        tokens=trace.budget.used if trace.budget else 0,
    )


def create_app(log_path: Path = DEFAULT_LOG_PATH, index_dir: Path | None = None,
               rate_limiter: RateLimiter | None = None) -> FastAPI:
    logger = JsonlLogger(log_path)
    limiter = rate_limiter or RateLimiter.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        tmp = Path(index_dir) if index_dir else Path(tempfile.mkdtemp(prefix="nykaa_api_index_"))
        client = chromadb.PersistentClient(path=str(tmp), settings=Settings(anonymized_telemetry=False))
        docs = load_knowledge_base()
        index_all(client, docs)
        collection = get_collection(client, RECOMMENDED_STRATEGY)
        app.state.client, app.state.doc_ids = client, {d.doc_id for d in docs}
        app.state.assistant = SupportAssistant(rag_tool_factory=lambda: RagLookupTool(collection=collection))
        logger.log(type="lifecycle", event="startup", documents=len(docs), index="temporary per-process")
        yield
        logger.log(type="lifecycle", event="shutdown")
        if index_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)

    app = FastAPI(title="Nykaa Support Agent (simulated policies)", version="1.0", lifespan=lifespan)
    app.state.logger = logger

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        trace_id, start = new_trace_id(), time.perf_counter()
        request.state.trace_id, request.state.log = trace_id, {}
        try:
            response = await call_next(request)
        except Exception as e:  # never echo the exception message: it may contain user input
            request.state.log["error_type"] = type(e).__name__
            response = _error(trace_id, 500, INTERNAL, "An internal error occurred. Please try again.")
        response.headers["X-Trace-Id"] = trace_id
        code = response.status_code
        logger.log(type="http", trace_id=trace_id, method=request.method, endpoint=request.url.path, status_code=code,
                   outcome="ok" if code < 400 else "client_error" if code < 500 else "server_error",
                   duration_ms=round((time.perf_counter() - start) * 1000, 2), **request.state.log)
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        details = _safe_validation_details(exc.errors())
        request.state.log.update(error_type="validation_error", error_fields=[".".join(d["loc"]) for d in details])
        return _error(request.state.trace_id, 422, "validation_error", "The request is invalid.", details)

    @app.get("/", include_in_schema=False)
    async def chat_page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

    @app.get("/health")
    async def health(request: Request) -> dict:
        settings = request.app.state.assistant.llm_settings
        return {"status": "ok", "provider": settings.provider,
                "model": settings.model if settings.is_real else "deterministic mock",
                "database": "postgresql" if database_url() else "in-memory",
                "rate_limit": {"per_minute": limiter.per_minute, "per_day": limiter.per_day}}

    @app.post("/ask", response_model=AskResponse, responses={429: {"model": ErrorResponse}})
    async def ask(body: AskRequest, request: Request) -> AskResponse | JSONResponse:
        retry_after = limiter.check(request.client.host if request.client else "unknown")
        if retry_after is not None:
            request.state.log.update(error_type=RATE_LIMITED)
            return _rate_limited(request.state.trace_id, retry_after)
        session_id = body.session_id or f"s-{request.state.trace_id[:12]}"
        new_session = not has_session(session_id)
        result = await run_in_threadpool(request.app.state.assistant.ask, session_id, body.question)
        request.state.log.update(session_id=session_id, new_session=new_session, masked_question=result.masked_question,
                                 response_status=result.status, guardrails=_guardrail_flags(result))
        return AskResponse(**result.model_dump(), trace_id=request.state.trace_id, new_session=new_session)

    @app.post("/add-document", response_model=AddDocumentResponse, status_code=status.HTTP_201_CREATED,
              responses={409: {"model": ErrorResponse}, 422: {"model": ErrorResponse}})
    async def add_document(body: AddDocumentRequest, request: Request):
        trace_id = request.state.trace_id
        try:
            doc = KBDocument(**body.model_dump())  # the KB's own validation (2-5 sentences etc.)
        except ValidationError as e:
            details = _safe_validation_details(e.errors())
            request.state.log.update(error_type="validation_error", error_fields=[".".join(d["loc"]) for d in details])
            return _error(trace_id, 422, "validation_error", "The document is invalid.", details)
        if doc.doc_id in request.app.state.doc_ids:
            request.state.log.update(error_type=CONFLICT, doc_id=doc.doc_id)
            return _error(trace_id, 409, CONFLICT, f"Document {doc.doc_id} already exists.")
        counts = await run_in_threadpool(
            lambda: {s: index_documents(request.app.state.client, [doc], s) for s in STRATEGIES})
        request.app.state.doc_ids.add(doc.doc_id)
        request.app.state.assistant.generation_cache.clear()  # owner decision: a KB change clears the cache
        request.state.log.update(doc_id=doc.doc_id, topic=doc.topic, chunks_indexed=counts)
        return AddDocumentResponse(trace_id=trace_id, doc_id=doc.doc_id, chunks_indexed=counts)

    @app.websocket("/ws/chat")
    async def chat(websocket: WebSocket):
        session_id = websocket.query_params.get("session_id") or f"ws-{new_trace_id()[:12]}"
        connection_id = new_trace_id()[:12]
        progress, explain = (websocket.query_params.get(k) == "1" for k in ("progress", "explain"))  # v2 opt-ins
        client = websocket.client.host if websocket.client else "unknown"
        if not re.fullmatch(SESSION_ID_PATTERN, session_id):
            logger.log(type="websocket", event="rejected", endpoint="/ws/chat", connection_id=connection_id,
                       outcome="client_error", error_type="invalid_session_id")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        await websocket.accept()
        logger.log(type="websocket", event="connect", endpoint="/ws/chat", connection_id=connection_id,
                   session_id=session_id, outcome="ok")
        try:
            while True:
                raw = await websocket.receive_text()
                trace_id, start = new_trace_id(), time.perf_counter()
                try:
                    question = WsQuestion.model_validate_json(raw).question
                except ValidationError as e:
                    details = _safe_validation_details(e.errors())
                    await websocket.send_json(ErrorResponse(trace_id=trace_id, error=ErrorBody(
                        type="validation_error", message="Send {\"question\": \"...\"}.", details=details)).model_dump())
                    logger.log(type="websocket", event="message", endpoint="/ws/chat", trace_id=trace_id,
                               connection_id=connection_id, session_id=session_id, outcome="client_error",
                               error_type="validation_error", duration_ms=round((time.perf_counter() - start) * 1000, 2))
                    continue
                retry_after = limiter.check(client)
                if retry_after is not None:
                    await websocket.send_json(ErrorResponse(trace_id=trace_id, error=ErrorBody(
                        type=RATE_LIMITED, message=f"Too many questions. Please try again in "
                                                   f"{math.ceil(retry_after)} s.")).model_dump())
                    logger.log(type="websocket", event="message", endpoint="/ws/chat", trace_id=trace_id,
                               connection_id=connection_id, session_id=session_id, outcome="client_error",
                               error_type=RATE_LIMITED, duration_ms=round((time.perf_counter() - start) * 1000, 2))
                    continue
                new_session = not has_session(session_id)
                result, trace = await _ask_with_progress(websocket, session_id, question, trace_id, progress)
                reply = AskResponse(**result.model_dump(), trace_id=trace_id, new_session=new_session).model_dump()
                if explain:
                    reply["explain"] = _explain(result, trace, websocket.app.state.assistant.llm_settings,
                                                round((time.perf_counter() - start) * 1000, 2)).model_dump()
                await websocket.send_json(reply)
                logger.log(type="websocket", event="message", endpoint="/ws/chat", trace_id=trace_id,
                           connection_id=connection_id, session_id=session_id, outcome="ok",
                           masked_question=result.masked_question, response_status=result.status,
                           guardrails=_guardrail_flags(result), duration_ms=round((time.perf_counter() - start) * 1000, 2))
        except WebSocketDisconnect as e:
            # The client went away (possibly mid-conversation): log it and end only this connection.
            logger.log(type="websocket", event="disconnect", endpoint="/ws/chat", connection_id=connection_id,
                       session_id=session_id, close_code=e.code, reason="client_disconnected", outcome="ok")
        except asyncio.CancelledError:
            # The server cancelled this handler because its client vanished while a turn was in progress.
            logger.log(type="websocket", event="disconnect", endpoint="/ws/chat", connection_id=connection_id,
                       session_id=session_id, reason="cancelled_mid_turn", outcome="ok")
            raise
        except Exception as e:  # any other failure ends this connection only, with a safe log line
            logger.log(type="websocket", event="error", endpoint="/ws/chat", connection_id=connection_id,
                       session_id=session_id, outcome="server_error", error_type=type(e).__name__)
            try:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            except Exception:
                pass  # the socket may already be gone

    return app


async def _ask_with_progress(websocket: WebSocket, session_id: str, question: str, trace_id: str,
                             progress: bool) -> tuple[SupportResponse, RequestTrace]:
    """Run one question in the thread pool; with progress=1, forward each pipeline stage as it really starts."""
    loop, steps = asyncio.get_running_loop(), asyncio.Queue()
    trace = RequestTrace(on_step=(lambda step: loop.call_soon_threadsafe(steps.put_nowait, step)) if progress else None)
    task = asyncio.ensure_future(run_in_threadpool(websocket.app.state.assistant.ask, session_id, question, trace))
    while progress and (not task.done() or not steps.empty()):
        getter = asyncio.ensure_future(steps.get())
        await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
        if getter.done():
            await websocket.send_json({"type": "progress", "step": getter.result(), "trace_id": trace_id})
        else:
            getter.cancel()
    return await task, trace


def _guardrail_flags(result) -> dict:
    g = result.guardrails
    return {"pii_masked": g.pii_masked, "injection_detected": g.injection_detected, "refused": g.refused,
            "grounded": g.grounded, "rejected_order": g.rejected_order, "schema_valid": g.schema_valid}


app = create_app()
