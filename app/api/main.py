"""FastAPI deployment (spec Task 11) with JSONL request logging (spec Task 12).

Every entry point calls `SupportAssistant.ask`, so PII masking, injection blocking, session memory, the CrewAI
crew, schema validation and the groundedness guardrail all apply over HTTP and WebSocket alike.

Index: at startup the app builds its own Chroma index in a temporary directory from the KB (both strategies).
`POST /add-document` adds to that index only: additions last until the process restarts, and the approved
`chroma_db/` collections used for the Phase 3 evidence are never modified.

Run: .venv\\Scripts\\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import asyncio
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
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.schemas import (
    AddDocumentRequest, AddDocumentResponse, AskRequest, AskResponse, ErrorBody, ErrorResponse, WsQuestion,
)
from app.api.schemas import SESSION_ID_PATTERN
from app.memory.session_memory import has_session
from app.observability.logging import DEFAULT_LOG_PATH, JsonlLogger, new_trace_id
from app.pipeline import SupportAssistant
from app.rag.chunking import STRATEGIES
from app.rag.documents import KBDocument, load_knowledge_base
from app.rag.generation import RECOMMENDED_STRATEGY
from app.rag.vector_store import get_collection, index_all, index_documents
from app.tools.crew_tools import RagLookupTool

CONFLICT, INTERNAL = "conflict", "internal_error"


def _safe_validation_details(errors: list[dict]) -> list[dict]:
    # pydantic's default error includes the raw `input` value, which could carry PII: drop it.
    return [{"loc": [str(p) for p in e.get("loc", ())], "type": e.get("type"), "msg": e.get("msg")} for e in errors]


def _error(trace_id: str, code: int, kind: str, message: str, details: list[dict] | None = None) -> JSONResponse:
    body = ErrorResponse(trace_id=trace_id, error=ErrorBody(type=kind, message=message, details=details or []))
    return JSONResponse(body.model_dump(), status_code=code, headers={"X-Trace-Id": trace_id})


def create_app(log_path: Path = DEFAULT_LOG_PATH, index_dir: Path | None = None) -> FastAPI:
    logger = JsonlLogger(log_path)

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

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest, request: Request) -> AskResponse:
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
                new_session = not has_session(session_id)
                result = await run_in_threadpool(websocket.app.state.assistant.ask, session_id, question)
                reply = AskResponse(**result.model_dump(), trace_id=trace_id, new_session=new_session)
                await websocket.send_json(reply.model_dump())
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


def _guardrail_flags(result) -> dict:
    g = result.guardrails
    return {"pii_masked": g.pii_masked, "injection_detected": g.injection_detected, "refused": g.refused,
            "grounded": g.grounded, "rejected_order": g.rejected_order, "schema_valid": g.schema_valid}


app = create_app()
