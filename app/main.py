"""FastAPI application: POST /chat and GET /health (TASK.pdf section 9).

Startup runs ingestion in a background thread. /health returns 503 until it has finished
successfully, and /chat returns 503 until then too.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool

from app.api.schemas import ChatRequest, ChatResponse, CitationOut, ToolCallOut
from app.chat.models import ChatResult
from app.chat.router import handle_message
from app.chat.services import Services
from app.config import Settings, app_today
from app.db import state_repo
from app.db.employees import get_requester
from app.ingestion.errors import IngestionError
from app.ingestion.pipeline import run_ingestion
from app.llm.provider import LLMConfigError, make_provider
from app.retrieval.embeddings import SentenceTransformerEmbedder
from app.security.pii import install_log_sanitizer

logger = logging.getLogger("app.api")
REQUEST_BUDGET_SECONDS = 30.0
SAFE_ERROR_ANSWER = "Something went wrong while handling your request. Please try again."


def create_app(
    settings: Settings | None = None,
    embedder=None,
    provider=None,
    start_background: bool = True,
    services: Services | None = None,
) -> FastAPI:
    """`services` lets callers (tests) supply an already-built index and database paths and skip
    ingestion. In production it is None and ingestion runs at startup."""
    settings = settings or Settings.from_env()
    app = FastAPI(title="Veltrix PeopleOps Assistant")
    app.state.settings = settings
    app.state.services = services
    app.state.startup_error = None
    app.state.ready_event = threading.Event()

    def _build(run_now: bool) -> None:
        try:
            logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
            install_log_sanitizer()
            state_repo.init_state_db(settings.state_db)
            chat_provider = provider or make_provider(
                settings.llm_provider,
                settings.groq_api_key,
                settings.groq_model,
                settings.llm_timeout_seconds,
                gemini_api_key=settings.gemini_api_key,
                gemini_model=settings.gemini_model,
            )
            if not settings.db_path.is_file():
                raise IngestionError(f"database not found at {settings.db_path}")
            chosen_embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model)
            report = run_ingestion(settings, embedder=chosen_embedder)
            app.state.services = Services(
                settings=settings,
                index=report.index,
                embedder=chosen_embedder,
                provider=chat_provider,
                db_path=settings.db_path,
                store_path=settings.vector_dir / "ingest.db",
                state_db=settings.state_db,
            )
            logger.info("service ready documents=%d chunks=%d provider=%s", report.documents, report.chunks, chat_provider.name)
        except (IngestionError, LLMConfigError) as exc:
            app.state.startup_error = str(exc)
            logger.error("startup failed: %s", exc)
        except Exception:
            app.state.startup_error = "unexpected startup error"
            logger.exception("startup failed")
        finally:
            app.state.ready_event.set()

    @app.on_event("startup")
    def _startup() -> None:
        if app.state.services is not None:
            install_log_sanitizer()
            app.state.ready_event.set()
            return
        if start_background:
            threading.Thread(target=_build, args=(False,), daemon=True).start()
        else:
            _build(True)

    @app.get("/health")
    def health() -> JSONResponse:
        if app.state.services is None:
            return JSONResponse(status_code=503, content={"status": "unavailable"})
        return JSONResponse(status_code=200, content={"status": "ok"})

    @app.post("/chat")
    async def chat(
        body: ChatRequest,
        x_employee_id: str | None = Header(default=None, alias="X-Employee-Id", description="Employee ID of the logged-in user"),
    ) -> JSONResponse:
        started = time.monotonic()
        parsed = body
        employee_id = (x_employee_id or "").strip()
        if not employee_id:
            return JSONResponse(status_code=401, content={"detail": "X-Employee-Id header is required"})

        services: Services | None = app.state.services
        if services is None:
            return JSONResponse(status_code=503, content={"detail": "service is starting"})

        requester = get_requester(settings.db_path, employee_id)
        if requester is None:
            return JSONResponse(status_code=401, content={"detail": "unknown employee"})

        now = datetime.now()
        owner = state_repo.claim_session(settings.state_db, parsed.session_id, requester.employee_id, now)
        session_tag = hashlib.sha256(parsed.session_id.encode("utf-8")).hexdigest()[:10]
        if owner != requester.employee_id:
            logger.warning("session ownership violation employee=%s session=%s", requester.employee_id, session_tag)
            return JSONResponse(status_code=403, content={"detail": "this session belongs to another employee"})

        request_id = str(uuid.uuid4())
        try:
            result: ChatResult = await run_in_threadpool(
                handle_message, services, requester, parsed.session_id, parsed.message, app_today(), now
            )
        except ValueError:
            logger.exception("invalid chat result request_id=%s", request_id)
            result = ChatResult("no_answer", SAFE_ERROR_ANSWER)
        except Exception:
            logger.exception("chat failed request_id=%s", request_id)
            result = ChatResult("no_answer", SAFE_ERROR_ANSWER)

        elapsed = time.monotonic() - started
        logger.info(
            "chat request_id=%s employee=%s session=%s route=%s tools=%s elapsed=%.2fs",
            request_id,
            requester.employee_id,
            session_tag,
            result.route,
            [c.name for c in result.tool_calls],
            elapsed,
        )
        response = ChatResponse(
            request_id=request_id,
            answer=result.answer,
            route=result.route,
            citations=[CitationOut(doc_id=c.doc_id, locator=c.locator) for c in result.citations],
            tool_calls=[ToolCallOut(name=c.name, args=c.args, status=c.status) for c in result.tool_calls],
            pending_confirmation=result.pending_confirmation,
        )
        return JSONResponse(status_code=200, content=response.model_dump())

    @app.exception_handler(ValidationError)
    @app.exception_handler(RequestValidationError)
    async def _validation(_request: Request, _exc: Exception) -> JSONResponse:
        # Invalid or missing body is always a 400 (TASK.pdf section 9), never FastAPI's default 422.
        return JSONResponse(status_code=400, content={"detail": "invalid request body"})

    return app


app = create_app()
