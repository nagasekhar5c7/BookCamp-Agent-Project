"""FastAPI application factory.

Everything that touches the framework lives here: app construction,
middleware, exception handlers, router mounting, and rate-limiter wiring.
By keeping this in a ``create_app`` factory (rather than a module-level
``app``) we can instantiate the app cleanly inside tests with different
dependency overrides.

A module-level ``app`` is still exposed at the bottom for the common
``uvicorn deep_research.api.main:app`` invocation used by
``scripts/run_local.py``.
"""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from deep_research.api.dependencies import limiter
from deep_research.api.routes import health, research
from deep_research.api.schemas import ErrorResponse
from deep_research.services.job_store import (
    InvalidJobStateError,
    JobNotFoundError,
)

log = structlog.get_logger(__name__)


def create_app() -> FastAPI:
    """Build and return a fully-configured FastAPI application."""
    app = FastAPI(
        title="Deep Research",
        description=(
            "Multi-agent deep research service. Submit a plain-text query "
            "and retrieve a fully cited Word document."
        ),
        version="0.1.0",
    )

    # --- Rate limiting -----------------------------------------------------
    # slowapi requires both: (a) the limiter attached to app.state so the
    # middleware can locate it, and (b) an exception handler for
    # RateLimitExceeded so the client gets a clean 429.
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # --- Exception handlers ------------------------------------------------
    app.add_exception_handler(JobNotFoundError, _handle_job_not_found)
    app.add_exception_handler(InvalidJobStateError, _handle_invalid_job_state)
    app.add_exception_handler(RequestValidationError, _handle_validation_error)

    # --- Routers -----------------------------------------------------------
    app.include_router(health.router)
    app.include_router(research.router)

    log.info("fastapi_app_created")
    return app


# --------------------------------------------------------------------------- #
# Exception handlers                                                          #
# --------------------------------------------------------------------------- #


def _handle_job_not_found(
    request: Request,  # noqa: ARG001 — FastAPI handler signature
    exc: JobNotFoundError,
) -> JSONResponse:
    """Translate :class:`JobNotFoundError` into a 404 with our error envelope."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error="job_not_found",
            detail=str(exc) or "The requested job does not exist.",
            job_id=str(exc) if str(exc) else None,
        ).model_dump(),
    )


def _handle_invalid_job_state(
    request: Request,  # noqa: ARG001 — FastAPI handler signature
    exc: InvalidJobStateError,
) -> JSONResponse:
    """Translate :class:`InvalidJobStateError` into a 409 Conflict."""
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content=ErrorResponse(
            error="invalid_job_state",
            detail=str(exc),
        ).model_dump(),
    )


def _handle_validation_error(
    request: Request,  # noqa: ARG001 — FastAPI handler signature
    exc: RequestValidationError,
) -> JSONResponse:
    """Normalise Pydantic validation errors into our error envelope."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="validation_error",
            detail="Request body failed validation.",
        ).model_dump()
        | {"errors": exc.errors()},
    )


# Module-level app for ``uvicorn deep_research.api.main:app``.
app = create_app()
