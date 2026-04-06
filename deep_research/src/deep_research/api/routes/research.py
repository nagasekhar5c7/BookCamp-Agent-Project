"""Research job endpoints: create, poll, review, download.

This module is the public face of the system. It owns nothing — it only:

1. Validates incoming requests (via Pydantic schemas).
2. Delegates state changes to :class:`~deep_research.services.job_store.JobStore`.
3. Delegates background execution to :mod:`deep_research.workers.runner`.
4. Shapes the response using the schemas in
   :mod:`deep_research.api.schemas`.

All business logic (planning, searching, synthesis, document generation)
lives behind the worker/graph layer. Keep it that way — the API layer
should stay thin and testable.
"""

from __future__ import annotations

from pathlib import Path

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse

from deep_research.api.dependencies import (
    get_job_store,
    get_settings,
    limiter,
)
from deep_research.api.schemas import (
    ApproveDecisionRequest,
    CreateResearchRequest,
    CreateResearchResponse,
    EditDecisionRequest,
    JobStatusResponse,
    PlanReviewResponse,
    ProgressView,
    RejectDecisionRequest,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
)
from deep_research.config import Settings
from deep_research.services.job_store import (
    ApproveDecision,
    EditDecision,
    Job,
    JobNotFoundError,
    JobStatus,
    JobStore,
    RejectDecision,
    ReviewDecision,
)
from deep_research.workers import runner

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/research", tags=["research"])


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _load_job_or_404(job_id: str, store: JobStore) -> Job:
    """Fetch a job from the store, translating absence into a 404."""
    try:
        return store.get(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        ) from exc


def _job_to_status_response(job: Job) -> JobStatusResponse:
    """Project a :class:`Job` onto the public :class:`JobStatusResponse`."""
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        progress=ProgressView(
            stage=job.progress.stage,
            task=job.progress.task,
            total=job.progress.total,
        ),
        cost_so_far_usd=job.cost_so_far_usd,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


# --------------------------------------------------------------------------- #
# POST /research — create a new job                                           #
# --------------------------------------------------------------------------- #


@router.post(
    "",
    response_model=CreateResearchResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Enqueue a new deep research job",
)
@limiter.limit(lambda: get_settings().rate_limit_post_research)
def create_research(
    request: Request,  # noqa: ARG001 — required by slowapi for keying
    body: CreateResearchRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
    settings: Settings = Depends(get_settings),
) -> CreateResearchResponse:
    """Create a job, kick off the pipeline in the background, and return 202."""
    job = store.create(query=body.query)
    log.info("research_job_created", job_id=job.job_id, query_len=len(body.query))

    background_tasks.add_task(
        runner.run_job,
        job.job_id,
        store=store,
        settings=settings,
    )

    return CreateResearchResponse(job_id=job.job_id, status=job.status)


# --------------------------------------------------------------------------- #
# GET /research/{job_id} — poll status                                        #
# --------------------------------------------------------------------------- #


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get the current status of a research job",
)
def get_research_status(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> JobStatusResponse:
    """Return current status, progress, cost, and any error."""
    job = _load_job_or_404(job_id, store)
    return _job_to_status_response(job)


# --------------------------------------------------------------------------- #
# GET /research/{job_id}/review — fetch the pending plan                      #
# --------------------------------------------------------------------------- #


@router.get(
    "/{job_id}/review",
    response_model=PlanReviewResponse,
    summary="Fetch the Lead's plan awaiting human review",
)
def get_pending_review(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> PlanReviewResponse:
    """Return the pending plan or 409 if the job is not awaiting review."""
    job = _load_job_or_404(job_id, store)

    if job.status != JobStatus.AWAITING_REVIEW or job.pending_plan is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} is not awaiting review "
                f"(current status: {job.status.value})"
            ),
        )

    return PlanReviewResponse(job_id=job.job_id, plan=job.pending_plan)


# --------------------------------------------------------------------------- #
# POST /research/{job_id}/review — submit approve / edit / reject             #
# --------------------------------------------------------------------------- #


@router.post(
    "/{job_id}/review",
    response_model=ReviewDecisionResponse,
    summary="Approve, edit, or reject the Lead's plan",
)
@limiter.limit(lambda: get_settings().rate_limit_post_review)
def submit_review_decision(
    request: Request,  # noqa: ARG001 — required by slowapi for keying
    job_id: str,
    body: ReviewDecisionRequest,
    background_tasks: BackgroundTasks,
    store: JobStore = Depends(get_job_store),
    settings: Settings = Depends(get_settings),
) -> ReviewDecisionResponse:
    """Resume the paused graph with the user's decision.

    On approve/edit we schedule :func:`runner.resume_job` as a background
    task so the HTTP request returns immediately. On reject the runner
    itself short-circuits to ``CANCELLED_BY_USER`` without touching the
    graph, but we still run it through the background task for a uniform
    code path.
    """
    job = _load_job_or_404(job_id, store)

    if job.status != JobStatus.AWAITING_REVIEW:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} is not awaiting review "
                f"(current status: {job.status.value})"
            ),
        )

    # Translate the api-layer request into the service-layer decision
    # union the runner expects. Kept as a separate hop so the wire schema
    # can evolve independently of the internal domain model.
    service_decision: ReviewDecision = _to_service_decision(body)

    # Flip the visible status up-front so a subsequent GET never sees a
    # stale 'awaiting_review' after the user has submitted a decision.
    if isinstance(body, RejectDecisionRequest):
        # Short-circuit: no further graph execution needed. The runner
        # will record the cancellation state; we leave status untouched
        # here so a concurrent poller can still see 'awaiting_review'
        # until the background task flips it.
        pass
    else:
        store.update_status(job_id, JobStatus.RESEARCHING)
        store.clear_pending_plan(job_id)

    log.info(
        "research_review_submitted",
        job_id=job_id,
        decision=body.decision,
    )

    background_tasks.add_task(
        runner.resume_job,
        job_id,
        service_decision,
        store=store,
        settings=settings,
    )

    # Re-read so the response reflects the status we just set.
    refreshed = store.get(job_id)
    return ReviewDecisionResponse(job_id=refreshed.job_id, status=refreshed.status)


# --------------------------------------------------------------------------- #
# GET /research/{job_id}/document — download the .docx                        #
# --------------------------------------------------------------------------- #


_DOCX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)


@router.get(
    "/{job_id}/document",
    summary="Download the generated .docx report",
    responses={
        200: {
            "content": {_DOCX_MEDIA_TYPE: {}},
            "description": "The generated Word document.",
        },
        404: {"description": "Job not found."},
        409: {"description": "Job is not yet done."},
    },
)
def download_document(
    job_id: str,
    store: JobStore = Depends(get_job_store),
) -> FileResponse:
    """Stream the final ``.docx`` once the job has completed successfully."""
    job = _load_job_or_404(job_id, store)

    if job.status != JobStatus.DONE or job.document_path is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id} is not ready for download "
                f"(current status: {job.status.value})"
            ),
        )

    path = Path(job.document_path)
    if not path.is_file():
        # Invariant violation — status says done but file is missing.
        log.error("document_missing", job_id=job_id, path=str(path))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Generated document is missing on disk",
        )

    return FileResponse(
        path=path,
        media_type=_DOCX_MEDIA_TYPE,
        filename=path.name,
    )


# --------------------------------------------------------------------------- #
# Schema → service-layer decision translation                                 #
# --------------------------------------------------------------------------- #


def _to_service_decision(body: ReviewDecisionRequest) -> ReviewDecision:
    """Convert an api-layer request into the internal decision union.

    The two unions are structurally identical today, but keeping them
    distinct protects us from accidentally leaking API-shaped types into
    the worker/graph layer.
    """
    if isinstance(body, ApproveDecisionRequest):
        return ApproveDecision(decision="approve")
    if isinstance(body, EditDecisionRequest):
        return EditDecision(decision="edit", plan=list(body.plan))
    if isinstance(body, RejectDecisionRequest):
        return RejectDecision(decision="reject")
    # Unreachable: FastAPI validation would have rejected the request.
    raise TypeError(f"Unknown review decision type: {type(body).__name__}")
