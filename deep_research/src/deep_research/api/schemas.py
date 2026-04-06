"""Request and response Pydantic models for the FastAPI layer.

These schemas are the **wire contract** — they are decoupled from the
internal domain models (``Job``, ``SubTask``, etc.) so we can evolve the
internal representation without breaking API clients. In practice some
shapes are identical today (e.g. :class:`PlanReviewResponse` just wraps
:class:`~deep_research.models.plan.SubTask`), and that is fine — the
indirection costs nothing and buys us forward compatibility.

Every response model has an explicit docstring and every field has a
``description`` so the auto-generated OpenAPI docs are useful out of the
box.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from deep_research.models.plan import SubTask
from deep_research.services.job_store import JobStatus


# --------------------------------------------------------------------------- #
# POST /research                                                              #
# --------------------------------------------------------------------------- #


class CreateResearchRequest(BaseModel):
    """Body of ``POST /research``."""

    query: str = Field(
        ...,
        description="The user's plain-text research question.",
        min_length=3,
        max_length=2000,
    )


class CreateResearchResponse(BaseModel):
    """Response returned when a new research job is enqueued."""

    job_id: str = Field(..., description="Opaque uuid identifying the job.")
    status: JobStatus = Field(
        ...,
        description="Initial status — always 'queued' immediately after creation.",
    )


# --------------------------------------------------------------------------- #
# GET /research/{job_id}                                                      #
# --------------------------------------------------------------------------- #


class ProgressView(BaseModel):
    """Fine-grained progress inside the current stage."""

    stage: str = Field(..., description="Current pipeline stage.")
    task: int | None = Field(
        default=None,
        description="1-based index of the currently running sub-task, if any.",
    )
    total: int | None = Field(
        default=None,
        description="Total number of sub-tasks in the approved plan, if known.",
    )


class JobStatusResponse(BaseModel):
    """Response for ``GET /research/{job_id}`` — the polling endpoint."""

    job_id: str
    status: JobStatus
    progress: ProgressView
    cost_so_far_usd: float = Field(
        ...,
        description="Running cost estimate in USD, checked against MAX_JOB_COST_USD.",
    )
    error: str | None = Field(
        default=None,
        description="Populated only when status == 'failed'.",
    )
    created_at: datetime
    updated_at: datetime


# --------------------------------------------------------------------------- #
# GET /research/{job_id}/review                                               #
# --------------------------------------------------------------------------- #


class PlanReviewResponse(BaseModel):
    """Response returned when the job is paused at the human-review gate."""

    job_id: str
    plan: list[SubTask] = Field(
        ...,
        description="The Lead's proposed sub-tasks, awaiting human approval.",
    )


# --------------------------------------------------------------------------- #
# POST /research/{job_id}/review                                              #
# --------------------------------------------------------------------------- #


class ApproveDecisionRequest(BaseModel):
    """Approve the Lead's plan as-is and proceed to research."""

    decision: Literal["approve"]


class EditDecisionRequest(BaseModel):
    """Substitute an edited plan before proceeding to research."""

    decision: Literal["edit"]
    plan: list[SubTask] = Field(
        ...,
        description="The user's edited sub-task list. Must be non-empty.",
        min_length=1,
    )


class RejectDecisionRequest(BaseModel):
    """Abandon the job. The graph transitions to 'cancelled_by_user'."""

    decision: Literal["reject"]


# Discriminated union: FastAPI dispatches on the ``decision`` field.
ReviewDecisionRequest = ApproveDecisionRequest | EditDecisionRequest | RejectDecisionRequest


class ReviewDecisionResponse(BaseModel):
    """Response returned after a review decision is accepted."""

    job_id: str
    status: JobStatus


# --------------------------------------------------------------------------- #
# GET /health                                                                 #
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    """Liveness response. Dependency checks live under ``checks``."""

    status: Literal["ok", "degraded"]
    version: str
    checks: dict[str, Literal["ok", "unknown"]] = Field(
        default_factory=dict,
        description="Per-dependency status (groq, tavily, ...).",
    )


# --------------------------------------------------------------------------- #
# Error envelope                                                              #
# --------------------------------------------------------------------------- #


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by all non-2xx responses."""

    error: str = Field(..., description="Short machine-readable error code.")
    detail: str = Field(..., description="Human-readable explanation.")
    job_id: str | None = Field(
        default=None,
        description="Included when the error is associated with a specific job.",
    )
