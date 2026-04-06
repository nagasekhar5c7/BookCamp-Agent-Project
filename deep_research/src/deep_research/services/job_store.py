"""In-memory job store and job lifecycle model.

The :class:`JobStore` abstraction owns every piece of state that survives
*outside* a single LangGraph invocation — job status, accumulated cost,
the pending plan while we wait for human review, the final document path,
and error messages. The graph pushes updates into the store at stage
boundaries; the FastAPI layer reads from it to answer polling requests.

**v1 scope**: single-process, in-memory dict with a threading lock. Good
enough for local runs. A Redis-backed implementation can slot in later by
subclassing :class:`JobStore` — the FastAPI dependency is the only place
that needs to change.

See ``ideas.md`` §9 for the lifecycle state machine.
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from deep_research.models.plan import SubTask


# --------------------------------------------------------------------------- #
# Public domain types                                                         #
# --------------------------------------------------------------------------- #


class JobStatus(str, Enum):
    """Job lifecycle states (see ideas.md §9).

    Terminal states are :attr:`DONE`, :attr:`FAILED`, and
    :attr:`CANCELLED_BY_USER`. All others are transient.
    """

    QUEUED = "queued"
    PLANNING = "planning"
    AWAITING_REVIEW = "awaiting_review"
    RESEARCHING = "researching"
    SYNTHESIZING = "synthesizing"
    GENERATING_DOCUMENT = "generating_document"
    DONE = "done"
    FAILED = "failed"
    CANCELLED_BY_USER = "cancelled_by_user"


TERMINAL_STATUSES: frozenset[JobStatus] = frozenset(
    {JobStatus.DONE, JobStatus.FAILED, JobStatus.CANCELLED_BY_USER}
)


class JobProgress(BaseModel):
    """Fine-grained progress within the current stage."""

    stage: str = Field(default="queued")
    task: int | None = Field(
        default=None,
        description="1-based index of the currently running sub-task, if any.",
    )
    total: int | None = Field(
        default=None,
        description="Total number of sub-tasks in the approved plan.",
    )


class Job(BaseModel):
    """Everything a caller can observe about a research job.

    This is intentionally denormalised into one flat object so the
    ``GET /research/{id}`` endpoint can return it with minimal shaping.
    """

    job_id: str
    query: str
    status: JobStatus = JobStatus.QUEUED
    progress: JobProgress = Field(default_factory=JobProgress)
    cost_so_far_usd: float = 0.0
    error: str | None = None

    # Populated when status == AWAITING_REVIEW; cleared once the user
    # submits their decision.
    pending_plan: list[SubTask] | None = None

    # Populated when status == DONE. Absolute path on the local filesystem
    # (v1 is local-only; remote storage would replace this with a URL).
    document_path: str | None = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# Review decision payload                                                     #
# --------------------------------------------------------------------------- #


class ApproveDecision(BaseModel):
    decision: Literal["approve"]


class EditDecision(BaseModel):
    decision: Literal["edit"]
    plan: list[SubTask] = Field(..., min_length=1)


class RejectDecision(BaseModel):
    decision: Literal["reject"]


ReviewDecision = ApproveDecision | EditDecision | RejectDecision


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #


class JobNotFoundError(LookupError):
    """Raised when a requested job_id does not exist in the store."""


class InvalidJobStateError(RuntimeError):
    """Raised when an operation is incompatible with the job's current status.

    Examples: fetching the review plan when the job is not awaiting review,
    downloading a document before the job is done.
    """


# --------------------------------------------------------------------------- #
# Store                                                                       #
# --------------------------------------------------------------------------- #


class JobStore:
    """Thread-safe in-memory job store.

    All mutating methods take the internal lock. Reads that return a copy
    of the job also take the lock briefly so a caller never observes a
    partially-updated record.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    # -- Lifecycle ----------------------------------------------------------

    def create(self, query: str) -> Job:
        """Create a new job in :attr:`JobStatus.QUEUED` state."""
        job = Job(job_id=str(uuid.uuid4()), query=query)
        with self._lock:
            self._jobs[job.job_id] = job
        return job.model_copy(deep=True)

    def get(self, job_id: str) -> Job:
        """Return a deep copy of the job or raise :class:`JobNotFoundError`."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return job.model_copy(deep=True)

    # -- Mutations ----------------------------------------------------------

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        error: str | None = None,
    ) -> Job:
        """Transition the job to a new status (and optionally attach an error)."""
        with self._lock:
            job = self._require(job_id)
            job.status = status
            job.progress.stage = status.value
            if error is not None:
                job.error = error
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def set_progress(
        self,
        job_id: str,
        *,
        stage: str | None = None,
        task: int | None = None,
        total: int | None = None,
    ) -> Job:
        """Update the fine-grained progress fields in place."""
        with self._lock:
            job = self._require(job_id)
            if stage is not None:
                job.progress.stage = stage
            if task is not None:
                job.progress.task = task
            if total is not None:
                job.progress.total = total
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def add_cost(self, job_id: str, delta_usd: float) -> Job:
        """Increment the running cost estimate."""
        with self._lock:
            job = self._require(job_id)
            job.cost_so_far_usd = round(job.cost_so_far_usd + delta_usd, 6)
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def set_pending_plan(self, job_id: str, plan: list[SubTask]) -> Job:
        """Attach the Lead's plan and flip status to :attr:`AWAITING_REVIEW`."""
        with self._lock:
            job = self._require(job_id)
            job.pending_plan = list(plan)
            job.status = JobStatus.AWAITING_REVIEW
            job.progress.stage = JobStatus.AWAITING_REVIEW.value
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def clear_pending_plan(self, job_id: str) -> Job:
        """Drop the pending plan (called when review is resolved)."""
        with self._lock:
            job = self._require(job_id)
            job.pending_plan = None
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    def set_document_path(self, job_id: str, path: str) -> Job:
        """Record the final .docx path and mark the job :attr:`DONE`."""
        with self._lock:
            job = self._require(job_id)
            job.document_path = path
            job.status = JobStatus.DONE
            job.progress.stage = JobStatus.DONE.value
            job.updated_at = datetime.now(timezone.utc)
            return job.model_copy(deep=True)

    # -- Internals ----------------------------------------------------------

    def _require(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFoundError(job_id)
        return job
