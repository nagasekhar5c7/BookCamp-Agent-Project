"""Background job runner — drives the LangGraph state machine.

This module is the single seam between the API layer and the graph.
The API never touches the graph directly; it only calls:

- :func:`run_job` — start a fresh job, drive the graph until it either
  pauses at the human-review interrupt or reaches a terminal state.
- :func:`resume_job` — hand a human review decision back to the paused
  graph so it can continue.

Both functions own their own error handling: they catch *any* exception,
log it with the bound ``job_id``, and mark the job as ``failed`` (or
``cancelled_by_user`` for rejections) in the store. This guarantees
that a crash inside a FastAPI ``BackgroundTask`` never leaves a job
stuck in a transient state.

**Compiled-graph lifetime**: the graph is built lazily once per process
via :func:`_get_graph` and reused across every job. The
:class:`~langgraph.checkpoint.memory.MemorySaver` attached at build time
is what lets :func:`resume_job` find the paused state for a given
``thread_id`` (which we set to the job id).
"""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.types import Command

from deep_research.config import Settings
from deep_research.graph import (
    CostCeilingExceededError,
    InsufficientResearchError,
    NodeDeps,
    ReviewRejectedError,
    build_graph,
    initial_state,
)
from deep_research.services.job_store import JobStatus, JobStore, ReviewDecision
from deep_research.tools import get_llm_client, get_search_client

log = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Graph lifecycle                                                             #
# --------------------------------------------------------------------------- #


_compiled_graph: Any | None = None


def _get_graph(store: JobStore, settings: Settings) -> Any:
    """Return the process-wide compiled graph, building it on first use.

    Cached because the attached :class:`MemorySaver` is the only
    persistence layer for HITL interrupts in v1 — if we rebuilt the
    graph on every call, each build would get its own empty saver and
    resume would never find the paused thread.

    Because the graph is per-process, every job shares the same
    injected :class:`LLMClient` / :class:`SearchClient` instances. That
    is acceptable today because both adapters will be stateless and
    thread-safe; if that changes, split the cache by job and key the
    checkpointer separately.
    """
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    # NOTE: the tools/ factories currently raise NotImplementedError.
    # Once the Groq and Tavily adapters are written, no change is
    # needed here — this code already invokes them correctly.
    llm = get_llm_client(settings=settings, store=store, job_id="<shared>")
    search = get_search_client(settings=settings)

    deps = NodeDeps(llm=llm, search=search, store=store, settings=settings)
    _compiled_graph = build_graph(deps)
    return _compiled_graph


def reset_graph_cache() -> None:
    """Reset the cached compiled graph.

    Exists for tests that need a fresh checkpointer between runs.
    Never called in production code.
    """
    global _compiled_graph
    _compiled_graph = None


# --------------------------------------------------------------------------- #
# Public entry points                                                         #
# --------------------------------------------------------------------------- #


def run_job(job_id: str, *, store: JobStore, settings: Settings) -> None:
    """Execute the research pipeline for ``job_id``.

    Called via ``BackgroundTasks.add_task`` from ``POST /research``.
    The graph will run until it either:

    1. Pauses at the ``human_review`` interrupt — the pending plan is
       already on the job store (written by the plan node) and the
       status was flipped to ``AWAITING_REVIEW`` by
       :meth:`JobStore.set_pending_plan`. This function simply returns;
       :func:`resume_job` will pick things up later.
    2. Reaches a terminal state (``DONE``) via the document node, which
       writes the output path and marks the job ``DONE`` itself.

    Any unexpected exception is caught and recorded on the job; the
    function never re-raises because it runs inside a background task
    where no one is listening.
    """
    bound = log.bind(job_id=job_id)
    bound.info("job_runner_started")

    try:
        graph = _get_graph(store, settings)
        config = {"configurable": {"thread_id": job_id}}
        job = store.get(job_id)
        state = initial_state(query=job.query, job_id=job_id)
        graph.invoke(state, config=config)
        _post_invoke_status_check(job_id, store, bound)
    except ReviewRejectedError:
        # Not reachable from run_job (review only happens on resume),
        # but handled symmetrically so the two entry points share a
        # single exception surface.
        store.clear_pending_plan(job_id)
        store.update_status(job_id, JobStatus.CANCELLED_BY_USER)
        bound.info("job_cancelled_by_user")
    except CostCeilingExceededError as exc:
        bound.error("cost_ceiling_exceeded", error=str(exc))
        store.update_status(
            job_id, JobStatus.FAILED, error=f"cost_ceiling_exceeded: {exc}"
        )
    except InsufficientResearchError as exc:
        bound.error("insufficient_research", error=str(exc))
        store.update_status(
            job_id, JobStatus.FAILED, error=f"insufficient_research: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 — background task boundary
        bound.error("job_runner_failed", error=str(exc))
        store.update_status(job_id, JobStatus.FAILED, error=str(exc))


def resume_job(
    job_id: str,
    decision: ReviewDecision,
    *,
    store: JobStore,
    settings: Settings,
) -> None:
    """Resume a job that was paused at the human-review interrupt.

    The API layer has already validated ``decision`` against the job's
    current state. This function translates the decision into a
    LangGraph ``Command(resume=...)`` payload and restarts the graph
    from the paused checkpoint:

    - ``approve`` → resume with the decision dict as-is; the human-review
      node sets ``plan_approved=True`` and execution flows to research.
    - ``edit`` → resume with the edited plan; the human-review node
      overwrites ``state.plan`` before approving.
    - ``reject`` → the human-review node raises
      :class:`ReviewRejectedError`, which we catch to flip the job to
      ``CANCELLED_BY_USER``.

    As with :func:`run_job`, any exception is converted into a failed
    job rather than re-raised.
    """
    bound = log.bind(job_id=job_id, decision=decision.decision)
    bound.info("job_resume_started")

    try:
        graph = _get_graph(store, settings)
        config = {"configurable": {"thread_id": job_id}}
        graph.invoke(
            Command(resume=decision.model_dump()),
            config=config,
        )
        _post_invoke_status_check(job_id, store, bound)
    except ReviewRejectedError:
        store.clear_pending_plan(job_id)
        store.update_status(job_id, JobStatus.CANCELLED_BY_USER)
        bound.info("job_cancelled_by_user")
    except CostCeilingExceededError as exc:
        bound.error("cost_ceiling_exceeded", error=str(exc))
        store.update_status(
            job_id, JobStatus.FAILED, error=f"cost_ceiling_exceeded: {exc}"
        )
    except InsufficientResearchError as exc:
        bound.error("insufficient_research", error=str(exc))
        store.update_status(
            job_id, JobStatus.FAILED, error=f"insufficient_research: {exc}"
        )
    except Exception as exc:  # noqa: BLE001 — background task boundary
        bound.error("job_resume_failed", error=str(exc))
        store.update_status(job_id, JobStatus.FAILED, error=str(exc))


# --------------------------------------------------------------------------- #
# Internals                                                                   #
# --------------------------------------------------------------------------- #


def _post_invoke_status_check(
    job_id: str,
    store: JobStore,
    log: structlog.stdlib.BoundLogger,
) -> None:
    """Verify the job is in a coherent state after ``graph.invoke``.

    ``invoke`` returns cleanly in two situations:

    1. The graph reached a terminal node (``document``), in which case
       that node already called :meth:`JobStore.set_document_path`
       which flipped the status to ``DONE``.
    2. The graph paused at an interrupt (``human_review``), in which
       case the plan node called :meth:`JobStore.set_pending_plan`
       which flipped the status to ``AWAITING_REVIEW``.

    Any other post-invoke status is an invariant violation — almost
    certainly a bug in a node — so we log it loudly and mark the job
    ``FAILED`` to avoid leaving it stuck in a transient state.
    """
    job = store.get(job_id)
    ok_states = {
        JobStatus.DONE,
        JobStatus.AWAITING_REVIEW,
        JobStatus.CANCELLED_BY_USER,
        JobStatus.FAILED,
    }
    if job.status in ok_states:
        return

    log.error("post_invoke_unexpected_state", status=job.status.value)
    store.update_status(
        job_id,
        JobStatus.FAILED,
        error=f"unexpected post-invoke status: {job.status.value}",
    )
