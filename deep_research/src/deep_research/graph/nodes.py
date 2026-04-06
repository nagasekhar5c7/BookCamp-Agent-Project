"""LangGraph node implementations.

Each node is a plain function of :class:`ResearchState` that returns a
partial state update. External collaborators (LLM client, search client,
settings, job store) are injected via closures created in
:func:`make_nodes`, which keeps the node functions themselves pure and
trivially testable.

Pipeline: ``plan → human_review → research → synthesize → document``.

Error handling follows ``ideas.md`` §11.3:

- Critical-path nodes (``plan``, ``synthesize``, ``document``) propagate
  exceptions — the runner catches them and marks the job ``failed``.
- The research node wraps each sub-task individually so one failure
  cannot abort the job; only breaching the 50% failure threshold or
  the cost ceiling raises out of the node.
- The cost ceiling is checked after every sub-task; breaching it raises
  :class:`CostCeilingExceededError` which the runner translates into
  ``failed / cost_ceiling_exceeded``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import structlog
from langgraph.types import interrupt

from deep_research.agents.lead import generate_plan, synthesize_report
from deep_research.agents.researcher import execute_subtask
from deep_research.config import Settings
from deep_research.graph.state import ResearchState, StepLogEntry
from deep_research.models.finding import Finding
from deep_research.models.plan import SubTask
from deep_research.services.citation_registry import CitationRegistry
from deep_research.services.document_writer import write_document
from deep_research.services.job_store import JobStatus, JobStore
from deep_research.tools.base import LLMClient, SearchClient

log = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #


class InsufficientResearchError(RuntimeError):
    """Raised when more than 50% of sub-tasks fail.

    Hard-fail for the job — writing a doc from mostly-failed research
    would produce a misleading artifact (ideas.md §11.3).
    """


class CostCeilingExceededError(RuntimeError):
    """Raised when the running cost estimate breaches ``MAX_JOB_COST_USD``."""


class ReviewRejectedError(RuntimeError):
    """Raised when the human reviewer rejects the plan.

    The runner catches this and flips the job to ``CANCELLED_BY_USER``
    rather than ``FAILED`` — it is not an error condition.
    """


# --------------------------------------------------------------------------- #
# Node dependency bundle                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NodeDeps:
    """Everything every node might need, bundled for injection.

    Keeping this as a ``frozen`` dataclass (not a BaseModel) means we
    can stash the live :class:`LLMClient` / :class:`SearchClient` /
    :class:`JobStore` instances in it without Pydantic trying to
    validate them — they are runtime objects, not data.
    """

    llm: LLMClient
    search: SearchClient
    store: JobStore
    settings: Settings


NodeFn = Callable[[ResearchState], dict[str, Any]]


def make_nodes(deps: NodeDeps) -> dict[str, NodeFn]:
    """Return the set of node functions with ``deps`` captured in closure.

    This is the *only* place the injection happens — the returned dict
    is what :mod:`deep_research.graph.builder` wires into the graph.
    Each node function is a small adapter that pulls the right slice of
    state, calls the appropriate agent/service, and returns a partial
    state update.
    """
    return {
        "plan": _plan_node(deps),
        "human_review": _human_review_node(deps),
        "research": _research_node(deps),
        "synthesize": _synthesize_node(deps),
        "document": _document_node(deps),
    }


# --------------------------------------------------------------------------- #
# plan                                                                        #
# --------------------------------------------------------------------------- #


def _plan_node(deps: NodeDeps) -> NodeFn:
    """Build the ``plan`` node closure — Lead decomposes the query."""

    def plan(state: ResearchState) -> dict[str, Any]:
        job_id = state["job_id"]
        bound = log.bind(job_id=job_id, stage="plan")
        bound.info("stage_entered")

        deps.store.update_status(job_id, JobStatus.PLANNING)

        subtasks = generate_plan(
            query=state["query"],
            llm=deps.llm,
            max_subtasks=deps.settings.max_subtasks,
        )

        # Surface the pending plan on the job so the API layer can
        # serve GET /research/{id}/review immediately when the graph
        # pauses at the human_review interrupt on the next tick.
        deps.store.set_pending_plan(job_id, subtasks)

        bound.info("stage_completed", subtasks=len(subtasks))
        return {
            "plan": subtasks,
            "step_log": [
                StepLogEntry(stage="plan", event="stage_completed", detail=f"{len(subtasks)} subtasks"),
            ],
        }

    return plan


# --------------------------------------------------------------------------- #
# human_review                                                                #
# --------------------------------------------------------------------------- #


def _human_review_node(deps: NodeDeps) -> NodeFn:
    """Build the ``human_review`` node — pauses via ``interrupt()``.

    On the first visit this calls ``interrupt({"plan": ...})`` which
    halts the graph until the runner resumes it with ``Command(resume=
    decision)``. On resume the value passed to ``resume=`` becomes the
    return value of ``interrupt()``, and execution continues inside
    this function body.

    Decision shape (matches :class:`~deep_research.services.job_store.ReviewDecision`):

    - ``{"decision": "approve"}``
    - ``{"decision": "edit", "plan": [...]}``
    - ``{"decision": "reject"}``
    """

    def human_review(state: ResearchState) -> dict[str, Any]:
        job_id = state["job_id"]
        bound = log.bind(job_id=job_id, stage="human_review")
        bound.info("stage_entered")

        # This call either pauses the graph (first visit) or returns
        # the decision the runner supplied via Command(resume=...).
        raw_decision = interrupt({"plan": state["plan"]})

        decision = _normalise_decision(raw_decision)
        bound.info("decision_received", decision=decision.get("decision"))

        if decision["decision"] == "reject":
            # The runner catches ReviewRejectedError and updates the
            # job to CANCELLED_BY_USER. Raising from inside the node
            # keeps the control-flow explicit in the graph layer.
            raise ReviewRejectedError("plan_rejected_by_reviewer")

        # Clear the pending plan on the store — we are about to leave
        # AWAITING_REVIEW and the API should no longer serve it.
        deps.store.clear_pending_plan(job_id)
        deps.store.update_status(job_id, JobStatus.RESEARCHING)

        update: dict[str, Any] = {
            "plan_approved": True,
            "step_log": [
                StepLogEntry(
                    stage="human_review",
                    event="approved" if decision["decision"] == "approve" else "edited",
                ),
            ],
        }

        if decision["decision"] == "edit":
            # Validate the edited plan against the SubTask schema. This
            # has already been enforced at the API boundary, but we
            # re-validate here because the graph is the authoritative
            # edge for this transition.
            edited = [SubTask.model_validate(t) for t in decision["plan"]]
            if not edited:
                raise ValueError("edit decision contained empty plan")
            update["plan"] = edited

        return update

    return human_review


def _normalise_decision(raw: Any) -> dict[str, Any]:
    """Coerce the resume payload into a plain ``dict`` for branching.

    The runner may pass either a Pydantic ``ReviewDecision`` model or
    an already-dict-shaped payload (e.g. from tests). Both are accepted
    and reduced to a single form before dispatch.
    """
    if hasattr(raw, "model_dump"):
        return raw.model_dump()  # type: ignore[no-any-return]
    if isinstance(raw, dict):
        return raw
    raise TypeError(f"Unexpected human-review resume payload: {type(raw).__name__}")


# --------------------------------------------------------------------------- #
# research                                                                    #
# --------------------------------------------------------------------------- #


def _research_node(deps: NodeDeps) -> NodeFn:
    """Build the ``research`` node — sequential sub-agent loop.

    One CitationRegistry is shared across all sub-tasks of this job so
    that citation ids remain globally stable. After each sub-task we
    (a) check the running cost against the ceiling and (b) update the
    job store's progress counters so the polling endpoint can show
    ``task i of N``.
    """

    def research(state: ResearchState) -> dict[str, Any]:
        job_id = state["job_id"]
        plan: list[SubTask] = state["plan"]
        bound = log.bind(job_id=job_id, stage="research")
        bound.info("stage_entered", total=len(plan))

        deps.store.update_status(job_id, JobStatus.RESEARCHING)
        deps.store.set_progress(job_id, stage="research", task=0, total=len(plan))

        registry = CitationRegistry()
        findings: list[Finding] = []
        limitations: list[str] = []

        for idx, subtask in enumerate(plan, start=1):
            sub_log = bound.bind(task_id=subtask.id, task_idx=idx)
            deps.store.set_progress(job_id, task=idx, total=len(plan))

            try:
                finding = execute_subtask(
                    query=state["query"],
                    subtask=subtask,
                    llm=deps.llm,
                    search=deps.search,
                    registry=registry,
                    max_sources=deps.settings.max_sources_per_subtask,
                )
            except Exception as exc:  # noqa: BLE001 — per-sub-task isolation
                # Unexpected exceptions inside the researcher are caught
                # here (see ideas.md §11.3 "Per-sub-task isolation") and
                # converted into a failed Finding so the job can continue
                # with the remaining sub-tasks.
                sub_log.error("subtask_exception", error=str(exc))
                finding = Finding(
                    task_id=subtask.id,
                    status="failed",
                    reason="unexpected_exception",
                    summary="",
                    key_points=[],
                )

            findings.append(finding)
            if finding.status == "failed":
                limitations.append(
                    f"{subtask.title}: {finding.reason or 'unknown failure'}"
                )

            # Cost ceiling check after every sub-task. Running total is
            # maintained by the LLM adapter; we read it off the store
            # which is the authoritative accumulator. If the adapter
            # has not been wired yet, cost_so_far_usd is 0.0 and this
            # check is a no-op — safe for v1 bring-up.
            current_cost = deps.store.get(job_id).cost_so_far_usd
            if current_cost > deps.settings.max_job_cost_usd:
                raise CostCeilingExceededError(
                    f"job cost ${current_cost:.4f} exceeded ceiling "
                    f"${deps.settings.max_job_cost_usd:.2f} after task {idx}/{len(plan)}"
                )

        # 50% failure threshold — writing a doc from mostly-failed
        # research would produce a misleading artifact.
        failed = sum(1 for f in findings if f.status == "failed")
        if failed * 2 > len(plan):
            raise InsufficientResearchError(
                f"{failed}/{len(plan)} sub-tasks failed — above the 50% threshold"
            )

        bound.info(
            "stage_completed",
            succeeded=len(plan) - failed,
            failed=failed,
            citations=len(registry),
        )
        return {
            "findings": findings,
            "citations": registry.as_dict(),
            "limitations": limitations,
            "step_log": [
                StepLogEntry(
                    stage="research",
                    event="stage_completed",
                    detail=f"{len(plan) - failed}/{len(plan)} ok",
                ),
            ],
        }

    return research


# --------------------------------------------------------------------------- #
# synthesize                                                                  #
# --------------------------------------------------------------------------- #


def _synthesize_node(deps: NodeDeps) -> NodeFn:
    """Build the ``synthesize`` node — Lead merges findings into an outline."""

    def synthesize(state: ResearchState) -> dict[str, Any]:
        job_id = state["job_id"]
        bound = log.bind(job_id=job_id, stage="synthesize")
        bound.info("stage_entered")

        deps.store.update_status(job_id, JobStatus.SYNTHESIZING)

        # Rebuild a live registry from the serialised state so the
        # Lead's synthesis prompt can render the numbered citation map.
        registry = CitationRegistry.from_dict(state.get("citations", {}))

        outline = synthesize_report(
            query=state["query"],
            findings=state.get("findings", []),
            registry=registry,
            limitations=state.get("limitations", []),
            llm=deps.llm,
        )

        bound.info("stage_completed", sections=len(outline.sections))
        return {
            "outline": outline,
            "step_log": [
                StepLogEntry(
                    stage="synthesize",
                    event="stage_completed",
                    detail=f"{len(outline.sections)} sections",
                ),
            ],
        }

    return synthesize


# --------------------------------------------------------------------------- #
# document                                                                    #
# --------------------------------------------------------------------------- #


def _document_node(deps: NodeDeps) -> NodeFn:
    """Build the ``document`` node — render the final .docx."""

    def document(state: ResearchState) -> dict[str, Any]:
        job_id = state["job_id"]
        bound = log.bind(job_id=job_id, stage="document")
        bound.info("stage_entered")

        deps.store.update_status(job_id, JobStatus.GENERATING_DOCUMENT)

        registry = CitationRegistry.from_dict(state.get("citations", {}))
        outline = state["outline"]

        path = write_document(
            outline=outline,
            registry=registry,
            output_dir=deps.settings.output_dir,
            job_id=job_id,
        )

        deps.store.set_document_path(job_id, str(path))

        bound.info("stage_completed", path=str(path))
        return {
            "document_path": str(path),
            "step_log": [
                StepLogEntry(
                    stage="document",
                    event="stage_completed",
                    detail=str(path.name),
                ),
            ],
        }

    return document
