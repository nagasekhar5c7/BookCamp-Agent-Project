"""LangGraph shared state schema.

This module defines the single :class:`ResearchState` ``TypedDict`` that
flows through every node of the pipeline. It is the **only** place the
graph's data shape is declared — nodes read from it and return partial
updates; LangGraph applies those updates through the reducers annotated
below.

Design choices (see ``ideas.md`` §5):

- **Additive reducers** on ``findings``, ``limitations``, ``errors``,
  and ``step_log`` let each node append without clobbering predecessors.
- **Dict merge reducer** on ``citations`` lets the research node grow
  the registry id-by-id without each update having to re-send the
  entire map.
- **``plan_approved``** is the gate the router checks after
  ``human_review`` resumes; the human can reject and end the job there.
- **``cost_estimate_usd``** is the running total checked against
  ``MAX_JOB_COST_USD`` after every LLM call in the research loop.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from deep_research.models.citation import Citation
from deep_research.models.finding import Finding
from deep_research.models.outline import ReportOutline
from deep_research.models.plan import SubTask


# --------------------------------------------------------------------------- #
# Step log                                                                    #
# --------------------------------------------------------------------------- #


class StepLogEntry(TypedDict, total=False):
    """Lightweight audit record appended by every node as it finishes.

    Kept as a ``TypedDict`` (not a Pydantic model) so LangGraph's
    reducer merging stays a plain list concatenation with no surprise
    validation inside the hot loop.
    """

    stage: str
    event: str
    duration_ms: float
    detail: str


# --------------------------------------------------------------------------- #
# Token usage                                                                 #
# --------------------------------------------------------------------------- #


class TokenUsage(TypedDict):
    """Running counters for tokens consumed during the job."""

    input_tokens: int
    output_tokens: int


def _empty_token_usage() -> TokenUsage:
    """Helper for constructing the initial state.

    Reducers do not accept a ``default_factory`` the way Pydantic does,
    so the runner calls this when seeding a brand-new job.
    """
    return {"input_tokens": 0, "output_tokens": 0}


# --------------------------------------------------------------------------- #
# Reducers                                                                    #
# --------------------------------------------------------------------------- #


def _merge_citations(
    left: dict[int, Citation] | None,
    right: dict[int, Citation] | None,
) -> dict[int, Citation]:
    """Reducer for :attr:`ResearchState.citations`.

    Right-biased dict merge: newer registrations (from the research
    loop) replace older ones on the same id. In practice the research
    node builds its registry in one pass and emits the full dict at
    once, so there is no id collision — but the merge semantics still
    need to be defined because LangGraph demands a reducer whenever the
    field is updated from more than one node.
    """
    if left is None:
        return dict(right or {})
    if not right:
        return dict(left)
    return {**left, **right}


def _last_write_wins(
    left: TokenUsage | None,
    right: TokenUsage | None,
) -> TokenUsage:
    """Reducer for :attr:`ResearchState.tokens_used`.

    Nodes emit **absolute** running totals rather than deltas, so the
    reducer just keeps the most recent non-None write. This matches
    how ``cost_estimate_usd`` is handled (no annotation = last-write-wins
    by default) and keeps token accounting consistent.
    """
    if right is not None:
        return right
    if left is not None:
        return left
    return _empty_token_usage()


# --------------------------------------------------------------------------- #
# State                                                                       #
# --------------------------------------------------------------------------- #


class ResearchState(TypedDict, total=False):
    """The single state object threaded through every LangGraph node.

    Every field is optional (``total=False``) because different nodes
    populate different slices. The runner seeds the dict with ``query``
    and ``job_id`` before invoking the graph; downstream nodes add the
    rest.
    """

    # ---- Input ------------------------------------------------------------
    query: str
    job_id: str

    # ---- Planning ---------------------------------------------------------
    plan: list[SubTask]
    plan_approved: bool

    # ---- Research accumulation -------------------------------------------
    findings: Annotated[list[Finding], operator.add]
    citations: Annotated[dict[int, Citation], _merge_citations]

    # ---- Synthesis --------------------------------------------------------
    outline: ReportOutline
    limitations: Annotated[list[str], operator.add]

    # ---- Output -----------------------------------------------------------
    document_path: str

    # ---- Cost & observability --------------------------------------------
    tokens_used: Annotated[TokenUsage, _last_write_wins]
    cost_estimate_usd: float
    errors: Annotated[list[str], operator.add]
    step_log: Annotated[list[StepLogEntry], operator.add]


def initial_state(*, query: str, job_id: str) -> ResearchState:
    """Construct a fresh :class:`ResearchState` for a new job.

    Centralised here so the runner and tests agree on the starting
    shape. All collection fields are pre-seeded with empty containers
    so reducers have something to append to.
    """
    return ResearchState(
        query=query,
        job_id=job_id,
        plan=[],
        plan_approved=False,
        findings=[],
        citations={},
        limitations=[],
        tokens_used=_empty_token_usage(),
        cost_estimate_usd=0.0,
        errors=[],
        step_log=[],
    )
