"""LangGraph orchestration package.

This package owns the state machine that drives the pipeline:

    plan → human_review → research → synthesize → document

Public API:

- :func:`build_graph` — compile a fresh graph with injected dependencies.
- :class:`NodeDeps` — the collaborator bundle passed into ``build_graph``.
- :class:`ResearchState` — the shared state TypedDict.
- :func:`initial_state` — seed a fresh state for a new job.

Error classes raised by nodes and observed by the runner:

- :class:`InsufficientResearchError` — >50% sub-task failure.
- :class:`CostCeilingExceededError` — running cost exceeded max budget.
- :class:`ReviewRejectedError` — human reviewer rejected the plan.
"""

from deep_research.graph.builder import build_graph
from deep_research.graph.nodes import (
    CostCeilingExceededError,
    InsufficientResearchError,
    NodeDeps,
    ReviewRejectedError,
)
from deep_research.graph.state import ResearchState, initial_state

__all__ = [
    "CostCeilingExceededError",
    "InsufficientResearchError",
    "NodeDeps",
    "ResearchState",
    "ReviewRejectedError",
    "build_graph",
    "initial_state",
]
