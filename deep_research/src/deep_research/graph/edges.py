"""Conditional routing functions for the graph.

LangGraph's ``add_conditional_edges`` takes a function that inspects
state and returns a string key; the edge table maps those keys to
destination nodes. Keeping the routing logic in its own module makes
the state-machine shape easy to audit at a glance — each function here
is a one-liner with an explanation.

The pipeline has exactly one branching point in v1: after
``human_review``. Approvals / edits flow on to ``research``; rejections
end the graph. Linear steps (``plan → human_review``, ``research →
synthesize``, ``synthesize → document``, ``document → END``) are plain
``add_edge`` calls in :mod:`deep_research.graph.builder` and do not
need routing functions.
"""

from __future__ import annotations

from typing import Literal

from deep_research.graph.state import ResearchState

# Sentinel destinations used by the router below. Kept as module-level
# constants so the builder and the router cannot drift out of sync.
ROUTE_RESEARCH: Literal["research"] = "research"
ROUTE_END: Literal["__end__"] = "__end__"


def route_after_review(state: ResearchState) -> Literal["research", "__end__"]:
    """Decide whether to start research after the human-review gate.

    The ``human_review`` node raises :class:`ReviewRejectedError` on
    rejection, which the runner catches *outside* the graph — so by the
    time this router runs, the only remaining possibilities are
    ``approve`` and ``edit``, both of which set ``plan_approved=True``.

    Keeping the router explicit (rather than wiring a plain edge) means
    a future "defer" or "ask for clarification" decision can be added
    without restructuring the graph shape.
    """
    if state.get("plan_approved"):
        return ROUTE_RESEARCH
    return ROUTE_END
