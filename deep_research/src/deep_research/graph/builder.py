"""Compile the LangGraph state machine.

:func:`build_graph` wires the nodes from :mod:`deep_research.graph.nodes`
together with the router from :mod:`deep_research.graph.edges`, attaches
a :class:`~langgraph.checkpoint.memory.MemorySaver` (so HITL interrupts
work within a single process lifetime — see ``ideas.md`` §4.4, §11.3),
and returns a compiled graph ready to be invoked by the worker.

The returned object is stateful with respect to the checkpointer: every
invocation scoped to the same ``thread_id`` (which we set to the job
id in the runner) will see the same persisted history. That is what
makes the pause/resume cycle work.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from deep_research.graph.edges import route_after_review
from deep_research.graph.nodes import NodeDeps, make_nodes
from deep_research.graph.state import ResearchState


def build_graph(deps: NodeDeps, *, checkpointer: Any | None = None) -> Any:
    """Build and compile the deep-research state graph.

    Args:
        deps: The injected collaborators (LLM, search, job store,
            settings) that every node closure captures.
        checkpointer: Optional custom checkpointer. Defaults to an
            in-process :class:`MemorySaver`, which is the right choice
            for v1 (local-only, single process, non-durable HITL).

    Returns:
        A compiled LangGraph ready to be invoked via ``graph.invoke``,
        ``graph.stream``, or ``graph.stream`` with a ``Command(resume=...)``
        payload.
    """
    nodes = make_nodes(deps)
    graph = StateGraph(ResearchState)

    # --- Register nodes ----------------------------------------------------
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # --- Linear spine ------------------------------------------------------
    # plan → human_review → (conditional) → research → synthesize → document → END
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "human_review")

    # --- Only branching point: review gate --------------------------------
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "research": "research",
            "__end__": END,
        },
    )

    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", "document")
    graph.add_edge("document", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())
