"""Plan domain models.

A **plan** is the Lead Researcher's decomposition of a user query into a
small number of atomic, independently executable sub-tasks. Each sub-task
carries just enough information for one researcher sub-agent to run end to
end: what to investigate, why, and a hint for the search query.

Plans are passed through three layers that all need a stable shape:

1. The Lead returns a plan from :func:`deep_research.agents.lead.generate_plan`.
2. The plan is persisted in the :class:`~deep_research.services.job_store.Job`
   while the graph is paused for human review.
3. The human may edit the plan via ``POST /research/{id}/review`` — the
   incoming JSON is validated against :class:`SubTask` before the graph
   resumes, so the researcher never sees a malformed sub-task.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """A single atomic research sub-task assigned to one sub-agent."""

    id: str = Field(
        ...,
        description="Stable short identifier, e.g. 't1', 't2'. Used in logs "
        "and finding-to-task linkage.",
        min_length=1,
        max_length=32,
    )
    title: str = Field(
        ...,
        description="Human-readable, one-line title.",
        min_length=1,
        max_length=200,
    )
    description: str = Field(
        ...,
        description="Detailed instructions for the sub-agent — what to find "
        "and why it matters to the overall query.",
        min_length=1,
    )
    search_hints: list[str] = Field(
        default_factory=list,
        description="Candidate search queries the sub-agent may use. The "
        "v1 researcher picks the first hint; future versions may merge "
        "results across hints.",
    )
