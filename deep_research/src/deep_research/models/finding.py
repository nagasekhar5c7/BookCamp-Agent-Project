"""Finding models — the structured output of a single researcher sub-task.

A :class:`Finding` is *always* produced for every sub-task, even on
failure — the researcher never raises on expected failure modes. The
``status`` field tells the downstream graph nodes whether to include
this finding in synthesis (``"ok"``) or to surface it in the Research
Limitations section (``"failed"``).

``source_ids`` on each :class:`KeyPoint` are **global** citation ids
assigned by the registry; the local-to-global remap has already happened
inside :func:`deep_research.agents.researcher.execute_subtask`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KeyPoint(BaseModel):
    """A single discrete claim with its supporting citation ids."""

    text: str = Field(
        ...,
        description="The claim, phrased so it can be dropped into the final report.",
        min_length=1,
    )
    source_ids: list[int] = Field(
        ...,
        description="Global citation ids (from the registry) that support this claim.",
        min_length=1,
    )


class Finding(BaseModel):
    """All information a researcher sub-agent extracted for one sub-task."""

    task_id: str = Field(..., description="The id of the sub-task this finding belongs to.")

    status: Literal["ok", "failed"] = Field(
        ...,
        description="'ok' if the researcher extracted usable key points, "
        "'failed' if the sub-task could not be completed.",
    )
    reason: str | None = Field(
        default=None,
        description="Machine-readable failure reason when status == 'failed' "
        "(e.g. 'no_sources_found', 'llm_output_unparseable', 'no_valid_citations').",
    )

    summary: str = Field(
        default="",
        description="High-level overview written by the researcher. May be "
        "empty on failure.",
    )
    key_points: list[KeyPoint] = Field(
        default_factory=list,
        description="Discrete citeable claims. Empty on failure.",
    )
