"""Citation model.

A :class:`Citation` is the structured form of a single source (typically
one Tavily search result) that the system has promised to render in the
final document's References section. Every factual claim in the report
traces back to one or more citations via integer ids assigned by the
:class:`~deep_research.services.citation_registry.CitationRegistry`.

Capture happens at the boundary: Tavily results are wrapped in
:class:`Citation` *before* the LLM ever sees them, so the LLM can never
hallucinate a URL we have no record of.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, HttpUrl


class Citation(BaseModel):
    """A single source that backs one or more claims in the report."""

    title: str = Field(..., description="Source title as reported by the search tool.")
    url: HttpUrl = Field(..., description="Canonical URL — the dedup key in the registry.")
    snippet: str = Field(
        default="",
        description="Short excerpt used for context in logs and for human review.",
    )
    accessed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the citation was first captured.",
    )
