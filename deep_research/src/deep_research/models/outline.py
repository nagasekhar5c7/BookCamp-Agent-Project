"""Report outline model — the Lead Synthesizer's output.

The :class:`ReportOutline` is a structured, citation-annotated skeleton
of the final report. The document writer turns this into a ``.docx`` by
rendering each section's heading and paragraphs verbatim; the outline is
the single source of truth for the report's content and ordering.

Citation markers of the form ``[1]`` / ``[2][5]`` appear inline in
paragraph strings. A validation step (see ``ideas.md`` §6 step 5) runs
before document generation to ensure every marker corresponds to an id
that exists in the :class:`~deep_research.services.citation_registry.CitationRegistry`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Section(BaseModel):
    """One heading-and-paragraphs block in the final report."""

    heading: str = Field(
        ...,
        description="Section heading as it will appear in the document.",
        min_length=1,
        max_length=200,
    )
    paragraphs: list[str] = Field(
        ...,
        description="Ordered paragraphs. Each string may contain inline "
        "citation markers like [1] or [2][5].",
        min_length=1,
    )


class ReportOutline(BaseModel):
    """The full synthesized report, ready for document rendering."""

    title: str = Field(
        ...,
        description="Report title — typically a restatement of the user query.",
        min_length=1,
        max_length=300,
    )
    sections: list[Section] = Field(
        ...,
        description="Ordered sections. If limitations were reported, the "
        "Lead is instructed to include a final 'Research Limitations' section.",
        min_length=1,
    )
