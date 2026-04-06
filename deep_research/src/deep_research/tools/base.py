"""Tool protocols — dependency-inversion boundary for LLM and search.

The agents and graph nodes depend on these :class:`Protocol` types
rather than on concrete Groq / Tavily clients. That keeps tests trivial
(hand-roll a fake), and makes it easy to swap providers later without
touching the business logic.

Two protocols, one value object:

- :class:`LLMClient` — structured JSON completion.
- :class:`SearchClient` — web search, returning :class:`SearchResult`.
- :class:`SearchResult` — a single search hit in a provider-agnostic shape.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Provider-agnostic representation of one search hit."""

    title: str = Field(..., description="Result title.")
    url: str = Field(..., description="Canonical URL of the result.")
    snippet: str = Field(
        default="",
        description="Short excerpt or description, shown in registry and logs.",
    )
    content: str = Field(
        default="",
        description="Full extracted content — what the LLM will actually read.",
    )


@runtime_checkable
class LLMClient(Protocol):
    """Abstract LLM client used by all agents.

    The system only ever needs JSON completion, so the protocol exposes
    a single method. Adapters (e.g. Groq) are responsible for prompt-
    level instructions that the model return raw JSON only.
    """

    def complete_json(self, prompt: str) -> str:
        """Run the LLM on ``prompt`` and return the raw JSON string.

        Implementations should:
        1. Apply any provider-specific JSON-mode request flags.
        2. Perform Layer-1 HTTP retries (tenacity) on transient failures.
        3. Track token usage so the runner can enforce the cost ceiling.
        4. Strip code fences if the model emits them anyway.
        """
        ...


@runtime_checkable
class SearchClient(Protocol):
    """Abstract web-search client used by the researcher sub-agent."""

    def search(self, *, query: str, max_results: int) -> list[SearchResult]:
        """Return up to ``max_results`` search hits for ``query``.

        Implementations must:
        1. Apply Layer-1 HTTP retries with a total wall-clock budget.
        2. Return an **empty list** (not raise) when the search tool
           returns zero results — the researcher interprets emptiness
           as ``no_sources_found``.
        3. Truncate/clean ``content`` so the LLM receives usable text.
        """
        ...
