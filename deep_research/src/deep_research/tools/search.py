"""Tavily search client factory.

Mirror of :mod:`deep_research.tools.llm` — the runner imports
:func:`get_search_client` today and the real adapter slots in later
without touching any caller.
"""

from __future__ import annotations

from deep_research.config import Settings
from deep_research.tools.base import SearchClient


def get_search_client(*, settings: Settings) -> SearchClient:
    """Return a :class:`SearchClient` wired to Tavily."""
    raise NotImplementedError(
        "Tavily search adapter not yet implemented — write tools/search.py::TavilySearchClient"
    )
