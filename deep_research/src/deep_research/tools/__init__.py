"""Tool adapters — concrete implementations of the protocols in ``base``.

Public API:

- :class:`LLMClient`, :class:`SearchClient`, :class:`SearchResult` — the
  protocol definitions that the rest of the codebase depends on.
- :func:`get_llm_client`, :func:`get_search_client` — factory functions
  the runner uses to obtain concrete instances per job.
"""

from deep_research.tools.base import LLMClient, SearchClient, SearchResult
from deep_research.tools.llm import get_llm_client
from deep_research.tools.search import get_search_client

__all__ = [
    "LLMClient",
    "SearchClient",
    "SearchResult",
    "get_llm_client",
    "get_search_client",
]
