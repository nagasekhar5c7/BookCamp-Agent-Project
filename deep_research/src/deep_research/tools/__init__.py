"""Tool adapters — concrete implementations of the protocols in ``base``.

Public API:

- :class:`LLMClient`, :class:`SearchClient`, :class:`SearchResult` — the
  protocol definitions that the rest of the codebase depends on.
- :func:`get_llm_client`, :func:`get_search_client` — factories the
  runner uses to obtain concrete instances per process.
- :func:`bind_job_id`, :func:`get_current_job_id` — contextvar helpers
  for per-invocation job attribution (used by the Groq adapter to push
  cost deltas onto the right job).
"""

from deep_research.tools.base import LLMClient, SearchClient, SearchResult
from deep_research.tools.context import bind_job_id, get_current_job_id
from deep_research.tools.llm import GroqLLMClient, get_llm_client
from deep_research.tools.search import TavilySearchClient, get_search_client

__all__ = [
    "GroqLLMClient",
    "LLMClient",
    "SearchClient",
    "SearchResult",
    "TavilySearchClient",
    "bind_job_id",
    "get_current_job_id",
    "get_llm_client",
    "get_search_client",
]
