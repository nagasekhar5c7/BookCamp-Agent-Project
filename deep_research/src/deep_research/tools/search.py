"""Tavily search adapter — concrete implementation of :class:`SearchClient`.

Wraps the official ``tavily-python`` SDK with:

1. **Layer-1 transport retries** via :mod:`tenacity` — same policy as
   the Groq adapter: exponential backoff, capped wall-clock, only on
   transient errors.
2. **Result normalisation** — every hit becomes a
   :class:`~deep_research.tools.base.SearchResult` with stable field
   names, so downstream code never sees raw Tavily dicts.
3. **Zero-result tolerance** — an empty result set is **not** an
   error. The researcher sub-agent interprets an empty list as
   ``no_sources_found`` and records that as a per-sub-task failure
   (see ``ideas.md`` §11.3, §13 row 13). Anything that raises here
   would instead cause an unhandled exception upstream.

The adapter is stateless; a single instance is shared across all jobs.
"""

from __future__ import annotations

from typing import Any, Final

import structlog
import httpx
from tavily import TavilyClient
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from deep_research.config import Settings
from deep_research.tools.base import SearchClient, SearchResult

log = structlog.get_logger(__name__)


# --------------------------------------------------------------------------- #
# Retry policy (Layer 1 — transport only)                                     #
# --------------------------------------------------------------------------- #
#
# Tavily's SDK raises plain exceptions rather than a typed hierarchy,
# so we retry on the generic HTTP transport errors that httpx emits
# under the hood. A 4xx from Tavily (bad api key, malformed query)
# will surface as an httpx.HTTPStatusError which we deliberately do
# **not** retry — fail fast on config errors.
_TRANSIENT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)

_MAX_ATTEMPTS: Final[int] = 3
_TOTAL_BUDGET_SEC: Final[float] = 90.0
_BACKOFF_MIN_SEC: Final[float] = 1.0
_BACKOFF_MAX_SEC: Final[float] = 15.0

# Truncate every result's content block to this many characters before
# it reaches the LLM. Tavily can return very long page extracts, and
# pumping 10k+ tokens into a single researcher prompt is both expensive
# and useless — the model only needs enough to produce 3-8 key points.
_MAX_CONTENT_CHARS: Final[int] = 4_000


class TavilySearchClient(SearchClient):
    """Tavily implementation of the :class:`SearchClient` protocol."""

    def __init__(self, *, settings: Settings) -> None:
        self._settings = settings
        self._client = TavilyClient(api_key=settings.tavily_api_key)

    # ------------------------------------------------------------------ API

    def search(self, *, query: str, max_results: int) -> list[SearchResult]:
        """Return up to ``max_results`` results for ``query``.

        Returns an **empty list** (not an exception) when Tavily finds
        nothing. The researcher treats this as
        :attr:`Finding.reason` = ``no_sources_found``.
        """
        bound = log.bind(provider="tavily", max_results=max_results)
        bound.debug("search_started", query=query)

        try:
            raw = self._call_with_retries(query=query, max_results=max_results)
        except _TRANSIENT_ERRORS as exc:
            # Retry budget exhausted. Treat the same as "no results" —
            # the researcher will mark the sub-task failed with a
            # clean reason instead of the whole job blowing up.
            bound.warning("search_transport_exhausted", error=str(exc))
            return []

        results = _normalise_results(raw)
        bound.info("search_completed", count=len(results))
        return results

    # --------------------------------------------------------------- Internals

    def _call_with_retries(self, *, query: str, max_results: int) -> dict[str, Any]:
        """Issue the Tavily call inside a tenacity retry wrapper."""

        @retry(
            reraise=True,
            stop=stop_after_attempt(_MAX_ATTEMPTS) | stop_after_delay(_TOTAL_BUDGET_SEC),
            wait=wait_exponential(
                multiplier=1.0, min=_BACKOFF_MIN_SEC, max=_BACKOFF_MAX_SEC
            ),
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
            before_sleep=_log_retry_attempt,
        )
        def _do_call() -> dict[str, Any]:
            # ``search_depth="advanced"`` pulls a longer extract per URL
            # which gives the researcher more material to cite. It costs
            # slightly more per call but stays well within the $1 ceiling
            # for typical jobs (7 sub-tasks × 5 sources).
            return self._client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
            )

        return _do_call()


# --------------------------------------------------------------------------- #
# Module-level helpers                                                        #
# --------------------------------------------------------------------------- #


def _normalise_results(raw: dict[str, Any]) -> list[SearchResult]:
    """Convert Tavily's raw response dict into our :class:`SearchResult` list.

    Tavily returns ``{"results": [{"title": ..., "url": ...,
    "content": ...}, ...], ...}``. We only keep the fields the
    researcher actually uses, and drop any entry missing a URL —
    citations are keyed by URL, so a URL-less result is worthless.
    """
    hits = raw.get("results") or []
    out: list[SearchResult] = []
    for hit in hits:
        url = (hit.get("url") or "").strip()
        if not url:
            continue
        content = (hit.get("content") or "").strip()
        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS] + "…"
        out.append(
            SearchResult(
                title=(hit.get("title") or url).strip(),
                url=url,
                # Tavily's "content" field doubles as a snippet-sized
                # summary, so we use the first ~240 chars as the snippet
                # shown in the citation registry and reserve the full
                # content for the LLM prompt.
                snippet=content[:240],
                content=content,
            )
        )
    return out


def _log_retry_attempt(retry_state) -> None:  # noqa: ANN001 — tenacity type
    """Tenacity ``before_sleep`` hook — emit a warning for each retry."""
    log.warning(
        "search_transport_retry",
        attempt=retry_state.attempt_number,
        next_wait_sec=round(retry_state.next_action.sleep, 2)
        if retry_state.next_action
        else None,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def get_search_client(*, settings: Settings) -> SearchClient:
    """Return a :class:`TavilySearchClient` configured from ``settings``."""
    return TavilySearchClient(settings=settings)
