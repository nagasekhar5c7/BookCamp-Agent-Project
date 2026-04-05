"""Researcher sub-agent: executes a single research sub-task.

A stateless worker. Given one :class:`SubTask`, it:

1. Runs a web search via the injected :class:`SearchClient`.
2. Registers every source URL in the shared :class:`CitationRegistry`
   (deduplicated across the whole job).
3. Asks the LLM to extract a structured :class:`Finding` that ties each
   claim to its source id.

Unlike the Lead agent, the researcher **never raises** on expected failure
modes (zero search results, unparseable LLM output, no valid citations). It
returns a :class:`Finding` with ``status="failed"`` and a machine-readable
reason so the graph can decide whether the overall run should continue (see
ideas.md §11.3, "Per-sub-task isolation").
"""

from __future__ import annotations

from importlib.resources import files
from typing import Final

import structlog
from pydantic import BaseModel, ValidationError

from deep_research.models.citation import Citation
from deep_research.models.finding import Finding, KeyPoint
from deep_research.models.plan import SubTask
from deep_research.services.citation_registry import CitationRegistry
from deep_research.tools.base import LLMClient, SearchClient, SearchResult

log = structlog.get_logger(__name__)

_MAX_FORMAT_REPAIR_ATTEMPTS: Final[int] = 2
_PROMPTS_PACKAGE: Final[str] = "deep_research.agents.prompts"


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def execute_subtask(
    *,
    query: str,
    subtask: SubTask,
    llm: LLMClient,
    search: SearchClient,
    registry: CitationRegistry,
    max_sources: int,
) -> Finding:
    """Run one research sub-task end-to-end.

    Always returns a :class:`Finding`. On expected failure modes the finding's
    ``status`` will be ``"failed"`` with a ``reason`` describing what went
    wrong; the caller (graph node) inspects ``status`` and updates job state
    accordingly. Unexpected exceptions are allowed to propagate so the node
    wrapper can log them and transition to the terminal ``failed`` state.

    Args:
        query: The original user query, passed into the prompt as context so
            the researcher can keep its extraction relevant.
        subtask: The single sub-task assigned to this researcher.
        llm: Injected LLM client.
        search: Injected search client (Tavily or a fake in tests).
        registry: Shared citation registry — each unique URL gets a stable
            global integer id here.
        max_sources: Per-sub-task source cap (from ``MAX_SOURCES_PER_SUBTASK``).

    Returns:
        A :class:`Finding` with either ``status="ok"`` and populated
        ``summary`` / ``key_points``, or ``status="failed"`` with a
        ``reason``.
    """
    bound_log = log.bind(task_id=subtask.id)

    # 1. Search --------------------------------------------------------------
    search_query = _pick_search_query(subtask)
    bound_log.info("search_started", query=search_query)
    results = search.search(query=search_query, max_results=max_sources)

    if not results:
        bound_log.warning("no_sources_found")
        return Finding(
            task_id=subtask.id,
            status="failed",
            reason="no_sources_found",
            summary="",
            key_points=[],
        )

    # 2. Register citations & build local → global id map -------------------
    #
    # The LLM is shown sources numbered 1..N *within this sub-task*. After the
    # LLM responds, we remap those local numbers to the registry's global ids
    # so citation markers remain stable across the whole report.
    local_to_global: dict[int, int] = {}
    for local_idx, result in enumerate(results, start=1):
        citation = Citation(
            title=result.title,
            url=result.url,
            snippet=result.snippet,
        )
        local_to_global[local_idx] = registry.register(citation)

    # 3. LLM extraction with format-repair retries --------------------------
    prompt = _load_prompt("researcher.md").format(
        query=query,
        title=subtask.title,
        description=subtask.description,
        search_results_block=_format_search_results(results),
    )

    parsed = _extract_finding(prompt=prompt, llm=llm, log=bound_log)
    if parsed is None:
        return Finding(
            task_id=subtask.id,
            status="failed",
            reason="llm_output_unparseable",
            summary="",
            key_points=[],
        )

    # 4. Remap local source ids → global citation ids -----------------------
    key_points: list[KeyPoint] = []
    for kp in parsed.key_points:
        remapped = [
            local_to_global[i] for i in kp.source_ids if i in local_to_global
        ]
        if not remapped:
            # Drop any claim the LLM failed to ground in a real source id.
            continue
        key_points.append(KeyPoint(text=kp.text, source_ids=remapped))

    if not key_points:
        bound_log.warning("no_valid_key_points_after_remap")
        return Finding(
            task_id=subtask.id,
            status="failed",
            reason="no_valid_citations",
            summary=parsed.summary,
            key_points=[],
        )

    finding = Finding(
        task_id=subtask.id,
        status="ok",
        summary=parsed.summary,
        key_points=key_points,
    )
    bound_log.info("subtask_completed", key_points=len(key_points))
    return finding


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


class _ResearcherKeyPoint(BaseModel):
    """LLM-side key-point schema. Source ids are *local* to this sub-task."""

    text: str
    source_ids: list[int]


class _ResearcherOutput(BaseModel):
    """LLM-side finding schema returned by the researcher prompt."""

    summary: str
    key_points: list[_ResearcherKeyPoint]
    used_source_ids: list[int] = []


def _load_prompt(name: str) -> str:
    """Load a prompt template file from the prompts package."""
    return files(_PROMPTS_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def _pick_search_query(subtask: SubTask) -> str:
    """Choose the best search query for a sub-task.

    For v1 we use the first search hint if present, else fall back to the
    sub-task title. Future work: try multiple hints and merge results.
    """
    if subtask.search_hints:
        return subtask.search_hints[0]
    return subtask.title


def _extract_finding(
    *,
    prompt: str,
    llm: LLMClient,
    log: structlog.stdlib.BoundLogger,
) -> _ResearcherOutput | None:
    """Call the LLM and parse its JSON, with format-repair retries.

    Returns the parsed output or ``None`` if it could not be parsed after
    all retries. Unlike the Lead helper, this never raises — researcher
    failures are recoverable at the graph level.
    """
    raw = llm.complete_json(prompt)

    for attempt in range(_MAX_FORMAT_REPAIR_ATTEMPTS + 1):
        try:
            return _ResearcherOutput.model_validate_json(raw)
        except ValidationError as exc:
            if attempt >= _MAX_FORMAT_REPAIR_ATTEMPTS:
                log.error("llm_output_unparseable", error=str(exc))
                return None
            log.warning("llm_format_repair", attempt=attempt + 1, error=str(exc))
            raw = llm.complete_json(_build_repair_prompt(raw, str(exc)))

    return None  # unreachable


def _build_repair_prompt(original: str, error: str) -> str:
    """Build the 'fix your JSON' prompt sent on a format-repair retry."""
    return (
        "Your previous response could not be parsed as valid JSON matching "
        "the required schema.\n\n"
        f"Parser error:\n{error}\n\n"
        f"Your previous response:\n{original}\n\n"
        "Return a corrected JSON response. Do not include any prose, "
        "markdown, or code fences — only raw JSON."
    )


def _format_search_results(results: list[SearchResult]) -> str:
    """Render search results into a numbered block the LLM can cite from."""
    lines: list[str] = []
    for idx, r in enumerate(results, start=1):
        lines.append(f"[source {idx}]")
        lines.append(f"Title: {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(f"Content: {r.content}")
        lines.append("")
    return "\n".join(lines).rstrip()
