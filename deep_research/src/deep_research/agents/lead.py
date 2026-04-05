"""Lead Researcher agent: planning and synthesis.

The Lead Researcher is strictly forbidden from performing research. Its only
responsibilities are:

1. :func:`generate_plan` — decompose a user query into atomic sub-tasks.
2. :func:`synthesize_report` — weave sub-agent findings into a cited report
   outline.

Both operations use an injected :class:`LLMClient` so the module is trivially
testable with a fake client, and neither touches the network directly. See
``ideas.md`` §4.1 and §11.2 for the design rationale (format-repair retries,
hard failure on unrecoverable plan/synthesis errors).
"""

from __future__ import annotations

from importlib.resources import files
from typing import Final

import structlog
from pydantic import BaseModel, ValidationError

from deep_research.models.finding import Finding
from deep_research.models.outline import ReportOutline
from deep_research.models.plan import SubTask
from deep_research.services.citation_registry import CitationRegistry
from deep_research.tools.base import LLMClient

log = structlog.get_logger(__name__)

# Format-repair retries for malformed LLM JSON output (see ideas.md §11.2).
# Separate from Layer 1 HTTP retries inside the LLMClient adapter.
_MAX_FORMAT_REPAIR_ATTEMPTS: Final[int] = 2

_PROMPTS_PACKAGE: Final[str] = "deep_research.agents.prompts"


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def generate_plan(
    *,
    query: str,
    llm: LLMClient,
    min_subtasks: int = 3,
    max_subtasks: int = 7,
) -> list[SubTask]:
    """Ask the Lead to decompose ``query`` into atomic research sub-tasks.

    Args:
        query: The user's raw research question.
        llm: Injected LLM client. Must implement :meth:`LLMClient.complete_json`.
        min_subtasks: Minimum number of sub-tasks the plan must contain.
        max_subtasks: Hard upper bound (enforced both in the prompt and post hoc).

    Returns:
        A list of :class:`SubTask` objects in execution order.

    Raises:
        PlanGenerationError: If the Lead cannot produce a schema-valid plan
            after the format-repair budget is exhausted, or if the plan has
            fewer than ``min_subtasks`` entries. Planning is critical-path,
            so callers must treat this as a terminal failure for the job.
    """
    prompt = _load_prompt("lead_plan.md").format(
        query=query,
        min_subtasks=min_subtasks,
        max_subtasks=max_subtasks,
    )

    parsed = _complete_json_with_repair(
        llm=llm,
        prompt=prompt,
        model=_PlanEnvelope,
        error_cls=PlanGenerationError,
        error_event="plan_generation_failed",
    )

    subtasks = parsed.subtasks[:max_subtasks]
    if len(subtasks) < min_subtasks:
        log.error(
            "plan_too_small",
            produced=len(subtasks),
            minimum=min_subtasks,
        )
        raise PlanGenerationError(
            f"Lead produced {len(subtasks)} sub-tasks, "
            f"fewer than the minimum {min_subtasks}"
        )

    log.info("plan_generated", count=len(subtasks))
    return subtasks


def synthesize_report(
    *,
    query: str,
    findings: list[Finding],
    registry: CitationRegistry,
    limitations: list[str],
    llm: LLMClient,
) -> ReportOutline:
    """Merge sub-agent findings into a fully-cited report outline.

    The Lead is given the complete citation registry so it can reference
    sources by their stable global integer id. Any marker it emits that is
    not in the registry will be caught by the downstream validation step
    in the graph (see ideas.md §6, step 5).

    Args:
        query: The user's original query, used to keep the synthesis focused.
        findings: All findings from researcher sub-agents. Failed findings
            are excluded from the findings block but their failure reasons
            should already be present in ``limitations``.
        registry: Global citation registry (source of truth for marker ids).
        limitations: Human-readable notes about sub-tasks that failed — will
            become a "Research Limitations" section in the final report if
            non-empty.
        llm: Injected LLM client.

    Returns:
        A :class:`ReportOutline` ready to be handed to the document writer.

    Raises:
        SynthesisError: If the Lead cannot produce a schema-valid outline
            after the format-repair budget is exhausted. Synthesis is
            critical-path and the job must fail.
    """
    prompt = _load_prompt("lead_synthesize.md").format(
        query=query,
        citation_map=_format_citation_map(registry),
        findings_block=_format_findings_block(findings),
        limitations_block=_format_limitations_block(limitations),
    )

    outline = _complete_json_with_repair(
        llm=llm,
        prompt=prompt,
        model=ReportOutline,
        error_cls=SynthesisError,
        error_event="synthesis_failed",
    )

    log.info("synthesis_completed", sections=len(outline.sections))
    return outline


# --------------------------------------------------------------------------- #
# Errors                                                                      #
# --------------------------------------------------------------------------- #


class PlanGenerationError(RuntimeError):
    """Raised when the Lead cannot produce a valid plan."""


class SynthesisError(RuntimeError):
    """Raised when the Lead cannot produce a valid report outline."""


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #


class _PlanEnvelope(BaseModel):
    """JSON envelope for the Lead's plan output."""

    subtasks: list[SubTask]


def _load_prompt(name: str) -> str:
    """Load a prompt template file from the prompts package."""
    return files(_PROMPTS_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def _complete_json_with_repair[T: BaseModel](
    *,
    llm: LLMClient,
    prompt: str,
    model: type[T],
    error_cls: type[RuntimeError],
    error_event: str,
) -> T:
    """Call the LLM and parse its JSON output, retrying with a repair prompt
    on :class:`ValidationError` up to :data:`_MAX_FORMAT_REPAIR_ATTEMPTS` times.

    This is the Layer-2 recovery path from ideas.md §11.2. It is deliberately
    distinct from the HTTP-level retries performed by the LLM adapter so that
    transport flakiness and model output malformation are handled at the
    correct layer.
    """
    raw = llm.complete_json(prompt)

    for attempt in range(_MAX_FORMAT_REPAIR_ATTEMPTS + 1):
        try:
            return model.model_validate_json(raw)
        except ValidationError as exc:
            if attempt >= _MAX_FORMAT_REPAIR_ATTEMPTS:
                log.error(error_event, attempts=attempt + 1, error=str(exc))
                raise error_cls(
                    f"LLM output failed schema validation after "
                    f"{attempt + 1} attempts"
                ) from exc
            log.warning(
                "llm_format_repair",
                attempt=attempt + 1,
                schema=model.__name__,
                error=str(exc),
            )
            raw = llm.complete_json(_build_repair_prompt(raw, str(exc)))

    # Unreachable: loop either returns or raises.
    raise error_cls("unreachable format-repair state")


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


def _format_citation_map(registry: CitationRegistry) -> str:
    """Render the registry as a numbered list the LLM can quote from."""
    lines = [f"[{cid}] {c.title} — {c.url}" for cid, c in registry.items()]
    return "\n".join(lines) if lines else "(no citations available)"


def _format_findings_block(findings: list[Finding]) -> str:
    """Render successful findings into a human-readable block for the prompt."""
    rendered: list[str] = []
    for f in findings:
        if f.status != "ok":
            continue
        rendered.append(f"### Sub-task: {f.task_id}")
        rendered.append(f"Summary: {f.summary}")
        for kp in f.key_points:
            source_ids = ", ".join(str(s) for s in kp.source_ids)
            rendered.append(f"- {kp.text}  (sources: [{source_ids}])")
        rendered.append("")
    return "\n".join(rendered).rstrip() if rendered else "(no successful findings)"


def _format_limitations_block(limitations: list[str]) -> str:
    """Render failed-sub-task notes as a bulleted list, or a placeholder."""
    if not limitations:
        return "(none)"
    return "\n".join(f"- {line}" for line in limitations)
