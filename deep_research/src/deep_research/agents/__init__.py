"""Agents package: Lead Researcher and Researcher sub-agent.

This package contains the LLM-backed logic for the two agent roles defined
in ``ideas.md`` §4. Prompt templates live in :mod:`deep_research.agents.prompts`
as plain ``.md`` files and are loaded at runtime via :mod:`importlib.resources`
so they can be edited without touching code.

Public API:

- :func:`generate_plan` — Lead decomposes a query into sub-tasks.
- :func:`synthesize_report` — Lead merges findings into a cited outline.
- :func:`execute_subtask` — Researcher executes a single sub-task.

Errors raised by the Lead (critical-path failures) are also re-exported:

- :class:`PlanGenerationError`
- :class:`SynthesisError`
"""

from deep_research.agents.lead import (
    PlanGenerationError,
    SynthesisError,
    generate_plan,
    synthesize_report,
)
from deep_research.agents.researcher import execute_subtask

__all__ = [
    "PlanGenerationError",
    "SynthesisError",
    "execute_subtask",
    "generate_plan",
    "synthesize_report",
]
