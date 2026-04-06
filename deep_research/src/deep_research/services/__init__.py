"""Services package: stateful collaborators used by the graph layer.

Thin re-exports for the most commonly imported public types. Each
submodule owns its own detailed docstring.
"""

from deep_research.services.citation_registry import CitationRegistry
from deep_research.services.document_writer import (
    OrphanCitationError,
    write_document,
)
from deep_research.services.job_store import (
    InvalidJobStateError,
    Job,
    JobNotFoundError,
    JobStatus,
    JobStore,
)
from deep_research.services.pricing import estimate_cost_usd

__all__ = [
    "CitationRegistry",
    "InvalidJobStateError",
    "Job",
    "JobNotFoundError",
    "JobStatus",
    "JobStore",
    "OrphanCitationError",
    "estimate_cost_usd",
    "write_document",
]
