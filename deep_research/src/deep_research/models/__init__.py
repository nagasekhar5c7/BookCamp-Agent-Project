"""Domain model package.

Thin re-exports so callers can ``from deep_research.models import Finding``
instead of reaching into each submodule. Every model here is a Pydantic
``BaseModel`` — they form the system's shared vocabulary between agents,
graph nodes, services, and the API layer.
"""

from deep_research.models.citation import Citation
from deep_research.models.finding import Finding, KeyPoint
from deep_research.models.outline import ReportOutline, Section
from deep_research.models.plan import SubTask

__all__ = [
    "Citation",
    "Finding",
    "KeyPoint",
    "ReportOutline",
    "Section",
    "SubTask",
]
