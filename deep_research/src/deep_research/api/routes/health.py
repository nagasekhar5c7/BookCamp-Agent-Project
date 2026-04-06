"""Health / liveness endpoint.

Kept deliberately cheap — no outbound calls to Groq or Tavily in v1 so
that `GET /health` can be hit on a tight loop by load balancers and
local smoke tests without burning API budget. The ``checks`` field is
returned as ``{"groq": "unknown", "tavily": "unknown"}`` so the shape is
stable for future deep-health work.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from fastapi import APIRouter

from deep_research.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


def _package_version() -> str:
    try:
        return version("deep_research")
    except PackageNotFoundError:
        return "0.0.0+dev"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return liveness status. Deep dependency checks are not wired in v1."""
    return HealthResponse(
        status="ok",
        version=_package_version(),
        checks={"groq": "unknown", "tavily": "unknown"},
    )
