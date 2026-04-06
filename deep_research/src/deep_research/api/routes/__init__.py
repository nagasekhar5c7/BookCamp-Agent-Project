"""Route modules for the FastAPI layer.

Each submodule exposes an :class:`fastapi.APIRouter` that
:mod:`deep_research.api.main` mounts on the application.
"""

from deep_research.api.routes import health, research

__all__ = ["health", "research"]
