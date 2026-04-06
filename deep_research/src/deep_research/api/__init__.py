"""FastAPI layer for the deep research service.

This package owns the HTTP surface — request/response schemas, route
handlers, dependency providers, and the application factory. It is the
only package allowed to import from ``fastapi`` / ``starlette``.

Public entry points:

- :func:`deep_research.api.main.create_app` — build a fresh
  :class:`fastapi.FastAPI` instance (used by tests).
- :data:`deep_research.api.main.app` — module-level app for
  ``uvicorn deep_research.api.main:app``.
"""

from deep_research.api.main import app, create_app

__all__ = ["app", "create_app"]
