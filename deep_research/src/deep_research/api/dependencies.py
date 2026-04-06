"""FastAPI dependency providers.

Every external collaborator the route handlers need — settings, job
store, rate limiter — is exposed as a ``Depends(...)`` provider here so
tests can override them with ``app.dependency_overrides[...]``.

The ``JobStore`` is a process-wide singleton: we want every request
handler and every background task to see the same in-memory dict. The
:func:`get_job_store` function is cached with ``lru_cache`` to guarantee
that. Swapping in a Redis-backed store later only requires changing this
one function.
"""

from __future__ import annotations

from functools import lru_cache

from slowapi import Limiter
from slowapi.util import get_remote_address

from deep_research.config import Settings, get_settings
from deep_research.services.job_store import JobStore

__all__ = [
    "get_job_store",
    "get_limiter",
    "get_settings",
    "limiter",
]


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    """Return the process-wide :class:`JobStore` singleton.

    Cached so that the FastAPI route handlers and the background worker
    observe the exact same object — critical for in-memory v1.
    """
    return JobStore()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
#
# slowapi's Limiter must be instantiated at import time and attached to the
# FastAPI app's state so the middleware can find it. We keep a module-level
# singleton here and expose it via get_limiter() for handlers that need to
# decorate themselves with `@limiter.limit("...")`.

limiter: Limiter = Limiter(key_func=get_remote_address)


def get_limiter() -> Limiter:
    """Return the module-level :class:`Limiter`."""
    return limiter


# Re-export get_settings so route modules can import everything dependency-
# related from a single place.
get_settings = get_settings
