"""Per-invocation context for tool adapters.

The Groq :class:`LLMClient` needs to push token / cost deltas onto the
currently-running job after every completion, but the protocol method
signature is just ``complete_json(prompt)`` — there is no ``job_id``
parameter we can plumb through.

Rather than rebuild the graph (and the LangGraph :class:`MemorySaver`)
per job just to bind a new adapter instance, we thread the current job
id through a :class:`~contextvars.ContextVar`. The runner sets it
before invoking the graph; any adapter call made downstream reads it.

Using a context var (not a thread-local) means this also works if we
ever run the graph from an async path — every task gets its own copy,
and nested graph calls in tests can override safely.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_current_job_id: ContextVar[str | None] = ContextVar(
    "deep_research_current_job_id", default=None
)


def get_current_job_id() -> str | None:
    """Return the job id currently being executed, or ``None`` if unset.

    Adapters use this to attribute token / cost deltas to the right
    job. A return value of ``None`` means the adapter is being invoked
    outside any job context (e.g. from a unit test that bypasses the
    runner) — callers must handle that gracefully by skipping the
    store update.
    """
    return _current_job_id.get()


@contextmanager
def bind_job_id(job_id: str) -> Iterator[None]:
    """Temporarily bind ``job_id`` as the current job for this task.

    Used by the runner as ``with bind_job_id(job_id): graph.invoke(...)``
    so that every downstream adapter call sees the correct id, and the
    binding is automatically cleared on exit even if the graph raises.
    """
    token = _current_job_id.set(job_id)
    try:
        yield
    finally:
        _current_job_id.reset(token)
