"""Groq LLM adapter — concrete implementation of :class:`LLMClient`.

Wraps the official ``groq`` Python SDK with:

1. **JSON mode** — every request sets ``response_format={"type": "json_object"}``
   so Groq guarantees the response is parseable JSON. The agents still
   run a Pydantic validation step on top and have their own Layer-2
   format-repair retry (see ``ideas.md`` §11.2), but JSON mode eliminates
   the common case of the model wrapping its output in markdown fences.
2. **Layer-1 transport retries** via :mod:`tenacity` — exponential
   backoff, capped total budget, only on transient errors
   (connection, 5xx, rate limit, timeout). 4xx errors fail fast.
3. **Cost accounting** — after every successful completion the adapter
   looks up the current job id (via :mod:`tools.context`), estimates
   the USD cost from :mod:`services.pricing`, and pushes the delta
   onto the :class:`~deep_research.services.job_store.JobStore`. The
   research node reads ``cost_so_far_usd`` back off the store to
   enforce the ``MAX_JOB_COST_USD`` ceiling.

The adapter is **stateless across calls** (no conversation history) —
each :meth:`complete_json` is an independent chat completion with a
single user message. This matches how all current callers use it: the
agents package does its own prompt construction and treats every call
as one-shot.
"""

from __future__ import annotations

from typing import Final

import structlog
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential,
)

from deep_research.config import Settings
from deep_research.services.job_store import JobStore
from deep_research.services.pricing import estimate_cost_usd
from deep_research.tools.base import LLMClient
from deep_research.tools.context import get_current_job_id

log = structlog.get_logger(__name__)

# --------------------------------------------------------------------------- #
# Retry policy (Layer 1 — transport only)                                     #
# --------------------------------------------------------------------------- #
#
# Exceptions treated as transient. 4xx (auth, bad request, model not
# found, context length exceeded) are **not** listed here — retrying
# them would just waste budget on a deterministic failure.
_TRANSIENT_ERRORS: Final[tuple[type[BaseException], ...]] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

# Budget: up to 3 attempts, 90s total wall-clock across retries, with
# exponential backoff capped at 15s between attempts. These match the
# limits spelled out in ideas.md §11.1.
_MAX_ATTEMPTS: Final[int] = 3
_TOTAL_BUDGET_SEC: Final[float] = 90.0
_BACKOFF_MIN_SEC: Final[float] = 1.0
_BACKOFF_MAX_SEC: Final[float] = 15.0


class GroqLLMClient(LLMClient):
    """Groq implementation of the :class:`LLMClient` protocol."""

    def __init__(self, *, settings: Settings, store: JobStore) -> None:
        self._settings = settings
        self._store = store
        self._model = settings.groq_model
        self._client = Groq(api_key=settings.groq_api_key)

    # ------------------------------------------------------------------ API

    def complete_json(self, prompt: str) -> str:
        """Return a JSON-mode completion for ``prompt``.

        The Pydantic-level parsing + repair loop lives in the agents
        package — this adapter only guarantees that the *transport*
        succeeded and that the raw string is charged to the current
        job. If Groq returns something that isn't valid JSON despite
        JSON mode, the agents layer catches the ``ValidationError``
        and triggers a format-repair retry.
        """
        bound = log.bind(provider="groq", model=self._model)
        bound.debug("llm_call_started")

        response = self._call_with_retries(prompt)

        # JSON mode guarantees a string content — but be defensive
        # about unexpected None values from the SDK.
        choice = response.choices[0]
        content = choice.message.content or ""
        content = _strip_code_fences(content)

        # Cost accounting — best effort. If we are running outside a
        # job context (e.g. in a test) we skip the store update.
        usage = getattr(response, "usage", None)
        if usage is not None:
            input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            cost_usd = estimate_cost_usd(
                model=self._model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
            self._charge_current_job(cost_usd)
            bound.info(
                "llm_call_completed",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
        else:
            bound.warning("llm_usage_missing")

        return content

    # --------------------------------------------------------------- Internals

    def _call_with_retries(self, prompt: str):  # noqa: ANN202 — Groq SDK type
        """Issue the HTTP call inside a tenacity retry wrapper.

        Defined as a nested ``@retry``-decorated function so the
        configuration stays next to the call site and so we can
        close over ``self`` without polluting the class namespace
        with a decorator.
        """

        @retry(
            reraise=True,
            stop=stop_after_attempt(_MAX_ATTEMPTS) | stop_after_delay(_TOTAL_BUDGET_SEC),
            wait=wait_exponential(
                multiplier=1.0, min=_BACKOFF_MIN_SEC, max=_BACKOFF_MAX_SEC
            ),
            retry=retry_if_exception_type(_TRANSIENT_ERRORS),
            before_sleep=_log_retry_attempt,
        )
        def _do_call():  # noqa: ANN202
            return self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

        return _do_call()

    def _charge_current_job(self, cost_usd: float) -> None:
        """Push a cost delta onto the current job, if one is bound.

        Silently no-ops when there is no current job id — that only
        happens when a test calls the adapter directly. In the normal
        runner path the contextvar is always set.
        """
        job_id = get_current_job_id()
        if job_id is None:
            return
        try:
            self._store.add_cost(job_id, cost_usd)
        except Exception as exc:  # noqa: BLE001 — never let billing kill the call
            log.warning("cost_update_failed", job_id=job_id, error=str(exc))


# --------------------------------------------------------------------------- #
# Module-level helpers                                                        #
# --------------------------------------------------------------------------- #


def _log_retry_attempt(retry_state) -> None:  # noqa: ANN001 — tenacity type
    """Tenacity ``before_sleep`` hook — emit a warning for each retry."""
    log.warning(
        "llm_transport_retry",
        attempt=retry_state.attempt_number,
        next_wait_sec=round(retry_state.next_action.sleep, 2)
        if retry_state.next_action
        else None,
        error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
    )


def _strip_code_fences(content: str) -> str:
    """Remove triple-backtick fences if Groq returned them despite JSON mode.

    JSON mode should eliminate this, but older models occasionally slip
    a ``` ```json ... ``` ``` wrapper around the payload. Stripping is
    cheap insurance.
    """
    text = content.strip()
    if not text.startswith("```"):
        return text

    # Drop the opening fence (which may have a language tag like ```json).
    first_newline = text.find("\n")
    if first_newline == -1:
        return text
    text = text[first_newline + 1 :]

    # Drop the trailing fence if present.
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #


def get_llm_client(*, settings: Settings, store: JobStore) -> LLMClient:
    """Return a :class:`GroqLLMClient` configured from ``settings``.

    The instance is **process-wide** — the runner caches it once inside
    :func:`deep_research.workers.runner._get_graph`. Per-job cost
    attribution is done via the :mod:`tools.context` contextvar rather
    than per-instance state.
    """
    return GroqLLMClient(settings=settings, store=store)
