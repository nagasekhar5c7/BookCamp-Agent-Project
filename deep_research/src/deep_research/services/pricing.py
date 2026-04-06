"""Hardcoded model price table and cost estimator.

Used by the graph runner to enforce the per-job cost ceiling defined by
``MAX_JOB_COST_USD`` (see ``ideas.md`` §13 row 16). Prices are quoted
per **1M tokens** in USD and must be updated manually when the provider
changes them.

| Last verified | 2026-04-05 (Groq public pricing)                     |
| Source        | https://groq.com/pricing                              |

Adding a new model: append an entry to :data:`_PRICE_TABLE`. Unknown
models are charged at a conservative fallback (``_FALLBACK_PRICE``) and
emit a warning log — never silently treated as free.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class ModelPrice(BaseModel):
    """Per-million-token pricing for one model."""

    input_per_million_usd: float = Field(..., ge=0)
    output_per_million_usd: float = Field(..., ge=0)


# Groq Cloud pricing as of 2026-04-05. See module docstring for source.
_PRICE_TABLE: dict[str, ModelPrice] = {
    "llama-3.3-70b-versatile": ModelPrice(
        input_per_million_usd=0.59,
        output_per_million_usd=0.79,
    ),
    "llama-3.1-8b-instant": ModelPrice(
        input_per_million_usd=0.05,
        output_per_million_usd=0.08,
    ),
}

# Conservative fallback used when the runner encounters a model id that
# isn't in the table. Chosen to be higher than any current Groq model so
# unknown models never *under*-bill against the ceiling.
_FALLBACK_PRICE: ModelPrice = ModelPrice(
    input_per_million_usd=2.00,
    output_per_million_usd=4.00,
)


def estimate_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return the estimated USD cost of a single LLM call.

    Args:
        model: The model id as returned by the LLM adapter.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.

    Returns:
        A float USD amount, rounded to 6 decimal places.
    """
    price = _PRICE_TABLE.get(model)
    if price is None:
        log.warning("unknown_model_pricing", model=model)
        price = _FALLBACK_PRICE

    cost = (
        input_tokens * price.input_per_million_usd / 1_000_000
        + output_tokens * price.output_per_million_usd / 1_000_000
    )
    return round(cost, 6)
