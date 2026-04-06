"""Typed application settings loaded from environment variables.

All runtime tunables live here. The single :class:`Settings` object is
constructed once per process via :func:`get_settings` (cached) and injected
into FastAPI handlers as a dependency. Keeping this module import-cheap
(no side effects beyond reading env vars) means it can be imported from
the api, worker, and graph layers without circular issues.

See ``ideas.md`` §10 for the canonical list of env vars.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-var driven configuration for the deep research service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---------------------------------------------------------------
    groq_api_key: str = Field(..., description="Groq Cloud API key.")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model id used by the Lead and Researcher agents.",
    )

    # --- Search ------------------------------------------------------------
    tavily_api_key: str = Field(..., description="Tavily search API key.")

    # --- Research limits ---------------------------------------------------
    max_subtasks: int = Field(default=7, ge=1, le=20)
    max_sources_per_subtask: int = Field(default=5, ge=1, le=20)
    subtask_timeout_sec: int = Field(default=180, ge=10)
    job_timeout_sec: int = Field(default=900, ge=60)

    # --- Cost ceiling ------------------------------------------------------
    max_job_cost_usd: float = Field(default=1.00, gt=0)

    # --- Human-in-the-loop -------------------------------------------------
    human_review_timeout_sec: int = Field(default=1800, ge=60)

    # --- Job store ---------------------------------------------------------
    job_store_backend: Literal["memory"] = Field(default="memory")

    # --- Rate limits (slowapi string format, e.g. "10/minute") -------------
    rate_limit_post_research: str = Field(default="10/minute")
    rate_limit_post_review: str = Field(default="30/minute")

    # --- Output & logging --------------------------------------------------
    output_dir: str = Field(default="./output")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_format: Literal["json", "pretty"] = Field(default="pretty")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance.

    Cached so that (a) env vars are read exactly once and (b) FastAPI
    dependency injection reuses the same object across requests. Tests
    that need to override values should call ``get_settings.cache_clear()``
    after mutating the environment.
    """
    return Settings()  # type: ignore[call-arg]
