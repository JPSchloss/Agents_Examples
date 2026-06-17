"""Pydantic models used for *structured* LLM output.

Two places use these:

  * `DataProfile` — the profiling tool returns a strongly-typed summary so the model (and
    any downstream code) gets consistent, parseable fields instead of free-form prose.
  * `RelevanceCheck` — the input guardrail forces its tiny classifier model to answer in
    this exact shape, which makes the tripwire decision deterministic to read.

Structured outputs are one of the biggest reliability wins in agentic systems: you trade a
little prompt rigidity for outputs you can branch on in code without regex-parsing English.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_pct: float
    distinct_count: int
    sample_values: list[str] = Field(default_factory=list)


class DataProfile(BaseModel):
    path: str
    row_count: int
    column_count: int
    duplicate_rows: int
    columns: list[ColumnProfile]
    issues: list[str] = Field(
        default_factory=list,
        description="Heuristic data-quality warnings (missing values, mixed types, etc.).",
    )


class RelevanceCheck(BaseModel):
    """Output shape for the input guardrail's classifier."""

    is_on_topic: bool = Field(
        description="True if the user's request is about data, pipelines, analytics, "
        "dashboards, or asking for engineering guidance on those topics."
    )
    reasoning: str = Field(description="One short sentence explaining the decision.")
