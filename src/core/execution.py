"""Execution vocabulary shared by models and couplers, owned by neither.

`RunStatus` and `CostEstimate` describe *running something* -- they say nothing
about rays, fields, or which side of a boundary you are on. They lived in
`solvers/base.py` until CHE-90, which meant `couplers/base.py` imported from
the solver package to describe a coupler's own result, and that import was one
half of a genuine cycle between the two.

Moving them here is not a tidying preference. The cause of the cycle was
historical -- a solver adapter was the first thing to need a run status -- and
leaving it in place would have made "solvers must not import couplers" a rule
with a documented exception, which is not a rule.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["CostEstimate", "RunStatus"]


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


class CostEstimate(BaseModel):
    """What a run is predicted to cost, and how much to trust the prediction.

    ``confidence`` defaults to ``"unknown"`` on purpose: a cost model that has
    not been calibrated should say so rather than present a number as a
    measurement.
    """

    model_config = ConfigDict(extra="forbid")

    wall_time_s: float | None = None
    peak_memory_bytes: int | None = None
    solver_calls: int = 1
    confidence: str = "unknown"
    notes: list[str] = Field(default_factory=list)
