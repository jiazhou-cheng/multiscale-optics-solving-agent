"""Stable interface implemented by external physics-solver adapters.

``RunStatus`` and ``CostEstimate`` are re-exported here for callers that expect
them at this path, but they are **owned by** ``core/execution.py``. CHE-90 moved
them because ``couplers/base.py`` imported them from here to describe a
*coupler* result, which made the solver package a dependency of the coupler
package for vocabulary belonging to neither.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from core.artifacts import ArtifactRecord
from core.execution import CostEstimate, RunStatus
from core.graph import ValidationReport
from core.specs import ModelSpec

__all__ = [
    "CostEstimate",
    "ModelAdapter",
    "ModelRunRequest",
    "ModelRunResult",
    "RunStatus",
]


class ModelRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    node_id: str
    inputs: dict[str, ArtifactRecord] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    design_parameters: dict[str, Any] = Field(default_factory=dict)
    require_gradients: bool = False


class ModelRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    outputs: dict[str, ArtifactRecord] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class ModelAdapter(Protocol):
    """Adapter protocol; implementations may remain solver-native internally."""

    @property
    def spec(self) -> ModelSpec: ...

    def estimate(self, request: ModelRunRequest) -> CostEstimate: ...

    def validate_request(self, request: ModelRunRequest) -> ValidationReport: ...

    def run(self, request: ModelRunRequest) -> ModelRunResult: ...
