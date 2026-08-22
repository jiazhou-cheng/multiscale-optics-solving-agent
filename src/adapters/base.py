"""Stable interface implemented by external physics-solver adapters."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from core.artifacts import ArtifactRecord
from core.graph import ValidationReport
from core.specs import ModelSpec


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PARTIAL = "partial"


class CostEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_time_s: float | None = None
    peak_memory_bytes: int | None = None
    solver_calls: int = 1
    confidence: str = "unknown"
    notes: list[str] = Field(default_factory=list)


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
