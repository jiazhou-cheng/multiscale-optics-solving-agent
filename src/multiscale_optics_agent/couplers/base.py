"""Stable interface for first-class physical couplers."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from multiscale_optics_agent.adapters.base import CostEstimate, RunStatus
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.graph import ValidationReport
from multiscale_optics_agent.core.specs import CouplerSpec


class CouplerRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    edge_id: str
    source: ArtifactRecord
    config: dict[str, Any] = Field(default_factory=dict)
    require_gradients: bool = False


class CouplerRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    target: ArtifactRecord | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_type: str | None = None
    error_message: str | None = None


class Coupler(Protocol):
    @property
    def spec(self) -> CouplerSpec: ...

    def estimate(self, request: CouplerRunRequest) -> CostEstimate: ...

    def validate_request(self, request: CouplerRunRequest) -> ValidationReport: ...

    def transform(self, request: CouplerRunRequest) -> CouplerRunResult: ...
