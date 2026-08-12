"""Stable interface for first-class physical couplers."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from multiscale_optics_agent.adapters.base import CostEstimate, RunStatus
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.graph import ValidationReport
from multiscale_optics_agent.core.specs import CouplerSpec

#: Port name used when a request supplies a single unnamed source. Keeps the
#: original one-artifact call style working while the ports form becomes the
#: general case.
DEFAULT_SOURCE_PORT = "source"


class CouplerRunRequest(BaseModel):
    """A coupler invocation.

    Originally this carried a single ``source: ArtifactRecord``, which cannot
    express a coupler with named ports -- the cascade of SI Algorithm S1, for
    instance, consumes both an incident ray bundle and a DOE transmission on the
    same plane. ``sources`` is the general form; ``source`` remains accepted and
    is mirrored into ``sources`` so existing single-artifact callers and the
    graph validator are unaffected.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    edge_id: str
    source: ArtifactRecord | None = None
    sources: dict[str, ArtifactRecord] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    require_gradients: bool = False

    @model_validator(mode="after")
    def _reconcile_source_forms(self) -> CouplerRunRequest:
        if self.source is None and not self.sources:
            raise ValueError("a coupler request must supply `source` or `sources`")
        if self.source is not None:
            existing = self.sources.get(DEFAULT_SOURCE_PORT)
            if existing is not None and existing != self.source:
                raise ValueError(
                    "`source` and `sources['source']` disagree; supply only one of them"
                )
            if existing is None:
                self.sources[DEFAULT_SOURCE_PORT] = self.source
        return self

    def require_source(self, port: str = DEFAULT_SOURCE_PORT) -> ArtifactRecord:
        """Fetch a named source port, or fail with a message naming what is missing."""
        try:
            return self.sources[port]
        except KeyError:
            available = ", ".join(sorted(self.sources)) or "none"
            raise KeyError(
                f"coupler request has no source port {port!r}; available: {available}"
            ) from None


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
