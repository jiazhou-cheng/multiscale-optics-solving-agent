"""Framework-neutral records for arrays and scientific artifacts produced by solvers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from multiscale_optics_agent.core.specs import ArtifactKind, Device, Framework


class ArtifactRecord(BaseModel):
    """A serializable reference to solver data plus its scientific semantics.

    Numerical arrays remain in solver-owned storage. The graph runtime passes this record and
    loads/converts data only through registered adapters or couplers.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: ArtifactKind
    uri: str
    sha256: str | None = None
    shape: tuple[int, ...] = ()
    dtype: str | None = None
    framework: Framework = Framework.INTERNAL
    device: Device = Device.CPU
    units: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
