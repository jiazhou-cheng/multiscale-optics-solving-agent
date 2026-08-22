"""The typed request, result and failure shapes for the standalone ray baseline.

Separated from the adapter for one reason: these are the *contract*
``run_standalone`` speaks, and a contract that lives inside the 1,180-line class
implementing it cannot be read without reading the implementation.

``_prescription_from_config`` is here too. Turning a request's config into an
``OpticalSystemSpec`` is request parsing rather than execution -- and it is where
supplying both ``prescription`` and ``sample`` is rejected as a conflict rather
than resolved as a precedence question, which is a statement about the request
form.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from core.optical_system import (
    OpticalSystemSpec,
    PrescriptionError,
)
from registry.prescriptions import (
    resolve_prescription,
)
from solvers.base import (
    RunStatus,
)
from solvers.optiland.constants import (
    _BASELINE_SEED,
    _DEFAULT_HX,
    _DEFAULT_HY,
    _DEFAULT_NUM_RAYS,
    _DEFAULT_SAMPLE,
    _DEFAULT_WAVELENGTH,
)


class OptilandRayRequest(BaseModel):
    """Typed contract for the single CHE-13 standalone ray baseline."""

    model_config = ConfigDict(extra="forbid")

    prescription: Literal["ReverseTelephoto"] = "ReverseTelephoto"
    backend: Literal["numpy"] = "numpy"
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float64"] = "float64"
    wavelength_um: float = Field(default=_DEFAULT_WAVELENGTH, gt=0)
    field_hx: float = Field(default=_DEFAULT_HX, ge=-1.0, le=1.0)
    field_hy: float = Field(default=_DEFAULT_HY, ge=-1.0, le=1.0)
    pupil_sampling: int = Field(default=_DEFAULT_NUM_RAYS, gt=0)
    output_directory: Path
    seed: int = _BASELINE_SEED
    require_gradients: Literal[False] = False


class OptilandRayFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    stage: str
    exception_type: str | None = None


class OptilandRayResult(BaseModel):
    """Structured success/failure result for :class:`OptilandRayRequest`."""

    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    package_version: str | None = None
    backend: str | None = None
    device: str | None = None
    cpu_device: str | None = None
    dtype: str | None = None
    requested_sampling: int | None = None
    surviving_ray_count: int | None = None
    runtime_seconds: float | None = None
    output_directory: str | None = None
    arrays_path: str | None = None
    summary_path: str | None = None
    scientific_array_sha256: str | None = None
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    failure: OptilandRayFailure | None = None




def _prescription_from_config(config: Mapping[str, Any]) -> OpticalSystemSpec:
    """The canonical prescription this request names, or supplies inline.

    ``config['prescription']`` accepts either an :class:`OpticalSystemSpec` or a
    serialized mapping, which is parsed through
    :meth:`OpticalSystemSpec.from_dict` so its schema version is checked rather
    than assumed. ``config['sample']`` names one of the registered canonical
    prescriptions. Supplying both is a conflict, not a precedence question, and
    is rejected.
    """
    inline = config.get("prescription")
    sample_name = config.get("sample")
    if inline is not None and sample_name is not None:
        raise PrescriptionError(
            "PRESCRIPTION_CONFLICTING_SOURCES",
            "config['prescription'] and config['sample'] both name a system",
            path="config",
            expected=(
                "exactly one of config['prescription'] (an inline canonical "
                "prescription) or config['sample'] (a registered prescription name)"
            ),
        )
    if inline is not None:
        if isinstance(inline, OpticalSystemSpec):
            return inline
        return OpticalSystemSpec.from_dict(inline)
    return resolve_prescription(str(sample_name or _DEFAULT_SAMPLE))
