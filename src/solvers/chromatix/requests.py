"""The typed request, result and failure shapes for the standalone wave baseline.

The contract `run_standalone` speaks, kept out of the class that implements it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from solvers.base import (
    RunStatus,
)
from solvers.chromatix.constants import (
    _BASELINE_DEVICE,
    _BASELINE_DTYPE,
    _BASELINE_FIELD_KIND,
    _BASELINE_SEED,
    _DEFAULT_MAX_OUTPUT_PIXELS,
    _EXPECTED_PHASOR,
    _SUPPORTED_PROPAGATION,
)

if TYPE_CHECKING:
    pass



class ChromatixWaveRequest(BaseModel):
    """Typed contract for the CHE-14 standalone scalar wave baseline.

    Exactly one scalar, monochromatic complex field enters, either in memory
    (``input_field_array``, used by the L1-WAVE-01 benchmark) or from a
    ``.npy`` file (``input_field_path``, used by the reproducible probe
    command). Everything else on this model is either an SI physical
    parameter or an explicitly declared convention that the result records
    verbatim -- nothing about the field is inferred.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # --- field source: exactly one of these two ---------------------------
    input_field_path: Path | None = None
    input_field_array: Any = None

    # --- SI physical parameters ------------------------------------------
    wavelength_m: float
    # (dy, dx) in that order, matching the (y, x) array axis order below.
    sample_pitch_m: tuple[float, float]
    z_m: float
    refractive_index: float = 1.0

    # --- declared conventions (recorded verbatim, validated non-empty) ----
    phasor: str = _EXPECTED_PHASOR
    coordinate_frame: str = (
        "right-handed Cartesian; array axes (y, x) row-major; +z is the propagation direction"
    )
    origin: str = "coordinate origin at array index n//2 along each spatial axis"
    reference_plane: str = "input plane at z=0; output plane at z=z_m"
    normalization: str = (
        "u stores complex field amplitude, not intensity; "
        "power = sum(|u|^2) * dy * dx in the supplied length units"
    )

    # --- solver-path selection (only one path is implemented) -------------
    propagation: str = _SUPPORTED_PROPAGATION
    field_kind: str = _BASELINE_FIELD_KIND
    device: str = _BASELINE_DEVICE
    dtype: str = _BASELINE_DTYPE
    require_gradients: bool = False

    # --- grid / padding policy -------------------------------------------
    padding_policy: str = "explicit"
    pad_width: int | None = None
    output_mode: str = "full"
    max_output_pixels: int = _DEFAULT_MAX_OUTPUT_PIXELS

    # --- run bookkeeping --------------------------------------------------
    output_directory: Path
    seed: int = _BASELINE_SEED


class ChromatixWaveFailure(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    stage: str
    exception_type: str | None = None


class ChromatixWaveResult(BaseModel):
    """Structured success/failure result for :class:`ChromatixWaveRequest`."""

    model_config = ConfigDict(extra="forbid")

    status: RunStatus
    package_version: str | None = None
    package_commit: str | None = None
    propagation: str | None = None
    device: str | None = None
    cpu_device: str | None = None
    jax_backend: str | None = None
    dtype: str | None = None
    input_shape: tuple[int, int] | None = None
    output_shape: tuple[int, int] | None = None
    input_sample_pitch_m: tuple[float, float] | None = None
    output_sample_pitch_m: tuple[float, float] | None = None
    pad_width: int | None = None
    padded: bool | None = None
    cropped: bool | None = None
    runtime_seconds: float | None = None
    output_directory: str | None = None
    input_field_path: str | None = None
    output_field_path: str | None = None
    summary_path: str | None = None
    input_field_sha256: str | None = None
    output_field_sha256: str | None = None
    scientific_array_sha256: str | None = None
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    field_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    failure: ChromatixWaveFailure | None = None
