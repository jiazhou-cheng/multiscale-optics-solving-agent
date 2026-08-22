"""The M1 standalone wave baseline, kept out of the graph-facing adapter.

`run_standalone` implements the frozen `M1-BASELINE-CPU-V1` contract: a fresh
process, no coupler and no ray engine, a fixed artifact set, and a structured
blocker rather than a fabricated value when the solver refuses. It is not the
`ModelAdapter` protocol and nothing in a graph reaches it.

Not archived with the gen1 suites, despite being M1 machinery, because
`knowledge/solvers/chromatix/probes/standalone_baseline.py` is a live consumer
and is the executable evidence behind a card claim.

`_BaselineError` is internal to this module by design: it exists so the several
validation stages can fail with a code, a message and the *stage* they failed
at, and be turned into one structured result at the top.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from core.errors import (
    AdapterDependencyError,
)
from solvers.base import (
    RunStatus,
)
from solvers.chromatix.constants import (
    _BASELINE_DEVICE,
    _BASELINE_DTYPE,
    _BASELINE_FIELD_KIND,
    _EXPECTED_PHASOR,
    _OUTPUT_MODES,
    _PADDING_POLICIES,
    _SUPPORTED_PROPAGATION,
)
from solvers.chromatix.execution import _import_chromatix, _uri_to_path
from solvers.chromatix.propagation import edge_energy_fraction

if TYPE_CHECKING:
    pass

from solvers.chromatix.provenance import (
    _cpu_device_name,
    _installed_chromatix_provenance,
    _scientific_array_hash,
)
from solvers.chromatix.requests import (
    ChromatixWaveFailure,
    ChromatixWaveRequest,
    ChromatixWaveResult,
)


class _BaselineError(Exception):
    """Internal control-flow signal carrying a structured baseline diagnostic."""

    def __init__(self, code: str, message: str, stage: str, exception_type: str | None = None):
        super().__init__(message)
        self.failure = ChromatixWaveFailure(
            code=code, message=message, stage=stage, exception_type=exception_type
        )




def run_standalone(
    adapter: Any, request: ChromatixWaveRequest | Mapping[str, Any]
) -> ChromatixWaveResult:
    """Run the one deterministic CHE-14 CPU wave baseline and persist its bundle.

    Unlike :meth:`run`, this entry point never raises for a rejected or
    failed request: every deliberately-unimplemented capability, invalid
    metadata value, unreadable input, resource-estimate overrun, and
    solver failure comes back as ``status=FAILED`` with a structured
    :class:`ChromatixWaveFailure`. No field, power, or convergence value
    is ever fabricated on a failure path.
    """
    started = time.perf_counter()
    try:
        typed = (
            request
            if isinstance(request, ChromatixWaveRequest)
            else ChromatixWaveRequest.model_validate(request)
        )
    except ValidationError as exc:
        return ChromatixWaveResult(
            status=RunStatus.FAILED,
            runtime_seconds=time.perf_counter() - started,
            failure=ChromatixWaveFailure(
                code="CHROMATIX_INVALID_BASELINE_REQUEST",
                message=str(exc),
                stage="request_validation",
                exception_type=type(exc).__name__,
            ),
        )

    try:
        return _execute_baseline(typed, started)
    except _BaselineError as exc:
        return ChromatixWaveResult(
            status=RunStatus.FAILED,
            device=typed.device,
            cpu_device=_cpu_device_name(),
            dtype=typed.dtype,
            propagation=typed.propagation,
            runtime_seconds=time.perf_counter() - started,
            output_directory=str(typed.output_directory),
            failure=exc.failure,
        )
    except Exception as exc:  # unexpected: still structured, never invented output
        return ChromatixWaveResult(
            status=RunStatus.FAILED,
            device=typed.device,
            cpu_device=_cpu_device_name(),
            dtype=typed.dtype,
            propagation=typed.propagation,
            runtime_seconds=time.perf_counter() - started,
            output_directory=str(typed.output_directory),
            failure=ChromatixWaveFailure(
                code="CHROMATIX_SOLVER_EXECUTION_FAILED",
                message=str(exc),
                stage="baseline_execution",
                exception_type=type(exc).__name__,
            ),
        )

def _baseline_problems(typed: ChromatixWaveRequest) -> list[tuple[str, str, str]]:
    """Return (code, message, stage) for every gate this baseline rejects.

    Pure Python: runs before chromatix/jax is imported so an unsupported
    request never reaches, or pays for, the solver.
    """
    import math

    problems: list[tuple[str, str, str]] = []

    if typed.propagation != _SUPPORTED_PROPAGATION:
        problems.append(
            (
                "CHROMATIX_UNSUPPORTED_PROPAGATION",
                f"propagation={typed.propagation!r} is not implemented; this baseline "
                f"implements only {_SUPPORTED_PROPAGATION!r} "
                "(chromatix.functional.asm_propagate). transform_propagate changes the "
                "sample pitch and has no forward-propagation oracle in this repository.",
                "capability_gate",
            )
        )
    if typed.field_kind != _BASELINE_FIELD_KIND:
        problems.append(
            (
                "CHROMATIX_UNSUPPORTED_FIELD_KIND",
                f"field_kind={typed.field_kind!r} is not implemented; only "
                f"{_BASELINE_FIELD_KIND!r} (chromatix ScalarField) has been probed. "
                "No Jones-basis ordering or vector propagation frame has been verified "
                "(knowledge/solvers/chromatix/capability_notes.md).",
                "capability_gate",
            )
        )
    if typed.require_gradients:
        problems.append(
            (
                "CHROMATIX_GRADIENTS_NOT_SUPPORTED",
                "require_gradients=True is rejected: no directional-derivative evidence "
                "exists for asm_propagate. The only gradient probe in this repository "
                "covers thin_lens -> transform_propagate, which this baseline does not "
                "implement. This baseline never claims a derivative it has not tested.",
                "capability_gate",
            )
        )
    if typed.device != _BASELINE_DEVICE:
        problems.append(
            (
                "CHROMATIX_UNSUPPORTED_DEVICE",
                f"device={typed.device!r} is not implemented; only {_BASELINE_DEVICE!r} "
                "has been exercised (no GPU was available in the probing container).",
                "capability_gate",
            )
        )
    if typed.dtype != _BASELINE_DTYPE:
        problems.append(
            (
                "CHROMATIX_UNSUPPORTED_DTYPE",
                f"dtype={typed.dtype!r} is not implemented; chromatix "
                "ScalarField.__init__ unconditionally casts to complex64, so "
                f"{_BASELINE_DTYPE!r} is the only honest declaration for this path.",
                "capability_gate",
            )
        )

    sources = [typed.input_field_path is not None, typed.input_field_array is not None]
    if sum(sources) != 1:
        problems.append(
            (
                "CHROMATIX_INVALID_BASELINE_REQUEST",
                "exactly one of input_field_path / input_field_array must be supplied; "
                f"got input_field_path={typed.input_field_path!r} and "
                f"input_field_array set={typed.input_field_array is not None}.",
                "request_validation",
            )
        )

    if not math.isfinite(typed.wavelength_m) or typed.wavelength_m <= 0.0:
        problems.append(
            (
                "CHROMATIX_INVALID_SAMPLING",
                f"wavelength_m must be a finite positive value in meters; "
                f"got {typed.wavelength_m!r}.",
                "sampling_validation",
            )
        )
    for axis, pitch in zip(("dy", "dx"), typed.sample_pitch_m, strict=True):
        if not math.isfinite(pitch) or pitch <= 0.0:
            problems.append(
                (
                    "CHROMATIX_INVALID_SAMPLING",
                    f"sample_pitch_m[{axis}] must be a finite positive value in meters; "
                    f"got {pitch!r}.",
                    "sampling_validation",
                )
            )
    if not math.isfinite(typed.z_m):
        problems.append(
            (
                "CHROMATIX_INVALID_SAMPLING",
                f"z_m must be a finite propagation distance in meters; got {typed.z_m!r}.",
                "sampling_validation",
            )
        )
    if not math.isfinite(typed.refractive_index) or typed.refractive_index <= 0.0:
        problems.append(
            (
                "CHROMATIX_INVALID_SAMPLING",
                "refractive_index must be a finite positive value; "
                f"got {typed.refractive_index!r}.",
                "sampling_validation",
            )
        )

    for name in ("phasor", "coordinate_frame", "origin", "reference_plane", "normalization"):
        if not str(getattr(typed, name)).strip():
            problems.append(
                (
                    "CHROMATIX_INVALID_METADATA",
                    f"declared convention {name!r} must be a non-empty string. Every model "
                    "boundary in this repository must declare its conventions explicitly.",
                    "metadata_validation",
                )
            )

    if typed.padding_policy not in _PADDING_POLICIES:
        problems.append(
            (
                "CHROMATIX_INVALID_PADDING",
                f"padding_policy={typed.padding_policy!r} must be one of "
                f"{list(_PADDING_POLICIES)!r}.",
                "padding_validation",
            )
        )
    if typed.padding_policy == "explicit" and typed.pad_width is None:
        problems.append(
            (
                "CHROMATIX_INVALID_PADDING",
                "padding_policy='explicit' requires pad_width. Use 'auto_transfer' to "
                "delegate to chromatix.compute_padding_transfer, or 'none' for pad_width=0.",
                "padding_validation",
            )
        )
    if typed.padding_policy != "explicit" and typed.pad_width is not None:
        problems.append(
            (
                "CHROMATIX_INVALID_PADDING",
                f"pad_width={typed.pad_width!r} was supplied but padding_policy is "
                f"{typed.padding_policy!r}; that would silently ignore the requested policy.",
                "padding_validation",
            )
        )
    if typed.pad_width is not None and typed.pad_width < 0:
        problems.append(
            (
                "CHROMATIX_INVALID_PADDING",
                f"pad_width must be non-negative; got {typed.pad_width!r}.",
                "padding_validation",
            )
        )
    if typed.output_mode not in _OUTPUT_MODES:
        problems.append(
            (
                "CHROMATIX_INVALID_PADDING",
                f"output_mode={typed.output_mode!r} must be one of {list(_OUTPUT_MODES)!r}.",
                "padding_validation",
            )
        )
    if typed.max_output_pixels <= 0:
        problems.append(
            (
                "CHROMATIX_INVALID_BASELINE_REQUEST",
                f"max_output_pixels must be positive; got {typed.max_output_pixels!r}.",
                "request_validation",
            )
        )

    return problems

def _load_baseline_field(typed: ChromatixWaveRequest) -> Any:
    """Load and structurally validate the input complex field. Never coerces."""
    import numpy as np

    if typed.input_field_path is not None:
        path = _uri_to_path(str(typed.input_field_path))
        if not path.exists():
            raise _BaselineError(
                "CHROMATIX_INPUT_FIELD_UNREADABLE",
                f"input_field_path does not exist: {path}",
                "input_field_load",
                "FileNotFoundError",
            )
        try:
            array = np.load(path)
        except Exception as exc:
            raise _BaselineError(
                "CHROMATIX_INPUT_FIELD_UNREADABLE",
                f"input_field_path could not be read as a .npy array ({path}): {exc}",
                "input_field_load",
                type(exc).__name__,
            ) from exc
    else:
        array = np.asarray(typed.input_field_array)

    if array.ndim != 2:
        raise _BaselineError(
            "CHROMATIX_INPUT_FIELD_NOT_2D",
            f"expected a 2D scalar field with axes (y, x); got ndim={array.ndim} "
            f"shape={tuple(array.shape)}.",
            "input_field_validation",
        )
    if not np.iscomplexobj(array):
        raise _BaselineError(
            "CHROMATIX_INPUT_FIELD_NOT_COMPLEX",
            f"input field dtype {array.dtype} is not complex. Complex fields in this "
            "repository store amplitude, not intensity; this baseline does not promote a "
            "real array to a complex field, because doing so would silently reinterpret "
            "an intensity map as an amplitude.",
            "input_field_validation",
        )
    if not np.all(np.isfinite(array)):
        nonfinite = int(np.count_nonzero(~np.isfinite(array)))
        raise _BaselineError(
            "CHROMATIX_INPUT_FIELD_NOT_FINITE",
            f"input field contains {nonfinite} non-finite sample(s) (NaN/Inf).",
            "input_field_validation",
        )
    if array.size == 0:
        raise _BaselineError(
            "CHROMATIX_INPUT_FIELD_NOT_2D",
            "input field is empty.",
            "input_field_validation",
        )
    return array

def _execute_baseline( typed: ChromatixWaveRequest, started: float) -> ChromatixWaveResult:
    problems = _baseline_problems(typed)
    if problems:
        code, message, stage = problems[0]
        joined = "; ".join(item[1] for item in problems)
        raise _BaselineError(code, joined if len(problems) > 1 else message, stage)

    try:
        jax, jnp, _chromatix, cf, compute_padding_transfer = _import_chromatix()
    except AdapterDependencyError as exc:
        raise _BaselineError(
            "CHROMATIX_DEPENDENCY_UNAVAILABLE",
            str(exc),
            "dependency_gate",
            type(exc).__name__,
        ) from exc

    import numpy as np

    u_raw = _load_baseline_field(typed)
    warnings: list[str] = []

    version, commit, provenance_warnings = _installed_chromatix_provenance()
    warnings.extend(provenance_warnings)

    if str(u_raw.dtype) != _BASELINE_DTYPE:
        warnings.append(
            f"input field dtype {u_raw.dtype} was cast to {_BASELINE_DTYPE} at the baseline "
            "boundary. chromatix.core.field.ScalarField.__init__ performs this cast "
            "unconditionally anyway; doing it here makes the saved input artifact identical "
            "to the array Chromatix actually propagated."
        )
    u_in = np.ascontiguousarray(u_raw.astype(np.complex64))

    if typed.phasor != _EXPECTED_PHASOR:
        warnings.append(
            f"declared phasor {typed.phasor!r} is not the project canonical "
            f"{_EXPECTED_PHASOR!r}. Chromatix declares no time convention and its spatial "
            "kernel is exp(+i k.r); this baseline applies no sign correction and forwards "
            "the field unchanged."
        )

    pitch_y_m, pitch_x_m = typed.sample_pitch_m
    height, width = int(u_in.shape[0]), int(u_in.shape[1])

    # --- padding policy ------------------------------------------------
    if typed.padding_policy == "none":
        pad_width = 0
    elif typed.padding_policy == "explicit":
        pad_width = int(typed.pad_width or 0)
    else:
        if pitch_y_m != pitch_x_m:
            raise _BaselineError(
                "CHROMATIX_INVALID_PADDING",
                "padding_policy='auto_transfer' needs square pixels: "
                "chromatix.compute_padding_transfer takes a single scalar dx, but "
                f"sample_pitch_m is ({pitch_y_m!r}, {pitch_x_m!r}). Pass an explicit "
                "pad_width instead.",
                "padding_validation",
            )
        pad_width = int(
            compute_padding_transfer(height, typed.wavelength_m, pitch_y_m, typed.z_m)
        )
        warnings.append(
            f"pad_width={pad_width} came from chromatix.compute_padding_transfer, a "
            "worst-case full-bandwidth estimator. For a band-limited input it is often "
            "far larger than necessary; the padded extent is recorded so a reviewer can "
            "check it against the physically occupied bandwidth."
        )

    padded_height = height + 2 * pad_width
    padded_width = width + 2 * pad_width
    padded_pixels = padded_height * padded_width
    if padded_pixels > typed.max_output_pixels:
        raise _BaselineError(
            "CHROMATIX_RESOURCE_ESTIMATE_EXCEEDED",
            f"padding policy {typed.padding_policy!r} implies a "
            f"{padded_height}x{padded_width} = {padded_pixels} pixel propagation grid, "
            f"above max_output_pixels={typed.max_output_pixels}. At complex64 the output "
            f"array alone would be {padded_pixels * 8 / 2**20:.1f} MiB and asm_propagate "
            "holds several such arrays during the FFT pair. Nothing was executed and no "
            "field was produced; raise max_output_pixels deliberately or reduce pad_width.",
            "resource_estimate",
        )

    # --- solver call -----------------------------------------------------
    try:
        field_in = cf.Field.build(
            jnp.asarray(u_in, dtype=jnp.complex64),
            jnp.asarray([[pitch_y_m, pitch_x_m]]),
            typed.wavelength_m,
        )
        field_out = cf.asm_propagate(
            field_in,
            z=typed.z_m,
            n=typed.refractive_index,
            pad_width=pad_width,
            mode=typed.output_mode,
        )
        u_out = np.ascontiguousarray(np.asarray(jax.device_get(field_out.u)).squeeze())
        dx_out = tuple(
            float(v) for v in np.asarray(jax.device_get(field_out.dx)).reshape(-1).tolist()
        )
        power_in = float(np.asarray(jax.device_get(field_in.power)).reshape(-1)[0])
        power_out = float(np.asarray(jax.device_get(field_out.power)).reshape(-1)[0])
    except Exception as exc:
        raise _BaselineError(
            "CHROMATIX_SOLVER_EXECUTION_FAILED",
            f"chromatix.functional.asm_propagate failed: {exc}",
            "solver_call",
            type(exc).__name__,
        ) from exc

    if u_out.ndim != 2:
        raise _BaselineError(
            "CHROMATIX_SOLVER_EXECUTION_FAILED",
            f"expected a 2D output field; got shape {tuple(u_out.shape)}.",
            "output_validation",
        )
    if not np.all(np.isfinite(u_out)):
        nonfinite = int(np.count_nonzero(~np.isfinite(u_out)))
        raise _BaselineError(
            "CHROMATIX_NONFINITE_OUTPUT",
            f"propagated field contains {nonfinite} non-finite sample(s) (NaN/Inf); "
            "no field or power metric is reported for this run.",
            "output_validation",
        )

    return _persist_baseline(
        typed=typed,
        started=started,
        u_in=u_in,
        u_out=u_out,
        dx_out=dx_out,
        pad_width=pad_width,
        power_in=power_in,
        power_out=power_out,
        version=version,
        commit=commit,
        jax_backend=jax.default_backend(),
        warnings=warnings,
    )


def _persist_baseline(
    *,
    typed: ChromatixWaveRequest,
    started: float,
    u_in: Any,
    u_out: Any,
    dx_out: tuple[float, ...],
    pad_width: int,
    power_in: float,
    power_out: float,
    version: str | None,
    commit: str | None,
    jax_backend: str,
    warnings: list[str],
) -> ChromatixWaveResult:
    import numpy as np

    output_directory = Path(typed.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    input_path = output_directory / "input_field.npy"
    output_path = output_directory / "output_field.npy"
    np.save(input_path, u_in)
    np.save(output_path, u_out)
    input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
    output_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    scientific_sha = _scientific_array_hash({"input_field": u_in, "output_field": u_out})

    input_shape = (int(u_in.shape[0]), int(u_in.shape[1]))
    output_shape = (int(u_out.shape[0]), int(u_out.shape[1]))
    expected_full = (input_shape[0] + 2 * pad_width, input_shape[1] + 2 * pad_width)
    cropped = typed.output_mode == "same" and pad_width > 0
    if not cropped and output_shape != expected_full:
        warnings.append(
            f"output shape {output_shape} does not match the padded shape {expected_full} "
            f"implied by pad_width={pad_width}; chromatix resized the grid in a way this "
            "baseline did not request."
        )

    pitch_y_m, pitch_x_m = typed.sample_pitch_m
    summary_metrics: dict[str, Any] = {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "input_sample_pitch_m": [pitch_y_m, pitch_x_m],
        "output_sample_pitch_m": list(dx_out),
        "sample_pitch_unchanged": bool(
            len(dx_out) == 2
            and np.isclose(dx_out[0], pitch_y_m, rtol=1e-6)
            and np.isclose(dx_out[1], pitch_x_m, rtol=1e-6)
        ),
        "pad_width": pad_width,
        "padding_policy": typed.padding_policy,
        "padded": bool(pad_width > 0),
        "output_mode": typed.output_mode,
        "cropped": bool(cropped),
        "resampled": False,
        "input_extent_m": [input_shape[0] * pitch_y_m, input_shape[1] * pitch_x_m],
        "output_extent_m": [
            output_shape[0] * (dx_out[0] if len(dx_out) == 2 else pitch_y_m),
            output_shape[1] * (dx_out[1] if len(dx_out) == 2 else pitch_x_m),
        ],
        "power_in": power_in,
        "power_out": power_out,
        "power_conservation_ratio": (power_out / power_in) if power_in else None,
        "input_edge_energy_fraction": edge_energy_fraction(u_in),
        "output_edge_energy_fraction": edge_energy_fraction(u_out),
        "finite_window_interpretation": (
            "power_in and power_out are discrete sums over the sampled window only "
            "(sum(|u|^2) * dy * dx), not radiometric watts. Energy that leaves the window "
            "is lost from power_out, so power_conservation_ratio is a window-truncation "
            "diagnostic, not a physical conservation law. The edge-energy fractions bound "
            "how much field is sitting at the window boundary on each plane."
        ),
        "input_amplitude_max": float(np.abs(u_in).max()),
        "output_amplitude_max": float(np.abs(u_out).max()),
        "all_finite": True,
    }

    field_metadata: dict[str, Any] = {
        "axis_order": "(y, x)",
        "array_layout": "row-major; array[i, j] is y=coords_y[i], x=coords_x[j]",
        "origin": typed.origin,
        "origin_index_rule": "index n//2 along each spatial axis is coordinate 0",
        "handedness": "right-handed",
        "plus_z": "propagation direction (+z), verified against asm_propagate for z >= 0",
        "coordinate_frame": typed.coordinate_frame,
        "reference_plane": typed.reference_plane,
        "wavelength_m": typed.wavelength_m,
        "sample_pitch_m": [pitch_y_m, pitch_x_m],
        "z_m": typed.z_m,
        "refractive_index": typed.refractive_index,
        "phasor": typed.phasor,
        "spatial_kernel_sign": "exp(+i k.r) (chromatix compute_asm_propagator, z >= 0)",
        "normalization": typed.normalization,
        "values_are": "complex field amplitude (never intensity); intensity is abs(u)**2",
        "polarization": "scalar (chromatix ScalarField; no polarization state tracked)",
        "coherence": "monochromatic, fully coherent single-wavelength field",
        "dtype": _BASELINE_DTYPE,
        "device": _BASELINE_DEVICE,
        "cpu_device": _cpu_device_name(),
        "jax_backend": jax_backend,
        "jax_enable_x64": False,
        "propagation_method": "chromatix.functional.asm_propagate",
        "package": "chromatix",
        "package_version": version,
        "package_commit": commit,
    }

    summary = {
        "schema_version": 1,
        "baseline_id": "CHE-14-CHROMATIX-WAVE",
        "propagation": typed.propagation,
        "field_kind": typed.field_kind,
        "device": typed.device,
        "dtype": typed.dtype,
        "seed": typed.seed,
        "seed_semantics": (
            "recorded; this baseline builds its field analytically and uses no RNG"
        ),
        "package_version": version,
        "package_commit": commit,
        "summary_metrics": summary_metrics,
        "field_metadata": {
            key: value
            for key, value in field_metadata.items()
            # cpu_device/jax_backend are environment facts, not part of the
            # deterministic scientific summary compared across two runs.
            if key not in {"cpu_device", "jax_backend"}
        },
        "input_field_sha256": input_sha,
        "output_field_sha256": output_sha,
        "scientific_array_sha256": scientific_sha,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    return ChromatixWaveResult(
        status=RunStatus.SUCCEEDED,
        package_version=version,
        package_commit=commit,
        propagation=typed.propagation,
        device=typed.device,
        cpu_device=field_metadata["cpu_device"],
        jax_backend=jax_backend,
        dtype=typed.dtype,
        input_shape=input_shape,
        output_shape=output_shape,
        input_sample_pitch_m=(pitch_y_m, pitch_x_m),
        output_sample_pitch_m=(dx_out[0], dx_out[1]) if len(dx_out) == 2 else None,
        pad_width=pad_width,
        padded=bool(pad_width > 0),
        cropped=bool(cropped),
        runtime_seconds=time.perf_counter() - started,
        output_directory=str(output_directory),
        input_field_path=str(input_path),
        output_field_path=str(output_path),
        summary_path=str(summary_path),
        input_field_sha256=input_sha,
        output_field_sha256=output_sha,
        scientific_array_sha256=scientific_sha,
        summary_metrics=summary_metrics,
        field_metadata=field_metadata,
        warnings=warnings,
    )

# ------------------------------------------------------------------
# Capability gate -- pure Python, no chromatix/jax import required.
# ------------------------------------------------------------------
