"""The angular-spectrum propagation itself, and the plane arithmetic around it.

`run_asm_propagate` is the graph-facing physics: load the incoming field,
resolve how far to propagate, call Chromatix, and hand back a `ComplexField`
with its conventions declared.

`WaveHandoffError` belongs here because an unresolvable source plane is a
geometry outcome, and carrying it as an exception rather than a sentinel is what
stops a caller mistaking "undeclared" for "at zero".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.arrays import array_state
from core.artifacts import ArtifactRecord
from core.precision import (
    DeviceKind,
    DType,
    ExecutionRequest,
    Precision,
)
from core.specs import ArtifactKind, Framework, ModelSpec
from solvers.base import (
    ModelRunRequest,
    ModelRunResult,
    RunStatus,
)
from solvers.chromatix.constants import (
    _CHROMATIX_SPATIAL_FACTOR,
    _DEFAULT_PROPAGATION_METHOD,
    _EDGE_ENERGY_REPORTING_THRESHOLD,
    _EXPECTED_PHASOR,
    _PHASOR_ESTABLISHED_BY,
    _PLANE_TOLERANCE_M,
    _PROPAGATION_METHODS,
    MODEL_ID,
)
from solvers.chromatix.execution import (
    _jax_device_for,
    _pitch_to_pair,
    _plan_input_bridge,
    _resolve_chromatix_execution,
    _uri_to_path,
)

if TYPE_CHECKING:
    import numpy as np



class WaveHandoffError(RuntimeError):
    """A declared boundary condition of the propagation does not hold.

    CHE-35. Distinct from ``SolverExecutionError``, which means Chromatix ran and
    failed: these are refusals made *before* any solver call, on a declaration
    that would otherwise produce a plausible field at the wrong place or with the
    wrong sign. Carried as an exception rather than a sentinel so no code path can
    mistake a refusal for a result; ``run()`` converts it to a structured failure.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code





def edge_energy_fraction(u: Any) -> float:
    """Fraction of |u|^2 sitting on the 1-pixel border of the sampled window.

    This is the observable finite-window indicator: a value far above the
    floor means the window truncates the field and any power or
    second-moment metric taken on this grid is window-limited.
    """
    import numpy as np

    intensity = np.abs(u) ** 2
    total = float(intensity.sum())
    if total <= 0.0 or min(intensity.shape) < 3:
        return 0.0
    border = float(
        intensity[0, :].sum()
        + intensity[-1, :].sum()
        + intensity[1:-1, 0].sum()
        + intensity[1:-1, -1].sum()
    )
    return border / total


def run_asm_propagate(
    spec: ModelSpec,
    request: ModelRunRequest,
    jax: Any,
    jnp: Any,
    cf: Any,
    compute_padding_transfer: Any,
) -> ModelRunResult:
    # Only reached after _import_chromatix() succeeded, so numpy (a jax
    # dependency) is guaranteed importable here even though it is not a
    # base dependency of this project.
    import numpy as np

    config = request.config
    if "input_field" not in request.inputs:
        raise KeyError("request.inputs is missing required 'input_field'")
    input_record = request.inputs["input_field"]

    u_in = _load_complex_array(input_record)
    if u_in.ndim != 2:
        raise ValueError(
            f"expected a 2D scalar field array (y, x), got ndim={u_in.ndim} shape={u_in.shape}"
        )
    if not np.iscomplexobj(u_in):
        raise ValueError(
            f"input_field array dtype {u_in.dtype} is not complex; this adapter "
            "does not silently promote a real array to a complex field "
            "(repository scientific conventions: complex fields store amplitude, "
            "not intensity)."
        )

    warnings: list[str] = []

    # CHE-35: the complex64 cast used to be a warning string. It is a number,
    # and the number is what the tolerance budget needs, so it is measured on
    # the field actually being propagated rather than described.
    truncation: dict[str, Any] | None = None
    if str(u_in.dtype) == "complex128":
        cast_back = u_in.astype(np.complex64).astype(np.complex128)
        reference_norm = float(np.linalg.norm(u_in))
        intensity_norm = float(np.linalg.norm(np.abs(u_in) ** 2))
        truncation = {
            "cause": (
                "chromatix.core.field.ScalarField.__init__ casts unconditionally to "
                "complex64 (`jnp.asarray(u, dtype=jnp.complex64)`), so a complex128 "
                "input is downcast inside Chromatix itself."
            ),
            "relative_field_error": (
                float(np.linalg.norm(cast_back - u_in) / reference_norm)
                if reference_norm
                else 0.0
            ),
            "relative_intensity_error": (
                float(
                    np.linalg.norm(np.abs(cast_back) ** 2 - np.abs(u_in) ** 2) / intensity_norm
                )
                if intensity_norm
                else 0.0
            ),
            "bridge_plan_metadata_key": "execution.input_bridge",
            "policy_note": (
                "accepted under ALLOW_DOWNCAST, this adapter's documented "
                "default for its own input port since CHE-35: Chromatix "
                "truncates a complex128 array inside ScalarField.__init__ "
                "whatever this adapter does, so refusing the input would "
                "remove the measurement without preventing the loss. Set "
                "config['bridge_policy']='safe' to refuse it instead."
            ),
            "scope": (
                "the input cast only. It does not include the transfer-function "
                "rounding, which depends on the represented phase and therefore on "
                "propagation_method -- see propagation_method below."
            ),
        }

    wavelength_m = float(input_record.metadata["wavelength"])
    pitch_y_m, pitch_x_m = _pitch_to_pair(input_record.metadata["sample_pitch"])
    phasor = input_record.metadata.get("phasor")
    if phasor != _EXPECTED_PHASOR:
        raise WaveHandoffError(
            "CHROMATIX_PHASOR_MISMATCH",
            f"input phasor metadata {phasor!r} is not the project canonical "
            f"{_EXPECTED_PHASOR!r}. CHE-35 established that Chromatix's ASM implements "
            f"{_CHROMATIX_SPATIAL_FACTOR}, which is this project's declared spatial "
            "factor, so the sign is no longer unknown and forwarding a mismatched "
            "field unchanged is not a neutral act: for a converging pupil field it is "
            "the difference between focusing and defocusing, and no downstream check "
            "distinguishes them. Re-express the field under the project convention, or "
            "conjugate it deliberately and say so in its metadata.",
        )

    refractive_index = float(config.get("refractive_index", 1.0))

    propagation_method = str(config.get("propagation_method", _DEFAULT_PROPAGATION_METHOD))
    if propagation_method not in _PROPAGATION_METHODS:
        raise WaveHandoffError(
            "CHROMATIX_UNSUPPORTED_PROPAGATION_METHOD",
            f"config['propagation_method']={propagation_method!r} is not one of "
            f"{list(_PROPAGATION_METHODS)!r}.",
        )

    z_m = _resolve_propagation_distance(config, input_record)

    pad_width = config.get("pad_width")
    if pad_width is None:
        if pitch_y_m != pitch_x_m:
            raise ValueError(
                "pad_width was not given and sample_pitch is non-square "
                f"({pitch_y_m!r}, {pitch_x_m!r}); "
                "chromatix.compute_padding_transfer only accepts a single scalar "
                "dx, so automatic padding is only supported for square pixels in "
                "this adapter. Pass config['pad_width'] explicitly."
            )
        pad_width = int(compute_padding_transfer(u_in.shape[0], wavelength_m, pitch_y_m, z_m))
    else:
        pad_width = int(pad_width)

    # chromatix.utils.shapes._broadcast_dx_to_grid requires a non-square dx
    # for a monochromatic field to be shaped (1, 2) -- i.e. (wavelengths,
    # 2) -- not a bare (2,) array (verified against the installed
    # package; a bare (2,) array raises "Number of wavelengths does not
    # match" because it is interpreted as one dx value per wavelength).
    # Resolve and PLACE, in that order. With a GPU present JAX puts arrays
    # on it by default, so a request for the CPU has to be honoured
    # explicitly or it silently becomes a GPU run -- the mirror image of the
    # failure CHE-55 was worried about.
    resolved = _resolve_chromatix_execution(config)
    input_plan = _plan_input_bridge(DType.parse(u_in.dtype), config)
    target_device = _jax_device_for(jax, resolved.device)
    field_in = cf.Field.build(
        jax.device_put(jnp.asarray(u_in, dtype=jnp.complex64), target_device),
        jax.device_put(jnp.asarray([[pitch_y_m, pitch_x_m]]), target_device),
        wavelength_m,
    )
    removed_carrier_phase_rad: float | None = None
    if propagation_method == "asm_carrier_removed":
        # CHE-40's kernel, over Chromatix's own FFTs, padding, frequency grid
        # and evanescent policy. Required by benchmarks/protocols/slice_protocol.yaml for
        # any phase-insensitive M3 PSF path; the field's ABSOLUTE phase is not
        # physical afterwards, which the output metadata states.
        from solvers.chromatix.carrier_removed_asm import (
            carrier_removed_asm_propagate,
        )

        propagation = carrier_removed_asm_propagate(
            field_in,
            z_m=z_m,
            refractive_index=refractive_index,
            pad_width=pad_width,
            wavelength_m=wavelength_m,
        )
        field_out = propagation.field
        removed_carrier_phase_rad = propagation.removed_carrier_phase_rad
    else:
        field_out = cf.asm_propagate(field_in, z=z_m, n=refractive_index, pad_width=pad_width)

    # Observed BEFORE anything is copied to the host: this is the one place
    # that can still tell where the computation actually happened. A
    # process-global platform pin produces exactly the state where the request
    # said cuda and this says cpu, and reporting the request here would hide
    # it (PB4a measured that; CHE-72 removed the dependency that caused it).
    output_state = array_state(field_out.u)
    device_mismatch = output_state.device.kind is not resolved.device.kind

    # From here on the data is on the host, as an explicit serialization
    # boundary for the .npy artifact -- not because the propagation needed it.
    u_out = np.asarray(jax.device_get(field_out.u))
    dx_out = tuple(
        float(v) for v in np.asarray(jax.device_get(field_out.dx)).reshape(-1).tolist()
    )
    power_in = float(np.asarray(jax.device_get(field_in.power)).reshape(-1)[0])
    power_out = float(np.asarray(jax.device_get(field_out.power)).reshape(-1)[0])

    output_root = Path(config.get("output_dir", "runs")) / request.run_id / request.node_id
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "output_field.npy"
    sha256 = _write_array(u_out, output_path)

    output_metadata = {
        # Requested / resolved / actual, all three kept apart. "actual" is
        # read off field_out.u; "requested" is what the caller asked for.
        "execution": {
            "requested": ExecutionRequest.from_config(
                MODEL_ID, config, default_precision=Precision.FP32
            ).as_dict(),
            "resolved": resolved.as_dict(),
            "actual": output_state.as_dict(),
            "device_mismatch": device_mismatch,
            "input_bridge": input_plan.as_dict(),
            "jax_default_backend": jax.default_backend(),
            "jax_enable_x64": False,
        },
        "serialization": {
            "boundary": "explicit_persistence",
            "host_copy": output_state.device.kind is not DeviceKind.CPU,
            "kind": (
                "serialization"
                if output_state.device.kind is not DeviceKind.CPU
                else "already_on_host"
            ),
            "reason": "npy persistence requires host bytes",
        },
        "wavelength": wavelength_m,
        "sample_pitch": dx_out,
        "coordinate_frame": (
            "axes=(y, x) row-major (chromatix Field.spatial_dims convention); "
            "right-handed Cartesian; +z is the propagation direction"
        ),
        "phasor": input_record.metadata.get("phasor", _EXPECTED_PHASOR),
        "polarization": "scalar (chromatix ScalarField; no polarization state tracked)",
        "normalization": (
            "u stores complex field amplitude, not intensity. field.power = "
            "sum(|u|^2) * prod(dx) is Chromatix's own bookkeeping in the length "
            "units supplied for dx/wavelength (meters, per project convention); "
            "it is not a calibrated SI radiometric power (W) unless the input "
            "amplitude already carried that convention."
        ),
        "propagation_method": propagation_method,
        "z_m": z_m,
        "refractive_index": refractive_index,
        "pad_width": pad_width,
        "padded": tuple(u_out.shape) != tuple(u_in.shape),
        "input_shape": tuple(int(s) for s in u_in.shape),
        "source_plane_z_m": input_record.metadata.get("z_m"),
        "source_reference_plane": input_record.metadata.get("reference_plane"),
        # CHE-40's policy, surfaced on the artifact rather than left in the
        # calling code: the removed exp(i k z) is recorded in float64 and never
        # folded back, so no consumer may read absolute optical phase off this
        # field. None on the absolute path, where the phase is physical.
        "removed_carrier_phase_rad": removed_carrier_phase_rad,
        "absolute_phase_is_physical": removed_carrier_phase_rad is None,
    }

    output_record = ArtifactRecord(
        id=f"{request.node_id}:output_field",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri=str(output_path),
        sha256=sha256,
        shape=tuple(int(s) for s in u_out.shape),
        dtype=str(u_out.dtype),
        framework=Framework.JAX,
        # OBSERVED placement of the output array, not jax.default_backend()
        # and not the request. jax.default_backend() is a process-wide fact
        # that says nothing about where THIS array landed, and the request
        # is the thing being checked.
        device=output_state.device.to_spec_device(),
        units=None,
        metadata=output_metadata,
    )

    if device_mismatch:
        # Never silent success reported as CUDA (PB4b section 13). The run is
        # still a valid complex64 propagation, so it is surfaced as a warning
        # with the ACTUAL device on the artifact rather than failed outright,
        # and the caller can gate on
        # metadata['execution']['device_mismatch'].
        warnings.append(
            f"requested device {resolved.device} but the output array landed on "
            f"{output_state.device}. The artifact records the ACTUAL device. The "
            "usual cause is a process-global JAX platform pin (jax_platform_name "
            "or JAX_PLATFORMS) set before Chromatix got a say. Any GPU-execution "
            "claim for this run is false."
        )
    # No warning for the input downcast, deliberately. CHE-35 replaced exactly
    # that warning string with a measured number, on the grounds that the
    # tolerance budget needs the magnitude and not the prose; PB4b adds the
    # recorded BridgePlan next to it rather than reinstating the warning. A
    # caller who wants the conversion refused instead of measured sets
    # config['bridge_policy']='safe'.

    input_edge_energy = edge_energy_fraction(u_in)
    output_edge_energy = edge_energy_fraction(u_out)
    if max(input_edge_energy, output_edge_energy) > _EDGE_ENERGY_REPORTING_THRESHOLD:
        warnings.append(
            f"edge-energy fraction is {max(input_edge_energy, output_edge_energy):.3g}, "
            f"above the {_EDGE_ENERGY_REPORTING_THRESHOLD:.2g} reporting threshold: the "
            "sampled window is truncating the field, so power and second-moment "
            "metrics taken on this grid are window-limited."
        )

    diagnostics = {
        # CHE-36 (M3.7). The graph path wrote dx_out onto the output artifact's
        # `sample_pitch` and reported it nowhere else, so a consumer had nothing
        # to check that metadata against -- reading the artifact and calling it
        # verified is circular. The downstream PSF measurement takes its axes
        # from this pitch, and taking the INPUT pupil pitch instead rescales
        # every distance it reports while leaving the intensity map plausible,
        # so the two are now stated separately and can be compared. The
        # baseline path has reported both since M1; this makes the graph path
        # symmetric with it.
        "input_sample_pitch_m": [pitch_y_m, pitch_x_m],
        "output_sample_pitch_m": list(dx_out),
        "sample_pitch_unchanged": bool(
            len(dx_out) == 2
            and np.isclose(dx_out[0], pitch_y_m, rtol=1e-6)
            and np.isclose(dx_out[1], pitch_x_m, rtol=1e-6)
        ),
        "power_in": power_in,
        "power_out": power_out,
        "power_conservation_ratio": (power_out / power_in) if power_in else None,
        "power_accounting": (
            "the ASM transfer function is unit-modulus wherever k_z is real, and on a "
            "grid with pitch > lambda/2 there are no evanescent bins, so total power "
            "over the INFINITE plane is conserved exactly -- M2 measured 1.0000000000 "
            "for a pure-phase DOE on that basis. The ratio above is taken on the "
            "sampled window, so any deficit is power that left the window, not power "
            "the propagation destroyed. On an unpadded run the same ratio reads 1.0 "
            "because wraparound recirculates that power, which is why a ratio of 1.0 "
            "here is not evidence of correctness."
        ),
        "input_edge_energy_fraction": input_edge_energy,
        "output_edge_energy_fraction": output_edge_energy,
        "edge_energy_reporting_threshold": _EDGE_ENERGY_REPORTING_THRESHOLD,
        "edge_energy_is_a_weak_wraparound_indicator": (
            "CHE-35 measured it moving by only 2x between a run carrying 1.4e-1 "
            "relative intensity error from wraparound and a correctly padded one. Use "
            "it to notice window truncation, not to certify padding."
        ),
        "propagation_method": propagation_method,
        "complex64_input_truncation": truncation,
        "phasor_convention": {
            "input": phasor,
            "chromatix_spatial_factor": _CHROMATIX_SPATIAL_FACTOR,
            "status": "established",
            "established_by": _PHASOR_ESTABLISHED_BY,
        },
        "chromatix_pinned_version": spec.source.pinned_version
        if spec.source
        else None,
        "jax_default_backend": jax.default_backend(),
        "derivative_note": (
            "field_out.u was converted from a JAX array to NumPy and written to "
            "disk (no shared JAX-native artifact store exists yet in this "
            "repository); this is a derivative boundary recorded under the repository "
            "scientific contract. require_gradients is rejected for this node before any "
            "solver call (see UnsupportedCapabilityError), so no differentiable "
            "path claim is made or broken by this conversion."
        ),
    }

    return ModelRunResult(
        status=RunStatus.SUCCEEDED,
        outputs={"output_field": output_record},
        diagnostics=diagnostics,
        warnings=warnings,
    )

def _resolve_propagation_distance(
    config: dict[str, Any], input_record: ArtifactRecord
) -> float:
    """The distance, checked against the two planes rather than taken on trust.

    CHE-35 AC6. The slice propagates from a declared exit pupil to a declared
    focus, and both planes are recorded -- the source plane on the incoming
    field's own metadata, the target on the edge. When the target is declared,
    the distance is *derived* from the pair, so a mismatch with an explicitly
    supplied ``z_m`` is a structured refusal instead of a silent defocus. A
    0.13 mm disagreement of exactly this kind was 0.311 waves of defocus on
    M3-SINGLET-REF (CHE-33, amendment A2), which is 300x the tightest gate in
    the budget and would have been charged to the slice.
    """
    target_plane_z_m = config.get("target_plane_z_m")
    declared_z_m = config.get("z_m")

    if target_plane_z_m is None:
        if declared_z_m is None:
            raise WaveHandoffError(
                "CHROMATIX_MISSING_PROPAGATION_DISTANCE",
                "config requires either 'z_m' or 'target_plane_z_m'.",
            )
        return float(declared_z_m)

    source_plane_z_m = input_record.metadata.get("z_m")
    if source_plane_z_m is None:
        raise WaveHandoffError(
            "CHROMATIX_SOURCE_PLANE_UNDECLARED",
            "config['target_plane_z_m'] was supplied, but the input field declares no "
            "z_m of its own, so the propagation distance between the two planes cannot "
            "be formed. A coupler-produced ComplexField always declares it.",
        )
    derived = float(target_plane_z_m) - float(source_plane_z_m)
    if declared_z_m is not None and abs(float(declared_z_m) - derived) > _PLANE_TOLERANCE_M:
        raise WaveHandoffError(
            "CHROMATIX_PROPAGATION_DISTANCE_MISMATCH",
            f"config['z_m']={float(declared_z_m)!r} m disagrees with the distance implied "
            f"by the declared planes: target {float(target_plane_z_m)!r} m minus the input "
            f"field's own reference plane {float(source_plane_z_m)!r} m = {derived!r} m "
            f"(offset {abs(float(declared_z_m) - derived):.6e} m). An axial disagreement "
            "between the plane a field is on and the distance it is propagated is a "
            "defocus, not a piston. Fix the declaration; do not widen the tolerance.",
        )
    return derived

def _load_complex_array(record: ArtifactRecord) -> np.ndarray[Any, Any]:
    import numpy as np

    path = _uri_to_path(record.uri)
    if not path.exists():
        raise FileNotFoundError(
            f"input_field artifact file not found for {record.id!r}: {path}"
        )
    array: np.ndarray[Any, Any] = np.load(path)
    return array

def _write_array(array: np.ndarray[Any, Any], path: Path) -> str:
    import numpy as np

    np.save(path, array)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest
