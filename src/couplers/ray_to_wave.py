"""C_RAY_TO_WAVE — coherent plane-wavelet accumulation onto a plane (CHE-24).

Implements main-text eq 2 of Cheng et al., ACS Photonics 2026
(DOI 10.1021/acsphotonics.6c00818):

    U(x, y) = sum_i  a_i * exp[ i k ( OPL_i + dr_i(x, y) ) ] * <n_hat, d_hat_i>

with ``dr_i(x, y) = d_x_i (x - x0_i) + d_y_i (y - y0_i)`` the extra path from
the ray's intersection point on the plane to the field point, along the wavelet
direction.

A ray is a plane wavelet, so each ray contributes a **linear phase ramp across
the whole plane**, not a point. That is the single most important structural
fact here: an implementation that deposits energy at ``(x0_i, y0_i)`` is doing
something else entirely, and will still produce plausible-looking output.

Two projection conventions
--------------------------
Main-text eq 2 carries the factor ``<n_hat, d_hat>``; SI eq S5, which derives
the same sum as an estimator of the angular-spectrum integral (eq S2), does
not. They are therefore **not the same operator**, and the difference is
physical, not cosmetic.

CHE-25 measured which one preserves a field. Summing every propagating mode of
a random field on a 16x16 grid reproduces that field to 7.1e-15 without the
factor, and misses it by 2.2 percent of peak amplitude with it -- consistent
with the smallest ``cos(theta)`` on that grid. So:

* :attr:`Projection.ASM_CONSISTENT` (no factor) is what a **coupler** must use,
  because a representation change has to preserve the field. It is the default.
* :attr:`Projection.SENSOR_OBLIQUITY` (with the factor) is main-text eq 2, and
  models a **detector** whose response depends on incidence angle. It is a
  sensor model, not a field reconstruction.

Choosing silently would have produced a coupler that loses a few percent
off-axis and round-trips inexactly for a reason no test would name.

Known limitation: no wavefront-curvature term (CHE-50)
------------------------------------------------------
The sum is linear in the transverse coordinate, so the reconstructed field
carries **no** ``exp(i k r^2 / 2R)`` term. The output is valid **at** the
declared reference plane; it is not a field a caller may propagate a further
distance and still trust in phase.

This is invisible in ``|U|^2``, which is why every M3 intensity gate converges
cleanly, and PB7 (CHE-58) confirmed the term does not surface in a PSF -- but
only because that configuration's post-handoff propagation distance is zero.
That is a property of the configuration, not of the operator. Measured on
M3-SINGLET-REF (CHE-38): about 1.2 rad of phase against an exact spherical-wave
reference at the 5-Airy-radius gate edge, while the intensity residual sits at
1e-3 and the complex-field residual (~0.127) is flat rather than convergent.

CHE-50 decided **no kernel change for now**. The discrepancy is tracked as a
known limitation rather than treated as a confirmed blocker, to be re-examined
when a propagation-sensitive hybrid composition -- M4's, first -- independently
requires it. Emitting the term, or refusing a further-propagation request, would
each be new and separately-verified physics, not a fix to verified behaviour.

Until then the limitation is declared rather than silently carried: every
emitted field states it in ``provenance["validity"]``, and the coupler card
carries it under ``known_limitations.no_wavefront_curvature_term`` (with the
derivation under
``the_plane_z_is_not_a_kernel_parameter.consequence_3_no_wavefront_curvature``).
To reconstruct on a different plane, advance the **ray state** there and
reconstruct again -- exact, not an approximation (``consequence_1`` on the card).

This module imports neither Optiland nor Chromatix. The coupler core is the
physics under test; if it could import an engine, a coupler defect could be
misattributed to engine behaviour and M1's independence evidence would stop
bounding the search. ``benchmarks/coupler_protocol.yaml`` declares the rule and
``tests/test_ray_to_wave.py`` enforces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np

from core.arrays import (
    asarray,
    device_of,
    dtype_of,
    matmul_precision_kwargs,
    namespace_of,
    numpy_dtype,
    verify_dtype,
    xp_for,
)
from core.boundary import (
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)
from core.capabilities import C_RAY_TO_WAVE_CAPABILITIES
from core.precision import ArrayState, DType, Precision

__all__ = [
    "Perturbation",
    "Projection",
    "ReconstructionDiagnostics",
    "collimated_bundle",
    "compute_precision_for",
    "grid_nyquist_direction_limit",
    "ray_to_wave",
]


class Projection(StrEnum):
    """Which of the paper's two wavelet-sum conventions to apply.

    See the module docstring: these differ by ``<n_hat, d_hat>`` and only the
    first preserves the field, which was measured rather than argued.
    """

    #: SI eq S5. No obliquity factor. Reproduces the angular-spectrum field on
    #: the plane exactly, so it round-trips. Correct for a coupler.
    ASM_CONSISTENT = "asm_consistent"
    #: Main-text eq 2. Applies <n_hat, d_hat>. Models an angle-dependent
    #: detector response. Correct for a sensor, not for a representation change.
    SENSOR_OBLIQUITY = "sensor_obliquity"

#: Above this ray count, the pairwise nearest-neighbour scan used for the
#: ray-density diagnostic is skipped rather than run at O(N^2). The diagnostic
#: then reports ``not_computed`` instead of a number, because a fabricated
#: estimate of a sampling condition is worse than an absent one.
_NEAREST_NEIGHBOUR_SCAN_LIMIT = 4096


@dataclass(frozen=True)
class Perturbation:
    """Deliberate defects, for negative tests only.

    Every field defaults to the correct physics. These exist so a negative test
    exercises *this* implementation with one term removed, rather than a
    parallel hand-written copy that could drift from it. A negative test that
    tests a different code path proves nothing about the code that ships.
    """

    #: Flip the phasor sign. Conjugates the wavefront: a converging beam diverges.
    phase_sign: Literal[1, -1] = 1
    #: Drop whichever projection factor the chosen :class:`Projection` implies.
    #: Undetectable at normal incidence, and a no-op under ASM_CONSISTENT.
    apply_projection_factor: bool = True
    #: Drop ``dr_i(x, y)``. Off-axis rays then deposit a piston instead of a ramp.
    apply_oblique_ramp: bool = True
    #: Transpose the output grid. Invisible in any rotationally symmetric case.
    transpose_axes: bool = False

    @property
    def is_identity(self) -> bool:
        return (
            self.phase_sign == 1
            and self.apply_projection_factor
            and self.apply_oblique_ramp
            and not self.transpose_axes
        )

    def describe(self) -> str:
        if self.is_identity:
            return "none"
        parts = []
        if self.phase_sign != 1:
            parts.append("phase_sign_flipped")
        if not self.apply_projection_factor:
            parts.append("projection_factor_omitted")
        if not self.apply_oblique_ramp:
            parts.append("oblique_ramp_omitted")
        if self.transpose_axes:
            parts.append("axes_transposed")
        return "+".join(parts)


@dataclass(frozen=True)
class ReconstructionDiagnostics:
    """Everything measured during a reconstruction, reported rather than judged."""

    ray_count: int
    wavelength_m: float
    grid_shape: tuple[int, int]
    sample_pitch_m: tuple[float, float]
    normalization: str
    projection: str
    perturbation: str
    max_transverse_direction: float
    grid_nyquist_direction_limit: float
    grid_nyquist_satisfied: bool
    max_projection_factor: float
    min_projection_factor: float
    reconstructed_discrete_power: float
    incident_amplitude_power_sum: float
    ray_spacing_estimate_m: float | None
    max_adjacent_ray_phase_rad: float | None
    ray_density_status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "ray_count": self.ray_count,
            "wavelength_m": self.wavelength_m,
            "grid_shape": list(self.grid_shape),
            "sample_pitch_m": list(self.sample_pitch_m),
            "normalization": self.normalization,
            "projection": self.projection,
            "perturbation": self.perturbation,
            "max_transverse_direction": self.max_transverse_direction,
            "grid_nyquist_direction_limit": self.grid_nyquist_direction_limit,
            "grid_nyquist_satisfied": self.grid_nyquist_satisfied,
            "max_projection_factor": self.max_projection_factor,
            "min_projection_factor": self.min_projection_factor,
            "reconstructed_discrete_power": self.reconstructed_discrete_power,
            "incident_amplitude_power_sum": self.incident_amplitude_power_sum,
            "ray_spacing_estimate_m": self.ray_spacing_estimate_m,
            "max_adjacent_ray_phase_rad": self.max_adjacent_ray_phase_rad,
            "ray_density_status": self.ray_density_status,
        }


def grid_nyquist_direction_limit(wavelength_m: float, pitch_m: float) -> float:
    """Largest transverse direction cosine a grid of this pitch can represent.

    A wavelet with transverse direction cosine ``d_t`` writes a phase ramp of
    spatial frequency ``d_t / lambda`` onto the plane. The grid resolves it only
    below the Nyquist frequency ``1 / (2 * pitch)``, giving

        |d_t| <= lambda / (2 * pitch)

    This is a condition on the **output grid**, and is distinct from whether the
    ray ensemble samples the wavefront densely enough. Both can fail
    independently, and refining one does not fix the other -- which is the usual
    wasted debugging step.
    """
    return wavelength_m / (2.0 * pitch_m)


def compute_precision_for(bundle: RayBundle) -> Precision:
    """The precision this coupler will accumulate phase in for ``bundle``.

    Taken from the data the bundle actually carries -- geometry and amplitude --
    and floored at the coupler's declared minimum. Distinct from the input dtype
    and from the output dtype on purpose (PB4b section 9): a bundle handed in as
    float16 is computed in float32, and calling that "float16 support" would be
    advertising a cast.
    """
    precisions = [dtype_of(bundle.positions_m).precision]
    if bundle.amplitude is not None:
        precisions.append(dtype_of(bundle.amplitude).precision)
    if bundle.optical_path_length_m is not None:
        precisions.append(dtype_of(bundle.optical_path_length_m).precision)
    floor = C_RAY_TO_WAVE_CAPABILITIES.minimum_compute_precision
    return max([*precisions, floor], key=lambda p: p.bits)


def _cis(xp: Any, phase: Any, complex_dtype: DType) -> Any:
    """``exp(i * phase)`` in an explicitly chosen complex dtype.

    Written out rather than left to ``xp.exp(1j * phase)`` because that relies
    on scalar-promotion rules that differ between NumPy versions and between
    NumPy and JAX-with-x64-disabled. The dtype of a reconstructed field is part
    of this coupler's contract, so it is stated, not inherited.
    """
    return xp.exp(phase.astype(numpy_dtype(complex_dtype)) * 1j)


def _ray_density_diagnostic(
    positions_xy: Any, directions_xy: Any, wavenumber: float, xp: Any
) -> tuple[float | None, float | None, str]:
    """Estimate the worst phase step between neighbouring rays at the plane.

    The wavelet picture holds locally only while adjacent rays differ by less
    than half a cycle. For each ray, find its nearest neighbour and evaluate the
    phase disagreement between the two ramps at their midpoint separation.
    """
    count = positions_xy.shape[0]
    if count < 2:
        return None, None, "not_applicable_single_ray"
    if count > _NEAREST_NEIGHBOUR_SCAN_LIMIT:
        return None, None, "not_computed_above_scan_limit"

    delta = positions_xy[:, None, :] - positions_xy[None, :, :]
    distances = xp.linalg.norm(delta, axis=2)
    # `np.fill_diagonal` mutates in place, which JAX arrays do not support. The
    # functional form masks the self-distance instead and is identical on both.
    distances = xp.where(xp.eye(count, dtype=bool), xp.inf, distances)
    neighbour = xp.argmin(distances, axis=1)
    separation = distances[xp.arange(count), neighbour]

    # Phase difference accumulated between a ray and its neighbour's ramp over
    # their separation: k * |(d_i - d_j) . (r_i - r_j)|.
    direction_difference = directions_xy - directions_xy[neighbour]
    offsets = positions_xy - positions_xy[neighbour]
    phase_step = wavenumber * xp.abs(xp.sum(direction_difference * offsets, axis=1))

    return float(xp.mean(separation)), float(xp.max(phase_step)), "computed"


def ray_to_wave(
    bundle: RayBundle,
    *,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    plane: ReferencePlane | None = None,
    normalization: Literal["none", "one_over_n"] | None = None,
    projection: Projection = Projection.ASM_CONSISTENT,
    perturbation: Perturbation = Perturbation(),
    enforce_grid_nyquist: bool = True,
    compute_precision: Precision | None = None,
) -> tuple[ComplexField, ReconstructionDiagnostics]:
    """Reconstruct the complex field a ray bundle produces on a plane.

    Parameters
    ----------
    bundle
        Rays carrying a complex amplitude and an OPL with a declared reference.
        A bundle carrying only an Optiland weight, or only ``opd_native``, is
        refused here -- see :meth:`RayBundle.require_coherent`.
    grid_shape, sample_pitch_m
        Output grid as ``(ny, nx)`` and ``(dy, dx)`` in metres. Coordinate zero
        is at index ``n // 2`` on each axis, matching the M1-pinned convention.
    normalization
        ``"none"`` sums a given physical ray ensemble (main-text eq 2).
        ``"one_over_n"`` applies the ``1/N`` of SI eqs S3/S5, which belongs only
        when the ensemble is a Monte Carlo sample of a spectrum. ``None``, the
        default, takes the bundle's own ``reconstruction_normalization``
        declaration -- the bundle knows which kind of ensemble it is, and making
        every caller restate it invites the two to disagree. The resolved choice
        is recorded in the output metadata either way.
    compute_precision
        Precision to accumulate phase in. ``None`` derives it from the bundle
        (:func:`compute_precision_for`), floored at the coupler's declared
        minimum. The output field is the complex dtype of this precision, so a
        float32 GPU bundle yields a complex64 field on the same device and a
        float64 host bundle yields complex128 on the host -- the historical
        behaviour, unchanged, for the historical input.

    Notes
    -----
    One implementation serves every device. The array module is taken from the
    bundle, so a NumPy bundle executes in NumPy on the host and a JAX bundle
    executes in JAX wherever that array lives; there is no CPU branch and no GPU
    branch to drift apart. Nothing here calls ``np.asarray`` on the bundle's
    data, which is what previously forced every reconstruction onto the host.

    Returns
    -------
    The reconstructed field, and diagnostics that are reported rather than
    judged -- power accounting, projection-factor range, and the two sampling
    conditions.
    """
    amplitude, optical_path_length = bundle.require_coherent()

    if normalization is None:
        normalization = bundle.reconstruction_normalization
    plane = plane or bundle.reference_plane
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dy, dx = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    if ny <= 0 or nx <= 0:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"grid_shape must be positive, got {grid_shape!r}",
            declaration="grid_shape",
        )

    # The one place the execution representation is chosen. Everything below is
    # written against `xp` and the two dtypes, so the same source runs on the
    # host and on a GPU.
    xp = xp_for(namespace_of(bundle.positions_m))
    # Dot products must compute at the dtype they claim; on a GPU that takes an
    # explicit request. See core.arrays.matmul_precision_kwargs.
    dot = matmul_precision_kwargs(namespace_of(bundle.positions_m))
    precision = compute_precision or compute_precision_for(bundle)
    real_dtype = precision.real_dtype
    complex_dtype = precision.complex_dtype
    if complex_dtype is None:  # pragma: no cover - guarded by the FP32 floor
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            f"{precision} has no complex dtype to reconstruct a field in",
            declaration="compute_precision",
        )
    real_np, complex_np = numpy_dtype(real_dtype), numpy_dtype(complex_dtype)

    wavenumber = bundle.wavenumber
    positions_xy = bundle.positions_m[:, :2].astype(real_np)
    directions_xy = bundle.directions[:, :2].astype(real_np)

    # Projection factor <n_hat, d_hat>. For the usual +z plane this is the
    # direction's z component; the general form keeps a tilted plane honest.
    # Always computed so it can be reported, but applied only under the
    # sensor convention -- see the module docstring for why the coupler
    # default omits it.
    normal = xp.asarray(plane.normal, dtype=real_np)
    projection_factor = xp.matmul(bundle.directions.astype(real_np), normal, **dot)
    apply_factor = (
        projection is Projection.SENSOR_OBLIQUITY and perturbation.apply_projection_factor
    )
    weight = (
        projection_factor.astype(complex_np)
        if apply_factor
        else xp.ones(bundle.count, dtype=complex_np)
    )

    # Grid coordinates, origin at index n // 2 (M1 convention, implemented in
    # ComplexField.coordinates and mirrored here for the phase ramps).
    y = (xp.arange(ny, dtype=real_np) - ny // 2) * dy
    x = (xp.arange(nx, dtype=real_np) - nx // 2) * dx

    sign = float(perturbation.phase_sign)

    # Constant per-ray phase: the OPL, minus the ramp evaluated back at the
    # ray's own intersection point so that dr is measured from there.
    constant_phase = optical_path_length.astype(real_np)
    if perturbation.apply_oblique_ramp:
        constant_phase = constant_phase - xp.sum(directions_xy * positions_xy, axis=1)

    coefficient = (
        amplitude.astype(complex_np)
        * weight
        * _cis(xp, sign * wavenumber * constant_phase, complex_dtype)
    )

    if perturbation.apply_oblique_ramp:
        # exp(i k (dx_i x + dy_i y)) is separable in x and y, so the O(N ny nx)
        # sum contracts from two O(N n) factors instead of materializing an
        # (N, ny, nx) tensor.
        ramp_y = _cis(xp, sign * wavenumber * xp.outer(directions_xy[:, 1], y), complex_dtype)
        ramp_x = _cis(xp, sign * wavenumber * xp.outer(directions_xy[:, 0], x), complex_dtype)
        u = xp.einsum("n,ny,nx->yx", coefficient, ramp_y, ramp_x, optimize=True, **dot)
    else:
        # Deliberately wrong: every ray deposits a piston over the whole plane.
        u = xp.full((ny, nx), coefficient.sum(), dtype=complex_np)

    # Resolved vs actual, checked rather than assumed. Under JAX with
    # jax_enable_x64 disabled a complex128 request comes back complex64 in
    # silence, so a coupler that reported its *resolved* dtype would be
    # reporting a precision it did not compute in.
    u = verify_dtype(u, complex_dtype, context="C_RAY_TO_WAVE")

    if normalization == "one_over_n":
        u = u / bundle.count

    if perturbation.transpose_axes:
        u = u.T
        ny, nx = nx, ny
        dy, dx = dx, dy

    # The Nyquist condition is per axis, not on the direction norm. A diagonal
    # FFT bin has |d| = sqrt(2) * lambda / (2 * pitch) yet is exactly
    # representable, because each component sits at its own axis limit. Testing
    # the norm rejects the corner modes of any square spectrum -- which is how
    # this was found: the CHE-26 round trip could not enumerate its own bins.
    limit_x = grid_nyquist_direction_limit(bundle.wavelength_m, dx)
    limit_y = grid_nyquist_direction_limit(bundle.wavelength_m, dy)
    max_du = float(xp.max(xp.abs(directions_xy[:, 0]))) if bundle.count else 0.0
    max_dv = float(xp.max(xp.abs(directions_xy[:, 1]))) if bundle.count else 0.0
    # Reported as the worst per-axis utilisation, so a single number still says
    # whether the grid is adequate.
    max_transverse = max(max_du, max_dv)
    nyquist_limit = min(limit_x, limit_y)
    nyquist_satisfied = bool(max_du <= limit_x and max_dv <= limit_y)
    if enforce_grid_nyquist and not nyquist_satisfied:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            (
                f"output grid cannot represent the steepest wavelet ramp: "
                f"|d_u|max = {max_du:.6f} against limit {limit_x:.6f}, "
                f"|d_v|max = {max_dv:.6f} against limit {limit_y:.6f} "
                f"(lambda / (2 * pitch), per axis)"
            ),
            declaration="sample_pitch_m",
            remedy=(
                "Refine the output pitch, or restrict the ray directions. Adding "
                "more rays will not help: this is a grid condition, not a ray-density one."
            ),
        )

    spacing, max_phase_step, density_status = _ray_density_diagnostic(
        positions_xy, directions_xy, wavenumber, xp
    )

    field = ComplexField(
        u=u,
        sample_pitch_m=(dy, dx),
        wavelength_m=bundle.wavelength_m,
        reference_plane=plane,
        frame=Frame(),
        normalization=(
            "u is complex amplitude; discrete power = sum(|u|^2) * dy * dx; "
            f"ray-sum normalization = {normalization}; projection = {projection}"
        ),
        provenance={
            "coupler": "C_RAY_TO_WAVE",
            "equation": (
                "ACS Photonics 2026 SI eq S5 (no obliquity factor)"
                if projection is Projection.ASM_CONSISTENT
                else "ACS Photonics 2026 main text eq 2 (with <n,d> obliquity)"
            ),
            "projection": str(projection),
            "ray_count": bundle.count,
            "perturbation": perturbation.describe(),
            "source_reference_plane": bundle.reference_plane.name,
            "optical_path_length_reference": bundle.optical_path_length_reference,
            # CHE-50, declared rather than silently carried. Static text derived
            # from CHE-38's measurement; no numerics read it and none depend on
            # it, so the kernel is unchanged. See this module's docstring.
            "validity": {
                "valid_at": "reference_plane as labelled, with zero further propagation",
                "wavefront_curvature_term": "absent",
                "further_propagation_verified": False,
                "note": (
                    "The reconstruction is linear in the transverse coordinate and "
                    "carries no exp(i k r^2 / 2R) term. Invisible in |U|^2; not "
                    "invisible to a subsequent propagation (~1.2 rad against an "
                    "exact spherical-wave reference at the 5-Airy-radius gate edge "
                    "on M3-SINGLET-REF, CHE-38). To move the handoff, advance the "
                    "ray state to the new plane and reconstruct again."
                ),
                "disposition": (
                    "CHE-50: tracked known limitation, no kernel change. Revisit "
                    "when a propagation-sensitive hybrid composition requires it."
                ),
            },
            # Requested / resolved / actual, all three distinguishable. The
            # input state is what arrived, the compute precision is what the
            # coupler chose, and the output state is read back off the array
            # that was actually produced -- never asserted from the request.
            "execution": {
                "input": bundle.state.as_dict(),
                "compute_precision": str(precision),
                "compute_dtype": str(complex_dtype),
                "output": ArrayState(
                    dtype_of(u), device_of(u), namespace_of(u)
                ).as_dict(),
            },
        },
    )

    diagnostics = ReconstructionDiagnostics(
        ray_count=bundle.count,
        wavelength_m=bundle.wavelength_m,
        grid_shape=(ny, nx),
        sample_pitch_m=(dy, dx),
        normalization=normalization,
        projection=str(projection),
        perturbation=perturbation.describe(),
        max_transverse_direction=max_transverse,
        grid_nyquist_direction_limit=nyquist_limit,
        grid_nyquist_satisfied=nyquist_satisfied,
        max_projection_factor=float(xp.max(projection_factor)),
        min_projection_factor=float(xp.min(projection_factor)),
        reconstructed_discrete_power=field.discrete_power(),
        incident_amplitude_power_sum=float(xp.sum(xp.abs(amplitude) ** 2)),
        ray_spacing_estimate_m=spacing,
        max_adjacent_ray_phase_rad=max_phase_step,
        ray_density_status=(
            density_status
            if density_status != "computed"
            else (
                "wavelet_approximation_holds"
                if max_phase_step is not None and max_phase_step < math.pi
                else "adjacent_ray_phase_step_exceeds_pi"
            )
        ),
    )

    return field, diagnostics


def collimated_bundle(
    *,
    positions_xy_m: np.ndarray,
    direction: tuple[float, float, float],
    wavelength_m: float,
    plane_z_m: float = 0.0,
    plane_name: str = "reconstruction plane",
    amplitude: complex = 1.0 + 0.0j,
    precision: Precision = Precision.FP64,
    namespace: Any = None,
) -> RayBundle:
    """Build the SI Figure S1c test bundle: one angular mode, many launch points.

    Rays share a direction but not a launch point, so each is given the OPL its
    lateral position implies, ``OPL_j = d_hat . r_j``. With those phases the
    ensemble represents a single plane-wave mode; without them it does not, and
    that is exactly what makes this the sharpest available check on the
    ray->wave direction. The analytic oracle ``exp(+i k d_hat . r)`` is exact,
    so the tolerance can be derived from dtype round-off rather than chosen.

    ``precision`` and ``namespace`` exist so the execution matrix can build the
    *same* bundle in float32 on a GPU and compare against the float64 host
    reference. The geometry is always constructed in float64 and cast once at
    the end: building it in float32 would fold construction round-off into the
    thing being measured, and the direction would fail its own unit-norm check
    for a reason that has nothing to do with the coupler.
    """
    positions_xy_m = np.asarray(positions_xy_m, dtype=np.float64)
    if positions_xy_m.ndim != 2 or positions_xy_m.shape[1] != 2:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"positions_xy_m must be (N, 2), got {positions_xy_m.shape}",
            declaration="positions_xy_m",
        )
    direction_array = np.asarray(direction, dtype=np.float64)
    direction_array = direction_array / np.linalg.norm(direction_array)

    count = positions_xy_m.shape[0]
    positions = np.column_stack(
        [positions_xy_m, np.full(count, plane_z_m, dtype=np.float64)]
    )
    directions = np.tile(direction_array, (count, 1))
    optical_path_length = positions @ direction_array
    amplitudes = np.full(count, amplitude, dtype=np.complex128)

    complex_dtype = precision.complex_dtype
    if complex_dtype is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            f"{precision} has no complex dtype, so it cannot carry a ray amplitude",
            declaration="precision",
        )
    # core.arrays.asarray, not xp.asarray: it verifies the cast landed. JAX with
    # jax_enable_x64 disabled accepts a float64 request and returns float32 with
    # only a UserWarning, so building the bundle directly would hand back an FP32
    # artifact labelled FP64.
    real, complex_ = precision.real_dtype, complex_dtype
    kwargs = {"namespace": namespace} if namespace is not None else {}

    return RayBundle(
        positions_m=asarray(positions, dtype=real, **kwargs),
        directions=asarray(directions, dtype=real, **kwargs),
        wavelength_m=wavelength_m,
        reference_plane=ReferencePlane(name=plane_name, z_m=plane_z_m),
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=asarray(amplitudes, dtype=complex_, **kwargs),
        optical_path_length_m=asarray(optical_path_length, dtype=real, **kwargs),
        optical_path_length_reference="origin of the global frame, along d_hat",
    )
