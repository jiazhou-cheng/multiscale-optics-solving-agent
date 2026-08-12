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

This module imports neither Optiland nor Chromatix. The coupler core is the
physics under test; if it could import an engine, a coupler defect could be
misattributed to engine behaviour and M1's independence evidence would stop
bounding the search. ``benchmarks/coupler_protocol.yaml`` declares the rule and
``tests/test_ray_to_wave.py`` enforces it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from multiscale_optics_agent.couplers.contracts import (
    ComplexField,
    ContractCode,
    ContractError,
    Frame,
    RayBundle,
    ReferencePlane,
)

__all__ = [
    "Perturbation",
    "ReconstructionDiagnostics",
    "ray_to_wave",
    "grid_nyquist_direction_limit",
]

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
    #: Drop the ``<n_hat, d_hat>`` projection factor. Undetectable at normal incidence.
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


def _ray_density_diagnostic(
    positions_xy: np.ndarray, directions_xy: np.ndarray, wavenumber: float
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
    distances = np.linalg.norm(delta, axis=2)
    np.fill_diagonal(distances, np.inf)
    neighbour = np.argmin(distances, axis=1)
    separation = distances[np.arange(count), neighbour]

    # Phase difference accumulated between a ray and its neighbour's ramp over
    # their separation: k * |(d_i - d_j) . (r_i - r_j)|.
    direction_difference = directions_xy - directions_xy[neighbour]
    offsets = positions_xy - positions_xy[neighbour]
    phase_step = wavenumber * np.abs(np.sum(direction_difference * offsets, axis=1))

    return float(np.mean(separation)), float(np.max(phase_step)), "computed"


def ray_to_wave(
    bundle: RayBundle,
    *,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    plane: ReferencePlane | None = None,
    normalization: Literal["none", "one_over_n"] = "none",
    perturbation: Perturbation = Perturbation(),
    enforce_grid_nyquist: bool = True,
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
        when the ensemble is a Monte Carlo sample of a spectrum. The choice is
        recorded in the output metadata rather than inferred.

    Returns
    -------
    The reconstructed field, and diagnostics that are reported rather than
    judged -- power accounting, projection-factor range, and the two sampling
    conditions.
    """
    amplitude, optical_path_length = bundle.require_coherent()

    plane = plane or bundle.reference_plane
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dy, dx = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    if ny <= 0 or nx <= 0:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"grid_shape must be positive, got {grid_shape!r}",
            declaration="grid_shape",
        )

    wavenumber = bundle.wavenumber
    positions_xy = bundle.positions_m[:, :2]
    directions_xy = bundle.directions[:, :2]

    # Projection factor <n_hat, d_hat>. For the usual +z plane this is the
    # direction's z component; the general form keeps a tilted plane honest.
    normal = np.asarray(plane.normal, dtype=np.float64)
    projection = bundle.directions @ normal
    if perturbation.apply_projection_factor:
        weight = projection.astype(np.complex128)
    else:
        weight = np.ones(bundle.count, dtype=np.complex128)

    # Grid coordinates, origin at index n // 2 (M1 convention, implemented in
    # ComplexField.coordinates and mirrored here for the phase ramps).
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx

    sign = float(perturbation.phase_sign)

    # Constant per-ray phase: the OPL, minus the ramp evaluated back at the
    # ray's own intersection point so that dr is measured from there.
    constant_phase = optical_path_length.astype(np.float64)
    if perturbation.apply_oblique_ramp:
        constant_phase = constant_phase - np.sum(directions_xy * positions_xy, axis=1)

    coefficient = amplitude * weight * np.exp(1j * sign * wavenumber * constant_phase)

    if perturbation.apply_oblique_ramp:
        # exp(i k (dx_i x + dy_i y)) is separable in x and y, so the O(N ny nx)
        # sum contracts from two O(N n) factors instead of materializing an
        # (N, ny, nx) tensor.
        ramp_y = np.exp(1j * sign * wavenumber * np.outer(directions_xy[:, 1], y))
        ramp_x = np.exp(1j * sign * wavenumber * np.outer(directions_xy[:, 0], x))
        u = np.einsum("n,ny,nx->yx", coefficient, ramp_y, ramp_x, optimize=True)
    else:
        # Deliberately wrong: every ray deposits a piston over the whole plane.
        u = np.full((ny, nx), coefficient.sum(), dtype=np.complex128)

    if normalization == "one_over_n":
        u = u / bundle.count

    if perturbation.transpose_axes:
        u = u.T
        ny, nx = nx, ny
        dy, dx = dx, dy

    transverse = np.linalg.norm(directions_xy, axis=1)
    max_transverse = float(np.max(transverse)) if bundle.count else 0.0
    nyquist_limit = min(
        grid_nyquist_direction_limit(bundle.wavelength_m, dy),
        grid_nyquist_direction_limit(bundle.wavelength_m, dx),
    )
    nyquist_satisfied = bool(max_transverse <= nyquist_limit)
    if enforce_grid_nyquist and not nyquist_satisfied:
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            (
                f"output grid cannot represent the steepest wavelet ramp: "
                f"max transverse direction cosine {max_transverse:.6f} exceeds the "
                f"grid limit lambda/(2*pitch) = {nyquist_limit:.6f}"
            ),
            declaration="sample_pitch_m",
            remedy=(
                "Refine the output pitch, or restrict the ray directions. Adding "
                "more rays will not help: this is a grid condition, not a ray-density one."
            ),
        )

    spacing, max_phase_step, density_status = _ray_density_diagnostic(
        positions_xy, directions_xy, wavenumber
    )

    field = ComplexField(
        u=u,
        sample_pitch_m=(dy, dx),
        wavelength_m=bundle.wavelength_m,
        reference_plane=plane,
        frame=Frame(),
        normalization=(
            "u is complex amplitude; discrete power = sum(|u|^2) * dy * dx; "
            f"ray-sum normalization = {normalization}"
        ),
        provenance={
            "coupler": "C_RAY_TO_WAVE",
            "equation": "ACS Photonics 2026 main text eq 2",
            "ray_count": bundle.count,
            "perturbation": perturbation.describe(),
            "source_reference_plane": bundle.reference_plane.name,
            "optical_path_length_reference": bundle.optical_path_length_reference,
        },
    )

    diagnostics = ReconstructionDiagnostics(
        ray_count=bundle.count,
        wavelength_m=bundle.wavelength_m,
        grid_shape=(ny, nx),
        sample_pitch_m=(dy, dx),
        normalization=normalization,
        perturbation=perturbation.describe(),
        max_transverse_direction=max_transverse,
        grid_nyquist_direction_limit=nyquist_limit,
        grid_nyquist_satisfied=nyquist_satisfied,
        max_projection_factor=float(np.max(projection)),
        min_projection_factor=float(np.min(projection)),
        reconstructed_discrete_power=field.discrete_power(),
        incident_amplitude_power_sum=float(np.sum(np.abs(amplitude) ** 2)),
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
) -> RayBundle:
    """Build the SI Figure S1c test bundle: one angular mode, many launch points.

    Rays share a direction but not a launch point, so each is given the OPL its
    lateral position implies, ``OPL_j = d_hat . r_j``. With those phases the
    ensemble represents a single plane-wave mode; without them it does not, and
    that is exactly what makes this the sharpest available check on the
    ray->wave direction. The analytic oracle ``exp(+i k d_hat . r)`` is exact,
    so the tolerance can be derived from dtype round-off rather than chosen.
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

    return RayBundle(
        positions_m=positions,
        directions=directions,
        wavelength_m=wavelength_m,
        reference_plane=ReferencePlane(name=plane_name, z_m=plane_z_m),
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=np.full(count, amplitude, dtype=np.complex128),
        optical_path_length_m=optical_path_length,
        optical_path_length_reference="origin of the global frame, along d_hat",
    )
