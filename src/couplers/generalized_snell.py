"""GENERALIZED_SNELL: the reduced-order diffractive interaction (CHE-143, M2.7).

The third model inside :mod:`couplers.interaction`. Unlike ``FULL_FIELD`` and
``LOCAL_PATCH``, no field is formed at all: each incident ray is redirected by
a local grating equation, evaluated at its own transverse position, and comes
out the other side still one ray.

    k_t^out = n_i k0 d_t^in + m grad_t(phi)(x, y)
    k_n^out = sqrt( (n_t k0)^2 - |k_t^out|^2 )
    opl^out = opl^in + m phi(x, y) / k0

``phi`` is read off the same complex ``DiffractiveSurface.transmission`` the
other two models use (``t = |t| exp(i phi)``), so a caller declares one surface
regardless of which model computes the interaction. The local phase gradient is
estimated directly from the complex transmission via

    d phi / du ~= angle( t[+1] * conj(t[-1]) ) / (2 du)

rather than by unwrapping and differencing ``angle(t)``: the two agree
wherever unwrapping would succeed, but the complex form has no unwrap step to
fail, and for a genuine phase ramp it is exact to round-off at any pixel
pitch, because ``angle`` of a unit-modulus product is exact -- which is what
the linear-ramp acceptance case needs. It silently returns the wrong answer
(aliased by a multiple of ``2 pi / (2 * pitch)``) when the true phase step
between the two samples exceeds ``pi``, which is exactly the condition the
local-gradient-smoothness predicate below is built to catch.

Only the planar substrate executes. A conformal one needs a per-ray local
tangent frame -- the surface's normal is position-dependent and this module
has no way to declare one -- so it is refused rather than approximated with
the flat-plane frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.boundary import ContractCode, ContractError, RayBundle, ReferencePlane

__all__ = [
    "GeneralizedSnellDiagnostics",
    "generalized_snell_step",
    "local_gradient_smoothness_margin",
    "propagating_order_margin",
    "single_order_dominance",
]


def _fractional_margin(value: float, limit: float) -> float:
    """``(limit - value) / limit`` for an upper bound. See predicates.py's twin."""
    if math.isinf(limit):
        return math.inf
    if limit <= 0.0 or not math.isfinite(limit):
        raise ValueError(f"a validity limit must be a positive finite number, got {limit!r}")
    return (limit - float(value)) / limit


def propagating_order_margin(
    k_t_out_sq: np.ndarray[Any, Any], *, n_transmitted: float, k0: float
) -> np.ndarray[Any, Any]:
    """Signed margin of ``|k_t^out| < n_t k0`` -- predicate 1, a hard limit.

    ``> 0`` the order propagates, ``0`` grazing, ``< 0`` evanescent: the
    requested order does not exist as an outgoing ray.
    """
    limit_sq = (float(n_transmitted) * float(k0)) ** 2
    return (limit_sq - np.asarray(k_t_out_sq, dtype=np.float64)) / limit_sq


def local_gradient_smoothness_margin(
    curvature_rad_per_m2: np.ndarray[Any, Any],
    worst_raw_step_rad: np.ndarray[Any, Any],
    *,
    transverse_scale_m: float,
) -> np.ndarray[Any, Any]:
    """Signed margin of predicate 2: the local plane-wave picture holds.

    The worse of two sub-checks, because either alone misses a real failure:

    * **Curvature vs. the declared transverse scale.** The phase change the
      local curvature accumulates over ``D`` must stay well under one fringe
      (``2 pi``, bounded here at ``pi`` for a symmetric two-sided budget) for
      "one locally redirected ray" to mean anything.
    * **The raw estimator step vs. the wrap boundary.** A wrapped phase step
      between two samples that approaches ``pi`` aliases the finite-difference
      gradient by a large, spurious amount -- and a *uniformly* undersampled
      grating aliases every tap the same wrong way, which the curvature check
      alone reads as "smooth" because the (wrong) gradient looks locally
      constant. This sub-check catches that case directly, from the raw step
      rather than from what the (possibly wrong) gradient implies.
    """
    curvature = np.abs(np.asarray(curvature_rad_per_m2, dtype=np.float64))
    phase_change = curvature * float(transverse_scale_m) ** 2
    curvature_margin = (math.pi - phase_change) / math.pi
    step_margin = (math.pi - np.abs(np.asarray(worst_raw_step_rad, dtype=np.float64))) / math.pi
    return np.minimum(curvature_margin, step_margin)


def _nearest_index(coord_m: np.ndarray[Any, Any], pitch_m: float, n: int) -> np.ndarray[Any, Any]:
    """Nearest sample index, matching ``couplers.patch.extract_patch``'s rule."""
    return np.round(coord_m / pitch_m).astype(np.int64) + n // 2


def _local_phase_gradient(
    transmission: np.ndarray[Any, Any],
    *,
    sample_pitch_m: tuple[float, float],
    iy: np.ndarray[Any, Any],
    ix: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], ...]:
    """Per-ray local phase, its gradient, its curvature, and the estimator's
    worst raw wrapped step.

    Returns ``(phase, grad_y, grad_x, curvature, worst_raw_step)``, all
    evaluated by a 5-tap stencil on the complex transmission at the ray's
    nearest sample. Indices are clipped to the interior so a ray near the edge
    reads the nearest interior stencil rather than wrapping or going out of
    bounds; a genuinely linear surface (the acceptance case) is unaffected by
    this because its gradient does not depend on which interior samples are
    read.

    ``worst_raw_step`` is the largest-magnitude wrapped phase step (each is
    ``angle(t_plus * conj(t_minus))``, in ``(-pi, pi]``) any of the six taps
    used. It exists because the curvature check alone can miss a real
    failure: a *uniformly* undersampled grating aliases every tap by
    consistently the same wrong amount, which looks perfectly smooth (zero
    curvature) while the gradient itself is nonsense. A step whose magnitude
    approaches ``pi`` is the direct symptom of that aliasing, independent of
    whether the resulting (wrong) gradient happens to look locally constant.
    """
    ny, nx = transmission.shape
    pitch_y, pitch_x = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    # The ray's own phase and amplitude are read at its TRUE nearest sample,
    # clipped only to stay in-array -- never at the interior-clamped `cy`/`cx`
    # below. The stencil needs an interior centre to keep every tap in bounds,
    # but using that same clamped location for the ray's own phase would read
    # a DIFFERENT pixel than the amplitude (which uses the true index), a
    # silent mismatch for any ray within 2 samples of an edge.
    phase = np.angle(transmission[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)])
    cy = np.clip(iy, 2, ny - 3)
    cx = np.clip(ix, 2, nx - 3)

    def step(t_plus: np.ndarray[Any, Any], t_minus: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return np.angle(t_plus * np.conj(t_minus))

    step_y = step(transmission[cy + 1, cx], transmission[cy - 1, cx])
    step_x = step(transmission[cy, cx + 1], transmission[cy, cx - 1])
    grad_y = step_y / (2.0 * pitch_y)
    grad_x = step_x / (2.0 * pitch_x)

    step_plus_y = step(transmission[cy + 2, cx], transmission[cy, cx])
    step_minus_y = step(transmission[cy, cx], transmission[cy - 2, cx])
    curvature_y = (step_plus_y / (2.0 * pitch_y) - step_minus_y / (2.0 * pitch_y)) / (2.0 * pitch_y)

    step_plus_x = step(transmission[cy, cx + 2], transmission[cy, cx])
    step_minus_x = step(transmission[cy, cx], transmission[cy, cx - 2])
    curvature_x = (step_plus_x / (2.0 * pitch_x) - step_minus_x / (2.0 * pitch_x)) / (2.0 * pitch_x)

    curvature = np.sqrt(curvature_y**2 + curvature_x**2)
    worst_raw_step = np.maximum.reduce(
        [np.abs(step_y), np.abs(step_x), np.abs(step_plus_y), np.abs(step_minus_y),
         np.abs(step_plus_x), np.abs(step_minus_x)]
    )
    return phase, grad_y, grad_x, curvature, worst_raw_step


def single_order_dominance(
    transmission: np.ndarray[Any, Any],
    *,
    sample_pitch_m: tuple[float, float],
    center_xy_m: tuple[float, float],
    patch_px: int,
    wavelength_m: float,
    target_dir_xy: tuple[float, float],
    pad_factor: int = 2,
) -> tuple[float, float]:
    """Fraction of local spectral power in the requested order, and its margin.

    Predicate 3. A local window of ``exp(i phi)`` (the phase alone -- amplitude
    modulation is not part of "how many orders the phase profile emits") is
    transformed, following exactly the window-then-FFT idiom
    ``couplers.patch.patch_secondary_rays`` uses for the same purpose.

    Dominance sums power over a disk around the requested order's direction,
    not a single bin: ``resolve_pad_px`` zero-pads well past ``patch_px`` (to
    avoid the reconstruction replica aliasing that machinery exists for), which
    *interpolates* the window's own DTFT onto a much finer grid than its native
    resolution. Reading a single interpolated bin therefore reports a small
    slice of one order's power, not the order's power -- the disk radius is the
    window's native angular resolution, ``lambda / (patch_px * pitch)``,
    matching the mainlobe width a rectangular window of that size actually has.
    ``margin = 2*dominance - 1`` so a bare majority is the boundary and full
    concentration is ``+1``.
    """
    from couplers.patch import extract_patch, resolve_pad_px

    phase_only = np.exp(1j * np.angle(np.asarray(transmission)))
    grid_n = int(np.asarray(transmission).shape[0])
    pad = resolve_pad_px(grid_n=grid_n, patch_px=int(patch_px))

    patch = extract_patch(
        phase_only, center_xy_m=center_xy_m, patch_px=int(patch_px), sample_pitch_m=sample_pitch_m
    )
    off = (pad - int(patch_px)) // 2
    padded = np.zeros((pad, pad), dtype=np.complex128)
    padded[off : off + int(patch_px), off : off + int(patch_px)] = patch
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(padded))) / (pad * pad)

    pitch_y, pitch_x = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    fy = np.fft.fftshift(np.fft.fftfreq(pad, d=pitch_y))
    fx = np.fft.fftshift(np.fft.fftfreq(pad, d=pitch_x))
    grid_fx, grid_fy = np.meshgrid(fx, fy)
    dir_x = grid_fx * float(wavelength_m)
    dir_y = grid_fy * float(wavelength_m)
    propagating = (dir_x**2 + dir_y**2) < 1.0

    power = np.abs(spectrum) ** 2
    total = float(power[propagating].sum())
    if total <= 0.0:
        return 0.0, -1.0

    du, dv = float(target_dir_xy[0]), float(target_dir_xy[1])
    resolution = float(wavelength_m) / (float(patch_px) * math.sqrt(pitch_y * pitch_x))
    within_order = propagating & (((dir_x - du) ** 2 + (dir_y - dv) ** 2) < resolution**2)
    dominance = float(power[within_order].sum()) / total
    margin = 2.0 * dominance - 1.0
    return dominance, margin


@dataclass(frozen=True)
class GeneralizedSnellDiagnostics:
    """What ran, in a form ``tests/`` and a benchmark record can both read."""

    order: int
    n_incident: float
    n_transmitted: float
    patch_px: int
    transverse_scale_m: float
    outgoing_ray_count: int
    worst_propagating_order_margin: float
    worst_local_gradient_smoothness_margin: float
    single_order_dominance: float
    single_order_dominance_margin: float
    substrate: str
    opl_convention: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "n_incident": self.n_incident,
            "n_transmitted": self.n_transmitted,
            "patch_px": self.patch_px,
            "transverse_scale_m": self.transverse_scale_m,
            "outgoing_ray_count": self.outgoing_ray_count,
            "worst_propagating_order_margin": self.worst_propagating_order_margin,
            "worst_local_gradient_smoothness_margin": self.worst_local_gradient_smoothness_margin,
            "single_order_dominance": self.single_order_dominance,
            "single_order_dominance_margin": self.single_order_dominance_margin,
            "substrate": self.substrate,
            "opl_convention": self.opl_convention,
        }


def generalized_snell_step(
    bundle: RayBundle,
    transmission: np.ndarray[Any, Any],
    *,
    sample_pitch_m: tuple[float, float],
    plane: ReferencePlane,
    n_incident: float,
    n_transmitted: float,
    order: int,
    patch_px: int,
) -> tuple[RayBundle, GeneralizedSnellDiagnostics]:
    """The per-ray local grating equation. See the module docstring for the physics.

    Refuses (``ContractError``) rather than returning a ray with no physical
    meaning when: the requested order is evanescent for any ray
    (``ContractCode.MODEL_NOT_APPLICABLE`` -- the order genuinely does not
    exist for this surface and indices, not a missing declaration), or the
    local phase varies too fast for the gradient estimate to be trusted
    (``ContractCode.MISSING_DECLARATION`` -- the surface needs a finer pitch
    or a smaller declared patch, which is a different remedy than a bad index).
    """
    amplitude_in, opl_in = bundle.require_coherent()
    transmission = np.asarray(transmission)
    if transmission.ndim != 2 or not np.iscomplexobj(transmission):
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"transmission must be a 2-D complex (ny, nx) grid, got shape "
            f"{transmission.shape}, dtype {transmission.dtype}",
            declaration="transmission",
        )

    k0 = bundle.wavenumber
    positions_xy = bundle.positions_m[:, :2].astype(np.float64)
    directions_xy = bundle.directions[:, :2].astype(np.float64)

    ny, nx = transmission.shape
    pitch_y, pitch_x = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    iy = _nearest_index(positions_xy[:, 1], pitch_y, ny)
    ix = _nearest_index(positions_xy[:, 0], pitch_x, nx)

    phase, grad_y, grad_x, curvature, worst_raw_step = _local_phase_gradient(
        transmission, sample_pitch_m=sample_pitch_m, iy=iy, ix=ix
    )

    transverse_scale_m = float(patch_px) * math.sqrt(pitch_y * pitch_x)
    smoothness_margin = local_gradient_smoothness_margin(
        curvature, worst_raw_step, transverse_scale_m=transverse_scale_m
    )
    worst_smoothness = float(np.min(smoothness_margin))
    if worst_smoothness < 0.0:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "the local phase gradient estimate is not reliable at the declared "
            f"transverse scale (worst signed margin {worst_smoothness:.3e} against "
            "predicate LOCAL_GRADIENT_SMOOTHNESS): the phase varies too fast, "
            "relative to the sample pitch and the declared patch scale, for a "
            "single local plane wave to describe the response here",
            declaration="patch_px",
            remedy=(
                "Sample the surface's phase on a finer grid, or declare a smaller "
                "patch_px. This is not the evanescent-order refusal: the gradient "
                "itself cannot be trusted yet, independent of where it points."
            ),
        )

    k_t_out_y = float(n_incident) * k0 * directions_xy[:, 1] + float(order) * grad_y
    k_t_out_x = float(n_incident) * k0 * directions_xy[:, 0] + float(order) * grad_x
    k_t_out_sq = k_t_out_y**2 + k_t_out_x**2

    order_margin = propagating_order_margin(k_t_out_sq, n_transmitted=n_transmitted, k0=k0)
    worst_order_margin = float(np.min(order_margin))
    if worst_order_margin < 0.0:
        raise ContractError(
            ContractCode.MODEL_NOT_APPLICABLE,
            "the requested diffraction order is evanescent for at least one ray "
            f"(worst signed margin {worst_order_margin:.3e} against predicate "
            f"PROPAGATING_ORDER_EXISTS): m={order} has no outgoing propagating "
            "direction at this position for n_incident="
            f"{n_incident}, n_transmitted={n_transmitted}",
            declaration="order",
            remedy=(
                "Request a different order, or accept that this configuration has "
                "no propagating solution here. Returning a normalized nonsense "
                "direction would be worse than refusing."
            ),
        )

    n_t_k0 = float(n_transmitted) * k0
    k_n_out = np.sqrt(np.clip(n_t_k0**2 - k_t_out_sq, 0.0, None))
    dir_x_out = k_t_out_x / n_t_k0
    dir_y_out = k_t_out_y / n_t_k0
    dir_z_out = k_n_out / n_t_k0
    directions_out = np.column_stack([dir_x_out, dir_y_out, dir_z_out])

    positions_out = np.column_stack(
        [positions_xy[:, 0], positions_xy[:, 1], np.full(positions_xy.shape[0], plane.z_m)]
    )

    local_transmission = transmission[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
    amplitude_out = amplitude_in * np.abs(local_transmission)
    # `m * phi`, not `phi`: the order factor belongs to the OPL for exactly the
    # same reason it belongs to the momentum. The local plane-wave factor of the
    # m-th order is exp(i m phi(x, y)) -- differentiate it and you get the
    # `m grad(phi)` above; evaluate it and you get `m phi`. Carrying `phi` alone
    # was CHE-148's finding (M2.12, B3-DOE-INLINE-ORDER-MINUS1), and two
    # code-independent arguments say which form is right:
    #
    #   * exp(i (-1) phi) and exp(i (+1) (-phi)) are the SAME complex factor, so
    #     `(order=-1, phi)` and `(order=+1, conj(t))` must return the same bundle.
    #     They already returned the same DIRECTION; with `phi` alone they returned
    #     opposite optical paths, which is a contradiction rather than a tolerance.
    #     (That equality is BITWISE only away from the branch cut: at phi = pi both
    #     `angle(t)` and `angle(conj(t))` return +pi, so the two differ there by one
    #     whole wave -- physically inert, and not a case the identity covers.)
    #   * `order=0` is the undiffracted transmission and picks up no ramp at all;
    #     with `phi` alone it was given the whole ramp phase on an undeflected ray.
    #
    # For `order=1` -- the default, every shipped test and every recorded run
    # before CHE-148 -- `float(1) * phase` is IEEE-exact, so this is bitwise the
    # previous behaviour there and changes only |m| != 1.
    opl_out = opl_in + float(order) * phase / k0

    outgoing = RayBundle(
        positions_m=positions_out,
        directions=directions_out,
        wavelength_m=bundle.wavelength_m,
        reference_plane=plane,
        frame=bundle.frame,
        amplitude=amplitude_out,
        optical_path_length_m=opl_out,
        optical_path_length_reference=(
            f"{bundle.optical_path_length_reference}; plus this generalized-Snell "
            f"surface's local phase m phi(x, y) / k0 at plane {plane.name!r}, order "
            f"m={order}"
        ),
        reconstruction_normalization=bundle.reconstruction_normalization,
    )

    centroid_xy = (float(np.mean(positions_xy[:, 0])), float(np.mean(positions_xy[:, 1])))
    centroid_target = (
        float(np.mean(dir_x_out)),
        float(np.mean(dir_y_out)),
    )
    dominance, dominance_margin = single_order_dominance(
        transmission,
        sample_pitch_m=sample_pitch_m,
        center_xy_m=centroid_xy,
        patch_px=patch_px,
        wavelength_m=bundle.wavelength_m,
        target_dir_xy=centroid_target,
    )

    diagnostics = GeneralizedSnellDiagnostics(
        order=int(order),
        n_incident=float(n_incident),
        n_transmitted=float(n_transmitted),
        patch_px=int(patch_px),
        transverse_scale_m=transverse_scale_m,
        outgoing_ray_count=int(outgoing.count),
        worst_propagating_order_margin=worst_order_margin,
        worst_local_gradient_smoothness_margin=worst_smoothness,
        single_order_dominance=dominance,
        single_order_dominance_margin=dominance_margin,
        substrate="planar",
        opl_convention=(
            "additive: outgoing OPL = incident OPL + m phi(x, y) / k0, at the ray's "
            "own transverse position; amplitude carries only |t(x, y)|. The order "
            "factor m is the same one that multiplies grad(phi) in the momentum "
            "equation, because both come from differentiating or evaluating the "
            "m-th order's local plane-wave factor exp(i m phi) -- see CHE-148"
        ),
    )
    return outgoing, diagnostics
