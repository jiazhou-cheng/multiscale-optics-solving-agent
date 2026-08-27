"""B3-DOE-INLINE / B4-DOE-INLINE, end to end: a DOE inside a refractive train.

CHE-148 (M2.12). The chain is the shipping path throughout::

    couplers.ray_to_wave.collimated_bundle                 (the incident bundle)
      [-> solvers.optiland.coherent_trace.trace_ray_batch  (M_RAY_OPTILAND, group 1)]
      -> couplers.interaction.diffractive_interaction       (model=GENERALIZED_SNELL)
      -> solvers.optiland.coherent_trace.trace_ray_batch    (M_RAY_OPTILAND, downstream)
      -> couplers.ray_to_wave.ray_to_wave                   (C_RAY_TO_WAVE, image plane)

Nothing here reimplements a coupler or a solver, and the diffractive model is
named at the call site rather than inferred. The one thing this module *does*
compute for itself is the reference: :func:`closed_form_outgoing` rebuilds the
outgoing bundle from the grating equation and the declared surface's phase,
sharing no line with ``couplers/generalized_snell.py``, which is what makes the
admissibility metrics an independent comparison rather than a tautology.

Like the other system drivers this runs outside ``GraphExecutor`` and builds its
record with ``record_from_probe``: there is no graph node for "trace a
caller-supplied ray population", the coherent-trace capability is reached
through the adapter directly, and inventing a graph around it would put a second
description of the chain beside the real one.

Run it::

    ./run.sh --no-build python benchmarks/systems/b3_doe_inline.py --family B3-DOE-INLINE --write
    ./run.sh --no-build python benchmarks/systems/b3_doe_inline.py --family B4-DOE-INLINE --write
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):  # direct `python benchmarks/systems/b3_doe_inline.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.boundary import ContractError, RayBundle, ReferencePlane
from core.capabilities import C_RAY_TO_WAVE_CAPABILITIES
from core.coherent_batch import CoherentRayBatch
from core.execution_record import DevicePrecisionObservation
from core.optical_system import (
    ApertureSpec,
    FieldSpec,
    IdealMaterialSpec,
    OpticalSystemSpec,
    PlaneGeometrySpec,
    SphericalGeometrySpec,
    SurfaceSpec,
    WavelengthSpec,
)
from core.paths import repository_root
from core.precision import DeviceKind, DevicePlacement, Precision
from couplers.interaction import (
    DiffractiveModel,
    DiffractiveSurface,
    GeneralizedSnellParameters,
    diffractive_interaction,
)
from couplers.ray_to_wave import Perturbation, Projection, collimated_bundle, ray_to_wave
from runtime.instance_runner import probe_refusal, record_from_probe
from solvers.optiland.builder import build_optiland_system
from solvers.optiland.coherent_trace import (
    configure_optiland_execution,
    plan_trace_bridges,
    surface_positions_m,
    trace_ray_batch,
)
from verification.evidence import InstanceRun, control_result, write_instance_record
from verification.families.b3_doe_inline import (
    B3_DOE_INLINE,
    B4_DOE_INLINE,
    DIFFRACTIVE_MODEL,
    PRESCRIPTION,
    PSF_WINDOW_HALF_EXTENT_FWHMS,
    TOPOLOGIES,
    airy_fwhm_m,
    analytic_order_direction,
    clear_aperture_margin,
    paraxial_image_side_na,
    paraxial_order_position_departure,
    psf_window_pitch_m,
    strehl_quantization,
)
from verification.result import (
    Measurement,
    NegativeControlOutcome,
    NegativeControlResult,
    UncertaintyBasis,
)
from verification.verifier import verify

__all__ = [
    "closed_form_outgoing",
    "declared_instance_ids",
    "differing_axes",
    "geometry",
    "measure",
    "run_chain",
    "run_instance",
]

ROOT = repository_root()
RECORDS_DIR = ROOT / "benchmarks" / "systems" / "records"
CPU = DevicePlacement(kind=DeviceKind.CPU, index=0)

LAMBDA_M = float(PRESCRIPTION["wavelength_m"])
K0 = 2.0 * math.pi / LAMBDA_M
R1_MM = float(PRESCRIPTION["radius_1_mm"])
R2_MM = float(PRESCRIPTION["radius_2_mm"])
T_MM = float(PRESCRIPTION["centre_thickness_mm"])
N_GLASS = float(PRESCRIPTION["refractive_index"])
CA_MM = float(PRESCRIPTION["clear_aperture_mm"])

#: Deterministic chain, so there is no ensemble to estimate an uncertainty from.
#: float64 round-off, taken on the largest quantity any metric divides by.
_FLOOR = 1e-15


# ---------------------------------------------------------------------------
# The paraxial ABCD of the prescription, by hand
# ---------------------------------------------------------------------------
#
# Hand-derived rather than read from Optiland's paraxial module, and that is the
# point: it is one of the four oracle pieces, so it must not share code with the
# thing it judges. `test_b3_doe_inline.py` checks it against PRESCRIPTION's own
# read-back values (2.3e-11 relative) so the two cannot drift apart silently.
# Convention: the state vector is (height, ray angle u = tan(theta)), so a
# translation carries the GEOMETRIC distance in the medium and a refraction is
# [[1, 0], [-(n2 - n1) / (r n2), n1 / n2]].


def _translate(distance_mm: float) -> np.ndarray:
    return np.array([[1.0, distance_mm], [0.0, 1.0]])


def _refract(radius_mm: float, n1: float, n2: float) -> np.ndarray:
    return np.array([[1.0, 0.0], [-(n2 - n1) / (radius_mm * n2), n1 / n2]])


def _singlet() -> np.ndarray:
    return _refract(R2_MM, N_GLASS, 1.0) @ _translate(T_MM) @ _refract(R1_MM, 1.0, N_GLASS)


def _bisect(f: Any, lo: float, hi: float) -> float:
    """Bisection on a monotone decreasing bracket, 200 iterations (float64 exact)."""
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


#: The back focal distance from the last vertex, where the paraxial A element of
#: (translate @ singlet) crosses zero for a collimated input. Independent of any
#: pre-lens distance, because A of translate(d) @ S @ translate(g) is
#: a_S + d c_S and carries no g.
BFD_MM = _bisect(lambda d: (_translate(d) @ _singlet())[0, 0], 40.0, 110.0)
EFL_MM = -1.0 / _singlet()[1, 0]


def _lens_spec(
    name: str, leads: tuple[tuple[float, str], ...], last_mm: float
) -> OpticalSystemSpec:
    """One singlet, with powerless dummy planes in front carrying the launch plane.

    ``trace_ray_batch`` requires the first traced surface to coincide with the
    bundle's declared plane to within a nanometre, and the builder always puts
    the first prescription surface at ``z = 0``. So a group whose rays arrive at
    ``z != 0`` gets dummy planes and is traced with ``skip=len(leads)``. The
    alternative -- translating the rays' z coordinate between groups -- would
    make the declared plane and the ray geometry two facts that could disagree.
    """
    surfaces = [
        SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=thickness, comment=comment)
        for thickness, comment in leads
    ]
    surfaces.append(
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=R1_MM),
            thickness_mm=T_MM,
            material=IdealMaterialSpec(refractive_index=N_GLASS),
            is_stop=True,
            comment="equiconvex front",
        )
    )
    surfaces.append(
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=R2_MM),
            thickness_mm=last_mm,
            comment="equiconvex back; thickness reaches the declared image plane",
        )
    )
    return OpticalSystemSpec(
        name=name,
        description=str(PRESCRIPTION["component"]),
        object_distance_mm=0.0,
        surfaces=tuple(surfaces),
        aperture=ApertureSpec(value_mm=CA_MM),
        fields=(FieldSpec(y_deg=0.0),),
        wavelengths=(WavelengthSpec(value_um=LAMBDA_M * 1e6, is_primary=True),),
    )


# ---------------------------------------------------------------------------
# Derived geometry
# ---------------------------------------------------------------------------


def geometry(params: dict[str, Any]) -> dict[str, Any]:
    """The physical realization of an instance's declared parameters.

    Every axial position follows from the prescription's own paraxial focal
    distances rather than from a nominal spacing, so the sensor really is the
    plane the analytic order position is derived at: on ``grating_then_lens`` the
    A element from the DOE plane to the sensor is 9.0e-17 and the B element is
    the EFL itself, which is why the reference collapses to
    ``f tan(arcsin(m lambda / Lambda))``.
    """
    topology = str(params["system_topology"])
    declared = TOPOLOGIES[topology]
    semi_m = float(params["used_semi_aperture_mm"]) * 1e-3
    pitch_m = float(params["doe_pitch_um"]) * 1e-6
    d_x, d_y, d_z = analytic_order_direction(params)

    if topology == "grating_then_lens":
        gap_mm = float(declared["doe_to_first_vertex_mm"])
        doe_z_mm = 0.0
        image_z_mm = gap_mm + T_MM + BFD_MM
        transfer = _translate(BFD_MM) @ _singlet() @ _translate(gap_mm)
        geom: dict[str, Any] = {
            "upstream_spec": None,
            "downstream_spec": _lens_spec(
                "grating-lens", ((gap_mm, "DOE plane; no power"),), BFD_MM
            ),
            "downstream_skip": 1,
            "magnification": None,
        }
        # The launched bundle's own semi-aperture is the used aperture, and the
        # DOE grid has to hold it.
        doe_footprint_m = semi_m
    else:
        d1_mm = float(declared["doe_before_intermediate_focus_mm"])
        group1_last_mm = BFD_MM - d1_mm
        object_mm = float(declared["second_group_object_distance_over_efl"]) * EFL_MM
        gap2_mm = d1_mm + object_mm
        # Where the second group's image is sharp for an object `object_mm` in
        # front of its first vertex: B = 0 at a conjugate.
        group2_last_mm = _bisect(
            lambda last: (_translate(last) @ _singlet() @ _translate(object_mm))[0, 1],
            20.0,
            5000.0,
        )
        lead_mm = 2.0  # the input plane sits 2 mm before group 1, as on the other topology
        doe_z_mm = lead_mm + T_MM + group1_last_mm
        image_z_mm = doe_z_mm + gap2_mm + T_MM + group2_last_mm
        transfer = _translate(group2_last_mm) @ _singlet() @ _translate(gap2_mm)
        conjugate = _translate(group2_last_mm) @ _singlet() @ _translate(object_mm)
        geom = {
            "upstream_spec": _lens_spec(
                "relay-group-1", ((lead_mm, "input plane; no power"),), group1_last_mm
            ),
            "downstream_spec": _lens_spec(
                "relay-group-2",
                ((doe_z_mm, "powerless dummy at z = 0; skipped"), (gap2_mm, "DOE plane")),
                group2_last_mm,
            ),
            "downstream_skip": 2,
            "magnification": float(conjugate[0, 0]),
            "second_group_object_distance_mm": object_mm,
            "second_group_gap_mm": gap2_mm,
            "second_group_last_mm": group2_last_mm,
            "group1_last_mm": group1_last_mm,
            "conjugate_residual_mm": float(conjugate[0, 1]),
        }
        # The converging bundle is narrower at the DOE than at the input, by the
        # paraxial ratio of the remaining distance to the focus.
        doe_footprint_m = semi_m * d1_mm / BFD_MM

    # A margin, because the paraxial footprint is not the traced one: the grid
    # must hold the widest ray, and a ray clipped to the edge sample would be
    # given the wrong phase silently.
    grid_n = int(2 * math.ceil(1.05 * doe_footprint_m / pitch_m))
    order_position_m = (
        float(transfer[0, 1]) * 1e-3 * d_x / d_z,
        float(transfer[0, 1]) * 1e-3 * d_y / d_z,
    )
    # Where the UNDIFFRACTED image sits, which is where the unmodulated arm's
    # window has to be centred. It is (0, 0) only on axis: at a declared incident
    # tilt the whole image moves, and centring the Strehl denominator on the axis
    # would put it where there is no light -- measured while authoring, that
    # reports a Strehl of 1.0e5 rather than 1.
    tilt_rad = math.radians(float(params["incident_tilt_deg"]))
    if topology != "grating_then_lens" and tilt_rad != 0.0:
        raise NotImplementedError(
            "a declared incident tilt on the relay topology enters BEFORE group 1, so "
            "the undiffracted image position is not B tan(theta) and this driver has no "
            "closed form for it. Refused rather than approximated; no declared instance "
            "combines the two axes."
        )
    plain_position_m = (
        0.0,
        float(transfer[0, 1]) * 1e-3 * math.sin(tilt_rad) / math.cos(tilt_rad),
    )
    geom.update(
        {
            "topology": topology,
            "doe_plane": ReferencePlane(name="doe", z_m=doe_z_mm * 1e-3),
            "image_plane": ReferencePlane(name="image", z_m=image_z_mm * 1e-3),
            "used_semi_aperture_m": semi_m,
            "doe_pitch_m": pitch_m,
            "doe_grid_n": grid_n,
            "doe_paraxial_footprint_m": doe_footprint_m,
            "transfer_a": float(transfer[0, 0]),
            "transfer_b_mm": float(transfer[0, 1]),
            "analytic_direction": (d_x, d_y, d_z),
            "analytic_order_position_m": order_position_m,
            "analytic_plain_position_m": plain_position_m,
            "efl_mm": EFL_MM,
            "back_focal_distance_mm": BFD_MM,
            "strehl_quantization": strehl_quantization(params),
            "paraxial_departure_law": paraxial_order_position_departure(params),
            "clear_aperture_margin": clear_aperture_margin(params),
        }
    )
    return geom


def _surface_phase(
    params: dict[str, Any], geom: dict[str, Any], *, conjugate: bool = False
) -> np.ndarray:
    """The declared surface's phase on its own grid: a ramp, or a constant.

    ``conjugate`` is the ``order-sign-flip`` control's surface arm, ``t ->
    conj(t)``, expressed as a negated phase so it goes through
    ``DiffractiveSurface.from_phase`` -- the shipping constructor with the
    repository phasor convention -- rather than around it.
    """
    grid_n = geom["doe_grid_n"]
    pitch_m = geom["doe_pitch_m"]
    kind = str(params["doe_phase_kind"])
    sign = -1.0 if conjugate else 1.0
    if kind == "flat_zero":
        row = np.zeros(grid_n, dtype=np.float64)
    elif kind == "flat_piston":
        row = np.full(grid_n, sign * 1.0, dtype=np.float64)
    else:
        x = (np.arange(grid_n) - grid_n // 2) * pitch_m
        row = sign * 2.0 * math.pi * x / (float(params["grating_period_um"]) * 1e-6)
    return np.tile(row, (grid_n, 1))


def _surface(
    params: dict[str, Any], geom: dict[str, Any], *, conjugate: bool = False
) -> DiffractiveSurface:
    return DiffractiveSurface.from_phase(
        _surface_phase(params, geom, conjugate=conjugate),
        sample_pitch_m=(geom["doe_pitch_m"], geom["doe_pitch_m"]),
        plane=geom["doe_plane"],
    )


def _launch_positions(semi_m: float, per_axis: int) -> np.ndarray:
    axis = np.linspace(-semi_m, semi_m, per_axis)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    keep = (xx**2 + yy**2) <= semi_m**2 * (1.0 + 1e-12)
    return np.column_stack([xx[keep], yy[keep]])


# ---------------------------------------------------------------------------
# Tracing, composition and reconstruction
# ---------------------------------------------------------------------------


def _trace(
    bundle: RayBundle, lens: Any, plane: ReferencePlane, *, skip: int
) -> tuple[RayBundle, dict[str, Any]]:
    batch = CoherentRayBatch(
        bundle=bundle,
        ray_id=np.arange(bundle.count, dtype=np.int64),
        valid=np.ones(bundle.count, dtype=bool),
    )
    plans = plan_trace_bridges(batch, home=C_RAY_TO_WAVE_CAPABILITIES, device=CPU)
    traced, diagnostics = trace_ray_batch(batch, lens, image_plane=plane, plans=plans, skip=skip)
    return traced.bundle, diagnostics


def _compose(
    traced: RayBundle,
    incoming_opl: np.ndarray,
    *,
    factor: float = 1.0,
    note: str = "composed",
) -> RayBundle:
    """Rebase the traced optical path onto the incoming bundle's own reference.

    ``trace_ray_batch`` returns Optiland's ``opd``, which starts at zero at the
    trace's first surface, so the path the incoming bundle already carried has to
    be added back. THIS is the composition the ``opl-not-rebased`` control
    mutates: ``factor = 0`` drops it and ``factor = 2`` (or adding the incident
    path a second time) counts it twice.
    """
    return dataclasses.replace(
        traced,
        optical_path_length_m=np.asarray(traced.optical_path_length_m)
        + factor * np.asarray(incoming_opl),
        optical_path_length_reference=note,
    )


def _shift_origin(bundle: RayBundle, dx_m: float, dy_m: float) -> RayBundle:
    """Exact change of transverse origin, so a window can be centred off axis.

    ``u'(x) = u(x + (dx, dy))``: the reconstruction grid's origin is index
    ``n // 2`` by convention, and the order does not sit there. Moving the rays
    rather than the grid keeps ``ray_to_wave``'s own coordinate convention the
    single one in play.
    """
    positions = np.asarray(bundle.positions_m).copy()
    positions[:, 0] -= dx_m
    positions[:, 1] -= dy_m
    return dataclasses.replace(bundle, positions_m=positions)


def _reconstruct(
    bundle: RayBundle,
    centre_xy_m: tuple[float, float],
    *,
    grid_n: int,
    pitch_m: float,
    plane: ReferencePlane,
    phase_sign: int = 1,
) -> tuple[np.ndarray, Any]:
    field, diagnostics = ray_to_wave(
        _shift_origin(bundle, *centre_xy_m),
        grid_shape=(grid_n, grid_n),
        sample_pitch_m=(pitch_m, pitch_m),
        plane=plane,
        normalization="none",
        projection=Projection.ASM_CONSISTENT,
        perturbation=Perturbation(phase_sign=phase_sign),
    )
    return np.asarray(field.u), diagnostics


def _parabolic_offset(v_lo: float, v_0: float, v_hi: float) -> float:
    denominator = v_lo - 2.0 * v_0 + v_hi
    if denominator == 0.0:
        return 0.0
    return 0.5 * (v_lo - v_hi) / denominator


def _peak_and_fwhm(intensity: np.ndarray, pitch_m: float) -> dict[str, float]:
    """Sub-sample peak, its offset from the window centre, and the FWHM.

    The peak is refined parabolically on the 3x3 neighbourhood of the brightest
    sample rather than taken as that sample's value. Without it the peak carries
    a sampling error of order ``(delta / FWHM)**2`` -- 7e-4 at this window's
    pitch -- which is the same size as the Strehl agreement being measured, and
    it would largely (not entirely) cancel between the two arms of a ratio. A
    quantity that is right only because two errors cancel is not a measurement.
    """
    n = intensity.shape[0]
    peak_y, peak_x = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    # Clamped one sample inside the border so the 3x3 neighbourhood the parabolic
    # fit reads is always in bounds. A peak ON the border is a window that does not
    # hold the response, which `psf_window_holds_*` reports rather than this hides.
    iy = int(np.clip(int(peak_y), 1, n - 2))
    ix = int(np.clip(int(peak_x), 1, n - 2))
    fx = _parabolic_offset(intensity[iy, ix - 1], intensity[iy, ix], intensity[iy, ix + 1])
    fy = _parabolic_offset(intensity[iy - 1, ix], intensity[iy, ix], intensity[iy + 1, ix])

    def vertex(v_lo: float, v_0: float, v_hi: float, frac: float) -> float:
        return v_0 - 0.25 * (v_lo - v_hi) * frac

    peak = 0.5 * (
        vertex(intensity[iy, ix - 1], intensity[iy, ix], intensity[iy, ix + 1], fx)
        + vertex(intensity[iy - 1, ix], intensity[iy, ix], intensity[iy + 1, ix], fy)
    )
    return {
        "peak": float(peak),
        "offset_x_m": (ix - n // 2 + fx) * pitch_m,
        "offset_y_m": (iy - n // 2 + fy) * pitch_m,
        "fwhm_m": _fwhm_m(intensity[iy, :], pitch_m),
    }


def _fwhm_m(profile: np.ndarray, pitch_m: float) -> float:
    peak = float(profile.max())
    if peak <= 0.0:
        return float("nan")
    index = int(np.argmax(profile))
    half = 0.5 * peak

    def edge(step: int) -> float:
        i = index
        while 0 <= i + step < len(profile) and profile[i + step] > half:
            i += step
        j = i + step
        if not (0 <= j < len(profile)):
            return float("nan")
        lo, hi = float(profile[i]), float(profile[j])
        if lo == hi:
            return float(i)
        return i + step * (lo - half) / (lo - hi)

    return abs(edge(1) - edge(-1)) * pitch_m


# ---------------------------------------------------------------------------
# The reference: the outgoing bundle, rebuilt from the closed form
# ---------------------------------------------------------------------------


def closed_form_outgoing(
    incident: RayBundle,
    params: dict[str, Any],
    geom: dict[str, Any],
    *,
    order: int | None = None,
    renormalize: bool = True,
    conjugate_surface: bool = False,
) -> tuple[RayBundle, np.ndarray]:
    """The admissible outgoing bundle, from first principles.

    Three closed forms, and no import from ``couplers/generalized_snell.py``:

    * **directions** -- ``k_t^out = k_t^in + m grad_t(phi)``, then
      ``k_n^out = sqrt((n_t k0)**2 - |k_t^out|**2)``, with ``grad_x(phi) =
      2 pi / Lambda`` for the declared ramp and zero for a constant. With
      ``n_i = n_t = 1`` this is ``d_x^out = d_x^in + m lambda / Lambda``.
    * **optical path** -- ``opl_in + m phi(x_snap) / k0``, with ``x_snap`` the
      ray's nearest sample of the declared grid. The snapping is the whole
      physical content of the Strehl law, so it is reproduced here rather than
      idealized away: an idealized reference would disagree with the shipping
      code by exactly the effect being measured.
    * **amplitude** -- ``|a_in| |t(x_snap)|``, which for every declared
      (phase-only) surface is ``a_in`` unchanged.

    Returns the bundle and the per-ray surface phase it used, because the phase is
    what ``outgoing_opl_rebase_residual_waves`` is measured against and recomputing
    it at the call site would be a second definition of the same convention.

    ``renormalize=False`` is the ``secondary-directions-not-renormalized``
    control: the transverse kick is added to the direction vector with ``d_z``
    left alone, so ``|d| = sqrt(1 + (m lambda / Lambda)**2)`` and the
    ``RayBundle`` contract refuses the result. That refusal is the control's
    first arm; the caller normalizes the same vector for the second.
    """
    positions = np.asarray(incident.positions_m, dtype=np.float64)
    directions = np.asarray(incident.directions, dtype=np.float64)
    opl_in = np.asarray(incident.optical_path_length_m, dtype=np.float64)
    amplitude = np.asarray(incident.amplitude)
    m = int(params["order"] if order is None else order)
    sign = -1.0 if conjugate_surface else 1.0

    kind = str(params["doe_phase_kind"])
    if kind == "linear_ramp":
        period_m = float(params["grating_period_um"]) * 1e-6
        grad_x = sign * 2.0 * math.pi / period_m
    else:
        grad_x = 0.0

    d_x = directions[:, 0] + m * grad_x / K0
    d_y = directions[:, 1].copy()
    if renormalize:
        d_z = np.sqrt(np.clip(1.0 - d_x**2 - d_y**2, 0.0, None))
    else:
        d_z = directions[:, 2].copy()

    # The phase at the ray's nearest sample of the DECLARED grid, computed from
    # the declaration. Index arithmetic mirrors couplers.patch.extract_patch's
    # rule -- round(coord / pitch) + n // 2 -- and then clips, which is what the
    # model does for a ray outside the array.
    pitch_m = geom["doe_pitch_m"]
    grid_n = geom["doe_grid_n"]
    ix = np.clip(np.round(positions[:, 0] / pitch_m).astype(np.int64) + grid_n // 2, 0, grid_n - 1)
    x_snap = (ix - grid_n // 2) * pitch_m
    if kind == "flat_zero":
        phase = np.zeros(positions.shape[0], dtype=np.float64)
    elif kind == "flat_piston":
        phase = np.full(positions.shape[0], sign * 1.0, dtype=np.float64)
    else:
        phase = sign * 2.0 * math.pi * x_snap / (float(params["grating_period_um"]) * 1e-6)
    # `from_phase` stores exp(i phi) and the model reads angle() back, so the
    # reference has to carry the same wrapped value rather than the unwrapped
    # ramp -- otherwise the two would differ by whole waves and the comparison
    # would be measuring numpy's branch cut. The order factor goes on AFTER the
    # wrap, for the same reason: the physical factor is exp(i m phi) but what the
    # surface array can hold is exp(i wrap(phi)), so the m-th order of the STORED
    # surface carries m wrap(phi). Applying m first would compare against a
    # different surface than the one declared.
    phase = float(m) * np.arctan2(np.sin(phase), np.cos(phase))

    return dataclasses.replace(
        incident,
        positions_m=np.column_stack(
            [positions[:, 0], positions[:, 1], np.full(positions.shape[0], geom["doe_plane"].z_m)]
        ),
        directions=np.column_stack([d_x, d_y, d_z]),
        amplitude=amplitude,
        optical_path_length_m=opl_in + phase / K0,
        optical_path_length_reference=(
            "closed form: incident path plus the declared ramp's phase at the ray's "
            "nearest sample, divided by k0"
        ),
        reference_plane=geom["doe_plane"],
    ), phase


def _local_modulus(
    params: dict[str, Any], geom: dict[str, Any], incident: RayBundle
) -> np.ndarray:
    """``|t|`` at each ray's nearest sample, from the declared surface array."""
    surface_phase = _surface_phase(params, geom)
    transmission = np.exp(1j * surface_phase)
    positions = np.asarray(incident.positions_m, dtype=np.float64)
    grid_n = geom["doe_grid_n"]
    pitch_m = geom["doe_pitch_m"]
    iy = np.clip(np.round(positions[:, 1] / pitch_m).astype(np.int64) + grid_n // 2, 0, grid_n - 1)
    ix = np.clip(np.round(positions[:, 0] / pitch_m).astype(np.int64) + grid_n // 2, 0, grid_n - 1)
    return np.abs(transmission[iy, ix])


# ---------------------------------------------------------------------------
# One pass through the chain
# ---------------------------------------------------------------------------


def _power(bundle: RayBundle) -> float:
    return float(np.sum(np.abs(np.asarray(bundle.amplitude)) ** 2))


def run_chain(params: dict[str, Any]) -> dict[str, Any]:
    """Every arm of one instance, in one pass.

    Six traces, and each one answers a different question:

    ``order``        the instance itself.
    ``plain``        the same geometry with NO diffractive surface, which is the
                     Strehl denominator and the FWHM reference.
    ``admissible``   the closed-form outgoing bundle, traced downstream, which is
                     what makes CHE-148's admissibility criterion a comparison.
    ``flat_zero``    the zero-phase exactness limit.
    ``flat_piston``  the zero-gradient exactness limit.
    ``tilt``         a plain ``collimated_bundle`` at the analytic order angle,
                     with no diffractive element anywhere in it. Only available
                     on ``grating_then_lens``, where the incident bundle is
                     collimated; ``None`` on the relay, and the record says so
                     rather than omitting the key.
    """
    started = time.perf_counter()
    geom = geometry(params)
    configure_optiland_execution(device=CPU, precision=Precision.FP64)

    downstream = build_optiland_system(geom["downstream_spec"])
    upstream = (
        build_optiland_system(geom["upstream_spec"]) if geom["upstream_spec"] is not None else None
    )

    # --- the incident bundle, at the DOE plane -----------------------------
    positions_xy = _launch_positions(
        geom["used_semi_aperture_m"], int(params["rays_per_axis"])
    )
    tilt_rad = math.radians(float(params["incident_tilt_deg"]))
    launched = collimated_bundle(
        positions_xy_m=positions_xy,
        direction=(0.0, math.sin(tilt_rad), math.cos(tilt_rad)),
        wavelength_m=LAMBDA_M,
        # Both topologies launch at z = 0: on grating_then_lens that IS the DOE
        # plane, and on the relay it is the input plane 2 mm in front of group 1,
        # whose own dummy surface carries the rays to the DOE.
        plane_z_m=0.0,
        plane_name="input" if upstream is not None else "doe",
    )
    upstream_diagnostics: dict[str, Any] | None = None
    if upstream is None:
        incident = launched
    else:
        traced, upstream_diagnostics = _trace(launched, upstream, geom["doe_plane"], skip=1)
        incident = _compose(traced, launched.optical_path_length_m, note="input plane")

    incident_positions = np.asarray(incident.positions_m)
    traced_footprint_m = float(np.abs(incident_positions[:, :2]).max())
    if traced_footprint_m > (geom["doe_grid_n"] // 2) * geom["doe_pitch_m"]:
        raise RuntimeError(
            f"the DOE grid ({geom['doe_grid_n']} at {geom['doe_pitch_m']} m) does not "
            f"hold the traced ray footprint ({traced_footprint_m} m); rays outside it "
            "would be given the edge sample's phase silently"
        )
    incident_opl = np.asarray(incident.optical_path_length_m)

    # --- the interaction ---------------------------------------------------
    surface = _surface(params, geom)
    interaction = diffractive_interaction(
        incident,
        surface,
        model=DiffractiveModel.GENERALIZED_SNELL,
        parameters=GeneralizedSnellParameters(
            order=int(params["order"]), patch_px=int(params["patch_px"])
        ),
    )
    outgoing = interaction.outgoing
    reference_bundle, reference_phase = closed_form_outgoing(incident, params, geom)

    # --- downstream traces ------------------------------------------------
    skip = geom["downstream_skip"]
    order_arm, order_diagnostics = _trace(outgoing, downstream, geom["image_plane"], skip=skip)
    plain_arm, plain_diagnostics = _trace(incident, downstream, geom["image_plane"], skip=skip)
    admissible_arm, admissible_diagnostics = _trace(
        reference_bundle, downstream, geom["image_plane"], skip=skip
    )

    arms: dict[str, Any] = {
        "order": (order_arm, np.asarray(outgoing.optical_path_length_m), order_diagnostics),
        "plain": (plain_arm, incident_opl, plain_diagnostics),
        "admissible": (
            admissible_arm,
            np.asarray(reference_bundle.optical_path_length_m),
            admissible_diagnostics,
        ),
    }

    # --- the two exactness arms, run on EVERY instance ---------------------
    exactness: dict[str, Any] = {}
    plain_positions = np.asarray(plain_arm.positions_m)
    plain_directions = np.asarray(plain_arm.directions)
    for kind, piston in (("flat_zero", 0.0), ("flat_piston", 1.0)):
        flat_params = {**params, "doe_phase_kind": kind}
        flat_surface = _surface(flat_params, geom)
        flat = diffractive_interaction(
            incident,
            flat_surface,
            model=DiffractiveModel.GENERALIZED_SNELL,
            parameters=GeneralizedSnellParameters(
                order=int(params["order"]), patch_px=int(params["patch_px"])
            ),
        )
        flat_traced, flat_diagnostics = _trace(
            flat.outgoing, downstream, geom["image_plane"], skip=skip
        )
        flat_total = np.asarray(flat_traced.optical_path_length_m) + np.asarray(
            flat.outgoing.optical_path_length_m
        )
        plain_total = np.asarray(plain_arm.optical_path_length_m) + incident_opl
        exactness[kind] = {
            "declared_piston_rad": piston,
            "expected_optical_path_piston_rad": int(params["order"]) * piston,
            "interaction_direction_residual": float(
                np.abs(np.asarray(flat.outgoing.directions) - np.asarray(incident.directions)).max()
            ),
            "interaction_position_residual_m": float(
                np.abs(
                    np.asarray(flat.outgoing.positions_m)[:, :2] - incident_positions[:, :2]
                ).max()
            ),
            "interaction_amplitude_residual": float(
                np.abs(
                    np.abs(np.asarray(flat.outgoing.amplitude))
                    - np.abs(np.asarray(incident.amplitude))
                ).max()
            ),
            "downstream_position_residual_m": float(
                np.abs(np.asarray(flat_traced.positions_m) - plain_positions).max()
            ),
            "downstream_direction_residual": float(
                np.abs(np.asarray(flat_traced.directions) - plain_directions).max()
            ),
            # `order * piston`, matching the interaction's own m phi / k0
            # convention: the m-th order of a constant-phase surface carries
            # m times that constant, and the m = -1 exactness arm is what
            # made that visible.
            "downstream_opl_residual_m": float(
                np.abs(flat_total - plain_total - int(params["order"]) * piston / K0).max()
            ),
            "invalid_rays": int(flat_diagnostics["invalid_rays"]),
            "bitwise_identical": bool(
                np.array_equal(np.asarray(flat_traced.positions_m), plain_positions)
                and np.array_equal(np.asarray(flat_traced.directions), plain_directions)
            ),
        }

    # --- the fully independent tilted arm, where it exists -----------------
    tilt: dict[str, Any] | None = None
    if upstream is None and str(params["doe_phase_kind"]) == "linear_ramp":
        d_x, d_y, d_z = geom["analytic_direction"]
        tilted = collimated_bundle(
            positions_xy_m=positions_xy,
            direction=(d_x, d_y, d_z),
            wavelength_m=LAMBDA_M,
            plane_z_m=0.0,
            plane_name="doe",
        )
        tilt_arm, tilt_diagnostics = _trace(tilted, downstream, geom["image_plane"], skip=skip)
        tilt = {
            "position_residual_m": float(
                np.abs(np.asarray(tilt_arm.positions_m) - np.asarray(order_arm.positions_m)).max()
            ),
            "centroid_x_m": _centroid_x(tilt_arm),
            "invalid_rays": int(tilt_diagnostics["invalid_rays"]),
        }

    snell = interaction.model_diagnostics
    return {
        "geometry": geom,
        "params": params,
        "arms": arms,
        "incident": incident,
        "incident_opl": incident_opl,
        "outgoing": outgoing,
        "reference_bundle": reference_bundle,
        "reference_phase_rad": reference_phase,
        "local_modulus": _local_modulus(params, geom, incident),
        "exactness": exactness,
        "tilt": tilt,
        "traced_footprint_m": traced_footprint_m,
        "incident_opl_sag_waves": float((incident_opl.max() - incident_opl.min()) / LAMBDA_M),
        "incident_direction_max_transverse": float(
            np.abs(np.asarray(incident.directions)[:, :2]).max()
        ),
        "upstream_invalid_rays": (
            None if upstream_diagnostics is None else int(upstream_diagnostics["invalid_rays"])
        ),
        "interaction_diagnostics": {
            key: value
            for key, value in interaction.diagnostics.items()
            if key
            in {
                "interaction",
                "model",
                "coupler",
                "substrate",
                "order",
                "patch_px",
                "transverse_scale_m",
                "outgoing_ray_count",
                "worst_propagating_order_margin",
                "worst_local_gradient_smoothness_margin",
                "single_order_dominance",
                "single_order_dominance_margin",
                "opl_convention",
            }
        },
        "single_order_dominance": float(snell.single_order_dominance),
        "measured_smoothness_margin": float(snell.worst_local_gradient_smoothness_margin),
        "surface_positions_downstream_m": surface_positions_m(downstream),
        "surface_positions_upstream_m": (
            None if upstream is None else surface_positions_m(upstream)
        ),
        "ray_count": int(incident.count),
        "wall_seconds": time.perf_counter() - started,
    }


def _centroid_x(bundle: RayBundle) -> float:
    weight = np.abs(np.asarray(bundle.amplitude)) ** 2
    total = float(weight.sum())
    if total <= 0.0:
        return float("nan")
    return float((np.asarray(bundle.positions_m)[:, 0] * weight).sum() / total)


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _blur(array: np.ndarray, sigma_px: float) -> np.ndarray:
    n = array.shape[0]
    fy = np.fft.fftfreq(n)[:, None]
    fx = np.fft.fftfreq(n)[None, :]
    kernel = np.exp(-2.0 * (math.pi * sigma_px) ** 2 * (fx**2 + fy**2))
    return np.real(np.fft.ifft2(np.fft.fft2(array) * kernel))


def _radial_profile(array: np.ndarray) -> np.ndarray:
    n = array.shape[0]
    axis = np.arange(n) - n // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy).astype(int)
    counts = np.bincount(radius.ravel())
    return np.bincount(radius.ravel(), weights=array.ravel()) / np.maximum(counts, 1)


def _fringe_contrast(profile: np.ndarray, floor: float = 1e-3) -> float:
    """Deepest interior minimum against the smaller of its flanking maxima.

    The ``floor`` excludes minima whose flanks are themselves noise: a profile
    that has decayed to 1e-3 of its peak has interior wiggles that are the radial
    average's own binning, not fringes.
    """
    p = np.asarray(profile, dtype=float)
    best = 0.0
    for i in range(1, len(p) - 1):
        if p[i] < p[i - 1] and p[i] <= p[i + 1]:
            envelope = min(float(p[:i].max()), float(p[i + 1 :].max()))
            if envelope > floor * float(p.max()) and envelope + p[i] > 0.0:
                best = max(best, float((envelope - p[i]) / (envelope + p[i])))
    return best


def _radial_bands(array: np.ndarray, bands: int = 4) -> list[float]:
    n = array.shape[0]
    total = float(array.sum())
    normalized = array / total if total > 0.0 else array
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(normalized))) ** 2
    axis = np.arange(n) - n // 2
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(xx, yy)
    power = float(spectrum.sum())
    out = []
    for b in range(bands):
        lo = b * n / (2 * bands)
        hi = (b + 1) * n / (2 * bands)
        mask = (radius >= lo) & (radius < hi)
        out.append(float(spectrum[mask].sum() / power) if power > 0.0 else float("nan"))
    return out


def measure(
    run: dict[str, Any],
    params: dict[str, Any],
    *,
    opl_factor: float = 1.0,
    double_count_incident: bool = False,
    conjugate_opl: bool = False,
    phase_sign: int = 1,
    with_structure: bool = True,
) -> tuple[dict[str, Measurement], dict[str, Any]]:
    """Every declared metric, plus the diagnostics a reader needs beside them.

    ``with_structure=False`` skips the five interference observables, which only
    B4-DOE-INLINE declares and which cost a reconstruction on a 256x256 window
    (``N * ny * nx`` = 8e8 terms at 12644 rays). Every negative-control arm sets
    it, because no control targets them.

    The four mutation keyword arguments are the negative-control defects, applied here
    rather than inside :func:`run_chain` because every one of them is a defect in
    the COMPOSITION and the reconstruction, downstream of the interaction -- so a
    control costs a re-measurement rather than a re-trace, and the arm it is
    compared against is bit-for-bit the arm the baseline used.
    """
    geom = run["geometry"]
    incident = run["incident"]
    outgoing = run["outgoing"]
    reference = run["reference_bundle"]
    order_arm, order_opl, order_diagnostics = run["arms"]["order"]
    plain_arm, plain_opl, plain_diagnostics = run["arms"]["plain"]
    admissible_arm, admissible_opl, admissible_diagnostics = run["arms"]["admissible"]

    # ---- the interaction's own conventions, against the closed form -------
    direction_residual = float(
        np.abs(np.asarray(outgoing.directions) - np.asarray(reference.directions)).max()
    )
    opl_rebase_residual_waves = float(
        np.abs(
            np.asarray(outgoing.optical_path_length_m)
            - np.asarray(incident.optical_path_length_m)
            - run["reference_phase_rad"] / K0
        ).max()
        / LAMBDA_M
    )
    amplitude_residual = float(
        np.abs(
            np.abs(np.asarray(outgoing.amplitude))
            - np.abs(np.asarray(incident.amplitude)) * run["local_modulus"]
        ).max()
    )
    incident_power = _power(incident)
    interaction_power_error = (
        abs(_power(outgoing) / incident_power - 1.0) if incident_power > 0.0 else float("nan")
    )

    # ---- admissibility, on the sensor ------------------------------------
    admissible_position_residual = float(
        np.abs(np.asarray(order_arm.positions_m) - np.asarray(admissible_arm.positions_m)).max()
    )

    # ---- the reconstruction windows --------------------------------------
    # Sized from the DECLARED parameters through the family's own two helpers, so
    # SENSOR_GRID_DIRECTION_CAPACITY's arithmetic and this window are one number.
    # The traced arm's measured |d|max is recorded beside it, not used for it.
    x_analytic, y_analytic = geom["analytic_order_position_m"]
    numerical_aperture = paraxial_image_side_na(params)
    fwhm_reference_m = airy_fwhm_m(numerical_aperture, LAMBDA_M)
    psf_n = int(params["psf_window_px"])
    psf_pitch = psf_window_pitch_m(params)

    def compose_order(bundle: Any, opl: np.ndarray) -> Any:
        composed = _compose(bundle, opl, factor=opl_factor, note="order arm")
        if double_count_incident:
            composed = dataclasses.replace(
                composed,
                optical_path_length_m=np.asarray(composed.optical_path_length_m)
                + run["incident_opl"],
                optical_path_length_reference="NEGATIVE CONTROL: incident path counted twice",
            )
        if conjugate_opl:
            opl_now = np.asarray(composed.optical_path_length_m)
            composed = dataclasses.replace(
                composed,
                optical_path_length_m=2.0 * float(opl_now.mean()) - opl_now,
                optical_path_length_reference=(
                    "NEGATIVE CONTROL: optical path conjugated about its own mean"
                ),
            )
        return composed

    order_field, order_recon = _reconstruct(
        compose_order(order_arm, order_opl),
        (x_analytic, y_analytic),
        grid_n=psf_n,
        pitch_m=psf_pitch,
        plane=geom["image_plane"],
        phase_sign=phase_sign,
    )
    plain_field, plain_recon = _reconstruct(
        _compose(plain_arm, plain_opl, note="plain arm"),
        geom["analytic_plain_position_m"],
        grid_n=psf_n,
        pitch_m=psf_pitch,
        plane=geom["image_plane"],
    )
    admissible_field, _ = _reconstruct(
        _compose(admissible_arm, admissible_opl, note="admissible arm"),
        (x_analytic, y_analytic),
        grid_n=psf_n,
        pitch_m=psf_pitch,
        plane=geom["image_plane"],
    )

    admissible_norm = float(np.linalg.norm(admissible_field))
    admissible_field_residual = (
        float(np.linalg.norm(order_field - admissible_field) / admissible_norm)
        if admissible_norm > 0.0
        else float("nan")
    )

    order_psf = _peak_and_fwhm(np.abs(order_field) ** 2, psf_pitch)
    plain_psf = _peak_and_fwhm(np.abs(plain_field) ** 2, psf_pitch)
    # Two ways the PSF window can fail to hold the response it is measuring, both
    # recorded rather than left for a reader to infer from a number that looks
    # plausible. The window is sized on the DIFFRACTION width and centred on the
    # ANALYTIC order position, so it misses (a) a response that is much wider
    # than diffraction-limited -- B4-DOE-INLINE-APERTURE-500 carries 0.99 waves
    # of spherical aberration and its unmodulated core is 2.4x the Airy width, and
    # (b) an order that is not where the analytic position says -- the aliased
    # instance's order lands 740 um away from a window of half-extent 34 um. In
    # both cases the peak, the FWHM and the Strehl are computed on the wrong
    # patch of field and are not comparable to the other instances.
    psf_half_extent_m = 0.5 * (psf_n - 1) * psf_pitch
    ray_centroid_x = _centroid_x(order_arm)
    window_holds_order = bool(abs(ray_centroid_x - x_analytic) < 0.5 * psf_half_extent_m)
    window_holds_core = bool(
        math.isfinite(plain_psf["fwhm_m"]) and plain_psf["fwhm_m"] < psf_half_extent_m
    )

    strehl = order_psf["peak"] / plain_psf["peak"] if plain_psf["peak"] > 0.0 else float("nan")
    strehl_predicted = geom["strehl_quantization"]
    strehl_error = abs(strehl / strehl_predicted - 1.0) if strehl_predicted > 0.0 else float("nan")

    # ---- order position, both estimators ---------------------------------
    denominator = max(abs(x_analytic), LAMBDA_M)
    order_position_error = abs(ray_centroid_x - x_analytic) / denominator
    field_position_x = x_analytic + order_psf["offset_x_m"]
    order_position_field_error = abs(field_position_x - x_analytic) / denominator

    psf_fwhm_error = (
        abs(order_psf["fwhm_m"] / plain_psf["fwhm_m"] - 1.0)
        if plain_psf["fwhm_m"] and math.isfinite(plain_psf["fwhm_m"])
        else float("nan")
    )

    # ---- the exactness limit, as one number ------------------------------
    refractive_limit_waves = 0.0
    for arm in run["exactness"].values():
        refractive_limit_waves = max(
            refractive_limit_waves,
            arm["downstream_position_residual_m"] / LAMBDA_M,
            arm["downstream_opl_residual_m"] / LAMBDA_M,
        )

    # ---- power accounting through the downstream trace -------------------
    entering = _power(outgoing)
    surviving = _power(order_arm)
    clipped = max(0.0, entering - surviving) / entering if entering > 0.0 else float("nan")

    # ---- interference structure (the B4 observables) ---------------------
    structure = (
        _structure_metrics(run, params, numerical_aperture, fwhm_reference_m)
        if with_structure
        else {
            "skipped": "not declared by this family, or a negative-control arm",
            "relative_l2": float("nan"),
            "fringe_coherent": float("nan"),
            "fringe_geometric": float("nan"),
            "bands_coherent": [float("nan")] * 4,
            "bands_geometric": [float("nan")] * 4,
        }
    )

    def m(value: float) -> Measurement:
        return Measurement(
            value=value,
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        )

    measurements = {
        "grating_equation_direction_residual": m(direction_residual),
        "outgoing_opl_rebase_residual_waves": m(opl_rebase_residual_waves),
        "outgoing_amplitude_residual": m(amplitude_residual),
        "order_position_vs_admissible_residual_m": m(admissible_position_residual),
        "admissible_bundle_field_residual": m(admissible_field_residual),
        "order_position_relative_error": m(order_position_error),
        "order_position_field_relative_error": m(order_position_field_error),
        "strehl_quantization_relative_error": m(strehl_error),
        "psf_fwhm_relative_error": m(psf_fwhm_error),
        "refractive_limit_residual_waves": m(refractive_limit_waves),
        "interaction_power_ratio_error": m(interaction_power_error),
        "downstream_clipped_power_fraction": m(clipped),
        "coherent_vs_geometric_relative_l2": m(structure["relative_l2"]),
        "fringe_contrast_coherent": m(structure["fringe_coherent"]),
        "fringe_contrast_geometric": m(structure["fringe_geometric"]),
        "high_band_spectrum_fraction_coherent": m(structure["bands_coherent"][1]),
        "high_band_spectrum_fraction_geometric": m(structure["bands_geometric"][1]),
    }

    diagnostics = {
        "diffractive_model": run["interaction_diagnostics"]["model"],
        "interaction": run["interaction_diagnostics"],
        "geometry_m": {
            "topology": geom["topology"],
            "doe_plane_z_m": geom["doe_plane"].z_m,
            "image_plane_z_m": geom["image_plane"].z_m,
            "used_semi_aperture_m": geom["used_semi_aperture_m"],
            "doe_pitch_m": geom["doe_pitch_m"],
            "doe_grid_n": geom["doe_grid_n"],
            "doe_paraxial_footprint_m": geom["doe_paraxial_footprint_m"],
            "doe_traced_footprint_m": run["traced_footprint_m"],
            "transfer_a_dimensionless": geom["transfer_a"],
            "transfer_b_mm": geom["transfer_b_mm"],
            "magnification": geom["magnification"],
            "efl_mm": geom["efl_mm"],
            "back_focal_distance_mm": geom["back_focal_distance_mm"],
            "analytic_direction": list(geom["analytic_direction"]),
            "analytic_order_position_m": list(geom["analytic_order_position_m"]),
            "analytic_plain_position_m": list(geom["analytic_plain_position_m"]),
            "clear_aperture_margin": geom["clear_aperture_margin"],
            "paraxial_departure_law": geom["paraxial_departure_law"],
        },
        "prescription": PRESCRIPTION,
        "surface_positions_downstream_m": run["surface_positions_downstream_m"],
        "surface_positions_upstream_m": run["surface_positions_upstream_m"],
        "incident_bundle": {
            "note": (
                "the bundle the interaction actually sees. Its optical-path sag and its "
                "transverse direction are what make the OPL rebasing and the incident "
                "direction term non-trivial, so they are recorded rather than implied."
            ),
            "ray_count": run["ray_count"],
            "optical_path_sag_waves": run["incident_opl_sag_waves"],
            "max_transverse_direction": run["incident_direction_max_transverse"],
            "upstream_invalid_rays": run["upstream_invalid_rays"],
        },
        "order_arm": {
            "ray_centroid_x_m": ray_centroid_x,
            "field_peak_x_m": field_position_x,
            "field_peak_offset_x_m": order_psf["offset_x_m"],
            "field_peak_offset_y_m": order_psf["offset_y_m"],
            "fwhm_m": order_psf["fwhm_m"],
            "invalid_rays": int(order_diagnostics["invalid_rays"]),
        },
        "unmodulated_arm": {
            "note": (
                "the same geometry with NO diffractive surface. It is the Strehl "
                "denominator and the FWHM reference, and its own agreement with the "
                "Airy width is reported here and deliberately NOT gated -- see "
                "psf_fwhm_relative_error's blind_to."
            ),
            "peak": plain_psf["peak"],
            "fwhm_m": plain_psf["fwhm_m"],
            "airy_fwhm_m": fwhm_reference_m,
            "airy_relative_error": (
                abs(plain_psf["fwhm_m"] / fwhm_reference_m - 1.0)
                if fwhm_reference_m > 0.0
                else float("nan")
            ),
            "paraxial_image_side_numerical_aperture": numerical_aperture,
            "measured_image_side_max_transverse_direction": _measured_image_side_na(run),
            "invalid_rays": int(plain_diagnostics["invalid_rays"]),
        },
        "admissible_arm": {
            "note": (
                "the outgoing bundle rebuilt from the grating equation and phi_snap / k0 "
                "in this driver, sharing no code with couplers/generalized_snell.py, and "
                "traced through the same downstream group."
            ),
            "position_residual_m": admissible_position_residual,
            "field_relative_l2": admissible_field_residual,
            "invalid_rays": int(admissible_diagnostics["invalid_rays"]),
        },
        "tilt_equivalent_arm": (
            run["tilt"]
            if run["tilt"] is None
            else {
                **run["tilt"],
                "note": (
                    "a plain collimated_bundle at the analytic order angle, traced "
                    "through the same singlet with NO diffractive element in it. This is "
                    "the arm that attributes order_position_relative_error: its own "
                    "departure from the paraxial position, and the order arm's, are the "
                    "same fraction."
                ),
                "own_relative_error": abs(run["tilt"]["centroid_x_m"] - x_analytic)
                / denominator,
                "double_ratio": abs(
                    (ray_centroid_x / x_analytic) / (run["tilt"]["centroid_x_m"] / x_analytic)
                    - 1.0
                )
                if x_analytic != 0.0 and run["tilt"]["centroid_x_m"] != 0.0
                else float("nan"),
            }
        ),
        "exactness_limits": run["exactness"],
        "strehl": {
            "measured": strehl,
            "predicted_sinc_squared": strehl_predicted,
            "sawtooth_half_width_rad": math.pi
            * float(params["doe_pitch_um"])
            / float(params["grating_period_um"])
            if str(params["doe_phase_kind"]) == "linear_ramp"
            else 0.0,
            "marechal_alternative": math.exp(
                -(
                    math.pi
                    * float(params["doe_pitch_um"])
                    / float(params["grating_period_um"])
                )
                ** 2
                / 3.0
            )
            if str(params["doe_phase_kind"]) == "linear_ramp"
            else 1.0,
        },
        "power_accounting": {
            "note": (
                "sum |a|**2 at each stage. Unlike the field-forming models, "
                "GENERALIZED_SNELL neither rebases the amplitude nor forms a field, so "
                "the incident and outgoing ray-power sums ARE directly comparable -- "
                "which is why interaction_power_ratio_error is an exact conservation law "
                "here rather than a discrete one."
            ),
            "incident_ray_power": incident_power,
            "outgoing_ray_power": entering,
            "after_downstream_ray_power": surviving,
            "downstream_invalid_rays": int(order_diagnostics["invalid_rays"]),
        },
        "reconstruction": {
            "psf_window_px": psf_n,
            "psf_window_pitch_m": psf_pitch,
            "psf_window_half_extent_fwhms": PSF_WINDOW_HALF_EXTENT_FWHMS,
            "psf_window_half_extent_m": psf_half_extent_m,
            "psf_window_holds_the_order": window_holds_order,
            "psf_window_holds_the_point_response": window_holds_core,
            "psf_window_validity_note": (
                "when either flag is False the peak, the FWHM and the Strehl-like ratio "
                "were computed on a patch of field that does not contain the thing they "
                "name, and are not comparable to the instances where both are True. The "
                "window is sized on the DIFFRACTION width and centred on the ANALYTIC "
                "order position, both by declaration."
            ),
            "max_transverse_direction": order_recon.max_transverse_direction,
            "grid_nyquist_direction_limit": order_recon.grid_nyquist_direction_limit,
            "grid_nyquist_satisfied": order_recon.grid_nyquist_satisfied,
            "ray_density_status": order_recon.ray_density_status,
            "max_adjacent_ray_phase_rad": order_recon.max_adjacent_ray_phase_rad,
            "ray_density_note": (
                "C_RAY_TO_WAVE declines the adjacent-ray phase scan above its ray-count "
                "limit, so max_adjacent_ray_phase_rad is None and the condition is "
                "UNDECLARED here rather than satisfied. See "
                "SENSOR_GRID_DIRECTION_CAPACITY's blind_to."
            ),
            "plain_arm_grid_nyquist_satisfied": plain_recon.grid_nyquist_satisfied,
        },
        "interference_structure": {
            "note": (
                "the coherent reconstruction against the traced ray density blurred to "
                "the diffraction limit -- the most generous thing a ray-only model can "
                "be given, since it grants it a resolution limit it cannot derive."
            ),
            # Only the real numbers reach the record. When the block is skipped the
            # metric slots carry NaN so the measurement dict stays shaped, but
            # writing NaN into a JSON record would leave a value nothing can read
            # back strictly, so the diagnostics carry the reason instead.
            **(
                {"skipped": structure["skipped"]}
                if "skipped" in structure
                else {k: v for k, v in structure.items() if k != "arrays"}
            ),
        },
        "single_order_dominance": run["single_order_dominance"],
        "measured_smoothness_margin": run["measured_smoothness_margin"],
        # `None`, not `inf`, for a flat surface: a JSON record has no infinity and
        # "this predicate has no content here" is a different statement from a
        # very large margin.
        "analytic_smoothness_margin": (
            1.0 - 4.0 * float(params["doe_pitch_um"]) / float(params["grating_period_um"])
            if str(params["doe_phase_kind"]) == "linear_ramp"
            else None
        ),
        "wall_seconds": run["wall_seconds"],
    }
    return measurements, diagnostics


def _measured_image_side_na(run: dict[str, Any]) -> float:
    """The traced unmodulated arm's own ``|d|max`` at the image plane.

    Recorded beside the paraxial value the window was sized from, never used to
    size it: a readout grid that came out of the run would put the validity
    predicate and the driver on two different numbers.
    """
    plain_arm = run["arms"]["plain"][0]
    return float(np.abs(np.asarray(plain_arm.directions)[:, :2]).max())


def _structure_metrics(
    run: dict[str, Any],
    params: dict[str, Any],
    numerical_aperture: float,
    fwhm_reference_m: float,
) -> dict[str, Any]:
    """The coherent field against the blurred ray density, on the wide window."""
    geom = run["geometry"]
    order_arm, order_opl, _ = run["arms"]["order"]
    n = int(params["structure_window_px"])
    pitch = float(params["structure_window_pitch_um"]) * 1e-6
    # Centred on the MEASURED ray centroid, not on the analytic order position,
    # and the difference is not cosmetic: this metric compares the coherent field
    # to the ray density OF THE SAME RAYS, so a window placed anywhere other than
    # where those rays are would be measuring window placement rather than
    # structure. Where the order lands is a separate question and
    # order_position_relative_error is what asks it -- which matters for
    # B4-DOE-INLINE-PITCH-ALIASED, whose order comes out on the wrong side of the
    # axis entirely: on the analytic centre the ray histogram is EMPTY and every
    # structure metric is a division by zero.
    centre = (
        _centroid_x(order_arm),
        float(
            (
                np.asarray(order_arm.positions_m)[:, 1]
                * np.abs(np.asarray(order_arm.amplitude)) ** 2
            ).sum()
            / max(float((np.abs(np.asarray(order_arm.amplitude)) ** 2).sum()), 1e-300)
        ),
    )
    x_analytic, y_analytic = centre

    field, diagnostics = _reconstruct(
        _compose(order_arm, order_opl, note="order arm"),
        centre,
        grid_n=n,
        pitch_m=pitch,
        plane=geom["image_plane"],
    )
    coherent = np.abs(field) ** 2
    total = float(coherent.sum())
    coherent = coherent / total if total > 0.0 else coherent

    positions = np.asarray(order_arm.positions_m)
    weights = np.abs(np.asarray(order_arm.amplitude)) ** 2
    half = n * pitch / 2.0
    histogram, _, _ = np.histogram2d(
        positions[:, 1] - y_analytic,
        positions[:, 0] - x_analytic,
        bins=n,
        range=[[-half, half], [-half, half]],
        weights=weights,
    )
    # The ray-only model gets the diffraction limit handed to it: a Gaussian of
    # the same FWHM as the Airy core. sigma = FWHM / 2.355.
    blur_px = fwhm_reference_m / pitch
    geometric = np.clip(_blur(histogram, blur_px / 2.3548), 0.0, None)
    geometric_total = float(geometric.sum())
    geometric = geometric / geometric_total if geometric_total > 0.0 else geometric

    geometric_norm = float(np.linalg.norm(geometric))
    relative_l2 = (
        float(np.linalg.norm(coherent - geometric) / geometric_norm)
        if geometric_norm > 0.0
        else float("nan")
    )
    return {
        "relative_l2": relative_l2,
        "fringe_coherent": _fringe_contrast(_radial_profile(coherent)[: n // 2]),
        "fringe_geometric": _fringe_contrast(_radial_profile(geometric)[: n // 2]),
        "bands_coherent": _radial_bands(coherent),
        "bands_geometric": _radial_bands(geometric),
        "window_px": n,
        "window_pitch_m": pitch,
        "window_centre_m": list(centre),
        "window_centre_note": (
            "the measured ray centroid, not the analytic order position: the comparison "
            "is between the coherent field and the ray density of the SAME rays."
        ),
        "geometric_power_in_window": geometric_total,
        "samples_per_diffraction_fwhm": fwhm_reference_m / pitch,
        "blur_fwhm_m": fwhm_reference_m,
        "blur_fwhm_px": blur_px,
        "image_side_numerical_aperture": numerical_aperture,
        "grid_nyquist_satisfied": diagnostics.grid_nyquist_satisfied,
        "coherent_power_in_window": total,
    }


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------


_COMMON: dict[str, Any] = {
    "system_topology": "grating_then_lens",
    "doe_phase_kind": "linear_ramp",
    "grating_period_um": 100.0,
    "order": 1.0,
    "used_semi_aperture_mm": 1.0,
    "incident_tilt_deg": 0.0,
    "doe_pitch_um": 1.0,
    "rays_per_axis": 128,
    "psf_window_px": 65,
    "structure_window_px": 256,
    "structure_window_pitch_um": 1.0,
    "diffractive_model": DIFFRACTIVE_MODEL,
    "patch_px": 65,
    "device": "cpu",
}


def _params(**overrides: Any) -> dict[str, Any]:
    return {**_COMMON, **overrides}


#: The gated set. PERIOD-100 is the reference every other instance is read
#: against, and every one of them moves exactly one declared axis.
GATED_PARAMETERS: dict[str, dict[str, Any]] = {
    "B3-DOE-INLINE-PERIOD-100": _params(),
    "B3-DOE-INLINE-PERIOD-200": _params(grating_period_um=200.0),
    "B3-DOE-INLINE-PERIOD-050": _params(grating_period_um=50.0),
    # m = -1 is not decoration: it is what found CHE-143's order/OPL inconsistency.
    # Before the fix in couplers/generalized_snell.py this instance reported a
    # Strehl-like ratio of 3e-6 against a predicted 0.99967 and a 47% PSF
    # broadening, because the outgoing rays were deflected as if the phase were
    # -phi while the optical path carried +phi. With `m phi` it reproduces
    # PERIOD-100's numbers exactly, which is the mirror symmetry the geometry has.
    "B3-DOE-INLINE-ORDER-MINUS1": _params(order=-1.0),
    "B3-DOE-INLINE-ZEROPHASE": _params(doe_phase_kind="flat_zero"),
    "B3-DOE-INLINE-ZEROGRAD": _params(doe_phase_kind="flat_piston"),
    "B3-DOE-INLINE-PITCH-5": _params(doe_pitch_um=5.0),
    # p / Lambda = 0.20, and this instance exists because a test caught the
    # declared set failing to IDENTIFY the Strehl law rather than merely agree
    # with one. (sin a / a)**2 and the Marechal exp(-a**2 / 3) share their first
    # two terms, so at p / Lambda = 0.05 they differ by 6.7e-6 while the measured
    # departure from the sinc law is 3.2e-5 -- the sweep could not tell them
    # apart. Here they differ by 1.56e-3, which is 4.5x the measured departure.
    "B3-DOE-INLINE-PITCH-20": _params(doe_pitch_um=20.0),
    "B3-DOE-INLINE-APERTURE-050": _params(used_semi_aperture_mm=0.5),
    # 2.0 mm is past PARAXIAL_ORDER_POSITION's bound and is here on purpose: the
    # declared law predicts 3.71e-4 against a 3e-4 gate, so the wide end of the
    # aperture axis is where the paraxial oracle stops holding and this instance
    # is what says so with a number rather than a footnote.
    "B3-DOE-INLINE-APERTURE-200": _params(used_semi_aperture_mm=2.0),
    "B3-DOE-INLINE-OFFAXIS-01": _params(incident_tilt_deg=0.5),
    "B3-DOE-INLINE-RELAY-01": _params(system_topology="lens_then_grating_then_lens"),
}

#: The characterization set. REFERENCE is what the others are read against.
CHARACTERIZATION_PARAMETERS: dict[str, dict[str, Any]] = {
    "B4-DOE-INLINE-REFERENCE": _params(doe_pitch_um=2.0),
    "B4-DOE-INLINE-APERTURE-300": _params(doe_pitch_um=2.0, used_semi_aperture_mm=3.0),
    "B4-DOE-INLINE-APERTURE-500": _params(doe_pitch_um=2.0, used_semi_aperture_mm=5.0),
    "B4-DOE-INLINE-RAYS-256": _params(doe_pitch_um=2.0, rays_per_axis=256),
    "B4-DOE-INLINE-RELAY-01": _params(
        doe_pitch_um=2.0, system_topology="lens_then_grating_then_lens"
    ),
    # p / Lambda = 0.30, past RAMP_GRADIENT_ESTIMATOR_UNALIASED's bound of 0.25.
    # The instance exists to record that the model's OWN smoothness margin comes
    # back POSITIVE there while the order is emitted on the wrong side of the
    # axis, and that single_order_dominance is the guard that catches it.
    "B4-DOE-INLINE-PITCH-ALIASED": _params(doe_pitch_um=30.0),
}

_REFERENCE_INSTANCE = {
    **dict.fromkeys(GATED_PARAMETERS, "B3-DOE-INLINE-PERIOD-100"),
    **dict.fromkeys(CHARACTERIZATION_PARAMETERS, "B4-DOE-INLINE-REFERENCE"),
}

_FAMILY_OF = {
    **dict.fromkeys(GATED_PARAMETERS, B3_DOE_INLINE),
    **dict.fromkeys(CHARACTERIZATION_PARAMETERS, B4_DOE_INLINE),
}

ALL_PARAMETERS: dict[str, dict[str, Any]] = {
    **GATED_PARAMETERS,
    **CHARACTERIZATION_PARAMETERS,
}

#: Which instances demonstrate which controls, and why only these. A control's
#: detection margin is a ratio against the baseline, so a baseline already broken
#: by real physics reports a false verdict -- B3-4F-IDEAL's precedent, and the
#: reason APERTURE-200 and RELAY-01 do not carry the position-metric controls.
_CONTROLS_ON: dict[str, tuple[str, ...]] = {
    # The reference instance: every metric is at its floor, so a margin measured
    # here is a statement about the control and nothing else.
    "B3-DOE-INLINE-PERIOD-100": (
        "opl-not-rebased",
        "phasor-sign-flip",
        "order-sign-flip",
        "secondary-directions-not-renormalized",
    ),
    # The only gated instance with a non-constant INCIDENT optical path, which is
    # the one configuration where double-counting it is not identically zero.
    "B3-DOE-INLINE-OFFAXIS-01": ("opl-not-rebased",),
    # And the other one: a converging incident bundle out of a real group.
    "B3-DOE-INLINE-RELAY-01": ("opl-not-rebased",),
}

#: Declared before the run, so a pass or a fail is read against a stated
#: expectation rather than reverse-engineered from the number.
_EXPECTED: dict[str, dict[str, str]] = {
    "B3-DOE-INLINE-ORDER-MINUS1": {
        "strehl_quantization_relative_error": "met, and expected to reproduce "
        "PERIOD-100's value EXACTLY: negating the order and conjugating the surface are "
        "the same operation on a pure ramp, so the m = -1 system is the m = +1 system "
        "mirrored and every scalar observable must agree to round-off. It did not before "
        "CHE-148 fixed the order factor in the interaction's OPL rebasing -- it reported "
        "3.2e-6 against a predicted 0.99967 -- which is what this instance is for",
        "psf_fwhm_relative_error": "met, and expected to reproduce PERIOD-100's value; it "
        "read 0.467 before the order factor was fixed",
    },
    "B3-DOE-INLINE-ZEROPHASE": {
        "refractive_limit_residual_waves": "met, and expected to be EXACTLY 0.0 rather "
        "than merely small: with phi == 0 the interaction's outputs are bitwise the "
        "incident bundle, so the downstream trace is bitwise the plain trace",
        "strehl_quantization_relative_error": "met, against a predicted Strehl of "
        "exactly 1 -- the sawtooth half-width is zero when there is no ramp",
    },
    "B3-DOE-INLINE-ZEROGRAD": {
        "refractive_limit_residual_waves": "met -- the zero-GRADIENT limit, which is the "
        "sharper of the two: the geometry must be untouched while the optical path picks "
        "up exactly the declared 1 rad piston and nothing else"
    },
    "B3-DOE-INLINE-PITCH-20": {
        "strehl_quantization_relative_error": "met, and this is the instance that lets "
        "the sweep IDENTIFY the law rather than agree with it: at p / Lambda = 0.20 the "
        "sinc law predicts 0.87514 and the Marechal form 0.87670, a 1.56e-3 separation "
        "against a measured departure expected near 3.5e-4",
        "psf_fwhm_relative_error": "met -- the sawtooth costs peak intensity, not width",
    },
    "B3-DOE-INLINE-PITCH-5": {
        "strehl_quantization_relative_error": "met, and this is the instance that makes "
        "the Strehl law a measurement rather than a formality: at p / Lambda = 0.05 the "
        "predicted Strehl is 0.99180 against 0.99967 at the reference, so the law is "
        "being read where it has moved"
    },
    "B3-DOE-INLINE-APERTURE-200": {
        "order_position_relative_error": "NOT met -- declared OUTSIDE by "
        "PARAXIAL_ORDER_POSITION, whose law predicts 3.71e-4 against a 3e-4 threshold. "
        "The far end of the aperture axis, not a failure, and the tilted arm is expected "
        "to depart by the SAME fraction, which is what attributes it to the singlet",
        "psf_fwhm_relative_error": "met -- a linear ramp is still a pure tilt at this "
        "aperture; what has grown is the spot, not the difference between the two arms",
    },
    "B3-DOE-INLINE-RELAY-01": {
        "order_position_field_relative_error": "expected NOT to be inside the recorded "
        "band, for the same reason as the ray metric -- the paraxial closed form does not "
        "describe this topology. The tolerance does not gate (may_gate=False), so this is "
        "a reported departure and not a verdict",
        "order_position_relative_error": "NOT met -- declared FAR_OUTSIDE by "
        "PARAXIAL_ORDER_POSITION. The order is relayed at -1.914x through a second real "
        "group and crosses it off axis, so its departure from the paraxial position is "
        "LINEAR in the aperture and 30x the collimated topology's. Every convention and "
        "admissibility metric is expected to be met: that is why this instance is in the "
        "gated family",
        "order_position_vs_admissible_residual_m": "met -- and this is the instance where "
        "it matters most, because the incident bundle is converging with a real per-ray "
        "optical path and direction, so nothing about the rebasing is trivially zero",
        "psf_fwhm_relative_error": "met, but at 1.3e-3 rather than the 1e-5 the collimated "
        "instances reach: the order sits 0.73 mm off axis behind a -1.914x second group, "
        "so the train is measurably non-aplanatic over the displacement. Checked against "
        "being a measurement artifact -- it is stable across a 4x window refinement",
    },
    "B3-DOE-INLINE-OFFAXIS-01": {
        "admissible_bundle_field_residual": "met -- and this is the instance the "
        "opl-not-rebased control's DOUBLE-COUNTED arm needs, because a 0.5 deg tilt puts "
        "29.2 waves of sag on the incident optical path where an on-axis collimated "
        "bundle has exactly zero"
    },
}


def declared_instance_ids(family_id: str | None = None) -> tuple[str, ...]:
    if family_id == "B3-DOE-INLINE":
        return tuple(GATED_PARAMETERS)
    if family_id == "B4-DOE-INLINE":
        return tuple(CHARACTERIZATION_PARAMETERS)
    return tuple(GATED_PARAMETERS) + tuple(CHARACTERIZATION_PARAMETERS)


def differing_axes(params: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """Which declared axes differ, checked rather than asserted in prose."""
    return sorted(k for k in params if params[k] != reference[k])


def canonical_instance(instance_id: str) -> Any:
    family = _FAMILY_OF[instance_id]
    params = ALL_PARAMETERS[instance_id]
    reference_id = _REFERENCE_INSTANCE[instance_id]
    expected = dict(_EXPECTED.get(instance_id, {}))
    if instance_id == reference_id:
        expected["differs_from"] = f"{reference_id} is this family's reference instance"
    else:
        differing = differing_axes(params, ALL_PARAMETERS[reference_id])
        expected["differs_from"] = (
            f"{reference_id} in exactly one axis: {differing[0]} "
            f"({ALL_PARAMETERS[reference_id][differing[0]]} -> {params[differing[0]]})"
        )
    return family.instantiate(instance_id, params, expected=expected)


def metric_names(family: Any) -> tuple[str, ...]:
    return tuple(metric.name for metric in family.metrics)


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def _controls(
    instance_id: str,
    params: dict[str, Any],
    run: dict[str, Any],
    baseline: dict[str, Measurement],
) -> dict[str, Any]:
    wanted = _CONTROLS_ON.get(instance_id, ())
    if not wanted:
        return {}
    family = _FAMILY_OF[instance_id]
    results: dict[str, Any] = {}

    def judge(control_id: str, metric: str, mutated: Measurement, note: str = "") -> None:
        tolerance = family.tolerance_for(metric)
        if baseline[metric].value > tolerance.threshold:
            results[control_id] = NegativeControlResult(
                control_id=control_id,
                outcome=NegativeControlOutcome.NOT_RUN,
                target_metric=metric,
                baseline=baseline[metric],
                note=(
                    f"not exercised: {metric} is already past its own gate at baseline "
                    f"({baseline[metric].value:.6g} > {tolerance.threshold:.6g}), so a "
                    "detection margin against it would be a ratio between two failures"
                ),
            )
            return
        results[control_id] = control_result(
            control_id,
            metric,
            baseline=baseline[metric],
            mutated=mutated,
            threshold=tolerance.threshold,
            note=note,
        )

    metric = "admissible_bundle_field_residual"
    if "opl-not-rebased" in wanted:
        dropped, _ = measure(run, params, opl_factor=0.0, with_structure=False)
        doubled, _ = measure(run, params, double_count_incident=True, with_structure=False)
        sag = run["incident_opl_sag_waves"]
        # The DROPPED arm is decisive on every topology; the DOUBLE-COUNTED arm is
        # inert wherever the incident path is a constant, and it is the larger of
        # the two that is scored so the control reports its own best evidence.
        worse = max(dropped[metric].value, doubled[metric].value)
        judge(
            "opl-not-rebased",
            metric,
            Measurement(
                value=worse,
                uncertainty=_FLOOR,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            ),
            note=(
                f"DROPPED arm: {metric} = {dropped[metric].value:.6g}, and the "
                "Strehl-like peak ratio departs from its own analytic value by "
                f"{dropped['strehl_quantization_relative_error'].value:.6g} "
                "against a 5e-3 gate -- the ramp never reaches the pupil, so the order "
                "is a defocused smear at the right position. DOUBLE-COUNTED arm: "
                f"{metric} = {doubled[metric].value:.6g}, with the incident optical path "
                f"carrying {sag:.3f} waves of sag across this instance's bundle. "
                + (
                    "That sag is zero here, so the double-counted arm is INERT by "
                    "construction rather than by weakness -- a collimated bundle "
                    "launched at its own reference plane has opl_in identically 0. It is "
                    "recorded so the null is on the record beside the arms that are not "
                    "null (OFFAXIS-01 at 29.235 waves, RELAY-01 at 9.658)."
                    if sag < 1e-9
                    else "That sag is what makes the double count detectable here."
                )
            ),
        )
    if "phasor-sign-flip" in wanted:
        flipped, _ = measure(run, params, phase_sign=-1, with_structure=False)
        conjugated, _ = measure(run, params, conjugate_opl=True, with_structure=False)
        judge(
            "phasor-sign-flip",
            metric,
            Measurement(
                value=max(flipped[metric].value, conjugated[metric].value),
                uncertainty=_FLOOR,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            ),
            note=(
                "RECONSTRUCTION arm (C_RAY_TO_WAVE's own Perturbation(phase_sign=-1) on "
                f"the interaction side only): {metric} = {flipped[metric].value:.6g}. "
                "OPTICAL-PATH arm (opl -> 2 mean(opl) - opl on the composed bundle): "
                f"{metric} = {conjugated[metric].value:.6g}. NOTE, measured and "
                "recorded because it is a real limitation rather than a caveat: the "
                "Strehl-like peak ratio under the reconstruction flip is "
                f"{flipped['strehl_quantization_relative_error'].value:.6g} against a "
                f"baseline of {baseline['strehl_quantization_relative_error'].value:.6g} "
                "-- i.e. an INTENSITY observable is blind to the phasor sign outright, "
                "because conjugating a field whose ray amplitudes are real leaves its "
                "modulus untouched. Only the field comparison sees it, and only because "
                "the reference arm keeps the correct convention"
            ),
        )
    if "order-sign-flip" in wanted:
        geom = run["geometry"]
        arms: dict[str, dict[str, float]] = {}
        for label, kwargs in (
            ("model order negated", {"order": -int(params["order"])}),
            ("surface transmission conjugated", {"conjugate_surface": True}),
        ):
            surface = _surface(params, geom, conjugate=bool(kwargs.get("conjugate_surface")))
            mutated_interaction = diffractive_interaction(
                run["incident"],
                surface,
                model=DiffractiveModel.GENERALIZED_SNELL,
                parameters=GeneralizedSnellParameters(
                    order=int(kwargs.get("order", params["order"])),
                    patch_px=int(params["patch_px"]),
                ),
            )
            residual = float(
                np.abs(
                    np.asarray(mutated_interaction.outgoing.directions)
                    - np.asarray(run["reference_bundle"].directions)
                ).max()
            )
            traced, _ = _trace(
                mutated_interaction.outgoing,
                build_optiland_system(geom["downstream_spec"]),
                geom["image_plane"],
                skip=geom["downstream_skip"],
            )
            x_analytic = geom["analytic_order_position_m"][0]
            arms[label] = {
                "direction_residual": residual,
                "centroid_x_m": _centroid_x(traced),
                "position_relative_error": abs(_centroid_x(traced) - x_analytic)
                / max(abs(x_analytic), LAMBDA_M),
            }
        worst = max(a["direction_residual"] for a in arms.values())
        judge(
            "order-sign-flip",
            "grating_equation_direction_residual",
            Measurement(
                value=worst,
                uncertainty=_FLOOR,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            ),
            note="; ".join(
                f"{label}: direction residual {a['direction_residual']:.6g}, order "
                f"centroid {a['centroid_x_m'] * 1e3:.9f} mm, "
                f"order_position_relative_error {a['position_relative_error']:.6g}"
                for label, a in arms.items()
            )
            + ". Both arms produce the same magnitude for different reasons, which is "
            "why both are run: the first negates the model's declared order, the second "
            "is the exp(-i phi) surface DiffractiveSurface.from_phase exists to prevent",
        )
    if "secondary-directions-not-renormalized" in wanted:
        results["secondary-directions-not-renormalized"] = _renormalization_control(
            params, run, baseline, family
        )
    return results


def _renormalization_control(
    params: dict[str, Any],
    run: dict[str, Any],
    baseline: dict[str, Measurement],
    family: Any,
) -> NegativeControlResult:
    """Add the grating kick to the direction VECTOR and leave ``d_z`` alone.

    Two arms, and together they are exhaustive rather than one margin.

    **Refused.** ``RayBundle.__post_init__`` rejects the result with
    ``NON_UNIT_DIRECTION`` before any trace, because ``|d| = sqrt(1 + dx**2)``.
    That is the mutation failing: it cannot reach a number.

    **Measured.** The same vector normalized after the fact, which is what a
    downstream solver that silently normalizes would produce. The direction
    cosine becomes ``dx / sqrt(1 + dx**2)``, so the residual is the analytic
    ``dx**3 / 2``. This arm exists because the first one only shows that OUR
    contract catches it, and the question a reader will ask is how large the
    error would have been if it had not.
    """
    metric = "grating_equation_direction_residual"
    geom = run["geometry"]
    refusal_text = ""
    try:
        closed_form_outgoing(run["incident"], params, geom, renormalize=False)
    except ContractError as error:
        refusal_text = str(error)[:200]

    directions = np.asarray(run["incident"].directions, dtype=np.float64)
    period_m = float(params["grating_period_um"]) * 1e-6
    kick = int(params["order"]) * LAMBDA_M / period_m
    naive = np.column_stack([directions[:, 0] + kick, directions[:, 1], directions[:, 2]])
    normalized = naive / np.linalg.norm(naive, axis=1, keepdims=True)
    residual = float(np.abs(normalized - np.asarray(run["reference_bundle"].directions)).max())
    worst_norm_deviation = float(np.abs(np.linalg.norm(naive, axis=1) - 1.0).max())

    tolerance = family.tolerance_for(metric)
    note = (
        f"REFUSED arm, which is the mutation failing: {refusal_text or 'NOT REFUSED'}. "
        f"The worst unit-norm deviation is {worst_norm_deviation:.6g} against the "
        "RayBundle contract's 1e-9 allowance, so no number could be produced at all. "
        f"MEASURED arm (the same vector normalized after the fact): {metric} = "
        f"{residual:.6g} against a threshold of {tolerance.threshold:.6g}, which is the "
        f"analytic dx**3 / 2 = {kick**3 / 2:.6g}. In POSITION the same defect is only a "
        f"relative order displacement of {abs(1.0 - math.sqrt(1.0 - kick**2)):.6g} -- a "
        "fifth of the real singlet's own paraxial departure at this aperture (9.271e-5) "
        "and so invisible against order_position_relative_error's 3e-4 gate. That is "
        "exactly why this control is scored on the direction: the position metric cannot "
        "see it, and a control declared against a metric that cannot see it would be a "
        "control that passed"
    )
    if not refusal_text:
        return NegativeControlResult(
            control_id="secondary-directions-not-renormalized",
            outcome=NegativeControlOutcome.DID_NOT_FIRE,
            target_metric=metric,
            baseline=baseline[metric],
            mutated=Measurement(
                value=residual,
                uncertainty=_FLOOR,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            ),
            note=(
                "the non-unit bundle was NOT refused by the RayBundle contract, which "
                "would be a real hole in the boundary. " + note
            ),
        )
    return NegativeControlResult(
        control_id="secondary-directions-not-renormalized",
        outcome=NegativeControlOutcome.FIRED,
        target_metric=metric,
        baseline=baseline[metric],
        mutated=Measurement(
            value=residual,
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        ),
        note=note,
    )


# ---------------------------------------------------------------------------
# Run + verify
# ---------------------------------------------------------------------------


def run_instance(instance_id: str, *, with_controls: bool = True) -> InstanceRun:
    instance = canonical_instance(instance_id)
    family = _FAMILY_OF[instance_id]
    params = dict(instance.parameters)

    reference_id = _REFERENCE_INSTANCE[instance_id]
    if instance_id != reference_id:
        differing = differing_axes(params, ALL_PARAMETERS[reference_id])
        if len(differing) != 1:
            raise AssertionError(
                f"{instance_id} differs from {reference_id} in {differing}, but CHE-148 "
                "requires exactly one axis per instance"
            )

    started = time.perf_counter()
    refusal, run = probe_refusal(lambda: run_chain(params))
    if refusal is not None:
        # A refusal is a measurement, not a driver failure: past
        # RAMP_GRADIENT_ESTIMATOR_UNALIASED or PROPAGATING_ORDER_EXISTS the model
        # says so with a structured code instead of returning a plausible ray.
        record = record_from_probe(
            instance,
            component="C_GENERALIZED_SNELL",
            node_id="doe_inline",
            refusal=refusal,
            observed_parameters={
                "diffractive_model": DIFFRACTIVE_MODEL,
                "analytic_smoothness_margin": (
                    1.0
                    - 4.0
                    * float(params["doe_pitch_um"])
                    / float(params["grating_period_um"])
                ),
                "paraxial_departure_law": paraxial_order_position_departure(params),
            },
            wall_seconds=time.perf_counter() - started,
            diagnostics=[
                {
                    "refusal_is_the_measurement": (
                        "a declared validity predicate puts this instance outside before "
                        "it runs, and the refusal is what makes that declaration "
                        "executable. The predicate's own margin is reported beside it."
                    ),
                    "predicate_margins": family.evaluate_validity(params)[1],
                }
            ],
        )
        result = verify(family, instance, record, measurements={})
        return InstanceRun(family=family, instance=instance, record=record, result=result)

    declared = metric_names(family)
    all_measurements, diagnostics = measure(
        run, params, with_structure="coherent_vs_geometric_relative_l2" in declared
    )
    measurements = {name: all_measurements[name] for name in declared}
    controls = _controls(instance_id, params, run, measurements) if with_controls else {}
    wall_seconds = time.perf_counter() - started

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="doe_inline",
        refusal=None,
        observed_parameters={
            # The model, named in the record because it is named at the call
            # site: CHE-142's rule is that the model is never inferred, and a
            # record that does not say which one ran cannot show that it wasn't.
            "diffractive_model": diagnostics["diffractive_model"],
            "system_topology": run["geometry"]["topology"],
            "doe_grid_n": run["geometry"]["doe_grid_n"],
            "ray_count": run["ray_count"],
            "analytic_order_position_x_m": run["geometry"]["analytic_order_position_m"][0],
            "predicted_strehl": run["geometry"]["strehl_quantization"],
        },
        device_precision=DevicePrecisionObservation(
            requested_device="cpu",
            actual_device="cpu",
            requested_dtype="complex128",
            actual_dtype="complex128",
        ),
        wall_seconds=wall_seconds,
        diagnostics=[diagnostics],
    )
    result = verify(
        family,
        instance,
        record,
        measurements=measurements,
        invariants=(
            {
                invariant.invariant_id: measurements[invariant.metric]
                for invariant in family.invariants
            }
            if family.invariants
            else None
        ),
        negative_controls=controls,
    )
    return InstanceRun(family=family, instance=instance, record=record, result=result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=None, choices=("B3-DOE-INLINE", "B4-DOE-INLINE"))
    parser.add_argument("--instance", default=None, help="run only this instance id")
    parser.add_argument(
        "--no-controls", action="store_true", help="skip the negative controls (faster)"
    )
    parser.add_argument(
        "--write", action="store_true", help="write benchmarks/systems/records/<id>.json"
    )
    args = parser.parse_args()

    ids = (args.instance,) if args.instance else declared_instance_ids(args.family)
    for instance_id in ids:
        run = run_instance(instance_id, with_controls=not args.no_controls)
        print(
            f"{instance_id}  status={run.record.status.value}  "
            f"verification={run.result.status.value}"
        )
        for metric in run.result.physics_accuracy:
            met = "" if metric.met is None else f"  met={metric.met}"
            tol = "n/a" if metric.tolerance is None else f"{metric.tolerance:.3g}"
            print(f"  {metric.metric}: {metric.measured.value:.6e}  tol={tol}{met}")
        for invariant in run.result.invariant_results:
            print(
                f"  [invariant] {invariant.invariant_id}: "
                f"{invariant.measured.value:.6e}  met={invariant.met}"
            )
        for control in run.result.negative_control_results:
            print(f"  [control] {control.control_id}: {control.outcome.value}")
            print(f"            {control.note}")
        print(
            f"  validity: declared={run.result.validity.declared.value} "
            f"observed={run.result.validity.observed.value}"
        )
        if args.write:
            path = write_instance_record(run, driver="systems/b3_doe_inline", directory=RECORDS_DIR)
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
