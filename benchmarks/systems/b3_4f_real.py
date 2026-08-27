"""B3-4F-REAL / B4-4F-REAL, end to end: the real aberrated 4f relay, executed.

CHE-145 (M2.9). The chain is the shipping path throughout::

    object field
      -> couplers.wave_to_ray.spectrum_to_rays          (C_WAVE_TO_RAY, enumerated)
      -> solvers.optiland.coherent_trace.trace_ray_batch (M_RAY_OPTILAND, group 1)
      -> couplers.interaction.diffractive_interaction    (model=FULL_FIELD)
      -> solvers.optiland.coherent_trace.trace_ray_batch (M_RAY_OPTILAND, group 2)
      -> couplers.ray_to_wave.ray_to_wave                (C_RAY_TO_WAVE, sensor)

Nothing here reimplements a coupler or a solver, and the diffractive model is
named at the call site rather than inferred. The 4F-1 reference is
``benchmarks.systems.b3_4f_ideal``'s own ``relay`` and its own ``_mask``, imported
rather than copied, so "the modulation is held unchanged from 4F-1" is true by
construction and not by inspection.

Like ``b3_4f_ideal.py`` this runs outside ``GraphExecutor`` and builds its record
with ``record_from_probe``: there is no graph node for "trace a caller-supplied
ray population", the coherent-trace capability is reached through the adapter
directly (the same way ``benchmarks/probes/ray_wave/demo3_hologram_lens.py``
reaches it), and inventing a graph around it would put a second description of
the chain beside the real one.

Run it::

    ./run.sh --no-build python benchmarks/systems/b3_4f_real.py --family B3-4F-REAL --write
    ./run.sh --no-build python benchmarks/systems/b3_4f_real.py --family B4-4F-REAL --write

Roughly 20-40 s per instance at ``grid_n = 48`` (5.1e6 outgoing rays), so the two
families are run separately rather than in one command.
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

if __package__ in (None, ""):  # direct `python benchmarks/systems/b3_4f_real.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.systems.b3_4f_ideal import (
    _mask as ideal_mask,
)
from benchmarks.systems.b3_4f_ideal import (
    _order_location_error_frac,
    _order_phase_error_rad,
    _order_power_relative_l2,
    _peak_search,
)
from benchmarks.systems.b3_4f_ideal import (
    build as build_ideal,
)
from core.boundary import ComplexField, Frame, ReferencePlane
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
    FullFieldParameters,
    diffractive_interaction,
)
from couplers.ray_to_wave import Perturbation, Projection, ray_to_wave
from couplers.wave_to_ray import (
    SamplingDensity,
    SamplingPerturbation,
    decompose,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)
from runtime.instance_runner import probe_refusal, record_from_probe
from solvers.optiland.builder import build_optiland_system
from solvers.optiland.coherent_trace import (
    configure_optiland_execution,
    plan_trace_bridges,
    surface_positions_m,
    trace_ray_batch,
)
from verification.evidence import InstanceRun, control_result, write_instance_record
from verification.families.b3_4f_ideal import CHECKED_ORDERS, order_coefficients
from verification.families.b3_4f_real import (
    B3_4F_REAL,
    B4_4F_REAL,
    DIFFRACTIVE_MODEL,
    PRESCRIPTION,
    field_angle_deg,
    object_pitch_m,
    peak_wave_aberration_waves,
    residual_ray_angle_rad,
)
from verification.result import (
    Measurement,
    NegativeControlOutcome,
    NegativeControlResult,
    UncertaintyBasis,
)
from verification.verifier import verify

__all__ = [
    "declared_instance_ids",
    "differing_axes",
    "run_chain",
    "run_instance",
]

ROOT = repository_root()
RECORDS_DIR = ROOT / "benchmarks" / "systems" / "records"
CPU = DevicePlacement(kind=DeviceKind.CPU, index=0)

LAMBDA_M = float(PRESCRIPTION["wavelength_m"])
F_M = float(PRESCRIPTION["effective_focal_length_mm"]) * 1e-3
FFD_MM = float(PRESCRIPTION["front_focal_distance_mm"])
BFD_MM = float(PRESCRIPTION["back_focal_distance_mm"])
T_MM = float(PRESCRIPTION["centre_thickness_mm"])
LEG_MM = FFD_MM + T_MM + BFD_MM

#: Deterministic chain, so there is no ensemble to estimate an uncertainty from.
#: float64 round-off on a sum over at most 256x256 samples.
_FLOOR = 1e-12

#: How far the two plane-related negative controls displace a plane, in mm.
#: Deliberately modest, and the point of the number is what it MEASURES rather
#: than what it proves: an axial displacement of the shared plane turns out not to
#: be detectable here at all (the field there is the object's own spectrum, whose
#: angular content is the object's own angular size seen from the group -- 8.4e-4
#: rad at used_semi_aperture_mm = 0.7 -- giving a depth of focus
#: of tens of metres), so both plane controls report that null beside a decisive
#: arm rather than claiming a 2 mm shift is a demonstration. See
#: ``_controls`` and ``_handoff_declaration_control``.
_CONTROL_PLANE_SHIFT_MM = 2.0

#: The launch positions on the shared focal plane skip the single outermost
#: sample per axis. That sample sits *exactly* at the sensor grid's Nyquist
#: direction limit -- the two grids are a Fourier pair, so |d|max = R/f =
#: lambda / (2 * pitch_object) identically -- and the real group's distortion
#: pushes it 0.6% over, which C_RAY_TO_WAVE refuses (correctly: it is a grid
#: condition). The field there is exp(-pi**2) = 5.2e-5 of the peak, so what is
#: dropped is 5e-5 of one row, and the alternative would be to disable a real
#: sampling check. Interior sampling stays complete: this is a rim trim, not a
#: subsample, and subsampling would alias (see the family docstring).
_LAUNCH_RIM_TRIM = 1

#: Rays per Optiland trace call. Bounds the (N_rays, n) ramp arrays inside
#: C_RAY_TO_WAVE, which are what actually set peak memory.
_TRACE_CHUNK = 1_200_000


# ---------------------------------------------------------------------------
# The two real refractive groups
# ---------------------------------------------------------------------------


def leg_spec(
    name: str, *, lead_mm: float | None = None, ffd_mm: float, bfd_mm: float
) -> OpticalSystemSpec:
    """One refractive group, with its launch plane at its own front focal plane.

    ``lead_mm``, when given, prepends a powerless dummy plane at ``z = 0`` whose
    thickness carries the launch plane to ``z = lead_mm``. The builder always
    puts the first prescription surface at ``z = 0`` (``object_distance_mm``
    moves the object surface, not the stack), and ``trace_ray_batch`` requires
    the first traced surface to coincide with the bundle's declared plane to
    within a nanometre. So group 2, whose rays arrive at ``z = LEG_MM``, gets one
    dummy plane and is traced with ``skip=2``. The alternative -- translating the
    rays' z coordinate between legs -- would make the declared plane and the ray
    geometry two facts that could disagree, which is exactly the defect
    ``handoff-plane-mis-declared`` exists to catch.
    """
    lead: tuple[SurfaceSpec, ...] = ()
    if lead_mm is not None:
        lead = (
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=lead_mm,
                comment="powerless dummy plane at z = 0; skipped, carries the launch plane",
            ),
        )
    return OpticalSystemSpec(
        name=name,
        description=(
            f"{PRESCRIPTION['component']}, placed so the launch plane is its front "
            "focal plane and the last surface's thickness reaches its back focal plane"
        ),
        object_distance_mm=0.0,
        surfaces=(
            *lead,
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=ffd_mm,
                comment="launch plane, at the front focal plane; no power",
            ),
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=PRESCRIPTION["radius_1_mm"]),
                thickness_mm=T_MM,
                material=IdealMaterialSpec(refractive_index=PRESCRIPTION["refractive_index"]),
                is_stop=True,
                comment="equiconvex front",
            ),
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=PRESCRIPTION["radius_2_mm"]),
                thickness_mm=bfd_mm,
                comment="equiconvex back; thickness reaches the back focal plane",
            ),
        ),
        aperture=ApertureSpec(value_mm=PRESCRIPTION["clear_aperture_mm"]),
        fields=(FieldSpec(y_deg=0.0),),
        wavelengths=(WavelengthSpec(value_um=LAMBDA_M * 1e6, is_primary=True),),
    )


# ---------------------------------------------------------------------------
# Derived geometry: everything follows from the used semi-aperture
# ---------------------------------------------------------------------------


def geometry(params: dict[str, Any]) -> dict[str, Any]:
    """The physical realization of an instance's declared parameters."""
    grid_n = int(params["grid_n"])
    r_m = float(params["used_semi_aperture_mm"]) * 1e-3
    pitch_focal = 2.0 * r_m / grid_n
    pitch_object = object_pitch_m(params)
    waist_px = float(params["object_waist_pixels"])
    return {
        "grid_n": grid_n,
        "object_grid_n": int(params["object_grid_n"]),
        "sensor_shape": (int(params["sensor_rows"]), int(params["sensor_cols"])),
        "offset_px": int(params["object_offset_px"]),
        "used_semi_aperture_m": r_m,
        "focal_plane_pitch_m": pitch_focal,
        "object_pitch_m": pitch_object,
        "object_waist_m": waist_px * pitch_object,
        "mask_period_m": float(params["samples_per_period"]) * pitch_focal,
        "order_spacing_px": grid_n / float(params["samples_per_period"]),
        "order_spacing_m": grid_n / float(params["samples_per_period"]) * pitch_object,
        "field_angle_deg": field_angle_deg(params),
        "peak_wave_aberration_waves": peak_wave_aberration_waves(params),
        "focal_plane_beam_1e_radius_m": r_m / math.pi,
    }


def object_field(
    params: dict[str, Any], geom: dict[str, Any], *, modulate_here: bool = False
) -> ComplexField:
    """The object: a Gaussian on the object grid, displaced along the column axis.

    The displacement is built into the ARRAY, and that is not the obvious choice
    -- displacing the launch positions instead would be free. It does not work,
    and the reason is worth recording because it looks like it should: the
    launch-position phase makes every position carry the same global spectrum, so
    the field a launch position "sees" is the spectrum's own Fourier series
    evaluated there, which is periodic with the object grid's extent. Shifting the
    launch set by an integer number of samples therefore reproduces the *same*
    object exactly, while still moving the rays -- a configuration where the
    aberration is sampled off axis and the field is on axis, which is not a
    physical system at all. Measured while authoring: it reports a 4.7e4 field
    departure, because the normalization peak lands where there is no light.

    So the object grid has to hold the displacement, and that is what caps the
    reachable field angle: see ``FOURIER_PLANE_FIELD_CAPACITY``.
    """
    n = geom["object_grid_n"]
    axis = np.arange(n) - n // 2
    ii, jj = np.meshgrid(axis, axis, indexing="ij")
    waist = float(params["object_waist_pixels"])
    offset = geom["offset_px"]
    u = np.exp(-(ii**2 + (jj - offset) ** 2) / waist**2).astype(np.complex128)
    if modulate_here:
        # The `modulation-off-the-focal-plane` control, and it is B3-4F-IDEAL's own
        # `modulation-in-image-plane` control transplanted: the same mask, built by
        # the same constructor in its own sample units, multiplied into the object
        # instead of into the field on the shared focal plane.
        u = u * ideal_mask(
            n,
            str(params["modulation_type"]),
            float(params["samples_per_period"]),
            float(params["phase_depth_rad"]),
        )
    return ComplexField(
        u=u,
        sample_pitch_m=(geom["object_pitch_m"], geom["object_pitch_m"]),
        wavelength_m=LAMBDA_M,
        reference_plane=ReferencePlane(name="object", z_m=0.0),
        frame=Frame(),
    )


def object_rays(
    params: dict[str, Any],
    geom: dict[str, Any],
    *,
    drop_launch_phase: bool = False,
    modulate_at_object_plane: bool = False,
) -> Any:
    """``C_WAVE_TO_RAY`` at full enumeration, launched across the object grid.

    ``drop_launch_phase`` is the ``omitted-object-space-opl-term`` control, taken
    through ``C_WAVE_TO_RAY``'s own ``SamplingPerturbation`` hook rather than a
    hand-written copy of the emitter. The term it removes,
    ``exp(i k (d_u x_p + d_v y_p))``, is the optical path from the plane wavefront
    through the origin -- perpendicular to the wavelet's own direction -- to its
    launch point: ``n_object * (d . r_launch)`` with ``n_object = 1``, the same
    physical quantity ``couplers/handoff.py`` adds on the pupil route.
    """
    field = object_field(params, geom, modulate_here=modulate_at_object_plane)
    perturbation = SamplingPerturbation(apply_launch_phase=not drop_launch_phase)
    spectrum = decompose(field, perturbation=perturbation)
    density = sampling_density(spectrum, SamplingDensity.UNIFORM)
    indices = enumerate_indices(density)

    n = geom["object_grid_n"]
    axis = (np.arange(n) - n // 2) * geom["object_pitch_m"]
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    bundle = spectrum_to_rays(
        spectrum, indices, density, launch_positions_xy_m=positions, perturbation=perturbation
    )
    return bundle, spectrum


# ---------------------------------------------------------------------------
# Tracing and reconstruction, chunked
# ---------------------------------------------------------------------------


def _trace_chunks(bundle: Any, lens: Any, plane: ReferencePlane, *, skip: int) -> list[Any]:
    pieces: list[Any] = []
    total = bundle.count
    for lo in range(0, total, _TRACE_CHUNK):
        hi = min(total, lo + _TRACE_CHUNK)
        sub = dataclasses.replace(
            bundle,
            positions_m=bundle.positions_m[lo:hi],
            directions=bundle.directions[lo:hi],
            amplitude=bundle.amplitude[lo:hi],
            optical_path_length_m=bundle.optical_path_length_m[lo:hi],
        )
        batch = CoherentRayBatch(
            bundle=sub,
            ray_id=np.arange(lo, hi, dtype=np.int64),
            valid=np.ones(hi - lo, dtype=bool),
        )
        plans = plan_trace_bridges(batch, home=C_RAY_TO_WAVE_CAPABILITIES, device=CPU)
        traced, diagnostics = trace_ray_batch(
            batch, lens, image_plane=plane, plans=plans, skip=skip
        )
        pieces.append((traced.bundle, diagnostics))
    return pieces


def _accumulate(
    pieces: list[Any],
    *,
    grid_shape: tuple[int, int],
    pitch_m: float,
    plane: ReferencePlane,
    perturbation: Perturbation | None = None,
) -> tuple[np.ndarray, Any]:
    """Sum ``C_RAY_TO_WAVE`` over trace chunks.

    Each chunk's bundle declares ``one_over_n``, and ``1/N`` over a chunk is not
    ``1/N`` over the run, so the reconstruction is asked for the unnormalized sum
    and divided by the whole run's ray count once -- the same rule
    ``couplers/streaming.py`` states for its shards.
    """
    perturbation = perturbation or Perturbation()
    total = sum(b.count for b, _ in pieces)
    accumulated: np.ndarray | None = None
    first = None
    for bundle, _ in pieces:
        field, diagnostics = ray_to_wave(
            bundle,
            grid_shape=grid_shape,
            sample_pitch_m=(pitch_m, pitch_m),
            plane=plane,
            normalization="none",
            projection=Projection.ASM_CONSISTENT,
            perturbation=perturbation,
        )
        u = np.asarray(field.u)
        accumulated = u if accumulated is None else accumulated + u
        first = first or diagnostics
    assert accumulated is not None
    return accumulated / total, first


def _power(bundle: Any) -> float:
    return float(np.sum(np.abs(np.asarray(bundle.amplitude)) ** 2))


def _flip_about_origin(a: np.ndarray) -> np.ndarray:
    """``x -> -x`` about the sample that carries coordinate zero.

    ``n // 2`` is coordinate zero on an even grid, so a plain reversal is off by
    one sample: ``roll(a[::-1], 1)[c + j] == a[c - j]`` and a plain
    ``a[::-1][c + j] == a[c - j - 1]``. A physical 4f is transform-then-transform
    and the ideal relay is transform-then-inverse-transform, so this is the whole
    difference between the two images -- and a one-sample parity error would look
    like a small distortion rather than a convention mistake.
    """
    return np.roll(a[::-1, ::-1], (1, 1), axis=(0, 1))


# ---------------------------------------------------------------------------
# One pass through the chain
# ---------------------------------------------------------------------------


def run_chain(
    params: dict[str, Any],
    *,
    modulated: bool = True,
    plane_shift_mm: float = 0.0,
    declared_plane_shift_mm: float = 0.0,
    opl_sign_flip: bool = False,
    phasor_sign: int = 1,
    drop_launch_phase: bool = False,
    rebuild_second_group_for_declaration: bool = True,
    modulate_at_object_plane: bool = False,
) -> dict[str, Any]:
    """Object field to sensor field, once.

    ``plane_shift_mm`` moves the shared plane for real: group 1 traces to it,
    the modulation sits on it, group 2 starts from it. ``declared_plane_shift_mm``
    moves only the *declaration*: group 1 still traces to the true focal plane
    and the surface claims to be elsewhere. The two are different defects and are
    the two plane-related negative controls.
    """
    started = time.perf_counter()
    geom = geometry(params)
    grid_n = geom["grid_n"]

    z_focal_mm = LEG_MM + plane_shift_mm
    declared_focal_mm = z_focal_mm + declared_plane_shift_mm
    traced_plane = ReferencePlane(name="focal", z_m=z_focal_mm * 1e-3)
    declared_plane = ReferencePlane(name="focal", z_m=declared_focal_mm * 1e-3)
    sensor_plane = ReferencePlane(name="sensor", z_m=(declared_focal_mm + LEG_MM) * 1e-3)

    configure_optiland_execution(device=CPU, precision=Precision.FP64)
    lens_a = build_optiland_system(
        leg_spec("4f-group-1", ffd_mm=FFD_MM, bfd_mm=BFD_MM + plane_shift_mm)
    )
    # `rebuild_second_group_for_declaration=False` is the handoff-plane control:
    # the second group stays built for the plane the rays are really at, so a
    # mis-declaration is refused by trace_ray_batch rather than absorbed.
    lens_b = build_optiland_system(
        leg_spec(
            "4f-group-2",
            lead_mm=declared_focal_mm if rebuild_second_group_for_declaration else z_focal_mm,
            ffd_mm=FFD_MM,
            bfd_mm=BFD_MM,
        )
    )

    # --- object -> rays -> group 1 -> shared plane ------------------------
    launched, spectrum = object_rays(
        params,
        geom,
        drop_launch_phase=drop_launch_phase,
        modulate_at_object_plane=modulate_at_object_plane,
    )
    launched_power = _power(launched)
    pieces_a = _trace_chunks(launched, lens_a, traced_plane, skip=1)
    if len(pieces_a) != 1:
        raise RuntimeError(
            "the FULL_FIELD interaction takes one incident bundle; raise _TRACE_CHUNK "
            f"above {launched.count} or lower object_grid_n"
        )
    incident, trace_a = pieces_a[0]
    if declared_plane_shift_mm:
        # The mis-declaration, injected exactly where a real one would live: the
        # bundle's declared plane, which is what the surface and the next trace
        # both read. The ray geometry is untouched.
        incident = dataclasses.replace(incident, reference_plane=declared_plane)
    incident_power = _power(incident)

    # --- the modulation, at the (possibly displaced) shared plane ---------
    if modulated and not modulate_at_object_plane:
        transmission = ideal_mask(
            grid_n,
            str(params["modulation_type"]),
            float(params["samples_per_period"]),
            float(params["phase_depth_rad"]),
        )
    else:
        transmission = np.ones((grid_n, grid_n), dtype=np.complex128)
    surface = DiffractiveSurface(
        transmission=np.asarray(transmission, dtype=np.complex128),
        sample_pitch_m=(geom["focal_plane_pitch_m"], geom["focal_plane_pitch_m"]),
        plane=declared_plane,
    )

    keep = np.arange(_LAUNCH_RIM_TRIM, grid_n - _LAUNCH_RIM_TRIM + 1)
    axis = (keep - grid_n // 2) * geom["focal_plane_pitch_m"]
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    launch_positions = np.column_stack([xx.ravel(), yy.ravel()])

    interaction = diffractive_interaction(
        incident,
        surface,
        model=DiffractiveModel.FULL_FIELD,
        parameters=FullFieldParameters(
            launch_positions_xy_m=launch_positions,
            # None enumerates every propagating bin: the deterministic limit the
            # coupler protocol makes mandatory and first, and the only setting
            # under which this family's stochastic policy is honest.
            secondary_count=None,
        ),
    )
    outgoing = interaction.outgoing
    outgoing_power = _power(outgoing)

    # --- group 2 -> sensor -----------------------------------------------
    pieces_b = _trace_chunks(outgoing, lens_b, sensor_plane, skip=2)
    if opl_sign_flip:
        flipped = []
        for bundle, diagnostics in pieces_b:
            opl = np.asarray(bundle.optical_path_length_m)
            mutated = 2.0 * float(opl.mean()) - opl
            flipped.append(
                (
                    dataclasses.replace(
                        bundle,
                        optical_path_length_m=mutated,
                        optical_path_length_reference=(
                            f"{bundle.optical_path_length_reference} "
                            "[NEGATIVE CONTROL: conjugated about its own mean]"
                        ),
                    ),
                    diagnostics,
                )
            )
        pieces_b = flipped
    survived_power = sum(_power(b) for b, _ in pieces_b)

    sensor_u, recon = _accumulate(
        pieces_b,
        grid_shape=geom["sensor_shape"],
        pitch_m=geom["object_pitch_m"],
        plane=sensor_plane,
        perturbation=Perturbation(phase_sign=-1) if phasor_sign < 0 else Perturbation(),
    )

    cascade = interaction.model_diagnostics
    return {
        "geometry": geom,
        "sensor_u": sensor_u,
        "focal_plane_incident_power": float(cascade.incident_discrete_power),
        "focal_plane_transmitted_power": float(cascade.transmitted_discrete_power),
        "evanescent_power_fraction": float(cascade.evanescent_power_fraction),
        "launched_rays": int(launched.count),
        "outgoing_rays": int(outgoing.count),
        "launched_power": launched_power,
        "incident_power": incident_power,
        "outgoing_amplitude_power": outgoing_power,
        "survived_power": survived_power,
        "invalid_rays_group_1": int(trace_a["invalid_rays"]),
        "invalid_rays_group_2": int(sum(d["invalid_rays"] for _, d in pieces_b)),
        "model": interaction.model.value,
        "interaction_diagnostics": {
            key: value
            for key, value in interaction.diagnostics.items()
            if key
            in {
                "interaction",
                "model",
                "coupler",
                "substrate",
                "launch_count",
                "secondary_count",
                "propagating_modes",
                "enumerated",
                "density_kind",
                "opl_convention",
                "amplitude_convention",
            }
        },
        "sensor_reconstruction": {
            "max_transverse_direction": recon.max_transverse_direction,
            "grid_nyquist_direction_limit": recon.grid_nyquist_direction_limit,
            "grid_nyquist_satisfied": recon.grid_nyquist_satisfied,
            "ray_density_status": recon.ray_density_status,
            "reconstruction": recon.reconstruction,
        },
        "surface_positions_group_1_m": surface_positions_m(lens_a),
        "surface_positions_group_2_m": surface_positions_m(lens_b),
        "traced_focal_plane_z_m": traced_plane.z_m,
        "declared_focal_plane_z_m": declared_plane.z_m,
        "spectrum_propagating_modes": int(spectrum.propagating_count),
        "wall_seconds": time.perf_counter() - started,
    }


# ---------------------------------------------------------------------------
# The 4F-1 reference, at this instance's own parameters
# ---------------------------------------------------------------------------


def ideal_reference(params: dict[str, Any], geom: dict[str, Any], *, modulated: bool) -> np.ndarray:
    """B3-4F-IDEAL's own relay, cropped onto this instance's sensor grid.

    Three conventions are applied, all of them derived rather than fitted:

    1. the ideal relay with no mask is the identity, so the unmodulated
       reference is the object field itself;
    2. a physical 4f inverts and the ideal realization does not, hence
       :func:`_flip_about_origin`;
    3. an object displaced by ``+q`` pixels images to ``-q`` through an inverting
       relay, and the ideal relay is shift-invariant, so the displacement is a
       roll of the centred answer rather than a second computation.
    """
    ideal_params = {
        "modulation_type": str(params["modulation_type"]),
        "samples_per_period": float(params["samples_per_period"]),
        "phase_depth_rad": float(params["phase_depth_rad"]),
        "grid_n": geom["grid_n"],
        "object_waist_pixels": float(params["object_waist_pixels"]),
        "device": "cpu",
    }
    object_u, _mask_u, image = build_ideal(ideal_params)
    if not modulated:
        image = object_u
    image = _flip_about_origin(image)
    if geom["offset_px"]:
        image = np.roll(image, -geom["offset_px"], axis=1)

    grid_n = geom["grid_n"]
    rows, cols = geom["sensor_shape"]
    if rows > grid_n or cols > grid_n:
        raise ValueError(
            f"sensor grid {geom['sensor_shape']} exceeds the ideal relay's own grid "
            f"{grid_n}; the reference would have to be extrapolated"
        )
    r0 = grid_n // 2 - rows // 2
    c0 = grid_n // 2 - cols // 2
    return image[r0 : r0 + rows, c0 : c0 + cols]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _peak_index(geom: dict[str, Any]) -> tuple[int, int]:
    """Where the unmodulated image's peak sits on the sensor grid."""
    rows, cols = geom["sensor_shape"]
    return rows // 2, cols // 2 - geom["offset_px"]


def _order_table(
    sensor_u: np.ndarray,
    reference_peak: complex,
    params: dict[str, Any],
    geom: dict[str, Any],
    *,
    predicted_spacing_px: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-order measured quantities, in the shape B3-4F-IDEAL's reductions read.

    The order axis is axis 0 -- B3-4F-IDEAL's ``_mask`` varies along axis 0, so
    its orders do too -- and the displacement is on axis 1, which is why the two
    are separable observables. The sign of the spacing is ``+n`` here against
    B3-4F-IDEAL's ``-n``, and that is the 4f inversion, not a free choice.
    """
    rows = geom["sensor_shape"][0]
    row0, col0 = _peak_index(geom)
    spacing = geom["order_spacing_px"]
    predicted = spacing if predicted_spacing_px is None else predicted_spacing_px
    analytic = order_coefficients(params)
    profile = np.abs(sensor_u[:, col0]) ** 2
    reference_intensity = float(abs(reference_peak) ** 2)

    table: dict[int, dict[str, Any]] = {}
    for n in CHECKED_ORDERS:
        true_idx = int(row0 + round(n * spacing)) % rows
        predicted_idx = int(row0 + round(n * predicted)) % rows
        c_analytic = analytic[n]
        table[n] = {
            "true_idx": true_idx,
            "predicted_idx": predicted_idx,
            "found_idx": _peak_search(profile, true_idx),
            "c_analytic": c_analytic,
            "c_measured": complex(sensor_u[true_idx, col0] / reference_peak),
            "p_analytic": abs(c_analytic) ** 2,
            "p_measured": float(profile[true_idx] / reference_intensity),
        }
    return table


def _fwhm_px(profile: np.ndarray, peak_index: int) -> float:
    """Full width at half maximum, by linear interpolation, in samples."""
    peak = float(profile[peak_index])
    if peak <= 0.0:
        return float("nan")
    half = 0.5 * peak

    def edge(step: int) -> float:
        i = peak_index
        while 0 <= i + step < len(profile) and profile[i + step] > half:
            i += step
        j = i + step
        if not (0 <= j < len(profile)):
            return float("nan")
        lo, hi = float(profile[i]), float(profile[j])
        if lo == hi:
            return float(i)
        return i + step * (lo - half) / (lo - hi)

    return abs(edge(1) - edge(-1))


def _centroid(intensity: np.ndarray) -> tuple[float, float]:
    total = float(intensity.sum())
    if total <= 0.0:
        return (float("nan"), float("nan"))
    rows = np.arange(intensity.shape[0])[:, None]
    cols = np.arange(intensity.shape[1])[None, :]
    return (
        float((intensity * rows).sum() / total),
        float((intensity * cols).sum() / total),
    )


def _field_metrics(
    sensor_u: np.ndarray, ideal_u: np.ndarray, scale: complex
) -> tuple[float, float]:
    scaled = scale * sensor_u
    norm = float(np.linalg.norm(ideal_u))
    rel_l2 = float(np.linalg.norm(scaled - ideal_u) / norm) if norm > 0.0 else float("nan")

    intensity = np.abs(ideal_u) ** 2
    bright = intensity > 1e-2 * float(intensity.max())
    if not bright.any():
        return rel_l2, float("nan")
    delta = np.angle(scaled[bright]) - np.angle(ideal_u[bright])
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return rel_l2, float(np.sqrt(np.mean(wrapped**2)))


def measure(
    run: dict[str, Any],
    reference: dict[str, Any],
    params: dict[str, Any],
) -> tuple[dict[str, Measurement], dict[str, Any]]:
    """Every declared metric, plus the diagnostics a reader needs beside them."""
    geom = run["geometry"]
    row0, col0 = _peak_index(geom)

    ideal_masked = ideal_reference(params, geom, modulated=True)
    ideal_plain = ideal_reference(params, geom, modulated=False)

    # THE one complex constant. Measured on the unmodulated pair of the same
    # geometry, not fitted to the modulated comparison: the chain's absolute
    # phase reference is not certified (Optiland's opd sign and reference plane
    # were recorded unverified by M1), so exactly one piston-and-scale is removed
    # and it is removed using a different measurement than the one being scored.
    measured_ref_peak = complex(reference["sensor_u"][row0, col0])
    ideal_ref_peak = complex(ideal_plain[row0, col0])
    scale = ideal_ref_peak / measured_ref_peak

    rel_l2, phase_rms = _field_metrics(run["sensor_u"], ideal_masked, scale)
    table = _order_table(run["sensor_u"], measured_ref_peak, params, geom)

    incident = run["focal_plane_incident_power"]
    transmitted = run["focal_plane_transmitted_power"]
    power_ratio = abs(transmitted / incident - 1.0) if incident > 0.0 else float("nan")

    launched = run["launched_power"]
    clipped = 0.0
    if launched > 0.0:
        clipped_group_1 = max(0.0, launched - run["incident_power"]) / launched
        outgoing = run["outgoing_amplitude_power"]
        clipped_group_2 = (
            max(0.0, outgoing - run["survived_power"]) / outgoing if outgoing > 0.0 else 0.0
        )
        clipped = clipped_group_1 + clipped_group_2

    ref_profile = np.abs(reference["sensor_u"][:, col0]) ** 2
    ideal_profile = np.abs(ideal_plain[:, col0]) ** 2
    measured_fwhm = _fwhm_px(ref_profile, int(np.argmax(ref_profile)))
    ideal_fwhm = _fwhm_px(ideal_profile, int(np.argmax(ideal_profile)))
    fwhm_error = (
        abs(measured_fwhm / ideal_fwhm - 1.0)
        if ideal_fwhm and np.isfinite(ideal_fwhm)
        else float("nan")
    )
    measured_centroid = _centroid(np.abs(reference["sensor_u"]) ** 2)
    ideal_centroid = _centroid(np.abs(ideal_plain) ** 2)
    centroid_shift = math.hypot(
        measured_centroid[0] - ideal_centroid[0], measured_centroid[1] - ideal_centroid[1]
    )

    def m(value: float) -> Measurement:
        return Measurement(
            value=value,
            uncertainty=_FLOOR,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
        )

    measurements = {
        "field_relative_l2_vs_ideal_4f": m(rel_l2),
        "field_phase_rms_vs_ideal_rad": m(phase_rms),
        "order_power_relative_l2": m(_order_power_relative_l2(table)),
        "order_phase_error_rad": m(_order_phase_error_rad(table)),
        "order_location_error_frac": m(
            _order_location_error_frac(table, geom["order_spacing_px"])
        ),
        "fourier_plane_power_relative_error": m(power_ratio),
        "clipped_power_fraction": m(clipped),
        "psf_fwhm_relative_error": m(fwhm_error),
        "psf_centroid_shift_px": m(centroid_shift),
    }

    diagnostics = {
        "diffractive_model": run["model"],
        "interaction": run["interaction_diagnostics"],
        "sensor_reconstruction": run["sensor_reconstruction"],
        "geometry_m": {
            key: geom[key]
            for key in (
                "used_semi_aperture_m",
                "focal_plane_pitch_m",
                "object_pitch_m",
                "object_waist_m",
                "mask_period_m",
                "order_spacing_m",
                "order_spacing_px",
                "field_angle_deg",
                "peak_wave_aberration_waves",
                "focal_plane_beam_1e_radius_m",
            )
        },
        "prescription": PRESCRIPTION,
        "surface_positions_group_1_m": run["surface_positions_group_1_m"],
        "surface_positions_group_2_m": run["surface_positions_group_2_m"],
        "traced_focal_plane_z_m": run["traced_focal_plane_z_m"],
        "declared_focal_plane_z_m": run["declared_focal_plane_z_m"],
        "ray_counts": {
            "launched": run["launched_rays"],
            "outgoing": run["outgoing_rays"],
            "object_spectrum_propagating_modes": run["spectrum_propagating_modes"],
        },
        "power_accounting": {
            "note": (
                "sum |a|**2 at each stage. The FULL_FIELD interaction rebases both the "
                "amplitude (to an importance-weighted spectral amplitude) and the "
                "optical path (to zero at the shared plane), so the incoming and "
                "outgoing ray-power sums are NOT comparable across it -- the "
                "comparable pair is the accumulated field's discrete power before and "
                "after the transmission multiply, which is what "
                "fourier_plane_power_relative_error reports."
            ),
            "launched_ray_power": run["launched_power"],
            "after_group_1_ray_power": run["incident_power"],
            "focal_plane_field_power_in": incident,
            "focal_plane_field_power_out": transmitted,
            "outgoing_ray_power": run["outgoing_amplitude_power"],
            "after_group_2_ray_power": run["survived_power"],
            "invalid_rays_group_1": run["invalid_rays_group_1"],
            "invalid_rays_group_2": run["invalid_rays_group_2"],
            "evanescent_power_fraction": run["evanescent_power_fraction"],
        },
        "psf": {
            "note": "measured on the unmodulated run of the same geometry",
            "measured_fwhm_px": measured_fwhm,
            "ideal_fwhm_px": ideal_fwhm,
            "measured_centroid_px": list(measured_centroid),
            "ideal_centroid_px": list(ideal_centroid),
        },
        "unmodulated_reference": {
            "note": (
                "the same chain with the modulation replaced by unit transmission. It "
                "supplies the one complex normalization constant, the point response, "
                "and the sampling floor the modulated departure has to be read against."
            ),
            "field_relative_l2_vs_ideal_4f": _field_metrics(
                reference["sensor_u"], ideal_plain, scale
            )[0],
            "wall_seconds": reference["wall_seconds"],
        },
        "measured_order_coefficients": {
            str(n): {
                "measured_real": table[n]["c_measured"].real,
                "measured_imag": table[n]["c_measured"].imag,
                "analytic_real": table[n]["c_analytic"].real,
                "analytic_imag": table[n]["c_analytic"].imag,
                "sensor_row": table[n]["true_idx"],
            }
            for n in CHECKED_ORDERS
        },
        "normalization": {
            "complex_scale_real": scale.real,
            "complex_scale_imag": scale.imag,
            "source": "unmodulated run of the same geometry, at the image peak sample",
        },
        "wall_seconds": run["wall_seconds"],
    }
    return measurements, diagnostics


# ---------------------------------------------------------------------------
# Instances
# ---------------------------------------------------------------------------

_COMMON: dict[str, Any] = {
    "modulation_type": "sinusoidal_phase",
    "samples_per_period": 8.0,
    "phase_depth_rad": 1.5,
    "object_offset_px": 0.0,
    "grid_n": 48,
    # 32 everywhere, not sized per instance: an off-axis instance has to differ
    # from its on-axis twin in exactly ONE axis, and the object grid IS the field
    # of view (see object_field()), so a grid sized to each offset would make
    # every field comparison a two-axis change.
    "object_grid_n": 32,
    "object_waist_pixels": 2.0,
    "sensor_rows": 32,
    # 48 everywhere, not 32, so that an off-axis instance differs from its
    # on-axis twin in exactly ONE axis. A sensor grid sized per instance would
    # make every field-angle comparison a two-axis change.
    "sensor_cols": 48,
    "diffractive_model": "full_field",
    "device": "cpu",
}


def _params(**overrides: Any) -> dict[str, Any]:
    return {**_COMMON, **overrides}


#: The paraxial-limit sweep. ``used_semi_aperture_mm`` is the only axis that
#: moves across APERTURE-01..04, and FIELD-01 moves only ``object_offset_px``
#: away from APERTURE-04.
LIMIT_PARAMETERS: dict[str, dict[str, Any]] = {
    "B3-4F-REAL-APERTURE-01": _params(used_semi_aperture_mm=4.0),
    "B3-4F-REAL-APERTURE-02": _params(used_semi_aperture_mm=2.0),
    "B3-4F-REAL-APERTURE-03": _params(used_semi_aperture_mm=1.0),
    "B3-4F-REAL-APERTURE-04": _params(used_semi_aperture_mm=0.7),
    "B3-4F-REAL-FIELD-01": _params(used_semi_aperture_mm=0.7, object_offset_px=6.0),
}

#: Characterization. Every instance moves exactly one axis away from
#: B4-4F-REAL-REFERENCE.
CHARACTERIZATION_PARAMETERS: dict[str, dict[str, Any]] = {
    "B4-4F-REAL-REFERENCE": _params(used_semi_aperture_mm=4.0),
    # 6.0 mm is past SHARED_PLANE_RAY_ANGLE_CAPACITY's ceiling and is here on
    # purpose: the chain refuses it, and the refusal is the measurement. The
    # margin crosses zero at 4.19 mm for grid_n = 48, so the wide end of the
    # aperture axis IS
    # B4-4F-REAL-REFERENCE at 4.0 mm and this instance is what says so.
    "B4-4F-REAL-APERTURE-WIDE": _params(used_semi_aperture_mm=6.0),
    "B4-4F-REAL-APERTURE-SMALL": _params(used_semi_aperture_mm=0.25),
    "B4-4F-REAL-FIELD-01": _params(used_semi_aperture_mm=4.0, object_offset_px=3.0),
    "B4-4F-REAL-FIELD-02": _params(used_semi_aperture_mm=4.0, object_offset_px=6.0),
    "B4-4F-REAL-FREQUENCY-01": _params(used_semi_aperture_mm=4.0, samples_per_period=4.0),
    "B4-4F-REAL-MODULATION-BINARY": _params(
        used_semi_aperture_mm=4.0, modulation_type="binary_phase"
    ),
    "B4-4F-REAL-GRID-64": _params(used_semi_aperture_mm=4.0, grid_n=64),
}

#: Which family owns each instance, and which instance it is read against.
_REFERENCE_INSTANCE = {
    **dict.fromkeys(LIMIT_PARAMETERS, "B3-4F-REAL-APERTURE-04"),
    **dict.fromkeys(CHARACTERIZATION_PARAMETERS, "B4-4F-REAL-REFERENCE"),
}

_FAMILY_OF = {
    **dict.fromkeys(LIMIT_PARAMETERS, B3_4F_REAL),
    **dict.fromkeys(CHARACTERIZATION_PARAMETERS, B4_4F_REAL),
}

ALL_PARAMETERS: dict[str, dict[str, Any]] = {**LIMIT_PARAMETERS, **CHARACTERIZATION_PARAMETERS}

#: Which instances demonstrate which negative controls, and why only these.
#: B3-4F-IDEAL's precedent: a control's detection margin is a ratio against the
#: baseline, so a baseline already broken by real physics reports a false
#: verdict. APERTURE-04 is the deepest-inside on-axis instance and FIELD-01 is
#: the only one whose object is displaced, which is the single instance where the
#: object-space reference term is not identically zero.
_CONTROLS_ON: dict[str, tuple[str, ...]] = {
    # The deepest-inside instance: every metric is at its floor, so a detection
    # margin measured here is a statement about the control and nothing else.
    "B3-4F-REAL-APERTURE-04": (
        "opl-sign-flip",
        "phasor-sign-flip",
        "modulation-off-the-focal-plane",
        "handoff-plane-mis-declared",
        # The on-axis arm of the object-space term, recorded so the claim that it
        # is not hidden by an on-axis instance is measured rather than argued.
        "omitted-object-space-opl-term",
    ),
    "B3-4F-REAL-FIELD-01": ("omitted-object-space-opl-term",),
}

#: Declared before the run, so a pass or a fail is read against a stated
#: expectation rather than reverse-engineered from the number.
_EXPECTED: dict[str, dict[str, str]] = {
    "B3-4F-REAL-APERTURE-01": {
        "field_relative_l2_vs_ideal_4f": "NOT met -- FAR_OUTSIDE by PARAXIAL_LIMIT "
        "(0.4067 waves of peak aberration). The far end of the convergence sweep, and "
        "expected near 9e-2 from the measured departure/aberration ratio of 0.0372",
        "order_power_relative_l2": "met, and that is the informative part: spherical "
        "aberration is a pure phase error, so an order's PEAK power stays at its floor "
        "even here",
    },
    "B3-4F-REAL-APERTURE-02": {
        "field_relative_l2_vs_ideal_4f": "NOT met -- FAR_OUTSIDE by PARAXIAL_LIMIT "
        "(0.02542 waves). Expected near 1/16 of APERTURE-01, i.e. 5.9e-3"
    },
    "B3-4F-REAL-APERTURE-03": {
        "field_relative_l2_vs_ideal_4f": "met -- INSIDE (1.589e-3 waves). Expected "
        "near 1/16 of APERTURE-02, i.e. 3.7e-4, which is the last point still on the "
        "fourth-power line"
    },
    "B3-4F-REAL-APERTURE-04": {
        "field_relative_l2_vs_ideal_4f": "met -- deepest inside (3.81e-4 waves). "
        "Expected to STOP falling as R**4, because the aberration term here is small "
        "enough that the comparison's own floor takes over: the ratio to APERTURE-03 "
        "should be well under 16. Expected NOT to be explained by the unmodulated arm "
        "alone -- that arm should keep falling closer to the law, so the two arms "
        "should SEPARATE here where they agreed at wider apertures, which is what "
        "attributes the broken ratio to the modulation's own floor rather than to the "
        "relay"
    },
    "B3-4F-REAL-FIELD-01": {
        "field_relative_l2_vs_ideal_4f": "met -- the same aperture as APERTURE-04 with "
        "the object displaced 6 px (0.144 deg). Expected slightly WORSE than "
        "APERTURE-04: the displacement adds to the height at which a ray crosses "
        "group 1, and PARAXIAL_LIMIT is blind to that term by declaration"
    },
}


def canonical_instance(instance_id: str) -> Any:
    family = _FAMILY_OF[instance_id]
    params = ALL_PARAMETERS[instance_id]
    reference_id = _REFERENCE_INSTANCE[instance_id]
    expected = dict(_EXPECTED.get(instance_id, {}))
    expected["compared_against_4f1"] = (
        "B3-4F-IDEAL's relay and Fourier series at THIS instance's own "
        f"modulation_type={params['modulation_type']}, "
        f"samples_per_period={params['samples_per_period']}, "
        f"phase_depth_rad={params['phase_depth_rad']} -- zero modulation axes differ "
        "across the comparison"
    )
    if instance_id == reference_id:
        expected["differs_from"] = f"{reference_id} is this family's reference instance"
    else:
        differing = differing_axes(params, ALL_PARAMETERS[reference_id])
        expected["differs_from"] = (
            f"{reference_id} in exactly one axis: {differing[0]} "
            f"({ALL_PARAMETERS[reference_id][differing[0]]} -> {params[differing[0]]})"
        )
    return family.instantiate(instance_id, params, expected=expected)


def differing_axes(params: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    """Which declared axes differ, checked rather than asserted in prose.

    CHE-145 requires exactly one axis to differ per instance from the instance it
    is compared against. Returning the list rather than a boolean is what lets
    :func:`canonical_instance` name the axis in the record's ``expected`` block,
    and :func:`run_instance` refuse to write a record that violates it.
    """
    return sorted(k for k in params if params[k] != reference[k])


def _predicted_shared_plane_angle(params: dict[str, Any]) -> float:
    """What SHARED_PLANE_RAY_ANGLE_CAPACITY predicts for |d|max at the shared plane.

    Recorded beside a refusal so the predicate's arithmetic and the refusal's own
    measured number can be compared rather than taken on trust.
    """
    r_m = float(params["used_semi_aperture_mm"]) * 1e-3
    return float(params["object_grid_n"]) * LAMBDA_M / (4.0 * r_m) + 2.0 * residual_ray_angle_rad(
        params
    )


def declared_instance_ids(family_id: str | None = None) -> tuple[str, ...]:
    if family_id == "B3-4F-REAL":
        return tuple(LIMIT_PARAMETERS)
    if family_id == "B4-4F-REAL":
        return tuple(CHARACTERIZATION_PARAMETERS)
    return tuple(LIMIT_PARAMETERS) + tuple(CHARACTERIZATION_PARAMETERS)


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def _controls(
    instance_id: str,
    params: dict[str, Any],
    reference: dict[str, Any],
    baseline: dict[str, Measurement],
) -> dict[str, Any]:
    wanted = _CONTROLS_ON.get(instance_id, ())
    if not wanted:
        return {}
    family = _FAMILY_OF[instance_id]
    geom = geometry(params)
    results: dict[str, Any] = {}

    def judge(
        control_id: str, metric: str, mutate: Any, note: str = ""
    ) -> None:
        """Run one control, unless its own target metric is already failing.

        B3-4F-IDEAL skips controls on instances it does not declare INSIDE, for
        the reason that a detection margin against an already-broken baseline is
        a ratio between two failures. The rule here is the sharper form of the
        same thing: what matters is whether the metric THIS control targets is
        within its own gate at baseline, not whether every metric is. That lets
        modulation-off-the-focal-plane be demonstrated at the wide aperture,
        where the plane displacement is a full depth of focus and the order
        powers are still clean, while still refusing to score a control against
        a metric that has already failed.
        """
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
        mutated, _diagnostics = measure(mutate(), reference, params)
        results[control_id] = control_result(
            control_id,
            metric,
            baseline=baseline[metric],
            mutated=mutated[metric],
            threshold=tolerance.threshold,
            note=note,
        )

    if "omitted-object-space-opl-term" in wanted:
        judge(
            "omitted-object-space-opl-term",
            "field_relative_l2_vs_ideal_4f",
            lambda: run_chain(params, drop_launch_phase=True),
            note=(
                f"object displaced {geom['offset_px']} px "
                f"({geom['field_angle_deg']:.4f} deg). Unlike the pupil-handoff route "
                "this term is NOT a piston on axis, because the object bundle is not "
                "collimated -- every wavelet has its own direction -- so it is "
                "demonstrated on BOTH an on-axis and a displaced instance rather than "
                "only off axis. CHE-145's requirement that it not be hidden by an "
                "on-axis instance is therefore met a fortiori"
            ),
        )
    if "opl-sign-flip" in wanted:
        judge(
            "opl-sign-flip",
            "field_phase_rms_vs_ideal_rad",
            lambda: run_chain(params, opl_sign_flip=True),
        )
    if "phasor-sign-flip" in wanted:
        judge(
            "phasor-sign-flip",
            "field_phase_rms_vs_ideal_rad",
            lambda: run_chain(params, phasor_sign=-1),
        )
    if "modulation-off-the-focal-plane" in wanted:
        # Two arms, because the obvious one measured nothing and the reason is a
        # property of the system rather than of the control. Reported together so
        # the null result is on record beside the decisive one.
        axial, _ = measure(
            run_chain(params, plane_shift_mm=_CONTROL_PLANE_SHIFT_MM), reference, params
        )
        beam_semi_radius_m = geom["focal_plane_beam_1e_radius_m"]
        object_na = geom["object_waist_m"] / F_M
        object_dof_m = LAMBDA_M / object_na**2
        judge(
            "modulation-off-the-focal-plane",
            "order_power_relative_l2",
            lambda: run_chain(params, modulate_at_object_plane=True),
            note=(
                "DECISIVE ARM: the modulation moved to the object plane -- the same "
                "mask from the same constructor, multiplied into the object instead of "
                "into the field on the shared focal plane, which is B3-4F-IDEAL's own "
                "modulation-in-image-plane control transplanted onto this rung. "
                "NULL ARM, measured and reported because it is a finding: displacing "
                f"the shared plane axially by {_CONTROL_PLANE_SHIFT_MM} mm "
                "(consistently -- group 1 traces to it, the modulation sits on it, "
                "group 2 starts from it) moves order_power_relative_l2 only from "
                f"{baseline['order_power_relative_l2'].value:.6g} to "
                f"{axial['order_power_relative_l2'].value:.6g}. That is not a weak "
                "control, it is the physics: the field on the shared plane is the "
                "object's own spectrum, so its angular content is set by the object's "
                f"angular size seen from the group ({object_na:.3e} rad for a "
                f"{geom['object_waist_m'] * 1e6:.2f} um waist), giving a depth of "
                f"focus of lambda / NA**2 = {object_dof_m:.1f} m. The beam is "
                f"{beam_semi_radius_m * 1e6:.1f} um wide there and 2 mm is 1e-4 of its "
                "Rayleigh range, and a displacement that also carries group 2 and the "
                "sensor is in any case a rigid translation of the second half of the "
                "system. So in THIS configuration the shared plane's axial position is "
                "not load-bearing, and a control that displaced it by 2 mm and called "
                "the result a demonstration would be reporting a null as a pass"
            ),
        )
    if "handoff-plane-mis-declared" in wanted:
        results["handoff-plane-mis-declared"] = _handoff_declaration_control(
            params, reference, baseline, family
        )
    return results


def _handoff_declaration_control(
    params: dict[str, Any],
    reference: dict[str, Any],
    baseline: dict[str, Measurement],
    family: Any,
) -> NegativeControlResult:
    """Mis-declare the ray/wave handoff plane, two ways, and report both.

    This control found something rather than confirming something, so it reports
    two arms:

    **Refused.** Declare the surface -- and therefore the outgoing rays -- at
    ``z_focal + 2 mm`` while the second group is still built for ``z_focal``, and
    ``trace_ray_batch`` refuses with ``REFERENCE_PLANE_MISMATCH`` before any
    number is produced. That is the control firing: the mis-declaration cannot
    reach a result.

    **Numerically inert.** Declare it at ``z_focal + 2 mm`` and *also* rebuild the
    second group around the declaration, and the answer is bit-identical to the
    correct one. That is not a hole in the check, it is a fact about the geometry:
    the outgoing rays are new rays created at the declared plane, the coherent
    accumulation that produced them ignores z entirely (``C_RAY_TO_WAVE`` sums
    plane wavelets over transverse coordinates), and translating the modulation
    plane, the second group and the sensor together by the same 2 mm is a rigid
    translation of the whole second half of the system. So an axial
    mis-declaration is *either* refused or *not a physical change* -- there is no
    third case where it silently corrupts the answer, and both arms are recorded
    so a reader is not left to assume the second one.
    """
    from core.boundary import ContractError

    inert, _diagnostics = measure(
        run_chain(params, declared_plane_shift_mm=_CONTROL_PLANE_SHIFT_MM),
        reference,
        params,
    )
    metric = "order_power_relative_l2"
    try:
        run_chain(
            params,
            declared_plane_shift_mm=_CONTROL_PLANE_SHIFT_MM,
            rebuild_second_group_for_declaration=False,
        )
    except ContractError as refusal:
        return NegativeControlResult(
            control_id="handoff-plane-mis-declared",
            outcome=NegativeControlOutcome.FIRED,
            target_metric=metric,
            baseline=baseline[metric],
            mutated=None,
            note=(
                f"REFUSED, which is the mutation failing: {str(refusal)[:200]}. "
                "The chain produced no result at all, so no "
                "metric could pass. Second arm, measured: when the second group is "
                "ALSO rebuilt around the mis-declaration the answer is inert -- "
                f"{metric} moves from {baseline[metric].value:.9g} to "
                f"{inert[metric].value:.9g} -- because translating the modulation "
                "plane, group 2 and the sensor together by 2 mm is a rigid "
                "translation of the second half of the system and the coherent "
                "accumulation is z-blind by construction. An axial mis-declaration is "
                "therefore either refused or not a physical change; it is never "
                "silently absorbed."
            ),
        )
    return NegativeControlResult(
        control_id="handoff-plane-mis-declared",
        outcome=NegativeControlOutcome.DID_NOT_FIRE,
        target_metric=metric,
        baseline=baseline[metric],
        mutated=inert[metric],
        note=(
            "the mis-declared plane was neither refused nor detected, which would be "
            "a real hole in the handoff contract"
        ),
    )


# ---------------------------------------------------------------------------
# Run + verify
# ---------------------------------------------------------------------------

#: The unmodulated run depends on the geometry and not on the modulation, and
#: several instances share a geometry. Cached per process because it is half the
#: cost of the family.
_REFERENCE_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}


def _reference_run(params: dict[str, Any]) -> dict[str, Any]:
    key = tuple(
        params[k]
        for k in (
            "used_semi_aperture_mm",
            "object_offset_px",
            "grid_n",
            "object_grid_n",
            "object_waist_pixels",
            "sensor_rows",
            "sensor_cols",
        )
    )
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = run_chain(params, modulated=False)
    return _REFERENCE_CACHE[key]


def run_instance(instance_id: str, *, with_controls: bool = True) -> InstanceRun:
    instance = canonical_instance(instance_id)
    family = _FAMILY_OF[instance_id]
    params = dict(instance.parameters)

    reference_id = _REFERENCE_INSTANCE[instance_id]
    if instance_id != reference_id:
        differing = differing_axes(params, ALL_PARAMETERS[reference_id])
        if len(differing) != 1:
            raise AssertionError(
                f"{instance_id} differs from {reference_id} in {differing}, but CHE-145 "
                "requires exactly one axis per instance"
            )

    started = time.perf_counter()
    refusal, run = probe_refusal(lambda: run_chain(params))
    if refusal is not None:
        # A refusal is a measurement here, not a failure of the driver. It is the
        # only executable form the aperture ceiling has: past
        # SHARED_PLANE_RAY_ANGLE_CAPACITY the aberrated ray angles cannot be
        # represented on the shared plane's grid, and C_RAY_TO_WAVE says so with a
        # structured code instead of returning a plausible field. The record
        # carries the refusal and no metrics, so the instance reads as refused
        # rather than as passing.
        record = record_from_probe(
            instance,
            component="C_RAY_TO_WAVE",
            node_id="real_4f_relay",
            refusal=refusal,
            observed_parameters={
                "diffractive_model": DIFFRACTIVE_MODEL,
                "peak_wave_aberration_waves": peak_wave_aberration_waves(params),
                "predicted_shared_plane_direction_cosine": _predicted_shared_plane_angle(params),
            },
            wall_seconds=time.perf_counter() - started,
            diagnostics=[
                {
                    "refusal_is_the_measurement": (
                        "SHARED_PLANE_RAY_ANGLE_CAPACITY declares this instance outside "
                        "before it runs, and the refusal is what makes that declaration "
                        "executable. The predicate's own predicted |d|max is reported "
                        "beside the refusal's measured one so the two can be compared."
                    ),
                    "predicate_margin": family.evaluate_validity(params)[1],
                }
            ],
        )
        result = verify(family, instance, record, measurements={})
        return InstanceRun(family=family, instance=instance, record=record, result=result)

    reference = _reference_run(params)
    measurements, diagnostics = measure(run, reference, params)
    controls = _controls(instance_id, params, reference, measurements) if with_controls else {}
    wall_seconds = time.perf_counter() - started

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="real_4f_relay",
        refusal=None,
        observed_parameters={
            # The model, named in the record because it is named at the call
            # site: CHE-142's rule is that the model is never inferred, and a
            # record that does not say which one ran cannot show that it wasn't.
            "diffractive_model": run["model"],
            "field_angle_deg": run["geometry"]["field_angle_deg"],
            "peak_wave_aberration_waves": run["geometry"]["peak_wave_aberration_waves"],
            "outgoing_rays": run["outgoing_rays"],
        },
        device_precision=DevicePrecisionObservation(
            requested_device="cpu",
            actual_device="cpu",
            requested_dtype="complex128",
            actual_dtype=str(run["sensor_u"].dtype),
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
            {"FOCAL_PLANE_POWER_CONSERVED": measurements["fourier_plane_power_relative_error"]}
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
    parser.add_argument("--family", default=None, choices=("B3-4F-REAL", "B4-4F-REAL"))
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
            print(f"  [control] {control.control_id}: {control.outcome.value}  {control.note}")
        print(
            f"  validity: declared={run.result.validity.declared.value} "
            f"observed={run.result.validity.observed.value}"
        )
        if args.write:
            path = write_instance_record(run, driver="systems/b3_4f_real", directory=RECORDS_DIR)
            print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
