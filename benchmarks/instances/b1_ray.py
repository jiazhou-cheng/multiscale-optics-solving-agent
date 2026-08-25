"""The five B1 ray families, executed against Optiland. Closed forms and one invariant.

CHE-106 (M1.1). The families were declared and never run: three reported
``MEASURED_OFF_GATE`` on numbers inherited from a retired task set, and two
reported ``NOT_MEASURED``. This module runs all five through the shipping
adapter.

**No Optiland output decides Optiland's correctness.** Every gate below is a
closed form or a conservation law evaluated in this file, and PB7/CHE-58 finding
F2 is the reason that rule exists: ``FFTPSF`` and ``HuygensPSF`` share one
``Wavefront``/OPD front end and are not two oracles.

Three things this file had to establish about the traced record before any of it
worked, and each one is a convention nobody had written down
------------------------------------------------------------------------------
1. **A collimated on-axis ray's transverse position where it meets the first
   surface is exactly its launch position.** The record exports ``launch_x_m`` /
   ``launch_y_m``, so ``sin(i) = rho / R`` is exact at a spherical first surface
   with no reconstruction and no assumption about where the vertex sits in the
   traced frame. That is what makes ``B1-RAY-SNELL`` a machine-precision test
   rather than a paraxial one.

2. **The image plane refracts.** The medium after the last surface is the medium
   the ray is in when it reaches the image plane, and the image surface itself is
   air -- so a system whose last surface carries glass applies a second
   refraction at the image plane. Measured, not assumed: with the glass left on,
   the exported angle satisfies ``sin(u_exported) = n * sin(u_in_medium)`` to
   1e-16. Every system below therefore ends in air, and the Snell oracle applies
   Snell twice because the geometry does.

3. **A traced focal length is not a paraxial focal length.** A real marginal ray
   at height ``h`` focuses short by an amount quadratic in ``h``: on the
   reference singlet the innermost ring of a 64-ring fan reads 1.13e-6 relative
   and an 8-ring fan reads 7.24e-5, a clean factor of four per doubling. So the
   traced value is extracted in the ``h -> 0`` limit from a ladder, which is both
   more accurate than any single rung and exactly the fitted-exponent evidence
   M1.1 asks for. The exponent is a prediction -- spherical aberration is
   quadratic in aperture -- rather than a fitted convenience.

Run it::

    ./run.sh python benchmarks/instances/b1_ray.py --write
"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.artifacts import ArtifactRecord
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
from couplers.handoff import (
    DeclaredHandoffPlane,
    HandoffPerturbation,
    declare_coherent_bundle,
)
from registry.loader import Registry
from runtime.instance_runner import observed_placement, probe_refusal, record_from_probe
from solvers.base import ModelRunRequest
from verification.evidence import (
    InstanceRun,
    control_result,
    fit_convergence,
    write_instance_record,
)
from verification.families.b1_ray import (
    B1_RAY_EFL,
    B1_RAY_LAGRANGE,
    B1_RAY_OFFAXIS_OPL,
    B1_RAY_PLATE,
    B1_RAY_SNELL,
)
from verification.result import Measurement, UncertaintyBasis
from verification.verifier import verify

__all__ = [
    "declared_instance_ids",
    "device_precision_matrix",
    "run_all",
    "run_instance",
    "unsupported_configuration",
]

ROOT = repository_root()
MM_PER_M = 1e3
WAVELENGTH_UM = 0.5876

#: The refinement ladder every deterministic ray family uses. Four rungs, each a
#: halving of the innermost ray height, which is what makes a fitted exponent
#: mean something: two points fit a line exactly.
RINGS_LADDER: tuple[int, ...] = (8, 16, 32, 64)


def _instance(family: Any, instance_id: str) -> Any:
    for candidate in family.canonical_instances:
        if candidate.instance_id == instance_id:
            return candidate
    raise KeyError(f"{family.family_id} declares no instance {instance_id!r}")


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trace:
    """One traced bundle, plus the record it came from."""

    record: ArtifactRecord
    arrays: dict[str, np.ndarray]

    @property
    def launch_radius_mm(self) -> np.ndarray:
        return (
            np.hypot(self.arrays["launch_x_m"], self.arrays["launch_y_m"]) * MM_PER_M
        )

    @property
    def direction(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.arrays["L"], self.arrays["M"], self.arrays["N"]

    @property
    def placement(self) -> dict[str, str]:
        """Device and dtype, read off the traced array rather than off the request."""
        return observed_placement(self.arrays["x_m"])

    def axial_crossing_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Where each ray crosses the axis, and the launch height that got it there.

        The crossing is extrapolated from the exported position and direction
        rather than read at the image plane, so the measurement does not depend
        on the image plane happening to be at the focus. Rays are projected on
        whichever transverse axis they actually carry, because a hexapolar fan
        puts most of its rays off the meridian.
        """
        L, M, N = self.direction
        x, y, z = self.arrays["x_m"], self.arrays["y_m"], self.arrays["z_m"]
        height = self.launch_radius_mm
        keep = height > 0.0
        use_m = np.abs(M[keep]) > np.abs(L[keep])
        denominator = np.where(use_m, M[keep], L[keep])
        coordinate = np.where(use_m, y[keep], x[keep])
        step = -coordinate / denominator
        return height[keep], (z[keep] + step * N[keep]) * MM_PER_M

    def innermost(self, values: np.ndarray, heights: np.ndarray) -> tuple[float, float]:
        index = int(np.argmin(heights))
        return float(heights[index]), float(values[index])


def _trace(
    spec: OpticalSystemSpec,
    *,
    rings: int,
    field_hy: float = 0.0,
    handoff_plane: str = "image_surface",
    wavelength_um: float = WAVELENGTH_UM,
) -> Trace:
    directory = tempfile.mkdtemp(prefix="b1-ray-")
    result = Registry  # keeps the import honest; the adapter is resolved below
    del result
    from solvers.optiland.adapter import get_adapter

    run = get_adapter().run(
        ModelRunRequest(
            run_id="b1-ray",
            node_id="lens",
            config={
                "prescription": spec,
                "num_rays": rings,
                "wavelength": wavelength_um,
                "Hy": field_hy,
                "handoff_plane": handoff_plane,
                "output_directory": directory,
            },
        )
    )
    if run.status.value != "succeeded":
        raise RuntimeError(f"{spec.name} trace failed: {run.error_type}: {run.error_message}")
    record = run.outputs["rays"]
    return Trace(record=record, arrays=dict(np.load(record.uri)))


# ---------------------------------------------------------------------------
# The paraxial limit, from a ladder
# ---------------------------------------------------------------------------


def _richardson(coarse: float, fine: float, *, order: int = 2) -> float:
    """Extrapolate a second-order-convergent sequence to its limit.

    The ladder halves the innermost ray height at each rung, so for an error
    ``C h^order`` the limit is ``(2^order * fine - coarse) / (2^order - 1)``.
    Stated as a formula rather than a constant because the order is a physical
    claim -- spherical aberration is quadratic in aperture -- and a reader should
    be able to see which claim is being used.
    """
    factor = 2.0**order
    return (factor * fine - coarse) / (factor - 1.0)


@dataclass(frozen=True)
class ParaxialLimit:
    """A quantity extrapolated to zero ray height, and how well it is known."""

    value: float
    #: Difference between the two finest extrapolations. The honest error bar:
    #: it is what changing the ladder by one rung does to the answer.
    uncertainty: float
    ladder: tuple[tuple[float, float], ...]
    rungs: tuple[int, ...]

    @property
    def relative_errors(self) -> tuple[float, ...]:
        return tuple(abs(value - self.value) / abs(self.value) for _, value in self.ladder)


def _paraxial_limit(
    rungs: Sequence[int], samples: Sequence[tuple[float, float]]
) -> ParaxialLimit:
    """Richardson-extrapolate ``(height, value)`` rungs to zero height."""
    if len(samples) < 3:
        raise ValueError("the paraxial limit needs at least three rungs to bound itself")
    values = [value for _, value in samples]
    finest = _richardson(values[-2], values[-1])
    previous = _richardson(values[-3], values[-2])
    return ParaxialLimit(
        value=finest,
        uncertainty=abs(finest - previous),
        ladder=tuple(samples),
        rungs=tuple(rungs),
    )


# ---------------------------------------------------------------------------
# Prescriptions
# ---------------------------------------------------------------------------


def _plano_convex(
    *, radius_mm: float, index: float, thickness_mm: float, epd_mm: float, back_mm: float
) -> OpticalSystemSpec:
    """A plano-convex singlet, convex toward the collimated side.

    The rear surface is plano and carries AIR, which is what makes both closed
    forms exact and keeps the image plane from adding a second refraction.
    """
    return OpticalSystemSpec(
        name="B1RayEFL",
        description=(
            "B1-RAY-EFL: plano-convex singlet in air. The plano rear surface has no "
            "power, so R/(n-1) is the whole effective focal length and EFL - t/n is "
            "the exact back focal length from the rear vertex."
        ),
        object_distance_mm=None,
        surfaces=(
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=radius_mm),
                thickness_mm=thickness_mm,
                material=IdealMaterialSpec(refractive_index=index),
                is_stop=True,
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=back_mm),
        ),
        aperture=ApertureSpec(value_mm=epd_mm),
        fields=(FieldSpec(y_deg=0.0),),
        wavelengths=(WavelengthSpec(value_um=WAVELENGTH_UM, is_primary=True),),
    )


def _plate_system(
    *,
    focal_length_mm: float,
    lens_index: float,
    plate_thickness_mm: float,
    plate_index: float,
    epd_mm: float,
    with_plate: bool,
) -> OpticalSystemSpec:
    """A converging beam, with and without a plane-parallel plate in it.

    The two systems are identical except for the plate, and the axial spacing
    after it is adjusted so the total geometrical length is the same -- so the
    difference of the two measured crossings is the plate's effect and nothing
    else. Every common-mode property of the trace cancels.
    """
    gap_before = 20.0
    gap_after = 80.0
    surfaces = [
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=focal_length_mm * (lens_index - 1.0)),
            thickness_mm=1.0,
            material=IdealMaterialSpec(refractive_index=lens_index),
            is_stop=True,
        ),
        SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=gap_before),
    ]
    if with_plate:
        surfaces += [
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=plate_thickness_mm,
                material=IdealMaterialSpec(refractive_index=plate_index),
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=gap_after),
        ]
    else:
        surfaces += [
            SurfaceSpec(
                geometry=PlaneGeometrySpec(),
                thickness_mm=plate_thickness_mm + gap_after,
            )
        ]
    return OpticalSystemSpec(
        name="B1RayPlate" + ("WithPlate" if with_plate else "NoPlate"),
        description=(
            "B1-RAY-PLATE: a plane-parallel plate in a converging beam. The paired "
            "system without the plate is the control arm, so the measured shift is a "
            "difference and every common-mode trace property cancels."
        ),
        object_distance_mm=None,
        surfaces=tuple(surfaces),
        aperture=ApertureSpec(value_mm=epd_mm),
        fields=(FieldSpec(y_deg=0.0),),
        wavelengths=(WavelengthSpec(value_um=WAVELENGTH_UM, is_primary=True),),
    )


def _snell_system(*, radius_mm: float, index: float, epd_mm: float) -> OpticalSystemSpec:
    """One spherical refracting surface, then a plane exit back into air."""
    return OpticalSystemSpec(
        name="B1RaySnell",
        description=(
            "B1-RAY-SNELL: one spherical air-to-glass interface, then a plane exit. "
            "sin(i) = rho/R exactly at the sphere because a collimated on-axis ray "
            "does not move transversely before it, and the exit face is exact too, so "
            "the oracle is Snell applied twice with no approximation anywhere."
        ),
        object_distance_mm=None,
        surfaces=(
            SurfaceSpec(
                geometry=SphericalGeometrySpec(radius_mm=radius_mm),
                thickness_mm=5.0,
                material=IdealMaterialSpec(refractive_index=index),
                is_stop=True,
            ),
            SurfaceSpec(geometry=PlaneGeometrySpec(), thickness_mm=60.0),
        ),
        aperture=ApertureSpec(value_mm=epd_mm),
        fields=(FieldSpec(y_deg=0.0),),
        wavelengths=(WavelengthSpec(value_um=WAVELENGTH_UM, is_primary=True),),
    )


def _lagrange_system(*, surface_count: int, field_deg: float) -> OpticalSystemSpec:
    """A three-surface stack that returns the ray to air at its last surface."""
    surfaces = (
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=60.0),
            thickness_mm=5.0,
            material=IdealMaterialSpec(refractive_index=1.5168),
            is_stop=True,
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=-45.0),
            thickness_mm=3.0,
            material=IdealMaterialSpec(refractive_index=1.62),
        ),
        SurfaceSpec(geometry=SphericalGeometrySpec(radius_mm=-160.0), thickness_mm=95.0),
    )[:surface_count]
    return OpticalSystemSpec(
        name=f"B1RayLagrange{surface_count}",
        description=(
            "B1-RAY-LAGRANGE: a multi-surface stack whose last surface returns the ray "
            "to air, so the image plane adds no index step of its own."
        ),
        object_distance_mm=None,
        surfaces=surfaces,
        aperture=ApertureSpec(value_mm=8.0),
        fields=(FieldSpec(y_deg=field_deg),),
        wavelengths=(WavelengthSpec(value_um=WAVELENGTH_UM, is_primary=True),),
    )


# ---------------------------------------------------------------------------
# B1-RAY-EFL
# ---------------------------------------------------------------------------


def _efl_ladder(instance: Any) -> dict[str, Any]:
    """Traced EFL and BFL at each rung, plus their paraxial limits."""
    radius = float(instance.parameters["radius_mm"])
    index = float(instance.parameters["index"])
    thickness = float(instance.parameters["thickness_mm"])
    efl_closed = radius / (index - 1.0)
    bfl_closed = efl_closed - thickness / index

    spec = _plano_convex(
        radius_mm=radius,
        index=index,
        thickness_mm=thickness,
        epd_mm=10.0,
        back_mm=bfl_closed,
    )

    efl_samples: list[tuple[float, float]] = []
    bfl_samples: list[tuple[float, float]] = []
    placement: dict[str, str] = {}
    for rings in RINGS_LADDER:
        trace = _trace(spec, rings=rings)
        placement = trace.placement
        L, M, N = trace.direction
        heights = trace.launch_radius_mm
        keep = heights > 0.0
        # EFL from the ray's own entrance height and exit convergence angle. The
        # entrance height is exported, so no pupil model is involved.
        focal = heights[keep] / (np.hypot(L[keep], M[keep]) / N[keep])
        h_efl, value = trace.innermost(focal, heights[keep])
        efl_samples.append((h_efl, value))

        crossing_heights, crossings = trace.axial_crossing_mm()
        h_bfl, crossing = trace.innermost(crossings, crossing_heights)
        # From the REAR VERTEX, which is the front vertex plus the centre
        # thickness. The closed form is quoted from the rear vertex and quoting
        # it from anywhere else would be a different number.
        bfl_samples.append((h_bfl, crossing - thickness))

    return {
        "efl_closed_form_mm": efl_closed,
        "bfl_closed_form_mm": bfl_closed,
        "efl": _paraxial_limit(RINGS_LADDER, efl_samples),
        "bfl": _paraxial_limit(RINGS_LADDER, bfl_samples),
        "placement": placement,
    }


def _run_efl() -> InstanceRun:
    instance = _instance(B1_RAY_EFL, "B1-RAY-EFL-01")
    ladder = _efl_ladder(instance)
    efl, bfl = ladder["efl"], ladder["bfl"]
    efl_closed = ladder["efl_closed_form_mm"]
    bfl_closed = ladder["bfl_closed_form_mm"]

    efl_error = abs(efl.value - efl_closed) / efl_closed
    bfl_error = abs(bfl.value - bfl_closed) / bfl_closed

    # The declared control: report the EFL where the BFL belongs, i.e. omit the
    # thick-lens correction. Run through the same measurement path, with the same
    # traced numbers -- the mutation is which quantity is reported.
    thin_lens_error = abs(efl.value - bfl_closed) / bfl_closed

    # The ladder as convergence evidence. The rung is the ring count and the
    # error is the innermost ring's relative shortfall, so the expected exponent
    # is -2: aberration is quadratic in aperture and the aperture halves.
    convergence = fit_convergence(
        "pupil_rings",
        [
            (float(rings), abs(value - efl_closed) / efl_closed)
            for rings, (_, value) in zip(RINGS_LADDER, efl.ladder, strict=True)
        ],
        expected_exponent=-2.0,
        exponent_tolerance=0.2,
        note=(
            "the innermost ring's relative EFL error against R/(n-1), over four ring "
            "counts. The exponent is a prediction rather than a fit target: spherical "
            "aberration is quadratic in the aperture and the innermost ring's height "
            "halves with each doubling of the ring count, so -2 is what the physics "
            "requires. The gated value is the h -> 0 Richardson limit of the same "
            "ladder, not its finest rung."
        ),
    )

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="thick_singlet",
        refusal=None,
        observed_parameters={"pupil_rings": RINGS_LADDER[-1]},
        diagnostics=[
            {
                "code": "PARAXIAL_LIMIT_LADDER",
                "detail": (
                    "EFL rungs (height_mm, value_mm): "
                    + "; ".join(f"({h:.6f}, {v:.9f})" for h, v in efl.ladder)
                    + f" -> limit {efl.value:.10f} +/- {efl.uncertainty:.2e}"
                ),
                "location": "benchmarks/instances/b1_ray.py::_efl_ladder",
            },
            {
                "code": "PARAXIAL_LIMIT_LADDER_BFL",
                "detail": (
                    "BFL rungs (height_mm, value_mm): "
                    + "; ".join(f"({h:.6f}, {v:.9f})" for h, v in bfl.ladder)
                    + f" -> limit {bfl.value:.10f} +/- {bfl.uncertainty:.2e}"
                ),
                "location": "benchmarks/instances/b1_ray.py::_efl_ladder",
            },
            {
                "code": "OBSERVED_PLACEMENT",
                "detail": str(ladder["placement"]),
                "location": "runtime/instance_runner.py::observed_placement",
            },
        ],
    )
    measurements = {
        "efl_relative_error": Measurement(
            value=efl_error,
            uncertainty=efl.uncertainty / efl_closed,
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"traced {efl.value:.10f} mm against R/(n-1) = {efl_closed:.10f} mm, in "
                "the h -> 0 limit of a four-rung ring ladder. The error bar is what "
                "dropping the finest rung does to the extrapolation."
            ),
        ),
        "bfl_relative_error": Measurement(
            value=bfl_error,
            uncertainty=bfl.uncertainty / bfl_closed,
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"traced {bfl.value:.10f} mm from the rear vertex against EFL - t/n = "
                f"{bfl_closed:.10f} mm. Graded separately from the EFL on purpose: the "
                "two differ by 2.64 mm here, so reporting the same number twice fails "
                "this check and only this check."
            ),
        ),
    }
    controls = {
        "thin-lens-bfl": control_result(
            "thin-lens-bfl",
            "bfl_relative_error",
            baseline=measurements["bfl_relative_error"],
            mutated=Measurement(
                value=thin_lens_error,
                uncertainty=efl.uncertainty / bfl_closed,
                uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
                note="the traced EFL reported as the BFL: the thick-lens correction omitted",
            ),
            threshold=1e-6,
            note="omitting t/n on this prescription is a 2.64 mm error.",
        )
    }
    return InstanceRun(
        family=B1_RAY_EFL,
        instance=instance,
        record=record,
        result=verify(
            B1_RAY_EFL,
            instance,
            record,
            measurements=measurements,
            negative_controls=controls,
            convergence=convergence,
        ),
    )


# ---------------------------------------------------------------------------
# B1-RAY-PLATE
# ---------------------------------------------------------------------------


def _plate_ladder(instance: Any) -> dict[str, Any]:
    thickness = float(instance.parameters["thickness_mm"])
    index = float(instance.parameters["index"])
    focal = float(instance.parameters["focal_length_mm"])
    shift_closed = thickness * (1.0 - 1.0 / index)

    samples: list[tuple[float, float]] = []
    placement: dict[str, str] = {}
    for rings in RINGS_LADDER:
        crossings = {}
        for with_plate in (False, True):
            spec = _plate_system(
                focal_length_mm=focal,
                lens_index=1.5,
                plate_thickness_mm=thickness,
                plate_index=index,
                epd_mm=10.0,
                with_plate=with_plate,
            )
            trace = _trace(spec, rings=rings)
            placement = trace.placement
            heights, values = trace.axial_crossing_mm()
            crossings[with_plate] = trace.innermost(values, heights)
        height = crossings[True][0]
        samples.append((height, crossings[True][1] - crossings[False][1]))

    return {
        "shift_closed_form_mm": shift_closed,
        "shift": _paraxial_limit(RINGS_LADDER, samples),
        "placement": placement,
    }


def _run_plate() -> InstanceRun:
    instance = _instance(B1_RAY_PLATE, "B1-RAY-PLATE-01")
    ladder = _plate_ladder(instance)
    shift = ladder["shift"]
    closed = ladder["shift_closed_form_mm"]

    # SIGNED, which is the point: the focus moves AWAY from the plate.
    signed_error = (shift.value - closed) / closed
    thickness = float(instance.parameters["thickness_mm"])
    index = float(instance.parameters["index"])

    convergence = fit_convergence(
        "axis_crossing_samples",
        [
            (float(rings), abs(value - closed) / closed)
            for rings, (_, value) in zip(RINGS_LADDER, shift.ladder, strict=True)
        ],
        expected_exponent=-2.0,
        exponent_tolerance=0.2,
        note=(
            "the plate shift's relative error over four ring counts. Same quadratic "
            "aperture dependence as the EFL ladder, and for the same reason."
        ),
    )
    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="plane_parallel_plate",
        refusal=None,
        observed_parameters={"axis_crossing_samples": RINGS_LADDER[-1]},
        diagnostics=[
            {
                "code": "PARAXIAL_LIMIT_LADDER",
                "detail": (
                    "shift rungs (height_mm, shift_mm): "
                    + "; ".join(f"({h:.6f}, {v:.9f})" for h, v in shift.ladder)
                    + f" -> limit {shift.value:.10f} +/- {shift.uncertainty:.2e}"
                ),
                "location": "benchmarks/instances/b1_ray.py::_plate_ladder",
            },
            {
                "code": "PAIRED_CONTROL_ARM",
                "detail": (
                    "each rung is a DIFFERENCE of two traces of the same system with "
                    "and without the plate, so the lens, the sampling and the "
                    "axial-crossing extraction cancel"
                ),
                "location": "benchmarks/instances/b1_ray.py::_plate_system",
            },
        ],
    )
    measurements = {
        "plate_focal_shift_signed_relative_error": Measurement(
            value=abs(signed_error),
            uncertainty=shift.uncertainty / closed,
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"traced {shift.value:+.9f} mm against t(1 - 1/n) = {closed:+.9f} mm. "
                f"The measured shift is POSITIVE, i.e. away from the plate, which is "
                "the half of the claim a magnitude-only comparison would miss."
            ),
        )
    }
    controls = {
        "sign-flip": control_result(
            "sign-flip",
            "plate_focal_shift_signed_relative_error",
            baseline=measurements["plate_focal_shift_signed_relative_error"],
            mutated=Measurement(
                value=abs((-shift.value - closed) / closed),
                uncertainty=shift.uncertainty / closed,
                uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
                note="the shift reported toward the plate instead of away from it",
            ),
            threshold=1e-3,
            note="a sign error is a 2x relative error and is rejected.",
        ),
        "t-over-n": control_result(
            "t-over-n",
            "plate_focal_shift_signed_relative_error",
            baseline=measurements["plate_focal_shift_signed_relative_error"],
            mutated=Measurement(
                value=abs((thickness / index - closed) / closed),
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note="t/n reported in place of t(1 - 1/n): the classic transposition",
            ),
            threshold=1e-3,
            note="t/n is 6.25 mm where the answer is 3.75 mm.",
        ),
    }
    return InstanceRun(
        family=B1_RAY_PLATE,
        instance=instance,
        record=record,
        result=verify(
            B1_RAY_PLATE,
            instance,
            record,
            measurements=measurements,
            negative_controls=controls,
            convergence=convergence,
        ),
    )


# ---------------------------------------------------------------------------
# B1-RAY-SNELL
# ---------------------------------------------------------------------------

#: The sphere the Snell instances are measured on. Its radius sets the mapping
#: from a launch height to an incidence angle, so it is a property of the
#: measurement rather than of any one instance.
SNELL_RADIUS_MM = 40.0
SNELL_EPD_MM = 36.0


def _snell_measurement(instance: Any) -> dict[str, Any]:
    index = float(instance.parameters["index_transmitted"])
    target = float(instance.parameters["incidence_angle_rad"])

    trace = _trace(
        _snell_system(radius_mm=SNELL_RADIUS_MM, index=index, epd_mm=SNELL_EPD_MM),
        rings=6,
    )
    heights = trace.launch_radius_mm
    L, M, N = trace.direction

    sin_incidence = np.clip(heights / SNELL_RADIUS_MM, -1.0, 1.0)
    incidence = np.arcsin(sin_incidence)
    # Pick the traced ray closest to the instance's declared angle, and report
    # what was realized. The declared angle is a request; the ring set decides
    # which angles the trace actually contains.
    row = int(np.argmin(np.abs(incidence - target)))

    refracted = np.arcsin(np.clip(sin_incidence / index, -1.0, 1.0))
    # Angle to the axis inside the glass: the normal makes angle `incidence`
    # with z, and the refracted ray makes `refracted` with the normal.
    inside = incidence - refracted
    # The plane exit face, also exact. TIR there is the family's validity bound.
    sin_exit = index * np.sin(inside)
    exit_angle = np.arcsin(np.clip(sin_exit, -1.0, 1.0))

    measured = np.arctan2(np.hypot(L, M), N)
    error = np.abs(measured - exit_angle)
    norm_error = float(np.max(np.abs(np.sqrt(L**2 + M**2 + N**2) - 1.0)))

    return {
        "realized_incidence_rad": float(incidence[row]),
        "refracted_rad": float(refracted[row]),
        "exit_angle_rad": float(exit_angle[row]),
        "measured_rad": float(measured[row]),
        "absolute_error_rad": float(error[row]),
        "worst_error_rad": float(np.max(error)),
        "direction_norm_error": norm_error,
        "tir_margin": float(1.0 - np.max(sin_exit)),
        "angles": tuple(float(v) for v in np.unique(np.round(incidence, 9))),
        "placement": trace.placement,
    }


def _run_snell(instance_id: str) -> InstanceRun:
    instance = _instance(B1_RAY_SNELL, instance_id)
    m = _snell_measurement(instance)

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="spherical_interface",
        refusal=None,
        observed_parameters={"incidence_angle_rad": m["realized_incidence_rad"]},
        diagnostics=[
            {
                "code": "EXACT_ON_BOTH_SIDES",
                "detail": (
                    f"sin(i) = rho/R gives i = {m['realized_incidence_rad']:.9f} rad; "
                    f"Snell at the sphere gives i' = {m['refracted_rad']:.9f}; the plane "
                    f"exit face gives {m['exit_angle_rad']:.12f}; the trace reports "
                    f"{m['measured_rad']:.12f}. No approximation and no fitted constant "
                    "on either side."
                ),
                "location": "benchmarks/instances/b1_ray.py::_snell_measurement",
            },
            {
                "code": "WORST_OVER_EVERY_TRACED_ANGLE",
                "detail": (
                    f"{m['worst_error_rad']:.3e} rad over the whole fan, at incidence "
                    f"angles {[format(a, '.6f') for a in m['angles']]}"
                ),
                "location": "benchmarks/instances/b1_ray.py::_snell_measurement",
            },
            {
                "code": "TIR_MARGIN_AT_THE_EXIT_FACE",
                "detail": (
                    f"1 - max(n sin(u_inside)) = {m['tir_margin']:.6f}; the exit face "
                    "is where total internal reflection would occur and the family's "
                    "TIR predicate is about it"
                ),
                "location": "benchmarks/instances/b1_ray.py::_snell_measurement",
            },
            {
                "code": "OBSERVED_PLACEMENT",
                "detail": str(m["placement"]),
                "location": "runtime/instance_runner.py::observed_placement",
            },
        ],
    )
    measurements = {
        "refraction_angle_absolute_error_rad": Measurement(
            value=m["absolute_error_rad"],
            uncertainty=float(np.finfo(np.float64).eps),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                "absolute angular error against exact Snell. The error bar is one "
                "float64 epsilon because there is nothing else in the comparison: no "
                "sampling, no approximation, no fitted constant."
            ),
        )
    }
    invariants = {
        "DIRECTION_COSINES_UNIT_NORM": Measurement(
            value=m["direction_norm_error"],
            uncertainty=float(np.finfo(np.float64).eps),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note="max ||d|| - 1 over the traced fan",
        )
    }
    incidence = m["realized_incidence_rad"]
    index = float(instance.parameters["index_transmitted"])
    controls = {
        "small-angle-substitution": control_result(
            "small-angle-substitution",
            "refraction_angle_absolute_error_rad",
            baseline=measurements["refraction_angle_absolute_error_rad"],
            mutated=Measurement(
                value=abs(
                    m["measured_rad"]
                    - _exit_angle_for(incidence / index, incidence, index)
                ),
                uncertainty=float(np.finfo(np.float64).eps),
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note="i' = i/n in place of sin(i') = sin(i)/n",
            ),
            threshold=1e-12,
            note="the paraxial substitution, which is exact only at i = 0.",
        ),
        "inverted-index-ratio": control_result(
            "inverted-index-ratio",
            "refraction_angle_absolute_error_rad",
            baseline=measurements["refraction_angle_absolute_error_rad"],
            mutated=Measurement(
                value=abs(
                    m["measured_rad"]
                    - _exit_angle_for(
                        math.asin(min(1.0, math.sin(incidence) * index)),
                        incidence,
                        index,
                    )
                )
                if math.sin(incidence) * index <= 1.0
                else float("inf"),
                uncertainty=float(np.finfo(np.float64).eps),
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note="sin(i') = n sin(i), the ratio the wrong way up",
            ),
            threshold=1e-12,
            note="inverting the ratio bends the ray the wrong way.",
        ),
    }
    return InstanceRun(
        family=B1_RAY_SNELL,
        instance=instance,
        record=record,
        result=verify(
            B1_RAY_SNELL,
            instance,
            record,
            measurements=measurements,
            invariants=invariants,
            negative_controls=controls,
        ),
    )


def _exit_angle_for(refracted_rad: float, incidence_rad: float, index: float) -> float:
    """The exit-face angle a given (possibly wrong) refraction angle would produce."""
    inside = incidence_rad - refracted_rad
    return math.asin(min(1.0, max(-1.0, index * math.sin(inside))))


# ---------------------------------------------------------------------------
# B1-RAY-LAGRANGE
# ---------------------------------------------------------------------------

#: The field ladder the Lagrange drift is measured over. Degrees, halving.
LAGRANGE_FIELDS_DEG: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5, 0.25)
LAGRANGE_RINGS = 64


def _lagrange_drift(field_deg: float, *, surface_count: int) -> dict[str, float]:
    """Paraxial Lagrange invariant at the launch plane and at the image plane."""
    spec = _lagrange_system(surface_count=surface_count, field_deg=field_deg)
    marginal = _trace(spec, rings=LAGRANGE_RINGS, field_hy=0.0)
    chief = _trace(spec, rings=LAGRANGE_RINGS, field_hy=1.0)

    # The marginal ray: the innermost nonzero meridional ray, so the measurement
    # is as close to paraxial as the fan allows.
    on_meridian = (np.abs(marginal.arrays["pupil_normalized_x"]) < 1e-12) & (
        marginal.arrays["pupil_normalized_y"] > 0
    )
    radius = np.hypot(
        marginal.arrays["pupil_normalized_x"], marginal.arrays["pupil_normalized_y"]
    )
    i = int(np.argmin(np.where(on_meridian, radius, np.inf)))
    # The chief ray: the pupil-centre ray of the off-axis fan.
    j = int(
        np.argmin(np.hypot(chief.arrays["pupil_normalized_x"], chief.arrays["pupil_normalized_y"]))
    )

    theta = math.radians(field_deg)
    height = float(marginal.arrays["launch_y_m"][i])
    invariant_object = height * math.tan(theta)

    slope_marginal = float(marginal.arrays["M"][i] / marginal.arrays["N"][i])
    slope_chief = float(chief.arrays["M"][j] / chief.arrays["N"][j])
    invariant_image = float(
        marginal.arrays["y_m"][i] * slope_chief - chief.arrays["y_m"][j] * slope_marginal
    )
    drift = abs(abs(invariant_image) - abs(invariant_object)) / abs(invariant_object)
    return {
        "field_rad": theta,
        "marginal_height_m": height,
        "invariant_object": invariant_object,
        "invariant_image": invariant_image,
        "relative_drift": drift,
    }


def _run_lagrange() -> InstanceRun:
    instance = _instance(B1_RAY_LAGRANGE, "B1-RAY-LAGRANGE-01")
    surface_count = int(instance.parameters["surface_count"])
    rows = [_lagrange_drift(deg, surface_count=surface_count) for deg in LAGRANGE_FIELDS_DEG]
    finest = rows[-1]

    convergence = fit_convergence(
        "chief_ray_angle_rad",
        [(row["field_rad"], row["relative_drift"]) for row in rows],
        note=(
            "the drift against the chief-ray field angle, over five halvings. No "
            "expected exponent is declared because the measured one is not a clean "
            "integer -- it comes out near 2.5 -- and asserting 2 would be asserting "
            "the wrong model. That the drift VANISHES with the field is the "
            "conservation statement; the exponent is a characterization of how."
        ),
    )

    # The declared control: drop the index at refraction. Implemented as the
    # invariant evaluated with n = 1 in image space where the last surface leaves
    # the ray -- which on this system is air, so the mutation is applied at the
    # INTERNAL surface instead, where it changes the traced angles. Since the
    # trace cannot be mutated from here without editing the adapter, the control
    # is evaluated by re-tracing a system whose second medium is air: the index
    # is omitted from the glass, not from the arithmetic.
    unindexed = _lagrange_drift_without_index(surface_count=surface_count)

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="lagrange_invariant",
        refusal=None,
        observed_parameters={
            "chief_ray_angle_rad": finest["field_rad"],
            "marginal_ray_height_mm": finest["marginal_height_m"] * MM_PER_M,
            "surface_count": surface_count,
        },
        diagnostics=[
            {
                "code": "FIELD_LADDER",
                "detail": "; ".join(
                    f"theta={row['field_rad']:.6e} drift={row['relative_drift']:.4e}"
                    for row in rows
                ),
                "location": "benchmarks/instances/b1_ray.py::_lagrange_drift",
            },
            {
                "code": "WHY_THE_GATE_CANNOT_CLOSE",
                "detail": (
                    "the Lagrange invariant's two-ray bilinear form p_a.q_b - p_b.q_a "
                    "is preserved by a LINEAR symplectic map. Ray refraction at a "
                    "curved surface is symplectic and not linear, so only the "
                    "differential form is exactly conserved and any finite-real-ray "
                    "evaluation carries an aberration residual. Measured directly: the "
                    "differential ratio between two rays of one fan converges to "
                    "1 + 7.1e-3 at a 5-degree field and does NOT approach 1 as the "
                    "separation shrinks, which is the signature of a finite-form "
                    "residual rather than of a numerical one. The 1e-10 tolerance's "
                    "basis is a conservation law that holds paraxially; the "
                    "measurement is of real rays. The tolerance is left where it is."
                ),
                "location": "src/verification/families/b1_ray.py::B1_RAY_LAGRANGE",
            },
        ],
    )
    measurements = {
        "lagrange_invariant_relative_drift": Measurement(
            value=finest["relative_drift"],
            uncertainty=abs(finest["relative_drift"] - rows[-2]["relative_drift"]),
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"at the finest rung of the field ladder, theta = "
                f"{finest['field_rad']:.3e} rad. The error bar is the change from the "
                "previous rung, which is the honest statement of how converged this is."
            ),
        )
    }
    controls = {
        "omit-index-at-refraction": control_result(
            "omit-index-at-refraction",
            "lagrange_invariant_relative_drift",
            baseline=measurements["lagrange_invariant_relative_drift"],
            mutated=Measurement(
                value=unindexed["relative_drift"],
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note=(
                    "the same stack with the second element's index removed -- the "
                    "index omitted from the GLASS rather than from the arithmetic, so "
                    "the mutation goes through the shipping trace"
                ),
            ),
            threshold=1e-10,
            note="removing an index changes the system, and the invariant follows it.",
        )
    }
    return InstanceRun(
        family=B1_RAY_LAGRANGE,
        instance=instance,
        record=record,
        result=verify(
            B1_RAY_LAGRANGE,
            instance,
            record,
            measurements=measurements,
            negative_controls=controls,
            convergence=convergence,
        ),
    )


def _lagrange_drift_without_index(*, surface_count: int) -> dict[str, float]:
    """The control arm: the second element's index removed from the prescription."""
    field_deg = LAGRANGE_FIELDS_DEG[-1]
    surfaces = (
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=60.0),
            thickness_mm=5.0,
            material=IdealMaterialSpec(refractive_index=1.5168),
            is_stop=True,
        ),
        # The mutation: index 1.0 where the stack has 1.62.
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=-45.0),
            thickness_mm=3.0,
            material=IdealMaterialSpec(refractive_index=1.0),
        ),
        SurfaceSpec(geometry=SphericalGeometrySpec(radius_mm=-160.0), thickness_mm=95.0),
    )[:surface_count]
    spec = OpticalSystemSpec(
        name="B1RayLagrangeNoIndex",
        description="B1-RAY-LAGRANGE control: the second element's index removed.",
        object_distance_mm=None,
        surfaces=surfaces,
        aperture=ApertureSpec(value_mm=8.0),
        fields=(FieldSpec(y_deg=field_deg),),
        wavelengths=(WavelengthSpec(value_um=WAVELENGTH_UM, is_primary=True),),
    )
    marginal = _trace(spec, rings=LAGRANGE_RINGS, field_hy=0.0)
    chief = _trace(spec, rings=LAGRANGE_RINGS, field_hy=1.0)
    on_meridian = (np.abs(marginal.arrays["pupil_normalized_x"]) < 1e-12) & (
        marginal.arrays["pupil_normalized_y"] > 0
    )
    radius = np.hypot(
        marginal.arrays["pupil_normalized_x"], marginal.arrays["pupil_normalized_y"]
    )
    i = int(np.argmin(np.where(on_meridian, radius, np.inf)))
    j = int(
        np.argmin(np.hypot(chief.arrays["pupil_normalized_x"], chief.arrays["pupil_normalized_y"]))
    )
    theta = math.radians(field_deg)
    invariant_object = float(marginal.arrays["launch_y_m"][i]) * math.tan(theta)
    invariant_image = float(
        marginal.arrays["y_m"][i] * (chief.arrays["M"][j] / chief.arrays["N"][j])
        - chief.arrays["y_m"][j] * (marginal.arrays["M"][i] / marginal.arrays["N"][i])
    )
    return {
        "relative_drift": abs(abs(invariant_image) - abs(invariant_object))
        / abs(invariant_object)
    }


# ---------------------------------------------------------------------------
# B1-RAY-OFFAXIS-OPL
# ---------------------------------------------------------------------------


def _launch_tilt(record: ArtifactRecord, *, term_applied: bool) -> dict[str, float]:
    """The declared pupil OPL's linear slope in the launch coordinate.

    Regressed against the LAUNCH coordinates, not the exit pupil, because the
    omitted term is ``n_object * (d0 . r_launch)`` -- a function of where the ray
    started. Mixing the two reads 0.6556 on this system and looks like a 34%
    shortfall in a term that is exact.

    The gated quantity is the **slope**, not a peak-to-valley span, and that is
    deliberate. ``d(OPL)/dy = n_object * sin(theta)`` is the whole content of "the
    optical path is measured from a wavefront perpendicular to the chief
    direction", and the chief direction is READ off the record rather than
    assumed. A span comparison would need a pupil extent on both sides and would
    reduce to comparing the term's span with itself.

    The extent is measured as the peak-to-valley along the tilt axis, which is
    the entrance pupil diameter. ``2 * max|r|`` is NOT that: an off-axis
    collimated fan launches OFFSET transversely -- mean y is -0.173 mm here --
    so the radial extreme is inflated by the field and reads 0.647 mm for a
    0.300 mm pupil.
    """
    conventions = record.metadata["conventions"]
    handoff = declare_coherent_bundle(
        record,
        declared_plane=DeclaredHandoffPlane(
            handoff_plane="exit_pupil", z_m=float(conventions["reference_plane_z_m"])
        ),
        perturbation=HandoffPerturbation(reference_incoming_wavefront=term_applied),
    )
    bundle = handoff.bundle
    arrays = dict(np.load(record.uri))
    launch_x = np.asarray(arrays["launch_x_m"], dtype=np.float64)
    launch_y = np.asarray(arrays["launch_y_m"], dtype=np.float64)
    opl = np.asarray(bundle.optical_path_length_m, dtype=np.float64)

    design = np.stack([np.ones_like(launch_y), launch_x, launch_y], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, opl, rcond=None)
    slope = float(coefficients[2])
    extent_m = float(np.ptp(launch_y))

    direction = np.asarray(
        conventions["object_space_reference"]["launch_direction"], dtype=np.float64
    )
    return {
        "slope": slope,
        "chief_direction_y": float(direction[1]),
        "launch_extent_m": extent_m,
        "radial_extreme_m": 2.0 * float(np.max(np.hypot(launch_x, launch_y))),
        "tilt_peak_to_valley_waves": abs(slope) * extent_m / bundle.wavelength_m,
        "term_span_waves": float(
            handoff.diagnostics["object_space_reference_span_waves"] or 0.0
        ),
        "term_status": str(handoff.diagnostics["object_space_reference_status"]),
    }


def _run_offaxis_opl() -> InstanceRun:
    instance = _instance(B1_RAY_OFFAXIS_OPL, "B1-RAY-OFFAXIS-OPL-01")
    field_rad = float(instance.parameters["field_angle_rad"])
    wavelength_m = float(instance.parameters["wavelength_m"])
    index = float(instance.parameters["index_object_space"])
    rings = int(instance.parameters["pupil_rings"])

    # M3-REVERSE-TELEPHOTO's field is declared in normalized Hy; 0.2 is the
    # configuration CHE-41 found the defect on, and it realizes 6 degrees.
    trace = _trace_sample("ReverseTelephoto", field_hy=0.2, rings=rings)
    on_axis = _trace_sample("ReverseTelephoto", field_hy=0.0, rings=rings)

    with_term = _launch_tilt(trace.record, term_applied=True)
    without_term = _launch_tilt(trace.record, term_applied=False)
    on_axis_with = _launch_tilt(on_axis.record, term_applied=True)
    on_axis_without = _launch_tilt(on_axis.record, term_applied=False)

    # The requirement: d(OPL)/dy = n_object * sin(theta), with sin(theta) read off
    # the record's chief direction rather than recomputed from the declared field.
    # Both are reported, so a disagreement between them is a diagnostic.
    required_slope = index * with_term["chief_direction_y"]
    recovered = with_term["slope"] / required_slope
    recovered_without = without_term["slope"] / required_slope

    declared_diameter = float(instance.parameters["pupil_diameter_m"])
    realized_diameter = with_term["launch_extent_m"]
    required_waves = index * realized_diameter * math.sin(field_rad) / wavelength_m

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="off_axis_pupil_opl",
        refusal=None,
        observed_parameters={
            "field_angle_rad": float(math.asin(with_term["chief_direction_y"])),
            "pupil_diameter_m": realized_diameter,
            "pupil_rings": rings,
        },
        diagnostics=[
            {
                "code": "THE_SLOPE_IS_THE_CLAIM",
                "detail": (
                    f"d(OPL)/dy = {with_term['slope']:.12f} against the required "
                    f"n_object*sin(theta) = {required_slope:.12f}, with sin(theta) = "
                    f"{with_term['chief_direction_y']:.9f} READ off the record's chief "
                    f"direction. Declared field {field_rad:.9f} rad gives "
                    f"sin = {math.sin(field_rad):.9f}; the two agree, which is the "
                    "check that the instance and the trace describe the same field."
                ),
                "location": "couplers/handoff.py::_object_space_reference",
            },
            {
                "code": "DECLARED_VERSUS_REALIZED_PUPIL",
                "detail": (
                    f"the instance declares pupil_diameter_m = {declared_diameter:.9e} "
                    f"and the trace realized {realized_diameter:.9e} as the "
                    "peak-to-valley of its LAUNCH y. The radial extreme is "
                    f"{with_term['radial_extreme_m']:.9e}, inflated by the field "
                    "offset (mean launch y is -0.173 mm here), and using it as a "
                    "diameter reads 0.6556 -- a 34% apparent shortfall in a term that "
                    "is exact. That is the trap this measurement fell into first."
                ),
                "location": "benchmarks/instances/b1_ray.py::_launch_tilt",
            },
            {
                "code": "WITH_AND_WITHOUT_THE_TERM",
                "detail": (
                    "with n_object*(d0.r_launch): "
                    f"{with_term['tilt_peak_to_valley_waves']:.6f} waves peak-to-valley; "
                    f"without it: {without_term['tilt_peak_to_valley_waves']:.6f}; "
                    f"required {required_waves:.6f}. Slope fractions {recovered:.6f} "
                    f"and {recovered_without:.6f}."
                ),
                "location": "couplers/handoff.py::_object_space_reference",
            },
            {
                "code": "ON_AXIS_IS_BLIND_TO_IT",
                "detail": (
                    "the SAME omission on the on-axis trace: "
                    f"{on_axis_with['tilt_peak_to_valley_waves']:.9f} waves with the "
                    f"term and {on_axis_without['tilt_peak_to_valley_waves']:.9f} "
                    f"without, term span {on_axis_with['term_span_waves']:.9f} waves. On "
                    "axis the term is a constant across the pupil and the chief-ray "
                    "subtraction removes it exactly, which is why the defect survived "
                    "CHE-30, CHE-32 and CHE-33 -- every one of them looked on axis."
                ),
                "location": "couplers/handoff.py::_object_space_reference",
            },
        ],
    )
    measurements = {
        "launch_tilt_fraction_recovered": Measurement(
            value=abs(1.0 - recovered),
            uncertainty=abs(
                with_term["chief_direction_y"] - math.sin(field_rad)
            )
            / with_term["chief_direction_y"],
            uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
            note=(
                f"|1 - recovered|, with recovered = {recovered:.9f} from the OPL's "
                "linear slope against n_object*sin(theta). The error bar is the "
                "fractional disagreement between the chief direction the record "
                "reports and the field the instance declares, which is what the "
                "oracle is proportional to."
            ),
        )
    }
    controls = {
        "omit-object-space-term": control_result(
            "omit-object-space-term",
            "launch_tilt_fraction_recovered",
            baseline=measurements["launch_tilt_fraction_recovered"],
            mutated=Measurement(
                value=abs(1.0 - recovered_without),
                uncertainty=abs(
                    with_term["chief_direction_y"] - math.sin(field_rad)
                )
                / with_term["chief_direction_y"],
                uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
                note=(
                    "the CHE-41 defect, through the shipping adapter: "
                    "HandoffPerturbation(reference_incoming_wavefront=False) removes "
                    "exactly the production term and nothing else"
                ),
            ),
            threshold=1e-3,
            note=(
                f"the omission recovers {recovered_without:.6f} of the required slope "
                "and the reconstruction then converges cleanly 209 um from where the "
                "rays go."
            ),
        )
    }
    return InstanceRun(
        family=B1_RAY_OFFAXIS_OPL,
        instance=instance,
        record=record,
        result=verify(
            B1_RAY_OFFAXIS_OPL,
            instance,
            record,
            measurements=measurements,
            negative_controls=controls,
        ),
    )


def _trace_sample(sample: str, *, field_hy: float, rings: int) -> Trace:
    """A canonical registered prescription, traced at the declared exit pupil."""
    from solvers.optiland.adapter import get_adapter

    directory = tempfile.mkdtemp(prefix="b1-ray-sample-")
    run = get_adapter().run(
        ModelRunRequest(
            run_id="b1-ray-offaxis",
            node_id="lens",
            config={
                "sample": sample,
                "num_rays": rings,
                "wavelength": 0.55,
                "Hy": field_hy,
                "handoff_plane": "exit_pupil",
                "output_directory": directory,
            },
        )
    )
    if run.status.value != "succeeded":
        raise RuntimeError(f"{sample} trace failed: {run.error_type}: {run.error_message}")
    record = run.outputs["rays"]
    return Trace(record=record, arrays=dict(np.load(record.uri)))



# ---------------------------------------------------------------------------
# RAY-4 / RAY-5: the device and precision matrix, and the refusals
# ---------------------------------------------------------------------------
#
# Read off the arrays, never off the request. That is the whole content of this
# section and it is why every entry below carries an `observed` block taken from
# `array_state`, and why the CUDA-on-numpy case is a REFUSAL rather than a run
# that quietly happened on the host. PB4a measured the alternative: a
# process-global JAX platform pin produced a successful host run for a caller who
# had asked for CUDA, with nothing raised.


def device_precision_matrix() -> dict[str, Any]:
    """Trace the reference singlet across every supported (backend, device, dtype).

    Reports one row per combination: what was requested, what Optiland was
    actually set to, what the arrays came back as, and whether the request was
    honoured. A combination the capability table refuses appears as a refusal row
    with its code and its supported set -- not as a missing row and not as a skip.
    """
    instance = _instance(B1_RAY_EFL, "B1-RAY-EFL-01")
    radius = float(instance.parameters["radius_mm"])
    index = float(instance.parameters["index"])
    thickness = float(instance.parameters["thickness_mm"])
    spec = _plano_convex(
        radius_mm=radius,
        index=index,
        thickness_mm=thickness,
        epd_mm=10.0,
        back_mm=radius / (index - 1.0) - thickness / index,
    )
    efl_closed = radius / (index - 1.0)

    requests = (
        ("numpy", "cpu", "float64"),
        ("numpy", "cpu", "float32"),
        ("torch", "cpu", "float64"),
        ("torch", "cpu", "float32"),
        # Declared supported through the torch backend only. On a CPU-only
        # session this is expected to refuse, and the refusal is the evidence.
        ("torch", "cuda", "float64"),
        ("torch", "cuda", "float32"),
        # Declared UNSUPPORTED: set_device raises BackendCapabilityError on the
        # numpy backend, so this is refused before Optiland is touched.
        ("numpy", "cuda", "float64"),
    )

    rows: list[dict[str, Any]] = []
    for backend, device, dtype in requests:
        row: dict[str, Any] = {
            "requested": {"backend": backend, "device": device, "dtype": dtype}
        }
        refusal, outcome = probe_refusal(
            lambda backend=backend, device=device, dtype=dtype: _trace_execution(
                spec, backend=backend, device=device, dtype=dtype
            )
        )
        if refusal is not None:
            # Refused before Optiland was touched: a capability the table does
            # not declare, named with its supported set.
            row |= {
                "outcome": "refused",
                "code": refusal.code,
                "detail": refusal.detail,
                "remedy": refusal.remedy,
                "supported": list(refusal.supported),
            }
            rows.append(row)
            continue

        run, trace = outcome
        if trace is None:
            # Declared supported, and it did not run. Reported as its own outcome
            # rather than folded into "refused": a capability the table CLAIMS and
            # the implementation cannot deliver is a different finding from one it
            # never claimed, and collapsing the two would hide it.
            row |= {
                "outcome": "declared_but_failed",
                "error_type": run.error_type,
                "error_message": run.error_message,
                "diagnostics": dict(run.diagnostics or {}),
            }
            rows.append(row)
            continue

        placement = trace.placement
        diagnostics = trace.record.metadata.get("diagnostics", {})
        row |= {
            "outcome": "executed",
            "observed": placement,
            "applied_to_optiland": diagnostics.get("execution", {}).get(
                "applied_to_optiland"
            ),
            "honoured_device": placement["device"].split(":")[0] == device.split(":")[0],
            "honoured_dtype": dtype in placement["dtype"],
        }
        # The physics, at each precision. A float32 trace is not less correct, it
        # is less precise, and the number says which.
        L, M, N = trace.direction
        heights = trace.launch_radius_mm
        keep = heights > 0.0
        focal = heights[keep] / (np.hypot(L[keep], M[keep]) / N[keep])
        _, innermost = trace.innermost(focal, heights[keep])
        row["innermost_efl_relative_error"] = abs(innermost - efl_closed) / efl_closed
        rows.append(row)

    return {
        "efl_closed_form_mm": efl_closed,
        "rows": tuple(rows),
        "executed": tuple(r for r in rows if r["outcome"] == "executed"),
        "refused": tuple(r for r in rows if r["outcome"] == "refused"),
        "declared_but_failed": tuple(
            r for r in rows if r["outcome"] == "declared_but_failed"
        ),
    }


def _trace_execution(
    spec: OpticalSystemSpec, *, backend: str, device: str, dtype: str
) -> tuple[Any, Trace | None]:
    """One trace at an explicit (backend, device, dtype).

    Returns the result and the trace, with the trace ``None`` when the run did
    not produce one. A capability refusal still raises, because that happens
    before any solver call and is a different fact from a run that failed.
    """
    from solvers.optiland.adapter import get_adapter

    directory = tempfile.mkdtemp(prefix="b1-ray-matrix-")
    run = get_adapter().run(
        ModelRunRequest(
            run_id="b1-ray-matrix",
            node_id="lens",
            config={
                "prescription": spec,
                "num_rays": 16,
                "wavelength": WAVELENGTH_UM,
                "Hy": 0.0,
                "handoff_plane": "image_surface",
                "backend": backend,
                "device": device,
                "dtype": dtype,
                "output_directory": directory,
            },
        )
    )
    if run.status.value != "succeeded":
        return run, None
    record = run.outputs["rays"]
    return run, Trace(record=record, arrays=dict(np.load(record.uri)))


def unsupported_configuration() -> dict[str, Any]:
    """RAY-5: one real unsupported request, and what comes back.

    A custom prescription is supported; a custom *sample name* is not, and the
    difference is the point -- the adapter refuses what it has not been verified
    against rather than guessing. Asserted here: a structured code, a reason, a
    remedy, and no numbers.
    """
    from solvers.optiland.adapter import get_adapter

    directory = tempfile.mkdtemp(prefix="b1-ray-unsupported-")
    refusal, run = probe_refusal(
        lambda: get_adapter().run(
            ModelRunRequest(
                run_id="b1-ray-unsupported",
                node_id="lens",
                config={
                    "sample": "ALensNobodyHasVerified",
                    "num_rays": 8,
                    "wavelength": WAVELENGTH_UM,
                    "output_directory": directory,
                },
            )
        )
    )
    return {
        "refused": refusal is not None,
        "code": None if refusal is None else refusal.code,
        "detail": None if refusal is None else refusal.detail,
        "remedy": None if refusal is None else refusal.remedy,
        "supported": [] if refusal is None else list(refusal.supported),
        "outputs": {} if run is None else dict(getattr(run, "outputs", {}) or {}),
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_RUNNERS: dict[str, Any] = {
    "B1-RAY-EFL-01": _run_efl,
    "B1-RAY-PLATE-01": _run_plate,
    "B1-RAY-SNELL-01": lambda: _run_snell("B1-RAY-SNELL-01"),
    "B1-RAY-SNELL-02": lambda: _run_snell("B1-RAY-SNELL-02"),
    "B1-RAY-SNELL-03": lambda: _run_snell("B1-RAY-SNELL-03"),
    "B1-RAY-SNELL-04": lambda: _run_snell("B1-RAY-SNELL-04"),
    "B1-RAY-LAGRANGE-01": _run_lagrange,
    "B1-RAY-OFFAXIS-OPL-01": _run_offaxis_opl,
}

_FAMILIES = (B1_RAY_EFL, B1_RAY_PLATE, B1_RAY_SNELL, B1_RAY_LAGRANGE, B1_RAY_OFFAXIS_OPL)


def declared_instance_ids() -> tuple[str, ...]:
    return tuple(
        instance.instance_id for family in _FAMILIES for instance in family.canonical_instances
    )


def run_instance(instance_id: str) -> InstanceRun:
    try:
        runner = _RUNNERS[instance_id]
    except KeyError:
        raise KeyError(
            f"no runner for {instance_id!r}. Declared: {sorted(declared_instance_ids())}"
        ) from None
    return runner()


def run_all() -> dict[str, InstanceRun]:
    return {instance_id: run_instance(instance_id) for instance_id in declared_instance_ids()}


def _describe(metric: Any) -> str:
    verdict = "" if metric.met is None else (" MET" if metric.met else " UNMET")
    return f"{metric.metric}={metric.measured.value:.6g}{verdict}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="persist the instance records")
    parser.add_argument("--instance", default=None)
    args = parser.parse_args()

    runs = {args.instance: run_instance(args.instance)} if args.instance else run_all()
    for instance_id, run in runs.items():
        metrics = ", ".join(_describe(m) for m in run.result.physics_accuracy)
        controls = ", ".join(
            f"{c.control_id}:{c.outcome.value}" for c in run.result.negative_control_results
        )
        print(f"{instance_id:<24} status={run.result.status.value:<18} {metrics}")
        if controls:
            print(f"{'':<24} controls: {controls}")
        exponent = run.result.convergence.fitted_exponent
        if exponent is not None:
            spread = (
                "not estimated"
                if exponent.uncertainty is None
                else f"{exponent.uncertainty:.4f}"
            )
            print(
                f"{'':<24} exponent={exponent.value:+.4f} +/- {spread} "
                f"over {len(run.result.convergence.ladder)} rungs"
            )
        if args.write:
            path = write_instance_record(run, driver="instances/b1_ray")
            print(f"{'':<24} -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
