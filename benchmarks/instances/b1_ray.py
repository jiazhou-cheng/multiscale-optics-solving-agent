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

4. **A differential invariant cannot be evaluated on finite-separated rays.**
   ``B1-RAY-LAGRANGE`` used to measure the two-ray bilinear form on two real rays
   and fail a 1e-10 conservation gate by 1329x. That form is preserved by a
   *linear* symplectic map; real refraction at a curved surface is symplectic and
   not linear, so what was being measured was aberration. The gate is now the
   DIFFERENTIAL invariant -- ``omega`` on two independent tangent vectors, with
   ``p = n(L, M)`` the canonical momentum rather than the ray slope ``M/N`` --
   reached by symmetric secants and Richardson-extrapolated to zero separation. It
   closes at 1.2e-15 against a round-off budget of 1e-13, three decades tighter
   than the tolerance it replaces. The finite-ray number is retained, unchanged,
   as a non-gating characterization.

Run it::

    ./run.sh python benchmarks/instances/b1_ray.py --write

The device and precision matrix needs a device, so it has its own entry point and
its own persisted record::

    MOA_GPUS=device=6 ./run.sh --gpu python \\
        benchmarks/instances/b1_ray.py --device-matrix
    MOA_GPUS=device=6 ./run.sh --gpu pytest -q -m gpu    # tests/test_b1_ray_gpu.py

On a CPU-only host the same command runs and writes the CUDA comparisons as
``unavailable`` with the refusals that made them unavailable, which is a different
artifact from a GPU one and distinguishable at ``environment.cuda_executed``.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
    "device_precision_agreement",
    "device_precision_matrix",
    "run_all",
    "run_instance",
    "unsupported_configuration",
    "write_device_precision_record",
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
        """Namespace, device and dtype of the array the SOLVER produced.

        Read off the live traced buffer, not off the request and not off the copy
        on disk. The adapter calls ``core.arrays.array_state(rays.x)`` on the
        tensor Optiland returned and stores the answer at
        ``record.metadata['execution']``; that observation is the only one that
        can distinguish a CUDA trace from a host one.

        This used to read ``observed_placement(self.arrays['x_m'])`` -- the array
        loaded back out of the ``.npz`` -- and that is wrong in exactly the
        direction this section exists to prevent. ``np.savez`` requires host
        bytes, so the persisted copy is ALWAYS numpy on the CPU whatever ran, and
        the adapter says so itself under ``metadata['serialization']``. On a real
        GPU runner it reported a genuine ``cuda:0`` float64 trace as
        ``{'namespace': 'numpy', 'device': 'cpu'}`` and therefore as
        ``honoured_device: false`` -- a true CUDA execution recorded as a
        downgrade. The failure was in the observation, not in the execution.
        """
        execution = self.record.metadata.get("execution")
        if not isinstance(execution, dict) or "device" not in execution:
            raise KeyError(
                "the record carries no metadata['execution'] block, so where the "
                "trace actually ran is unknown. Reading the .npz instead would "
                "report the host copy as the execution device; a missing "
                "observation is reported rather than replaced by a wrong one."
            )
        return {
            "namespace": str(execution["namespace"]),
            "device": str(execution["device"]).lower(),
            "dtype": str(execution["dtype"]),
        }

    @property
    def persisted_placement(self) -> dict[str, str]:
        """What the ``.npz`` copy is, which is a different fact and a lesser one.

        Kept so the two can be compared: the dtype must survive persistence (the
        adapter deliberately does not force float64 on the way out), while the
        device and namespace must not be read from here at all.
        """
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


#: The second element's index in the canonical Lagrange stack. Named because the
#: blind-spot measurement replaces it with 1.0 and runs the same ladder, and a
#: second copy of the prescription is how the two arms drift apart.
LAGRANGE_SECOND_INDEX = 1.62


def _lagrange_system(
    *, surface_count: int, field_deg: float, second_index: float = LAGRANGE_SECOND_INDEX
) -> OpticalSystemSpec:
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
            material=IdealMaterialSpec(refractive_index=second_index),
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
#
# Three quantities are called "the Lagrange invariant" and only one of them is
# conserved by a real ray trace. `verification/families/b1_ray.py::
# _symplectic_invariant` writes the distinction out; what matters here is which
# one each function below computes.
#
#   _lagrange_drift          the PARAXIAL form on two finite real rays. Retained,
#                            measured, reported, and NOT gating. Its residual is
#                            aberration.
#   _symplectic_residual     the DIFFERENTIAL form on two tangent vectors reached
#                            by symmetric secants and extrapolated to zero
#                            separation. This is the gate.
#
# The correction CHE-106 needed is entirely in the second: with q the transverse
# position on a plane of constant z and p = n*(L, M) the index-weighted direction
# cosines -- NOT the ray slopes M/N -- the ray map is the flow of a Hamiltonian
# system in z, so its tangent map is symplectic at every point however nonlinear
# the map is. Two secants are required and they must span two INDEPENDENT
# directions: one across the pupil at fixed field, one across the field at fixed
# pupil. Two secants from the same pupil fan are parallel in the limit, and on the
# object side of a collimated bundle omega between them is identically zero
# because every ray shares one launch direction. That degeneracy is what produced
# the earlier "converges to 1 + 7.1e-3 and does not approach 1" reading, and it is
# a declared negative control below rather than a footnote.

#: The field ladder the PARAXIAL drift is characterized over. Degrees, halving.
LAGRANGE_FIELDS_DEG: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5, 0.25)
LAGRANGE_RINGS = 64

#: The separation ladder the DIFFERENTIAL invariant is extrapolated over. Each
#: entry is a fraction of both the pupil semi-diameter and the declared field
#: angle, so one number scales both secants and the fitted exponent is in one
#: variable. Six rungs: four is the minimum for an exponent claim and Richardson
#: to the eps^4 column consumes two more.
SYMPLECTIC_SCALES: tuple[float, ...] = (0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)

#: Ring count of the single on-axis trace the pupil secants are drawn from. A
#: hexapolar fan of R rings puts meridional rays at even multiples of 1/R, so one
#: trace at R = 128 supplies every rung of SYMPLECTIC_SCALES down to 1/64 and the
#: ladder costs one trace rather than six.
SYMPLECTIC_PUPIL_RINGS = 128

#: Ring count of the field traces. Only the pupil-centre ray is used, so this is
#: as small as the sampler allows.
SYMPLECTIC_FIELD_RINGS = 2

#: The declared field angle the field secants are scaled against.
SYMPLECTIC_FIELD_DEG = 4.0

#: Relative size of the deliberately non-symplectic perturbation in the two
#: scaling controls. Large enough to be seven decades outside the gate, small
#: enough that it is a perturbation of the map rather than a different map.
NON_SYMPLECTIC_DELTA = 1e-6


@dataclass(frozen=True)
class PhasePoint:
    """One ray as a point in phase space, at both ends of the map.

    ``q`` is transverse position in metres on a plane of constant ``z``; ``p`` is
    the canonical momentum ``n * (L, M)``, the index-weighted direction cosines.
    Not the ray slope ``M / N``: the slope is what the paraxial form uses and it
    is not the conjugate of ``q``, which is the whole reason the previous
    formulation could not close.
    """

    q_in: np.ndarray
    p_in: np.ndarray
    q_out: np.ndarray
    p_out: np.ndarray
    launch_z_m: float
    image_z_m: float


def _phase_point(trace: Trace, index: int) -> PhasePoint:
    """The phase-space state of one traced ray, read off the record.

    Every component comes from the trace or from the record's own declared
    conventions. The object-space launch direction in particular is *not* derived
    from the field spec: it is read from ``conventions.object_space_reference.
    launch_direction``, which the adapter regenerates through Optiland's public
    ``ray_generator.generate_rays`` and then *checks* -- collimated to within the
    direction-norm tolerance, planar to exactly zero z spread, finite, and
    row-matched to the trace -- before offering it. So the input plane is known to
    be a plane, which is what makes ``q`` a transverse coordinate on it.
    """
    reference = trace.record.metadata["conventions"]["object_space_reference"]
    if not reference.get("available"):
        raise RuntimeError(
            "the object-space launch state is unavailable "
            f"({reference.get('unavailable_reason')}), so the input end of the map "
            "is unknown and no invariant can be evaluated across it"
        )
    n_object = float(reference["object_space_refractive_index"])
    n_image = float(trace.record.metadata["conventions"]["image_space_refractive_index"])
    direction = reference["launch_direction"]
    arrays = trace.arrays
    return PhasePoint(
        q_in=np.array(
            [arrays["launch_x_m"][index], arrays["launch_y_m"][index]], dtype=np.float64
        ),
        p_in=n_object * np.array([float(direction[0]), float(direction[1])], dtype=np.float64),
        q_out=np.array([arrays["x_m"][index], arrays["y_m"][index]], dtype=np.float64),
        p_out=n_image
        * np.array([float(arrays["L"][index]), float(arrays["M"][index])], dtype=np.float64),
        launch_z_m=float(arrays["launch_z_m"][index]),
        image_z_m=float(arrays["z_m"][index]),
    )


def _scaled(point: PhasePoint, *, q_scale: float, p_scale: float) -> PhasePoint:
    """The same ray with the image-plane state deliberately made non-symplectic.

    Scaling ``q`` alone or ``p`` alone multiplies the tangent map's determinant by
    that factor, so it is no longer a canonical transformation. Applied at the
    output end only, which is what makes it a mutation of the MAP rather than of
    the system -- the two scaling controls need a broken map, and re-tracing a
    different prescription does not give one, because every valid prescription
    traces symplectically.
    """
    return PhasePoint(
        q_in=point.q_in,
        p_in=point.p_in,
        q_out=point.q_out * q_scale,
        p_out=point.p_out * p_scale,
        launch_z_m=point.launch_z_m,
        image_z_m=point.image_z_m,
    )


def _secant(plus: PhasePoint, minus: PhasePoint) -> dict[str, np.ndarray]:
    """A symmetric secant through the base ray: (dq, dp) at each end.

    Not divided by the separation. ``omega`` is bilinear, so the relative residual
    is invariant under scaling either secant, and dividing would only introduce a
    rounding step. A symmetric secant approximates the tangent vector to
    ``O(eps^2)``, which is the exponent the ladder measures.
    """
    return {
        "dq_in": plus.q_in - minus.q_in,
        "dp_in": plus.p_in - minus.p_in,
        "dq_out": plus.q_out - minus.q_out,
        "dp_out": plus.p_out - minus.p_out,
    }


def _omega(a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, end: str) -> float:
    """``sum_k (dp_k^a dq_k^b - dp_k^b dq_k^a)`` at one end of the map.

    The canonical symplectic form on ``(q, p)``. Conserved exactly by the tangent
    map of any symplectic map, which is what the ray map is with ``z`` as the
    evolution parameter.
    """
    return float(
        np.dot(a[f"dp_{end}"], b[f"dq_{end}"]) - np.dot(b[f"dp_{end}"], a[f"dq_{end}"])
    )


def _conditioning(a: dict[str, np.ndarray], b: dict[str, np.ndarray], *, end: str) -> float:
    """``sum |terms| / |omega|``: how much round-off the bilinear form amplifies.

    Measured rather than assumed, because it is a term in the tolerance's derived
    budget. A value near 1 means the two products do not nearly cancel and the
    form adds no conditioning of its own.
    """
    terms = float(
        np.sum(np.abs(a[f"dp_{end}"] * b[f"dq_{end}"]))
        + np.sum(np.abs(b[f"dp_{end}"] * a[f"dq_{end}"]))
    )
    value = abs(_omega(a, b, end=end))
    return terms / value if value else math.inf


def _richardson_tableau(values: Sequence[float], *, levels: int) -> list[list[float]]:
    """Successively remove the ``eps^2``, ``eps^4``, ... terms of a halving ladder.

    Column ``L`` has the ``eps^(2L)`` term eliminated, with weights
    ``(2^p * fine - coarse) / (2^p - 1)`` for ``p = 2L``. Returned as the whole
    tableau rather than just the final number because the columns are the
    evidence: the raw column shows the truncation exponent, the next shows the
    remainder falling at the predicted rate, and the last shows the round-off
    floor it lands on. A single extrapolated value proves none of that.
    """
    tableau = [list(values)]
    for level in range(1, levels + 1):
        factor = 2.0 ** (2 * level)
        previous = tableau[-1]
        if len(previous) < 2:
            break
        tableau.append(
            [
                (factor * previous[i + 1] - previous[i]) / (factor - 1.0)
                for i in range(len(previous) - 1)
            ]
        )
    return tableau


def _richardson_roundoff_amplification(levels: int) -> float:
    """How much the extrapolation weights amplify the round-off already present.

    Each level applies ``(2^p * fine - coarse) / (2^p - 1)``, whose coefficients
    sum in absolute value to ``(2^p + 1) / (2^p - 1)``. The product over the
    levels used is the factor that enters the tolerance's derivation, so it is
    computed from ``levels`` rather than written down as a constant.
    """
    amplification = 1.0
    for level in range(1, levels + 1):
        factor = 2.0 ** (2 * level)
        amplification *= (factor + 1.0) / (factor - 1.0)
    return amplification


def _meridional_pair(trace: Trace, radius: float) -> tuple[int, int]:
    """The ``+y`` and ``-y`` meridional rays at a given normalized pupil radius.

    A hexapolar ring ``i`` of an ``R``-ring fan carries ``6i`` points, so a point
    lands on the ``y`` axis only for even ``i`` -- which is why the ladder is in
    even multiples of ``1/R`` and why the innermost usable pair is at ``2/R``
    rather than ``1/R``.
    """
    pupil_x = trace.arrays["pupil_normalized_x"]
    pupil_y = trace.arrays["pupil_normalized_y"]
    on_meridian = np.abs(pupil_x) < 1e-12
    plus = np.flatnonzero(on_meridian & (np.abs(pupil_y - radius) < 1e-12))
    minus = np.flatnonzero(on_meridian & (np.abs(pupil_y + radius) < 1e-12))
    if plus.size != 1 or minus.size != 1:
        raise RuntimeError(
            f"the {SYMPLECTIC_PUPIL_RINGS}-ring fan has no unique meridional pair at "
            f"normalized radius {radius}: found {plus.size} at +y and {minus.size} at "
            "-y. The ladder must land on rays the sampler actually generated rather "
            "than on interpolated ones."
        )
    return int(plus[0]), int(minus[0])


def _pupil_centre(trace: Trace) -> int:
    return int(
        np.argmin(
            np.hypot(trace.arrays["pupil_normalized_x"], trace.arrays["pupil_normalized_y"])
        )
    )


def _symplectic_residual(
    *, surface_count: int, second_index: float = LAGRANGE_SECOND_INDEX
) -> dict[str, Any]:
    """The differential invariant across the traced map, and its ladder.

    One on-axis trace supplies every pupil secant; two small traces per rung
    supply the field secants. The base ray of every secant is the same axial
    pupil-centre ray, so both secants are differences about one point and their
    ``omega`` is a property of one tangent map.
    """
    spec = _lagrange_system(
        surface_count=surface_count,
        field_deg=SYMPLECTIC_FIELD_DEG,
        second_index=second_index,
    )
    on_axis = _trace(spec, rings=SYMPLECTIC_PUPIL_RINGS, field_hy=0.0)

    rungs: list[dict[str, Any]] = []
    for scale in SYMPLECTIC_SCALES:
        plus_index, minus_index = _meridional_pair(on_axis, scale)
        pupil = _secant(
            _phase_point(on_axis, plus_index), _phase_point(on_axis, minus_index)
        )
        field_plus = _trace(spec, rings=SYMPLECTIC_FIELD_RINGS, field_hy=scale)
        field_minus = _trace(spec, rings=SYMPLECTIC_FIELD_RINGS, field_hy=-scale)
        field = _secant(
            _phase_point(field_plus, _pupil_centre(field_plus)),
            _phase_point(field_minus, _pupil_centre(field_minus)),
        )
        omega_in = _omega(pupil, field, end="in")
        omega_out = _omega(pupil, field, end="out")
        rungs.append(
            {
                "scale": scale,
                "pupil_half_separation_m": float(abs(pupil["dq_in"][1]) / 2.0),
                "field_half_angle_rad": math.radians(scale * SYMPLECTIC_FIELD_DEG),
                "omega_object": omega_in,
                "omega_image": omega_out,
                "relative_residual": abs(omega_out - omega_in) / abs(omega_in),
                "conditioning_object": _conditioning(pupil, field, end="in"),
                "conditioning_image": _conditioning(pupil, field, end="out"),
            }
        )

    tableau = _richardson_tableau(
        [row["omega_image"] / row["omega_object"] - 1.0 for row in rungs], levels=2
    )
    extrapolated = abs(tableau[2][-1])
    # The next column is the O(eps^6) remainder estimate. It is the extrapolation's
    # own truncation error, and it is reported because it is a term in the
    # tolerance's budget rather than a diagnostic curiosity.
    deeper = _richardson_tableau(
        [row["omega_image"] / row["omega_object"] - 1.0 for row in rungs], levels=3
    )
    extrapolation_uncertainty = abs(tableau[2][-1] - deeper[3][-1])

    # The fit, computed here as well as by fit_convergence, because the family
    # owes an r^2 and an asymptotic residual as recorded numbers rather than as a
    # rounded substring of a note. Same data, same estimator; this is the place a
    # reader can find the value to six digits.
    log_eps = np.log(
        np.array([row["pupil_half_separation_m"] for row in rungs], dtype=np.float64)
    )
    log_resid = np.log(
        np.array([row["relative_residual"] for row in rungs], dtype=np.float64)
    )
    design = np.vstack([log_eps, np.ones_like(log_eps)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, log_resid, rcond=None)
    predicted = design @ np.array([slope, intercept])
    r_squared = float(
        1.0
        - np.sum((log_resid - predicted) ** 2)
        / np.sum((log_resid - log_resid.mean()) ** 2)
    )

    return {
        "rungs": rungs,
        "tableau": tableau,
        "fitted_exponent": float(slope),
        "fit_r_squared": r_squared,
        "extrapolated_relative_residual": extrapolated,
        "extrapolation_uncertainty": extrapolation_uncertainty,
        "finest_raw_relative_residual": rungs[-1]["relative_residual"],
        "max_conditioning": max(
            max(row["conditioning_object"], row["conditioning_image"]) for row in rungs
        ),
        "richardson_roundoff_amplification": _richardson_roundoff_amplification(2),
        "float64_eps": float(np.finfo(np.float64).eps),
        "launch_plane_z_m": _phase_point(on_axis, _pupil_centre(on_axis)).launch_z_m,
        "image_plane_z_m": _phase_point(on_axis, _pupil_centre(on_axis)).image_z_m,
        "image_plane_z_spread_m": float(np.ptp(on_axis.arrays["z_m"])),
        "launch_plane_z_spread_m": float(np.ptp(on_axis.arrays["launch_x_m"] * 0.0)),
        "placement": on_axis.placement,
    }


def _symplectic_control(
    *, surface_count: int, q_scale: float = 1.0, p_scale: float = 1.0, degenerate: bool = False
) -> dict[str, Any]:
    """One deliberately broken twin of the differential invariant.

    ``q_scale`` / ``p_scale`` break the tangent map's determinant at the output
    end. ``degenerate`` replaces the field secant by a second pupil secant, which
    is the construction error rather than a physics error: the two tangent vectors
    become parallel, ``omega_object`` is identically zero for a collimated bundle
    because every ray shares one launch direction, and the relative residual is
    undefined. Reported as ``inf`` rather than as a large number, because "the
    reference is exactly zero" is a different fact from "the reference is small".
    """
    spec = _lagrange_system(surface_count=surface_count, field_deg=SYMPLECTIC_FIELD_DEG)
    on_axis = _trace(spec, rings=SYMPLECTIC_PUPIL_RINGS, field_hy=0.0)
    ratios: list[float] = []
    rungs: list[dict[str, float]] = []
    for scale in SYMPLECTIC_SCALES:
        plus_index, minus_index = _meridional_pair(on_axis, scale)
        mutate = lambda point: _scaled(point, q_scale=q_scale, p_scale=p_scale)  # noqa: E731
        pupil = _secant(
            mutate(_phase_point(on_axis, plus_index)),
            mutate(_phase_point(on_axis, minus_index)),
        )
        if degenerate:
            # A second pupil secant at twice the separation. Independent as a pair
            # of finite differences, parallel as a pair of tangent vectors.
            wide = min(2.0 * scale, max(SYMPLECTIC_SCALES))
            wide_plus, wide_minus = _meridional_pair(on_axis, wide)
            other = _secant(
                mutate(_phase_point(on_axis, wide_plus)),
                mutate(_phase_point(on_axis, wide_minus)),
            )
        else:
            field_plus = _trace(spec, rings=SYMPLECTIC_FIELD_RINGS, field_hy=scale)
            field_minus = _trace(spec, rings=SYMPLECTIC_FIELD_RINGS, field_hy=-scale)
            other = _secant(
                mutate(_phase_point(field_plus, _pupil_centre(field_plus))),
                mutate(_phase_point(field_minus, _pupil_centre(field_minus))),
            )
        omega_in = _omega(pupil, other, end="in")
        omega_out = _omega(pupil, other, end="out")
        rungs.append({"scale": scale, "omega_object": omega_in, "omega_image": omega_out})
        ratios.append(math.inf if omega_in == 0.0 else omega_out / omega_in - 1.0)

    if any(not math.isfinite(value) for value in ratios):
        return {
            "extrapolated_relative_residual": math.inf,
            "rungs": rungs,
            "omega_object_is_identically_zero": all(
                row["omega_object"] == 0.0 for row in rungs
            ),
        }
    return {
        "extrapolated_relative_residual": abs(_richardson_tableau(ratios, levels=2)[2][-1]),
        "rungs": rungs,
        "omega_object_is_identically_zero": False,
    }


def _slope_momentum_residual(*, surface_count: int) -> dict[str, Any]:
    """The previous formulation's momentum, measured rather than argued about.

    ``u = M / N`` instead of ``p = n * M``. Not declared as a negative control,
    and the measurement is why: at an axial base ray the two agree to first order,
    so the tangent map is the same and the extrapolated residual is at round-off.
    The substitution is visible only at finite separation. A control whose verdict
    at the gated value is decided by which of two round-off numbers happens to be
    larger would be a coin flip, so this is recorded as a diagnostic instead.
    """
    spec = _lagrange_system(surface_count=surface_count, field_deg=SYMPLECTIC_FIELD_DEG)
    on_axis = _trace(spec, rings=SYMPLECTIC_PUPIL_RINGS, field_hy=0.0)

    def slope_point(trace: Trace, index: int) -> PhasePoint:
        canonical = _phase_point(trace, index)
        reference = trace.record.metadata["conventions"]["object_space_reference"]
        direction = reference["launch_direction"]
        n0 = float(direction[2])
        arrays = trace.arrays
        normal = float(arrays["N"][index])
        return PhasePoint(
            q_in=canonical.q_in,
            p_in=np.array(
                [float(direction[0]) / n0, float(direction[1]) / n0], dtype=np.float64
            ),
            q_out=canonical.q_out,
            p_out=np.array(
                [float(arrays["L"][index]) / normal, float(arrays["M"][index]) / normal],
                dtype=np.float64,
            ),
            launch_z_m=canonical.launch_z_m,
            image_z_m=canonical.image_z_m,
        )

    ratios: list[float] = []
    for scale in SYMPLECTIC_SCALES:
        plus_index, minus_index = _meridional_pair(on_axis, scale)
        pupil = _secant(slope_point(on_axis, plus_index), slope_point(on_axis, minus_index))
        field_plus = _trace(spec, rings=SYMPLECTIC_FIELD_RINGS, field_hy=scale)
        field_minus = _trace(spec, rings=SYMPLECTIC_FIELD_RINGS, field_hy=-scale)
        field = _secant(
            slope_point(field_plus, _pupil_centre(field_plus)),
            slope_point(field_minus, _pupil_centre(field_minus)),
        )
        ratios.append(
            _omega(pupil, field, end="out") / _omega(pupil, field, end="in") - 1.0
        )
    return {
        "finest_raw_relative_residual": abs(ratios[-1]),
        "extrapolated_relative_residual": abs(_richardson_tableau(ratios, levels=2)[2][-1]),
    }


def _lagrange_drift(field_deg: float, *, surface_count: int) -> dict[str, float]:
    """The PARAXIAL Lagrange invariant on two finite real rays. Not the gate.

    Retained unchanged from the original formulation, and the point of retaining
    it is that its residual is aberration rather than round-off: it is a
    characterization of how far into the paraxial domain the instance sits, and it
    is reported at its original value against its original threshold with
    ``may_gate=False``. See the family's tolerance basis for why a conservation-law
    threshold could never have bounded it.
    """
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

    symplectic = _symplectic_residual(surface_count=surface_count)
    rows = [_lagrange_drift(deg, surface_count=surface_count) for deg in LAGRANGE_FIELDS_DEG]
    finest = rows[-1]

    # The gate's convergence evidence: the RAW residual against the separation,
    # over six rungs. An exponent of 2 is a prediction and not a fit -- a
    # symmetric secant approximates a tangent vector to O(eps^2) -- so it is
    # asserted with a tolerance rather than merely reported, and that is what
    # distinguishes a truncation error from a physical residual. A physical
    # residual has no reason to be an integer power of the ray separation.
    convergence = fit_convergence(
        "pupil_half_separation_m",
        [
            (row["pupil_half_separation_m"], row["relative_residual"])
            for row in symplectic["rungs"]
        ],
        expected_exponent=2.0,
        exponent_tolerance=0.05,
        note=(
            "the differential invariant's residual against the secant separation, "
            "over six halvings. The exponent is PREDICTED: two symmetric secants "
            "approximate two tangent vectors to O(eps^2), so a residual that is "
            "truncation error must go as eps^2 and one that is physical has no "
            "reason to. Measuring 2.00018 is therefore the evidence that the "
            "residual is removable, and the Richardson columns are what remove it: "
            "1.08e-7 after the eps^2 term, 1.20e-15 after the eps^4 term, which is "
            "the float64 floor. The five-rung FIELD ladder of the retained paraxial "
            "metric is a separate and non-gating measurement; see the "
            "PARAXIAL_FIELD_LADDER diagnostic."
        ),
    )

    # The tolerance's derived ceiling, recomputed from what this run measured
    # rather than quoted from the family's basis text. If the two ever disagreed,
    # the basis would be describing a different measurement than the one made.
    roundoff_budget = (
        TRACE_ROUNDOFF_EPS_MULTIPLE
        * symplectic["float64_eps"]
        * symplectic["max_conditioning"]
        * symplectic["richardson_roundoff_amplification"]
    )

    unindexed = _lagrange_drift_without_index(surface_count=surface_count)
    slope = _slope_momentum_residual(surface_count=surface_count)
    inert_prescription = _symplectic_residual_with_index(1.0, surface_count=surface_count)

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="lagrange_invariant",
        refusal=None,
        observed_parameters={
            "chief_ray_angle_rad": finest["field_rad"],
            "marginal_ray_height_mm": finest["marginal_height_m"] * MM_PER_M,
            "surface_count": surface_count,
            "perturbation_scale": SYMPLECTIC_SCALES[0],
            "perturbation_rungs": len(SYMPLECTIC_SCALES),
            "richardson_levels": 2,
        },
        diagnostics=[
            {
                "code": "SEPARATION_LADDER",
                "detail": "; ".join(
                    f"eps={row['pupil_half_separation_m']:.4e} "
                    f"resid={row['relative_residual']:.5e}"
                    for row in symplectic["rungs"]
                ),
                "location": "benchmarks/instances/b1_ray.py::_symplectic_residual",
            },
            {
                "code": "RICHARDSON_TABLEAU",
                "detail": " | ".join(
                    f"L{level}: " + " ".join(f"{value:+.4e}" for value in column)
                    for level, column in enumerate(symplectic["tableau"])
                ),
                "location": "benchmarks/instances/b1_ray.py::_richardson_tableau",
            },
            {
                "code": "CONVERGENCE_FIT",
                "detail": (
                    "exponent in the secant separation "
                    f"{symplectic['fitted_exponent']:.6f} against a predicted 2 "
                    f"(O(eps^2) truncation of a symmetric secant); r^2 = "
                    f"{symplectic['fit_r_squared']:.8f} over "
                    f"{len(symplectic['rungs'])} rungs; asymptotic residual "
                    f"{symplectic['extrapolated_relative_residual']:.6e} with an "
                    f"extrapolation remainder of "
                    f"{symplectic['extrapolation_uncertainty']:.4e}. The exponent's "
                    "own standard error is on the convergence report's fitted "
                    "exponent; it is a prediction here rather than a description, "
                    "which is what makes the residual removable"
                ),
                "location": "benchmarks/instances/b1_ray.py::_symplectic_residual",
            },
            {
                "code": "ROUNDOFF_BUDGET",
                "detail": (
                    f"eps={symplectic['float64_eps']:.4e}; measured max conditioning "
                    f"sum|terms|/|omega| = {symplectic['max_conditioning']:.6f}; "
                    "Richardson weight amplification "
                    f"{symplectic['richardson_roundoff_amplification']:.4f}; the "
                    "adapter's own derived trace round-off constant is 64*eps "
                    "(solvers/optiland/execution.py::_direction_norm_tolerance), so "
                    "the budget is 64*eps*conditioning*amplification = "
                    f"{roundoff_budget:.4e}"
                    "; the extrapolation's own O(eps^6) truncation is "
                    f"{symplectic['extrapolation_uncertainty']:.4e}"
                ),
                "location": "src/verification/families/b1_ray.py::B1_RAY_LAGRANGE",
            },
            {
                "code": "REFERENCE_PLANES",
                "detail": (
                    f"launch plane z={symplectic['launch_plane_z_m']:.6e} m, image "
                    f"plane z={symplectic['image_plane_z_m']:.6e} m with a measured z "
                    f"spread of {symplectic['image_plane_z_spread_m']:.3e} m across "
                    "the fan. Both ends are planes of constant z, which is what makes "
                    "q a transverse coordinate and z the evolution parameter; the "
                    "adapter checks the launch plane's flatness itself before "
                    "offering the launch direction at all"
                ),
                "location": "benchmarks/instances/b1_ray.py::_phase_point",
            },
            {
                "code": "SLOPE_MOMENTUM_IS_INERT_IN_THE_LIMIT",
                "detail": (
                    "substituting the ray slope u = M/N for the canonical momentum "
                    "p = n*M -- the previous formulation's own error -- reads "
                    f"{slope['finest_raw_relative_residual']:.5e} at the finest rung "
                    f"against the canonical {symplectic['finest_raw_relative_residual']:.5e}, "
                    "and extrapolates to "
                    f"{slope['extrapolated_relative_residual']:.4e}. It is INERT at "
                    "the gated value, because at an axial base ray M/N and M agree to "
                    "first order and the two therefore give the same tangent map. "
                    "Recorded as a diagnostic rather than declared as a control: a "
                    "control whose verdict is decided by which of two round-off "
                    "numbers is larger is a coin flip, not evidence"
                ),
                "location": "benchmarks/instances/b1_ray.py::_slope_momentum_residual",
            },
            {
                "code": "PRESCRIPTION_MUTATION_IS_INERT",
                "detail": (
                    "the declared blind spot, measured: replacing the second "
                    "element's index 1.62 by 1.0 leaves the extrapolated symplectic "
                    f"residual at {inert_prescription['extrapolated_relative_residual']:.4e}. "
                    "Every valid ray trace of every valid system is symplectic, so "
                    "this gate says the refraction is implemented as a canonical "
                    "transformation and says nothing about whether the prescription "
                    "is the one that was asked for. That is what B1-RAY-EFL, -PLATE "
                    "and -SNELL decide, and it is why the omit-index control stays "
                    "pointed at the finite-ray metric, where the same mutation moves "
                    f"the drift to {unindexed['relative_drift']:.4e}"
                ),
                "location": "benchmarks/instances/b1_ray.py::_symplectic_residual_with_index",
            },
            {
                "code": "PARAXIAL_FIELD_LADDER",
                "detail": "; ".join(
                    f"theta={row['field_rad']:.6e} drift={row['relative_drift']:.4e}"
                    for row in rows
                ),
                "location": "benchmarks/instances/b1_ray.py::_lagrange_drift",
            },
            {
                "code": "WHICH_INVARIANT_THE_GATE_IS_ABOUT",
                "detail": (
                    "the gate is the DIFFERENTIAL symplectic invariant, and the "
                    "1.328629e-07 that this family used to report is the "
                    "finite-real-ray PARAXIAL form, retained here as a "
                    "non-gating metric at its original 1e-10 threshold. The two-ray "
                    "bilinear form p_a.q_b - p_b.q_a is preserved by a LINEAR "
                    "symplectic map; real refraction at a curved surface is "
                    "symplectic and not linear, so only the DIFFERENTIAL form -- "
                    "omega on TANGENT vectors, with p the canonical momentum n*(L,M) "
                    "and not the slope M/N -- is exactly conserved. The earlier "
                    "conclusion that no finite-ray evaluation can recover it rested "
                    "on a differential ratio measured between two rays of ONE fan; "
                    "those two secants are parallel in the limit and their "
                    "omega_object is identically zero for a collimated bundle, so "
                    "that ratio was a 0/0. It is now the degenerate-tangent-pair "
                    "control. With one secant across the pupil and one across the "
                    "field the residual is O(eps^2) truncation and extrapolates to "
                    f"{symplectic['extrapolated_relative_residual']:.4e}, i.e. to "
                    "float64 round-off"
                ),
                "location": "src/verification/families/b1_ray.py::_symplectic_invariant",
            },
        ],
    )
    measurements = {
        "symplectic_invariant_relative_residual": Measurement(
            value=symplectic["extrapolated_relative_residual"],
            uncertainty=max(
                symplectic["extrapolation_uncertainty"],
                symplectic["float64_eps"]
                * symplectic["max_conditioning"]
                * symplectic["richardson_roundoff_amplification"],
            ),
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                "Richardson-extrapolated to zero ray separation over "
                f"{len(SYMPLECTIC_SCALES)} halvings, two levels. The error bar is the "
                "larger of the extrapolation's own O(eps^6) remainder "
                f"({symplectic['extrapolation_uncertainty']:.3e}, estimated from the "
                "next column) and the float64 round-off floor of the extrapolation "
                "weights -- because below that floor the two are indistinguishable "
                "and quoting the smaller would claim a precision the arithmetic does "
                "not have"
            ),
        ),
        "lagrange_invariant_relative_drift": Measurement(
            value=finest["relative_drift"],
            uncertainty=abs(finest["relative_drift"] - rows[-2]["relative_drift"]),
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"at the finest rung of the field ladder, theta = "
                f"{finest['field_rad']:.3e} rad. The error bar is the change from the "
                "previous rung, which is the honest statement of how converged this "
                "is. Non-gating: this is the paraxial form on finite real rays and "
                "its residual is aberration"
            ),
        ),
    }
    controls = {
        "non-symplectic-momentum-scale": control_result(
            "non-symplectic-momentum-scale",
            "symplectic_invariant_relative_residual",
            baseline=measurements["symplectic_invariant_relative_residual"],
            mutated=Measurement(
                value=_symplectic_control(
                    surface_count=surface_count, p_scale=1.0 + NON_SYMPLECTIC_DELTA
                )["extrapolated_relative_residual"],
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note=(
                    f"image-plane canonical momenta scaled by 1 + {NON_SYMPLECTIC_DELTA:g} "
                    "with the positions untouched, so the tangent map's determinant "
                    "is no longer one. omega is bilinear in p, so the expected "
                    "residual is the scale factor itself"
                ),
            ),
            threshold=1e-13,
            note=(
                "a map that is not a canonical transformation, by a known amount, "
                "and the metric reads that amount back."
            ),
        ),
        "non-symplectic-position-scale": control_result(
            "non-symplectic-position-scale",
            "symplectic_invariant_relative_residual",
            baseline=measurements["symplectic_invariant_relative_residual"],
            mutated=Measurement(
                value=_symplectic_control(
                    surface_count=surface_count, q_scale=1.0 + NON_SYMPLECTIC_DELTA
                )["extrapolated_relative_residual"],
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note=(
                    "the conjugate half: image-plane positions scaled by "
                    f"1 + {NON_SYMPLECTIC_DELTA:g} with the momenta untouched. "
                    "Declared separately because a metric sensitive to one factor of "
                    "a bilinear form is not automatically sensitive to the other"
                ),
            ),
            threshold=1e-13,
            note="the other factor of the bilinear form, and it is not blind to it.",
        ),
        "degenerate-tangent-pair": control_result(
            "degenerate-tangent-pair",
            "symplectic_invariant_relative_residual",
            baseline=measurements["symplectic_invariant_relative_residual"],
            mutated=Measurement(
                value=_symplectic_control(surface_count=surface_count, degenerate=True)[
                    "extrapolated_relative_residual"
                ],
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note=(
                    "both secants drawn from the pupil fan. omega_object is "
                    "IDENTICALLY zero at every rung -- a collimated bundle gives "
                    "every ray one launch direction, so dp_object = 0 for both "
                    "secants -- while omega_image is not, so the relative residual "
                    "is infinite rather than merely large. This is the construction "
                    "error behind the earlier 'converges to 1 + 7.1e-3 and does not "
                    "approach 1'"
                ),
            ),
            threshold=1e-13,
            note=(
                "the oracle needs two LINEARLY INDEPENDENT tangent directions, and "
                "that requirement is executable rather than advisory."
            ),
        ),
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
            note=(
                "removing an index changes the system, and the paraxial invariant "
                "follows it. Pointed at the finite-ray metric deliberately: the "
                "symplectic gate is blind to the prescription by construction, and "
                "the measured inertness is in the PRESCRIPTION_MUTATION_IS_INERT "
                "diagnostic."
            ),
        ),
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


def _symplectic_residual_with_index(index: float, *, surface_count: int) -> dict[str, Any]:
    """The same ladder on a MUTATED prescription, which is the declared blind spot.

    The differential invariant is a property of the map and not of the system, so
    a different-but-valid stack is still symplectic and this is expected to be
    inert. It is measured rather than asserted, because "the gate cannot see X" is
    a claim like any other and an unmeasured one reads as an excuse.
    """
    return _symplectic_residual(surface_count=surface_count, second_index=index)


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
#
# Which array is the one to read is itself a place this went wrong, and it is
# worth stating because the failure was invisible on a CPU-only host. The `.npz`
# beside a record is a persistence copy and `np.savez` requires host bytes, so
# reading placement out of it reports numpy-on-CPU whatever executed. On the GPU
# runner that turned a genuine `cuda:0` float64 trace into `honoured_device:
# false` -- a real CUDA execution recorded as a downgrade, which is the same class
# of defect as a downgrade recorded as success, only in the other direction. The
# authoritative observation is `array_state(rays.x)` on the live traced tensor,
# which the adapter performs and stores at `record.metadata['execution']`; see
# `Trace.placement`.
#
# Executing the matrix is not the same as measuring agreement, and the second is
# what the acceptance criterion asks for. `device_precision_agreement` compares
# the traced arrays across configurations, keyed on what the arrays ACTUALLY are,
# with a tolerance derived from the coarser of the two precisions involved. In a
# CPU-only session the CUDA comparisons report `unavailable` and carry the
# refusal that made them unavailable; they are never reported as agreeing.

#: The device/dtype combinations the matrix requests, in the order it requests
#: them. The two CUDA rows through the torch backend are declared supported and
#: are expected to execute on a GPU runner and to refuse on a CPU-only one -- both
#: are results, and which one happened is read off the arrays afterwards.
DEVICE_PRECISION_REQUESTS: tuple[tuple[str, str, str], ...] = (
    ("numpy", "cpu", "float64"),
    ("numpy", "cpu", "float32"),
    ("torch", "cpu", "float64"),
    ("torch", "cpu", "float32"),
    # Declared supported through the torch backend only. On a CPU-only session
    # this refuses, and the refusal is the evidence.
    ("torch", "cuda", "float64"),
    ("torch", "cuda", "float32"),
    # Declared UNSUPPORTED: set_device raises BackendCapabilityError on the
    # numpy backend, so this is refused before Optiland is touched.
    ("numpy", "cuda", "float64"),
)

#: Ring count of the matrix traces. Small: this section is about placement and
#: precision, and the physics gates live in the families above.
DEVICE_PRECISION_RINGS = 16

#: The agreement comparisons, as ``(id, reference request, compared request,
#: class)``. Every class the acceptance criterion names is here, and the last
#: pair is what makes it a statement about both supported backends rather than
#: about one of them twice.
#: ``(comparison_id, reference request, compared request, class)``.
_Request = tuple[str, str, str]
DEVICE_PRECISION_COMPARISONS: tuple[tuple[str, _Request, _Request, str], ...] = (
    (
        "cpu_fp64_vs_cuda_fp64",
        ("torch", "cpu", "float64"),
        ("torch", "cuda", "float64"),
        "same_dtype_cross_device",
    ),
    (
        "cpu_fp32_vs_cuda_fp32",
        ("torch", "cpu", "float32"),
        ("torch", "cuda", "float32"),
        "same_dtype_cross_device",
    ),
    (
        "cpu_fp32_vs_cpu_fp64",
        ("torch", "cpu", "float64"),
        ("torch", "cpu", "float32"),
        "cross_dtype_same_device",
    ),
    (
        "cuda_fp32_vs_cuda_fp64",
        ("torch", "cuda", "float64"),
        ("torch", "cuda", "float32"),
        "cross_dtype_same_device",
    ),
    (
        "numpy_cpu_fp64_vs_torch_cpu_fp64",
        ("numpy", "cpu", "float64"),
        ("torch", "cpu", "float64"),
        "same_dtype_cross_backend",
    ),
)

#: Round-off constant of one trace, as a multiple of the dtype's epsilon. Not
#: chosen here: it is the same 64 the adapter derives for its own direction-norm
#: check (``solvers/optiland/execution.py::_direction_norm_tolerance``) from the
#: same argument -- a few operations per surface on quantities of order one -- and
#: reusing it is what keeps the agreement bound and the artifact boundary's bound
#: from drifting apart.
TRACE_ROUNDOFF_EPS_MULTIPLE = 64.0


def _dtype_epsilon(dtype: str) -> float:
    return float(np.finfo(np.dtype(dtype)).eps)


def _coarser_dtype(first: str, second: str) -> str:
    """The dtype whose round-off dominates a comparison between the two.

    A float32-vs-float64 comparison cannot be held to float64 round-off: the
    float32 arm's own floor is the bound, and pretending otherwise would report a
    precision cost as a correctness failure.
    """
    return "float32" if "float32" in (first, second) else first


def _agreement_tolerance(reference_dtype: str, compared_dtype: str) -> dict[str, Any]:
    """The bound for one comparison, derived from the coarser precision.

    Dimensionless throughout: direction cosines are unit-norm so their absolute
    error IS their relative error, and positions are normalized by the axial lever
    arm between the launch and image planes, which is the length that converts a
    direction error into a position error. Normalizing positions by their own
    magnitude instead would be meaningless near a focus, where the magnitude
    passes through zero.
    """
    dtype = _coarser_dtype(reference_dtype, compared_dtype)
    eps = _dtype_epsilon(dtype)
    return {
        "threshold": TRACE_ROUNDOFF_EPS_MULTIPLE * eps,
        "dominant_dtype": dtype,
        "basis": (
            f"{TRACE_ROUNDOFF_EPS_MULTIPLE:.0f} * eps({dtype}) = "
            f"{TRACE_ROUNDOFF_EPS_MULTIPLE * eps:.4e}. The two arms evaluate the same "
            "refraction arithmetic; what differs is the order of operations and the "
            "kernels, so the admissible difference is the round-off of one trace at "
            "the coarser of the two precisions. The multiple is the adapter's own "
            "derived constant for a trace of this depth (a few operations per surface "
            "on quantities of order one) rather than a number chosen here, and it is "
            "the coarser dtype because a float32 arm cannot be held to a float64 "
            "floor. Positions are compared against the axial lever arm between the "
            "launch and image planes, because that is the length an angular error is "
            "multiplied by."
        ),
    }


def _compared_quantities(
    reference: Trace, compared: Trace, *, efl_closed_mm: float
) -> dict[str, dict[str, float]]:
    """Three quantities, each with the value on both arms at the worst element.

    A max-error scalar with no accompanying pair of values cannot be re-checked,
    so each entry reports the reference value and the compared value AT the
    element where the error is largest, not summary statistics of two unrelated
    places.
    """
    if reference.arrays["x_m"].shape != compared.arrays["x_m"].shape:
        raise RuntimeError(
            "the two arms traced different ray counts "
            f"({reference.arrays['x_m'].shape} vs {compared.arrays['x_m'].shape}); "
            "an elementwise comparison would be comparing different rays"
        )
    pupil_gap = float(
        np.max(
            np.abs(
                reference.arrays["pupil_normalized_y"].astype(np.float64)
                - compared.arrays["pupil_normalized_y"].astype(np.float64)
            )
        )
    )
    if pupil_gap > 1e-6:
        raise RuntimeError(
            f"the two arms sampled different pupil coordinates (max gap {pupil_gap:.3e}), "
            "so row i of one is not row i of the other"
        )

    lever_arm_m = abs(
        float(reference.arrays["z_m"][0]) - float(reference.arrays["launch_z_m"][0])
    )

    def entry(name: str, scale: float, scale_note: str) -> dict[str, float]:
        a = reference.arrays[name].astype(np.float64)
        b = compared.arrays[name].astype(np.float64)
        errors = np.abs(a - b)
        worst = int(np.argmax(errors))
        absolute = float(errors[worst])
        return {
            "reference_value": float(a[worst]),
            "compared_value": float(b[worst]),
            "absolute_error": absolute,
            "normalized_error": absolute / scale,
            "normalization_scale": scale,
            "normalization": scale_note,
            "worst_element": worst,
        }

    def traced_efl_mm(trace: Trace) -> float:
        L, M, N = trace.direction
        heights = trace.launch_radius_mm
        keep = heights > 0.0
        focal = heights[keep] / (np.hypot(L[keep], M[keep]) / N[keep])
        _, innermost = trace.innermost(focal, heights[keep])
        return float(innermost)

    efl_reference = traced_efl_mm(reference)
    efl_compared = traced_efl_mm(compared)
    return {
        "image_plane_y_m": entry(
            "y_m", lever_arm_m, f"axial lever arm launch->image = {lever_arm_m:.6f} m"
        ),
        "image_plane_x_m": entry(
            "x_m", lever_arm_m, f"axial lever arm launch->image = {lever_arm_m:.6f} m"
        ),
        "direction_cosine_M": entry(
            "M", 1.0, "direction cosines are unit-norm, so absolute error is relative"
        ),
        "direction_cosine_N": entry(
            "N", 1.0, "direction cosines are unit-norm, so absolute error is relative"
        ),
        "traced_efl_mm": {
            "reference_value": efl_reference,
            "compared_value": efl_compared,
            "absolute_error": abs(efl_reference - efl_compared),
            "normalized_error": abs(efl_reference - efl_compared) / abs(efl_closed_mm),
            "normalization_scale": abs(efl_closed_mm),
            "normalization": (
                "the closed-form EFL R/(n-1), so this is a genuine relative error on "
                "a physical scalar rather than a difference of two large numbers"
            ),
            "worst_element": -1,
        },
    }


def device_precision_matrix() -> dict[str, Any]:
    """Trace the reference singlet across every supported (backend, device, dtype).

    Reports one row per combination: what was requested, what Optiland was
    actually set to, what the arrays came back as, and whether the request was
    honoured. A combination the capability table refuses appears as a refusal row
    with its code and its supported set -- not as a missing row and not as a skip.

    The traced arrays are retained on each executed row so that
    :func:`device_precision_agreement` can compare them. Nothing here compares
    them: this function establishes *where each run happened*, and agreement is a
    separate question with a separate tolerance.
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

    rows: list[dict[str, Any]] = []
    traces: dict[tuple[str, str, str], Trace] = {}
    for backend, device, dtype in DEVICE_PRECISION_REQUESTS:
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

        # OBSERVED, from the live traced tensor. `trace.placement` reads
        # `record.metadata['execution']`, which the adapter fills from
        # `array_state(rays.x)` before anything is written to disk.
        placement = trace.placement
        execution = dict((run.diagnostics or {}).get("execution", {}))
        row |= {
            "outcome": "executed",
            "observed": placement,
            # The persistence copy, kept alongside so the two can be contrasted
            # rather than confused. Its dtype must match; its device says nothing.
            "persisted": trace.persisted_placement,
            # What Optiland was actually told, read back out of Optiland's own
            # getters rather than echoed from the request.
            "applied_to_optiland": execution.get("applied_to_optiland"),
            "adapter_observed_actual": execution.get("actual"),
            "adapter_reported_mismatches": list(execution.get("mismatches") or ()),
            "honoured_device": placement["device"].split(":")[0] == device.split(":")[0],
            "honoured_dtype": dtype in placement["dtype"],
            "record_device": str(trace.record.device),
            "record_framework": str(trace.record.framework),
            "record_dtype": str(trace.record.dtype),
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
        traces[(backend, device, dtype)] = trace

    return {
        "efl_closed_form_mm": efl_closed,
        "pupil_rings": DEVICE_PRECISION_RINGS,
        "rows": tuple(rows),
        "executed": tuple(r for r in rows if r["outcome"] == "executed"),
        "refused": tuple(r for r in rows if r["outcome"] == "refused"),
        "declared_but_failed": tuple(
            r for r in rows if r["outcome"] == "declared_but_failed"
        ),
        "traces": traces,
        "cuda_executed": any(
            r["outcome"] == "executed" and r["observed"]["device"].startswith("cuda")
            for r in rows
        ),
    }


def device_precision_agreement(matrix: dict[str, Any] | None = None) -> dict[str, Any]:
    """Numerical agreement across device and precision, measured from the arrays.

    One entry per comparison class. An entry is either ``measured`` -- with both
    arms' requested and actual placement, the reference and compared value of each
    quantity at its worst element, the absolute and normalized error, the derived
    tolerance and the verdict -- or ``unavailable``, carrying the outcome of
    whichever arm did not execute. There is no third state, and in particular
    there is no state in which a CUDA comparison reports agreement without a CUDA
    execution behind it: the actual device of each arm is asserted against the
    comparison's own name before any number is computed.
    """
    matrix = matrix if matrix is not None else device_precision_matrix()
    traces: dict[tuple[str, str, str], Trace] = matrix["traces"]
    by_request = {
        (r["requested"]["backend"], r["requested"]["device"], r["requested"]["dtype"]): r
        for r in matrix["rows"]
    }
    efl_closed = float(matrix["efl_closed_form_mm"])

    def named(request: _Request) -> dict[str, str]:
        return dict(zip(("backend", "device", "dtype"), request, strict=True))

    comparisons: list[dict[str, Any]] = []
    for comparison_id, reference_key, compared_key, kind in DEVICE_PRECISION_COMPARISONS:
        entry: dict[str, Any] = {
            "comparison_id": comparison_id,
            "class": kind,
            "requested_reference": named(reference_key),
            "requested_compared": named(compared_key),
        }
        missing = [key for key in (reference_key, compared_key) if key not in traces]
        if missing:
            entry |= {
                "status": "unavailable",
                "unavailable_because": [
                    {
                        "requested": named(key),
                        "outcome": by_request[key]["outcome"],
                        "code": by_request[key].get("code"),
                        "detail": by_request[key].get("detail"),
                    }
                    for key in missing
                ],
                "note": (
                    "not measured in this environment. The arm that did not execute "
                    "is named with the outcome that stopped it, so this reads as an "
                    "absence rather than as agreement."
                ),
            }
            comparisons.append(entry)
            continue

        reference, compared = traces[reference_key], traces[compared_key]
        actual_reference, actual_compared = reference.placement, compared.placement
        # The guard the criterion asks for in as many words: a CUDA arm must have
        # actually been on CUDA, read off the arrays, or this comparison is not
        # the comparison it is named after and must not be reported as measured.
        for label, key, actual in (
            ("reference", reference_key, actual_reference),
            ("compared", compared_key, actual_compared),
        ):
            if actual["device"].split(":")[0] != key[1]:
                raise RuntimeError(
                    f"{comparison_id}: the {label} arm requested {key[1]} and the "
                    f"arrays came back on {actual['device']}. A comparison named for a "
                    "device it did not run on would be a fabricated measurement, so "
                    "this raises instead of recording a number."
                )
            if key[2] not in actual["dtype"]:
                raise RuntimeError(
                    f"{comparison_id}: the {label} arm requested {key[2]} and the "
                    f"arrays came back as {actual['dtype']}."
                )

        tolerance = _agreement_tolerance(actual_reference["dtype"], actual_compared["dtype"])
        quantities = _compared_quantities(reference, compared, efl_closed_mm=efl_closed)
        worst = max(q["normalized_error"] for q in quantities.values())
        entry |= {
            "status": "measured",
            "actual_reference": actual_reference,
            "actual_compared": actual_compared,
            "tolerance": tolerance,
            "quantities": quantities,
            "worst_normalized_error": worst,
            "met": worst <= tolerance["threshold"],
            "identical": worst == 0.0,
        }
        comparisons.append(entry)

    measured = [c for c in comparisons if c["status"] == "measured"]
    return {
        "comparisons": tuple(comparisons),
        "measured": tuple(measured),
        "unavailable": tuple(c for c in comparisons if c["status"] == "unavailable"),
        "all_measured_met": all(c["met"] for c in measured),
        "cuda_comparisons_measured": tuple(
            c["comparison_id"]
            for c in measured
            if "cuda" in c["actual_reference"]["device"]
            or "cuda" in c["actual_compared"]["device"]
        ),
    }


def write_device_precision_record(directory: Path | None = None) -> Path:
    """Persist the matrix and the agreement, with provenance, as evidence.

    A number that exists only in a CI log is not evidence anybody can re-check,
    which is why this writes to ``benchmarks/probes/records/`` and stamps itself
    through ``core.provenance.record_provenance``: the code half of that stamp is
    re-verified against the tree on every default-gate run by
    ``tests/test_provenance_fingerprint.py``, so a later change to the adapter or
    to this driver makes the record stale rather than leaving it silently wrong.

    The record states which environment produced it. A CPU-only run writes the
    CUDA comparisons as ``unavailable`` with their refusals, and a GPU run
    replaces them with measurements; the two are distinguishable at a glance from
    ``environment.cuda_executed``.
    """
    from core.provenance import record_provenance

    matrix = device_precision_matrix()
    agreement = device_precision_agreement(matrix)
    directory = directory or (ROOT / "benchmarks" / "probes" / "records" / "optiland")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "b1_ray_device_precision.json"

    payload: dict[str, Any] = {
        "probe": "instances/b1_ray::write_device_precision_record",
        "question": (
            "does the Optiland ray trace agree across device and precision, and is "
            "the device and precision of every arm read off the arrays it produced?"
        ),
        "system": {
            "prescription": "B1-RAY-EFL-01 plano-convex singlet in air",
            "efl_closed_form_mm": matrix["efl_closed_form_mm"],
            "pupil_rings": matrix["pupil_rings"],
            "wavelength_um": WAVELENGTH_UM,
        },
        "measurement_method": (
            "one trace per (backend, device, dtype); placement read from "
            "core.arrays.array_state on the live traced tensor via "
            "record.metadata['execution']; agreement computed elementwise on the "
            "traced arrays after asserting the two arms sampled the same pupil "
            "coordinates; tolerance derived from 64 * eps of the coarser dtype."
        ),
        "environment": {
            "cuda_executed": matrix["cuda_executed"],
            "cuda_unavailable_reason": (
                None
                if matrix["cuda_executed"]
                else next(
                    (
                        r.get("detail")
                        for r in matrix["rows"]
                        if r["requested"]["device"] == "cuda" and r["outcome"] == "refused"
                    ),
                    "no CUDA row refused and none executed",
                )
            ),
            "torch": _torch_build(),
            "optiland_version": _optiland_version(),
        },
        "rows": [
            {key: value for key, value in row.items() if key != "diagnostics"}
            for row in matrix["rows"]
        ],
        "agreement": {
            "comparisons": list(agreement["comparisons"]),
            "all_measured_met": agreement["all_measured_met"],
            "cuda_comparisons_measured": list(agreement["cuda_comparisons_measured"]),
        },
    }
    payload["record_provenance"] = record_provenance(
        probe="instances/b1_ray::write_device_precision_record", root=ROOT
    )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _torch_build() -> dict[str, Any]:
    """Which torch is installed, and whether it can see a device.

    Part of the record's provenance in the sense that matters here: the default
    image ships ``2.13.0+cpu`` and the GPU image ``2.13.0+cu126``, and that is
    the difference between a CUDA row that refuses and one that executes.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is pinned in both images
        return {"available": False}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_toolkit": torch.version.cuda,
        "cuda_is_available": bool(torch.cuda.is_available()),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def _optiland_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("optiland")
    except Exception:  # pragma: no cover - defensive
        return None


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
                "num_rays": DEVICE_PRECISION_RINGS,
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
    parser.add_argument(
        "--device-matrix",
        action="store_true",
        help=(
            "run only the device/precision matrix and agreement, and persist it. This "
            "is the GPU entry point: `MOA_GPUS=device=6 ./run.sh --gpu python "
            "benchmarks/instances/b1_ray.py --device-matrix`."
        ),
    )
    args = parser.parse_args()

    if args.device_matrix:
        matrix = device_precision_matrix()
        agreement = device_precision_agreement(matrix)
        for row in matrix["rows"]:
            requested = row["requested"]
            label = f"{requested['backend']}/{requested['device']}/{requested['dtype']}"
            if row["outcome"] == "executed":
                observed = row["observed"]
                print(
                    f"{label:<28} executed  observed="
                    f"{observed['namespace']}/{observed['device']}/{observed['dtype']}"
                    f"  honoured={row['honoured_device'] and row['honoured_dtype']}"
                )
            else:
                print(f"{label:<28} {row['outcome']:<9} {row.get('code') or row.get('error_type')}")
        print()
        for comparison in agreement["comparisons"]:
            if comparison["status"] != "measured":
                arms = ", ".join(
                    f"{a['requested']['device']}/{a['requested']['dtype']}:{a['outcome']}"
                    for a in comparison["unavailable_because"]
                )
                print(f"{comparison['comparison_id']:<34} UNAVAILABLE  ({arms})")
                continue
            print(
                f"{comparison['comparison_id']:<34} "
                f"worst={comparison['worst_normalized_error']:.4e} "
                f"tol={comparison['tolerance']['threshold']:.4e} "
                f"met={comparison['met']}"
            )
        path = write_device_precision_record()
        print(f"\n-> {path.relative_to(ROOT)}")
        return 0

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
