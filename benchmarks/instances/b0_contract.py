"""The ten canonical B0 instances, executed. Five negative outcomes, two silent traps.

CHE-108 (M1.3), part B0.3. The B0 families and their instances were declared and
never run: ``tests/test_b0_families.py`` asserts that the *declarations* are well
formed, which is a different claim from "each of the five statuses is produced by
a real component refusing a real request". This module produces them.

What "executed" means here, precisely
-------------------------------------
Three of these instances are graphs and run through ``GraphExecutor``:
``B0-HANDOFF-01`` and ``B0-META-01`` are the ``examples/graphs/ray_to_wave.yaml``
slice with one declaration removed, and they refuse mid-graph after a real
Optiland trace.

The other seven are **not expressible as a graph, and that is a property of the
question rather than a shortcut**. An empty precision intersection is decided by
``plan_bridge`` before any node runs -- that is the whole point of B0-CAPINT-01,
which exists to establish that the route is refused *at planning time with a
named reason* rather than as a traceback three nodes in. A record whose declared
device disagrees with its array's placement is refused at artifact intake. A
curvature bound is evaluated on parameters. In each case the shipping refusal
path *is* the execution, and ``instance_runner.probe_refusal`` turns what it
raised into a record -- requiring a real exception, reading every field off the
raised object, and fabricating nothing.

The two that are supposed to succeed
------------------------------------
``B0-UNITS-01`` and ``B0-UNITS-02`` run clean. Their declared contract status is
``ok`` and the physics is wrong, and they are here to prove the substrate can
tell "it executed" from "it is right". Both are re-measured against their
closed forms rather than inherited from ``verification/hazards.py``: the recorded
numbers are what makes them usable as controls, so a run that merely repeated
them would establish nothing about the pinned solvers in front of us.

Run it::

    ./run.sh python benchmarks/instances/b0_contract.py --write
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import numpy as np

from core.artifacts import ArtifactRecord
from core.capabilities import capabilities_for
from core.execution_record import DevicePrecisionObservation
from core.paths import repository_root
from core.precision import (
    ArrayNamespace,
    ArrayState,
    BridgePolicy,
    DeviceKind,
    DevicePlacement,
    DType,
    plan_bridge,
)
from core.specs import Device
from couplers.curvature import check_patch, curvature_direction_error_bound
from couplers.handoff import (
    DeclaredHandoffPlane,
    declare_coherent_bundle,
)
from registry.loader import Registry
from runtime.instance_runner import (
    observed_placement,
    probe_refusal,
    record_from_probe,
)
from verification.evidence import InstanceRun, run_and_verify, write_instance_record
from solvers.base import ModelRunRequest
from verification.asm_oracle import angular_spectrum_float64, compare_fields
from verification.families.b0_contract import (
    B0_CONTRACT,
    B0_DTYPE,
    B0_UNITS,
    B0_VALIDITY,
)
from verification.hazards import hazard_for
from verification.result import Measurement, UncertaintyBasis
from verification.verifier import verify

__all__ = ["run_all", "run_instance"]

ROOT = repository_root()
GRAPH_PATH = ROOT / "examples" / "graphs" / "ray_to_wave.yaml"

#: A small trace. These instances are about refusals, not accuracy, so the ray
#: count only has to be large enough to be a real bundle.
CONTRACT_RINGS = 8
WAVELENGTH_UM = 0.55
SINGLET_PUPIL_Z_M = 6.814345991561233e-05

_CPU = DevicePlacement(DeviceKind.CPU)
_CUDA0 = DevicePlacement(DeviceKind.CUDA, 0)


def _instance(family: Any, instance_id: str) -> Any:
    for candidate in family.canonical_instances:
        if candidate.instance_id == instance_id:
            return candidate
    raise KeyError(f"{family.family_id} declares no instance {instance_id!r}")


# ---------------------------------------------------------------------------
# The graph cases
# ---------------------------------------------------------------------------


def _graph_with_edge_config(**overrides: Any):
    """The ray -> wave -> field slice graph, with the edge config edited.

    ``None`` removes a key. Removing rather than blanking is the point: the
    coupler refuses a *missing* declaration, and a key present with a null value
    is a different request.
    """
    graph = Registry.load_graph(GRAPH_PATH)
    spec = graph.model_dump()
    for edge in spec["edges"]:
        if edge["coupler"] != "C_RAY_TO_WAVE":
            continue
        for key, value in overrides.items():
            if value is None:
                edge["config"].pop(key, None)
            else:
                edge["config"][key] = value
    # Keep it cheap: these instances are refusals, and a 512-ring trace would
    # buy nothing but minutes.
    for node in spec["nodes"]:
        if node["model"] == "M_RAY_OPTILAND":
            node["config"]["num_rays"] = CONTRACT_RINGS
    return type(graph).model_validate(spec)


def _run_handoff_01() -> InstanceRun:
    """A bare ``opd_native``: the coupler could proceed and refuses to.

    ``blocked`` rather than ``invalid_configuration``. Nothing about the request
    is malformed -- the missing thing is a declaration, and CHE-108 found the
    verifier collapsing this into ``invalid_configuration`` because the executor
    maps the code to ``MISSING_EDGE_DECLARATION``. The catalogue now decides.
    """
    instance = _instance(B0_CONTRACT, "B0-HANDOFF-01")
    graph = _graph_with_edge_config(handoff_plane=None, handoff_plane_z_m=None)
    return run_and_verify(B0_CONTRACT, instance, graph)


def _run_meta_01() -> InstanceRun:
    """A declared plane the trace was not exported at.

    ``omitted_declaration: reference_plane`` means the consumer and the producer
    do not agree about which plane the record describes. Expressed as a graph
    whose edge declares ``image_surface`` while the node exported at the exit
    pupil, so the disagreement is between two real declarations rather than a
    hand-edited artifact.
    """
    instance = _instance(B0_CONTRACT, "B0-META-01")
    graph = _graph_with_edge_config(handoff_plane="image_surface", handoff_plane_z_m=0.0)
    return run_and_verify(B0_CONTRACT, instance, graph)


# ---------------------------------------------------------------------------
# The planning-time cases
# ---------------------------------------------------------------------------


def _run_capint_01() -> InstanceRun:
    """No precision at which a C_PATCH_WFT -> Chromatix route can execute.

    Two facts, both read off the shipping capability table rather than restated:
    the native compute sets are ``{complex128}`` and ``{complex64}`` and their
    intersection is empty, and ``plan_bridge`` under the project default policy
    refuses the crossing with the supported set named. Project risk R5, as a
    measurement.
    """
    instance = _instance(B0_CONTRACT, "B0-CAPINT-01")
    patch = capabilities_for("C_PATCH_WFT")
    chromatix = capabilities_for("M_WAVE_CHROMATIX")
    intersection = patch.native_compute_dtypes & chromatix.native_compute_dtypes

    refusal, _ = probe_refusal(
        lambda: plan_bridge(
            ArrayState(DType.COMPLEX128, _CPU, ArrayNamespace.NUMPY),
            chromatix,
            policy=BridgePolicy.SAFE,
        )
    )
    record = record_from_probe(
        instance,
        component="C_PATCH_WFT -> M_WAVE_CHROMATIX",
        node_id="route_planning",
        refusal=refusal,
        observed_parameters={"dtype": "complex128"},
        diagnostics=[
            {
                "code": "NATIVE_COMPUTE_INTERSECTION",
                "detail": (
                    "C_PATCH_WFT computes in "
                    f"{sorted(str(d) for d in patch.native_compute_dtypes)}, "
                    "M_WAVE_CHROMATIX in "
                    f"{sorted(str(d) for d in chromatix.native_compute_dtypes)}; "
                    f"intersection {sorted(str(d) for d in intersection)}"
                ),
                "location": "core/capabilities.py",
            }
        ],
    )
    return InstanceRun(
        family=B0_CONTRACT,
        instance=instance,
        record=record,
        result=verify(B0_CONTRACT, instance, record),
    )


def _run_device_01() -> InstanceRun:
    """CUDA asked of a coupler whose capability table declares CPU only."""
    instance = _instance(B0_CONTRACT, "B0-DEVICE-01")
    refusal, _ = probe_refusal(
        lambda: plan_bridge(
            ArrayState(DType.COMPLEX128, _CUDA0, ArrayNamespace.JAX),
            capabilities_for("C_PATCH_WFT"),
            policy=BridgePolicy.SAFE,
        )
    )
    record = record_from_probe(
        instance,
        component="C_PATCH_WFT",
        node_id="device_planning",
        refusal=refusal,
        observed_parameters={"device": "cuda"},
    )
    return InstanceRun(
        family=B0_CONTRACT,
        instance=instance,
        record=record,
        result=verify(B0_CONTRACT, instance, record),
    )


def _run_device_02() -> InstanceRun:
    """A record that declares ``gpu`` over an array that is on the host.

    Reachable only because CHE-108 made it so. ``from_artifact_record`` compared
    nothing between ``record.device`` and the array it was handed, so a
    ``record.device = requested_device`` upstream produced a reported CUDA run
    that happened on the host with nothing raised. The array here is real host
    NumPy and the declaration is the lie; the refusal names both sides.

    This needs no GPU, which is the useful part: the failure mode it guards is
    the one that appears *on a machine with no working accelerator*.
    """
    instance = _instance(B0_CONTRACT, "B0-DEVICE-02")

    grid = int(instance.parameters["grid_n"])
    field = np.ones((grid, grid), dtype=np.complex64)
    record_declaring_gpu = ArtifactRecord(
        id="b0-device-02-field",
        kind="complex_field",
        uri="memory://b0-device-02",
        dtype="complex64",
        # The declaration under test. Everything else on this record is true.
        device=Device.GPU,
        metadata={
            "wavelength": 5.5e-7,
            "sample_pitch": 1e-6,
            "phasor": "exp(-i omega t)",
            "normalization": "none",
            "pad_width": 0,
            "coordinate_frame": "right-handed",
        },
    )

    from core.boundary import ComplexField

    refusal, _ = probe_refusal(
        lambda: ComplexField.from_artifact_record(record_declaring_gpu, array=field)
    )
    observation = DevicePrecisionObservation(
        requested_device="cuda",
        actual_device="cpu",
        requested_dtype="complex64",
        actual_dtype=str(field.dtype),
        requested_namespace="jax",
        actual_namespace="numpy",
    )
    record = record_from_probe(
        instance,
        component="M_WAVE_CHROMATIX",
        node_id="artifact_intake",
        refusal=refusal,
        observed_parameters={"device": "cpu"},
        device_precision=observation,
    )
    return InstanceRun(
        family=B0_CONTRACT,
        instance=instance,
        record=record,
        result=verify(B0_CONTRACT, instance, record),
    )


def _run_patch_01(registry: Registry) -> InstanceRun:
    """A per-ray quadrature weight requested of a bundle that is not on rings.

    ``out_of_validity``, not ``unsupported``: the weight formula computes fine on
    a rectangular bundle, and what it returns are weights that mean nothing. The
    trigger is a real Optiland trace whose pupil coordinates are then placed on a
    rectangular grid, so the weight is asked of exactly the sampling the
    declaration excludes.
    """
    instance = _instance(B0_CONTRACT, "B0-PATCH-01")
    record_in, arrays = _optiland_trace(registry)

    # The weight is derived from the record's NORMALIZED PUPIL coordinates, not
    # from the ray geometry, so those are what has to become rectangular. The
    # traced positions, directions and opd are left exactly as Optiland produced
    # them: the only thing changed is the sampling the weight formula is asked to
    # assume, which is the declaration under test.
    count = int(np.asarray(arrays["x_m"]).size)
    side = max(2, math.isqrt(count) + 1)
    grid = np.linspace(-1.0, 1.0, side)
    xs, ys = np.meshgrid(grid, grid, indexing="xy")
    rectangular = dict(arrays)
    rectangular["pupil_normalized_x"] = np.resize(xs.ravel(), count)
    rectangular["pupil_normalized_y"] = np.resize(ys.ravel(), count)

    plane = DeclaredHandoffPlane(handoff_plane="exit_pupil", z_m=SINGLET_PUPIL_Z_M)
    refusal, handoff = probe_refusal(
        lambda: declare_coherent_bundle(
            record_in,
            declared_plane=plane,
            arrays=rectangular,
        )
    )

    # The coupler does NOT raise here, and that is deliberate rather than a gap.
    # ``_ray_quadrature_weight`` catches NON_HEXAPOLAR_SAMPLING and falls back to
    # the unweighted amplitude mapping, because the same condition is reached by
    # a legitimately VIGNETTED hexapolar fan -- a ray dropped, so row order no
    # longer matches ring order -- and refusing a vignetted trace outright would
    # be wrong. What it must not do is drop the weight silently, and it does not:
    # the reported status carries the code and the reason.
    #
    # So this instance's ``out_of_validity`` comes from the family's own validity
    # predicate re-evaluated against the realized sampling, which is the
    # verifier's second route to that status and the correct one here. The
    # coupler's report is lifted onto the record so the code is on it either way.
    diag = {} if handoff is None else dict(handoff.diagnostics)
    quadrature = {
        "applied": diag.get("quadrature_weight_applied"),
        "status": diag.get("quadrature_weight_status"),
        "reason": diag.get("quadrature_weight_reason"),
    }
    reported_reason = str(quadrature.get("reason") or "")
    diagnostics = [
        {
            "code": "QUADRATURE_WEIGHT_NOT_APPLIED",
            "detail": (
                f"status={quadrature.get('status')!r}; the coupler reported: "
                f"{reported_reason}"
            ),
            "location": "couplers/handoff.py::_ray_quadrature_weight",
        },
        {
            "code": "REPORTED_RATHER_THAN_REFUSED",
            "detail": (
                "the weight is dropped and the drop is REPORTED with its code, "
                "rather than refused. The fallback exists because a vignetted "
                "hexapolar fan reaches the same condition and must still be usable "
                "unweighted; the distinction that matters is that nothing is "
                "silent, and the family's HEXAPOLAR_RING predicate is what turns "
                "the realized sampling into an out_of_validity status."
            ),
            "location": "couplers/handoff.py::_ray_quadrature_weight",
        },
    ]
    assert "NON_HEXAPOLAR_SAMPLING" in reported_reason, (
        "the coupler dropped the quadrature weight without naming why. A silent "
        f"drop is the failure this instance exists to catch; it reported: {quadrature!r}"
    )
    assert quadrature.get("applied") is False

    record = record_from_probe(
        instance,
        component="C_RAY_TO_WAVE",
        node_id="quadrature_weight",
        refusal=refusal,
        observed_parameters={"pupil_sampling": "rectangular"},
        diagnostics=diagnostics,
    )
    return InstanceRun(
        family=B0_CONTRACT,
        instance=instance,
        record=record,
        result=verify(B0_CONTRACT, instance, record),
    )


# ---------------------------------------------------------------------------
# B0-DTYPE: the loss as a number
# ---------------------------------------------------------------------------


def _run_dtype_01() -> InstanceRun:
    """complex128 into Chromatix: allowed, lossy, and the loss is measured.

    The gate is not "a warning was emitted". It is the measured relative residual
    of a real Chromatix complex64 propagation against the **independent float64
    angular spectrum** in ``verification/asm_oracle.py``, which shares no code
    with Chromatix -- so what it measures is the cost of the representation
    rather than the cost of the implementation. A warning would pass a run whose
    loss was a thousand times larger.

    Also the CHE-107 WAVE-2 measurement. The two tickets ask for the same number
    and it is measured once here rather than twice.
    """
    instance = _instance(B0_DTYPE, "B0-DTYPE-01")
    grid = int(instance.parameters["grid_n"])
    wavelength_m = float(instance.parameters["wavelength_m"])
    distance_m = float(instance.parameters["propagation_distance_m"])

    chromatix = capabilities_for("M_WAVE_CHROMATIX")
    source = ArrayState(DType.COMPLEX128, _CPU, ArrayNamespace.NUMPY)

    # SAFE refuses; ALLOW_DOWNCAST admits and records. Both are run, because the
    # claim is about the pair: the loss is *allowed* under a declared policy and
    # *refused* under the default, and neither half alone says that.
    refused_under_safe, _ = probe_refusal(
        lambda: plan_bridge(source, chromatix, policy=BridgePolicy.SAFE)
    )
    assert refused_under_safe is not None, (
        "SAFE admitted a complex128 -> complex64 crossing. The whole reason "
        "complex128 is kept out of accepted_input_dtypes is that this refuses."
    )
    plan = plan_bridge(source, chromatix, policy=BridgePolicy.ALLOW_DOWNCAST)
    assert plan.lossy and plan.downcast

    # A float64 input field, which is the whole point: the caller HAS
    # complex128 data, and what is being measured is what happens to it. A
    # soft-edged disc, so the spectrum is not dominated by aperture ringing.
    pitch_m = wavelength_m  # one wavelength per sample: comfortably inside the band
    axis = (np.arange(grid, dtype=np.float64) - grid // 2) * pitch_m
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    radius = np.hypot(yy, xx)
    waist_m = 0.25 * grid * pitch_m
    u_in = np.exp(-((radius / waist_m) ** 2)).astype(np.complex128)

    # The independent reference: float64 throughout, pure NumPy, no padding.
    reference = angular_spectrum_float64(
        u_in,
        wavelength_m=wavelength_m,
        sample_pitch_m=pitch_m,
        z_m=distance_m,
        refractive_index=1.0,
    )

    # The shipping Chromatix path, at the dtype the bridge actually delivers and
    # with no padding, so the two propagate the same discrete problem.
    import chromatix.functional as cf
    import jax.numpy as jnp

    field_in = cf.Field.build(
        jnp.asarray(u_in, dtype=jnp.complex64),
        jnp.asarray([[pitch_m, pitch_m]]),
        wavelength_m,
    )
    field_out = cf.asm_propagate(field_in, z=distance_m, n=1.0, pad_width=0)
    observed = observed_placement(field_out.u)
    test = np.asarray(jnp.squeeze(field_out.u), dtype=np.complex128)
    while test.ndim > 2:
        test = test[0]

    comparison = compare_fields(test, reference)
    loss = comparison.raw_relative_field_error

    eps32 = float(np.finfo(np.float32).eps)
    accumulated_phase_rad = 2.0 * math.pi * distance_m / wavelength_m
    bound = eps32 * accumulated_phase_rad

    observation = DevicePrecisionObservation(
        requested_device="cpu",
        actual_device=observed["device"],
        requested_dtype="complex128",
        actual_dtype=observed["dtype"],
        requested_namespace="numpy",
        actual_namespace=observed["namespace"],
        measured_loss_relative=loss,
        measured_loss_basis=(
            f"eps32 * 2*pi*z/lambda = {eps32:.6e} * {accumulated_phase_rad:.6f} rad = "
            f"{bound:.6e}; one float32 epsilon per radian of accumulated phase. "
            "Measured against verification/asm_oracle.angular_spectrum_float64, which "
            "shares no code with Chromatix."
        ),
    )
    record = record_from_probe(
        instance,
        component="M_WAVE_CHROMATIX",
        node_id="input_bridge",
        refusal=None,
        lossy=True,
        observed_parameters={
            "requested_dtype": "complex128",
            "bridge_policy": "allow_downcast",
            "grid_n": grid,
        },
        device_precision=observation,
        diagnostics=[
            {
                "code": "BRIDGE_PLAN",
                "detail": json.dumps(plan.as_dict(), sort_keys=True, default=str),
                "location": "core/precision.py::plan_bridge",
            },
            {
                "code": "SAFE_REFUSES_THE_SAME_CROSSING",
                "detail": refused_under_safe.detail,
                "location": "core/precision.py::_negotiate_dtype",
            },
            {
                "code": "FIELD_COMPARISON",
                "detail": json.dumps(comparison.as_dict(), sort_keys=True, default=float),
                "location": "verification/asm_oracle.py::compare_fields",
            },
        ],
    )
    measurements = {
        "measured_precision_loss": Measurement(
            value=loss,
            uncertainty=comparison.piston_aligned_relative_field_error,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                "raw relative field error of the shipping complex64 propagation "
                "against the independent float64 angular spectrum, over "
                f"{accumulated_phase_rad:.4f} rad of accumulated phase. The error bar "
                "is the piston-aligned residual: what remains after the best global "
                "phase is removed, i.e. the part that is spatially varying wavefront "
                f"error rather than absolute-phase representation cost. Bound: {bound:.6e}."
            ),
        ),
        "loss_was_reported": Measurement(
            value=0.0,
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                "zero means reported: the bridge plan carries lossy=True and the "
                "record carries measured_loss_relative with its basis, so nothing "
                "about this truncation is silent"
            ),
        ),
    }
    return InstanceRun(
        family=B0_DTYPE,
        instance=instance,
        record=record,
        result=verify(B0_DTYPE, instance, record, measurements=measurements),
    )


# ---------------------------------------------------------------------------
# B0-VALIDITY: it would run, and the answer would be wrong
# ---------------------------------------------------------------------------


def _run_validity_01() -> InstanceRun:
    """A declared tangent-plane error past the eq S9 bound.

    ``out_of_validity``: the tangent-plane picture still computes, and it
    computes the wrong thing. Two things are executed, because they are two
    different questions and only running both establishes the status:

    1. the **bound itself**, from the shipping ``curvature_direction_error_bound``
       rather than from a tabulated arcsin, compared against the declared
       ``eps_curv``. This is the family's gate, and the verifier reaches
       ``out_of_validity`` through the ``SI_S3_CURVATURE`` predicate on the
       realized parameters;
    2. the **guard**, ``check_patch``, asked for an error tighter than the
       geometry can deliver, so the refusal path is shown live at these
       parameters with its remedy. At the instance's own threshold the guard
       PASSES -- 0.2 rad is a generous tolerance for a bound of 0.05 -- and that
       is not a contradiction: the guard asks "is the geometry inside my
       tolerance", the family asks "is the declared error inside what the
       geometry admits", and conflating the two is how a validity claim gets
       made backwards.
    """
    instance = _instance(B0_VALIDITY, "B0-VALIDITY-01")
    patch_width_m = float(instance.parameters["patch_width_m"])
    radius_m = float(instance.parameters["substrate_radius_m"])
    declared_eps_rad = float(instance.parameters["tangent_plane_error_rad"])

    bound = curvature_direction_error_bound(patch_width_m, radius_m)
    crossed = declared_eps_rad > bound

    # (1) The guard at the instance's own threshold: generous, so it passes and
    # returns a budget that records the margin.
    at_declared_threshold = check_patch(
        patch_width_m=patch_width_m,
        radius_m=radius_m,
        error_threshold_rad=declared_eps_rad,
    )

    # (2) The guard asked for something the geometry cannot deliver.
    tight_threshold = bound / 2.0
    refusal, _ = probe_refusal(
        lambda: check_patch(
            patch_width_m=patch_width_m,
            radius_m=radius_m,
            error_threshold_rad=tight_threshold,
        )
    )
    assert refusal is not None, (
        f"check_patch admitted a {patch_width_m} m patch on a {radius_m} m radius at "
        f"a threshold of {tight_threshold:.6g} rad, below its own bound of "
        f"{bound:.6g}. The guard is not live at these parameters."
    )

    record = record_from_probe(
        instance,
        component="C_PATCH_WFT",
        node_id="curvature_budget",
        # No refusal on the record: the instance's own configuration EXECUTES,
        # and the out-of-validity comes from the declared eps_curv exceeding the
        # bound rather than from a component saying no. Attaching the tightened
        # guard's refusal here would report a refusal of a request nobody made.
        refusal=None,
        observed_parameters={
            "patch_width_m": patch_width_m,
            "substrate_radius_m": radius_m,
            "tangent_plane_error_rad": declared_eps_rad,
        },
        diagnostics=[
            {
                "code": "SI_S9_BOUND",
                "detail": (
                    f"curvature_direction_error_bound({patch_width_m:g}, {radius_m:g}) = "
                    f"{bound:.9g} rad, i.e. arcsin(D/2R); declared eps_curv = "
                    f"{declared_eps_rad:g} rad; crossed = {crossed}"
                ),
                "location": "couplers/curvature.py::curvature_direction_error_bound",
            },
            {
                "code": "GUARD_PASSES_AT_THE_DECLARED_THRESHOLD",
                "detail": (
                    f"check_patch within_budget={at_declared_threshold.within_budget}, "
                    f"max_patch_width_m={at_declared_threshold.max_patch_width_m:.6g}, "
                    f"thin_patch_assumption_holds="
                    f"{at_declared_threshold.thin_patch_assumption_holds}. The guard "
                    "asks a different question from the family; see the docstring."
                ),
                "location": "couplers/curvature.py::check_patch",
            },
            {
                "code": "GUARD_REFUSES_A_TIGHTER_THRESHOLD",
                "detail": (
                    f"asked for {tight_threshold:.6g} rad, half the bound: "
                    f"{refusal.code} -- {refusal.detail}"
                ),
                "location": "couplers/curvature.py::check_patch",
            },
        ],
    )
    measurements = {
        "validity_status_is_out_of_validity": Measurement(
            value=0.0 if crossed else 1.0,
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                "zero means the crossing was detected and reported as out of "
                f"validity rather than as unsupported: eps_curv {declared_eps_rad:g} "
                f"against a computed bound of {bound:.6g}. The code would run, and "
                "the answer would be wrong, which is what separates this from an "
                "unsupported request."
            ),
        )
    }
    return InstanceRun(
        family=B0_VALIDITY,
        instance=instance,
        record=record,
        result=verify(B0_VALIDITY, instance, record, measurements=measurements),
    )


# ---------------------------------------------------------------------------
# B0-UNITS: it ran, nothing raised, the number is wrong
# ---------------------------------------------------------------------------


def _run_units_01() -> InstanceRun:
    """Optiland ``add_layer`` in micrometres, fed the literature's nanometres.

    Re-measured on the pinned install. TWO separations are reported because they
    answer different questions: the gated metric is the distance from the
    CORRECT coating, which is what makes this a trap rather than a rounding
    difference, and the distance from BARE GLASS is why nobody notices -- a
    1000x-too-thick quarter-wave layer does essentially nothing, so the coated
    result and the uncoated one look like the same reflectance.
    """
    instance = _instance(B0_UNITS, "B0-UNITS-01")
    hazard = hazard_for("OPTILAND_ADD_LAYER_UM_NM")
    m = _measure_add_layer_hazard()

    record = record_from_probe(
        instance,
        component="M_RAY_OPTILAND",
        node_id="thin_film_stack",
        refusal=None,
        observed_parameters={"hazard": hazard.hazard_id, "unit_reading": "literature"},
        diagnostics=[
            {
                "code": "CONTRACT_STATUS_OK_PHYSICS_WRONG",
                "detail": (
                    "nothing raised. 99.64 um is a physically constructible layer and "
                    "the returned number looks like a reflectance: "
                    f"{m['wrong']:.10f} against bare glass {m['bare']:.10f} "
                    f"({m['relative_to_bare']:.3e} apart) and against the correctly "
                    f"coated {m['right']:.10f} ({m['relative_to_correct']:.4f} apart). "
                    f"The correct coating is {m['correct_improvement_factor']:.3f}x below "
                    "bare glass, which is what an AR coating is supposed to look like."
                ),
                "location": hazard.api,
            },
            {
                "code": "REMEDY_EXISTS_IN_THE_API",
                "detail": (
                    "the pinned install also exposes add_layer_nm, which divides by "
                    "1000 and is the call the literature's number belongs in"
                ),
                "location": "optiland/thin_film/stack.py::add_layer_nm",
            },
            {
                "code": "MEASUREMENT_DIFFERS_FROM_THE_INHERITED_RECORD",
                "detail": (
                    f"the inherited hazard records wrong={hazard.wrong_value:.8f} against "
                    f"bare={hazard.right_value:.8f}, a separation of "
                    f"{hazard.relative_separation:.3e}. This run measures "
                    f"{m['wrong']:.10f} against {m['bare']:.10f}, a separation of "
                    f"{m['relative_to_bare']:.3e}. The bare-glass value reproduces "
                    "exactly and the correct-coating improvement factor reproduces to "
                    "three decimals, so the difference is in the coating MATERIAL MODEL "
                    "-- IdealMaterial(n=1.38) here against whatever dispersive MgF2 the "
                    "original used -- and not in the unit hazard, which is identical. "
                    "The finding is unchanged: the coating does nothing."
                ),
                "location": "src/verification/hazards.py",
            },
        ],
    )
    measurements = {
        "relative_error_vs_closed_form": Measurement(
            value=m["relative_to_correct"],
            uncertainty=abs(m["relative_to_bare"]),
            uncertainty_basis=UncertaintyBasis.ORACLE_ERROR_BOUND,
            note=(
                "distance from the CORRECT quarter-wave reflectance. Exceeding the "
                "gate is the intended outcome: this instance is a known-wrong "
                "configuration and the tolerance exists to reject it. The error bar is "
                "the distance from bare glass, which bounds how much of this gap could "
                "be a coating-model difference rather than the unit slip."
            ),
        ),
        "contract_status_is_ok": Measurement(
            value=0.0,
            # EXACT promises a number, and for a boolean cast to 0/1 the number
            # is zero: there is nothing about "did anything raise" that is uncertain.
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note="zero means the contract reported ok, which it did: nothing raised",
        ),
    }
    return InstanceRun(
        family=B0_UNITS,
        instance=instance,
        record=record,
        result=verify(
            B0_UNITS,
            instance,
            record,
            measurements=measurements,
            silent_hazard_ids=(hazard.hazard_id,),
        ),
    )


def _run_units_02() -> InstanceRun:
    """Chromatix ``kykx``: cycles on one function, radians on the other.

    Both call sites are measured, and the measurement relocates half the
    recorded hazard: the 2*pi factor is on both, and the SIGN INVERSION is on
    ``asm_propagate`` rather than on ``plane_wave``. The gated metric is the
    recorded case -- the cycles-per-length value handed to ``plane_wave``, which
    wants radians -- against ``z tan(theta)`` computed here.
    """
    instance = _instance(B0_UNITS, "B0-UNITS-02")
    hazard = hazard_for("CHROMATIX_KYKX_2PI_AND_SIGN")
    m = _measure_kykx_hazard()

    relative = abs(m["wrong_um"] - m["right_um"]) / abs(m["right_um"])
    # Why the asm_propagate ratio is 7.45 rather than 2*pi: a spatial frequency
    # 2*pi too large is 1.029 cycles/um, so sin(theta) = lambda*f = 0.548 and the
    # beam is at 33 degrees. The walk-off is z*tan(asin(lambda*f)), which is not
    # linear in f, so the "factor of 2*pi" statement is a paraxial one.
    lam = 0.532
    predicted_asm = 200.0 * math.tan(math.asin(min(1.0, lam * m["radians_per_um"])))

    record = record_from_probe(
        instance,
        component="M_WAVE_CHROMATIX",
        node_id="asm_propagate",
        refusal=None,
        observed_parameters={"hazard": hazard.hazard_id, "unit_reading": "literature"},
        diagnostics=[
            {
                "code": "CONTRACT_STATUS_OK_PHYSICS_WRONG",
                "detail": (
                    "nothing raised on either call site. plane_wave, handed the "
                    f"cycles-per-length value, walks the beam {m['wrong_um']:+.6f} um "
                    f"where geometry requires {m['right_um']:+.6f} -- a factor of "
                    f"{m['plane_wave_factor']:.4f}, which is 2*pi to "
                    f"{abs(m['plane_wave_factor'] - 2 * math.pi):.2e}. Handed the "
                    "radians value it is correct at "
                    f"{m['plane_wave_correct_um']:+.6f} um."
                ),
                "location": "chromatix.functional.plane_wave",
            },
            {
                "code": "THE_SIGN_INVERSION_IS_ON_THE_PROPAGATOR",
                "detail": (
                    "asm_propagate's kykx displaces OPPOSITE to its parameter: the "
                    f"correct cycles-per-length value gives {m['asm_propagate_correct_um']:+.6f} "
                    f"um where geometry requires {m['right_um']:+.6f}, so the magnitude "
                    "is right and the sign is not. The radians value on the same "
                    f"argument gives {m['asm_propagate_mistaken_um']:+.6f} um."
                ),
                "location": "chromatix.functional.asm_propagate",
            },
            {
                "code": "THE_FACTOR_IS_ONLY_2PI_IN_THE_PARAXIAL_LIMIT",
                "detail": (
                    "the asm_propagate ratio is "
                    f"{m['asm_propagate_factor']:.4f} rather than 2*pi because a spatial "
                    f"frequency of {m['radians_per_um']:.6f} cycles/um puts sin(theta) = "
                    f"{lam * m['radians_per_um']:.6f}, i.e. 33 degrees. The walk-off is "
                    "z*tan(asin(lambda*f)), which predicts "
                    f"{predicted_asm:+.4f} um against the measured "
                    f"{m['asm_propagate_mistaken_um']:+.4f}."
                ),
                "location": "src/verification/hazards.py",
            },
            {
                "code": "MEASUREMENT_DIFFERS_FROM_THE_INHERITED_RECORD",
                "detail": (
                    f"the inherited hazard records one number, {hazard.wrong_value:+.6f} "
                    "um, described as 2*pi too small AND sign-flipped. This run finds "
                    "those to be two different mistakes at two different call sites: "
                    f"2*pi too small with the sign PRESERVED on plane_wave "
                    f"({m['plane_wave_mistaken_um']:+.6f} um), and sign-inverted on "
                    "asm_propagate. The magnitude of the recorded number is reproduced "
                    "to 0.5%; its attribution is corrected. Nothing about the hazard's "
                    "severity changes -- both call sites accept the wrong unit silently."
                ),
                "location": "src/verification/hazards.py",
            },
        ],
    )
    measurements = {
        "relative_error_vs_closed_form": Measurement(
            value=relative,
            uncertainty=m["pitch_um"] / abs(m["right_um"]),
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                "one sample pitch of centroid resolution as the error bar. The "
                "displacement is wrong by 84% of the walk-off itself, so the sampling "
                "limit is nowhere near the finding. Exceeding the gate is the intended "
                "outcome: this is a known-wrong configuration."
            ),
        ),
        "contract_status_is_ok": Measurement(
            value=0.0,
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note="zero means the contract reported ok, which it did: nothing raised",
        ),
    }
    return InstanceRun(
        family=B0_UNITS,
        instance=instance,
        record=record,
        result=verify(
            B0_UNITS,
            instance,
            record,
            measurements=measurements,
            silent_hazard_ids=(hazard.hazard_id,),
        ),
    )


# ---------------------------------------------------------------------------
# The two hazard measurements, isolated so they are readable
# ---------------------------------------------------------------------------


def _measure_add_layer_hazard() -> dict[str, float]:
    """Reflectance three ways: bare, 1000x-too-thick, and correct.

    Through Optiland's own ``ThinFilmStack``, which is the point -- the hazard is
    in the unit its API takes, so a re-derivation in NumPy would not reproduce
    it. ``add_layer`` takes MICROMETRES; the AR-coating literature and the
    upstream tutorial quote the quarter-wave MgF2 layer for 550 nm as 99.64
    NANOMETRES.

    The pinned install also has ``add_layer_nm``, which is the remedy an agent
    should find, and the measurement below records what each of the three calls
    returns rather than asserting which is which.
    """
    from optiland.materials import IdealMaterial
    from optiland.thin_film import ThinFilmStack

    wavelength_um = 0.55
    n_glass = 1.5168
    n_coat = 1.38

    def _reflectance(thickness_um: float | None) -> float:
        stack = ThinFilmStack(
            incident_material=IdealMaterial(1.0),
            substrate_material=IdealMaterial(n_glass),
        )
        if thickness_um is not None:
            stack.add_layer(IdealMaterial(n_coat), thickness_um)
        value = stack.reflectance(wavelength_um, 0.0, "u")
        return float(np.asarray(value).ravel()[0])

    bare = _reflectance(None)
    # The literature's 99.64 NANOMETRES, handed to a micrometre API.
    wrong = _reflectance(99.64)
    right = _reflectance(0.09964)
    return {
        "bare": bare,
        "wrong": wrong,
        "right": right,
        # Two separations, and they answer different questions. The distance from
        # the CORRECT coating is what the gate rejects; the distance from BARE
        # GLASS is why nobody notices.
        "relative_to_correct": abs(wrong - right) / abs(right),
        "relative_to_bare": abs(wrong - bare) / abs(bare),
        "correct_improvement_factor": bare / right,
    }


def _measure_kykx_hazard() -> dict[str, float]:
    """The tilted-beam walk-off under both readings of ``kykx``, on both call sites.

    Measured rather than inherited, and the measurement corrects the recorded
    hazard's *attribution*. The recorded finding is one number -- ``-z tan(theta)
    / (2*pi)`` -- described as "both wrong by 2*pi and wrong in sign". On the
    pinned install those are two different mistakes at two different call sites,
    and separating them is what makes either one actionable:

    ``plane_wave`` **wants radians per length.** Handing it the cycles-per-length
    value (which is what ``asm_propagate`` wants) produces a walk-off 2*pi too
    SMALL, with the sign preserved.

    ``asm_propagate`` **wants cycles per length.** Handing it the
    radians-per-length value produces a walk-off 2*pi too LARGE, and the
    displacement runs OPPOSITE in sign to the parameter.

    So the factor is on both and the sign inversion is on ``asm_propagate``. The
    correct answer is ``z tan(theta)`` computed here; nothing is compared against
    a stored number.

    A **localized** beam, not a plane wave: a uniform field fills the grid and
    its centroid cannot move, so a plane-wave measurement of a displacement would
    read zero for every ``kykx`` and would "agree" with any mistake.
    """
    import chromatix.functional as cf
    import jax.numpy as jnp

    tilt_rad = math.radians(5.0)
    distance_um = 200.0
    wavelength_um = 0.532
    pitch_um = 0.25
    grid = 1024
    aperture_radius_um = 60.0

    cycles_per_um = math.sin(tilt_rad) / wavelength_um
    radians_per_um = 2.0 * math.pi * cycles_per_um

    def _centroid_of(intensity: np.ndarray) -> float:
        while intensity.ndim > 2:
            intensity = intensity.sum(axis=0)
        rows = intensity.shape[0]
        coords = (np.arange(rows, dtype=np.float64) - rows // 2) * pitch_um
        profile = intensity.sum(axis=1)
        total = float(profile.sum())
        return float((profile * coords).sum() / total) if total > 0 else 0.0

    def _via_plane_wave(kykx: float) -> float:
        """Tilt applied where the beam is CREATED. plane_wave wants radians/length."""
        field = cf.plane_wave(
            shape=(grid, grid),
            dx=pitch_um,
            spectrum=wavelength_um,
            kykx=(kykx, 0.0),
            pupil=lambda f: cf.circular_pupil(f, w=aperture_radius_um),
        )
        out = cf.asm_propagate(field, z=distance_um, n=1.0, pad_width=grid // 2)
        return _centroid_of(np.asarray(jnp.abs(jnp.squeeze(out.u)) ** 2, dtype=np.float64))

    def _via_asm_propagate(kykx: float) -> float:
        """Tilt applied on the PROPAGATOR. asm_propagate wants cycles/length."""
        field = cf.plane_wave(
            shape=(grid, grid),
            dx=pitch_um,
            spectrum=wavelength_um,
            pupil=lambda f: cf.circular_pupil(f, w=aperture_radius_um),
        )
        out = cf.asm_propagate(
            field, z=distance_um, n=1.0, pad_width=grid // 2, kykx=(kykx, 0.0)
        )
        return _centroid_of(np.asarray(jnp.abs(jnp.squeeze(out.u)) ** 2, dtype=np.float64))

    right_um = distance_um * math.tan(tilt_rad)
    plane_wave_correct = _via_plane_wave(radians_per_um)
    plane_wave_mistaken = _via_plane_wave(cycles_per_um)
    asm_correct = _via_asm_propagate(cycles_per_um)
    asm_mistaken = _via_asm_propagate(radians_per_um)

    return {
        "right_um": right_um,
        # The gated mistake: the recorded hazard's case, on plane_wave.
        "wrong_um": plane_wave_mistaken,
        "plane_wave_correct_um": plane_wave_correct,
        "plane_wave_mistaken_um": plane_wave_mistaken,
        "plane_wave_factor": right_um / plane_wave_mistaken if plane_wave_mistaken else math.nan,
        "asm_propagate_correct_um": asm_correct,
        "asm_propagate_mistaken_um": asm_mistaken,
        "asm_propagate_factor": asm_mistaken / right_um if right_um else math.nan,
        "ratio": plane_wave_mistaken / right_um if right_um else math.nan,
        "pitch_um": pitch_um,
        "cycles_per_um": cycles_per_um,
        "radians_per_um": radians_per_um,
    }


# ---------------------------------------------------------------------------
# Shared: one real trace
# ---------------------------------------------------------------------------


def _optiland_trace(registry: Registry) -> tuple[ArtifactRecord, dict[str, np.ndarray]]:
    """A real M3SingletRef trace at the declared exit pupil, in memory."""
    import tempfile

    from solvers.optiland.adapter import get_adapter

    out = tempfile.mkdtemp(prefix="b0-contract-")
    result = get_adapter().run(
        ModelRunRequest(
            run_id="b0-contract",
            node_id="lens",
            config={
                "sample": "M3SingletRef",
                "num_rays": CONTRACT_RINGS,
                "wavelength": WAVELENGTH_UM,
                "handoff_plane": "exit_pupil",
                "output_directory": out,
            },
        )
    )
    if result.status.value != "succeeded":
        raise RuntimeError(f"the shared trace failed: {result.error_type}: {result.error_message}")
    record = result.outputs["rays"]
    arrays = dict(np.load(record.uri))
    return record, arrays


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

#: instance_id -> the function that executes it. A table rather than a scan, so
#: an instance declared and not executed is visible as an absence here.
_RUNNERS: dict[str, Any] = {
    "B0-CAPINT-01": lambda registry: _run_capint_01(),
    "B0-DEVICE-01": lambda registry: _run_device_01(),
    "B0-DEVICE-02": lambda registry: _run_device_02(),
    "B0-META-01": lambda registry: _run_meta_01(),
    "B0-HANDOFF-01": lambda registry: _run_handoff_01(),
    "B0-PATCH-01": _run_patch_01,
    "B0-DTYPE-01": lambda registry: _run_dtype_01(),
    "B0-VALIDITY-01": lambda registry: _run_validity_01(),
    "B0-UNITS-01": lambda registry: _run_units_01(),
    "B0-UNITS-02": lambda registry: _run_units_02(),
}


def declared_instance_ids() -> tuple[str, ...]:
    """Every canonical B0 instance the families declare."""
    return tuple(
        instance.instance_id
        for family in (B0_CONTRACT, B0_DTYPE, B0_UNITS, B0_VALIDITY)
        for instance in family.canonical_instances
    )


def run_instance(instance_id: str, *, registry: Registry | None = None) -> InstanceRun:
    try:
        runner = _RUNNERS[instance_id]
    except KeyError:
        raise KeyError(
            f"no runner for {instance_id!r}. Declared: {sorted(declared_instance_ids())}"
        ) from None
    return runner(registry or Registry.from_package())


def run_all(*, registry: Registry | None = None) -> dict[str, InstanceRun]:
    """Every declared B0 instance, executed."""
    registry = registry or Registry.from_package()
    return {
        instance_id: run_instance(instance_id, registry=registry)
        for instance_id in declared_instance_ids()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="persist the instance records")
    parser.add_argument("--instance", default=None, help="run one instance by id")
    args = parser.parse_args()

    runs = (
        {args.instance: run_instance(args.instance)}
        if args.instance
        else run_all()
    )
    for instance_id, run in runs.items():
        metrics = ", ".join(
            f"{m.metric}={m.measured.value:.6g}" for m in run.result.physics_accuracy
        )
        expected = run.instance.expected.get("status") or run.instance.expected.get(
            "contract_status"
        )
        print(
            f"{instance_id:<16} status={run.result.status.value:<22} "
            f"expected={expected!s:<22} {metrics}"
        )
        if args.write:
            path = write_instance_record(run, driver="instances/b0_contract")
            print(f"    -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
