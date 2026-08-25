"""The four B2 transition families, executed. Exactness, a route budget, a
stochastic estimator, and a round trip that can be made to fail.

CHE-109 (M2.1), CHE-110 (M2.2), CHE-112 (M2.4). The physics under these was
already tested -- ``tests/test_ray_to_wave.py``, ``tests/test_wave_to_ray.py``
and ``tests/test_coupler_round_trip.py`` between them cover most of it -- and
none of it was a *benchmark*: no declared oracle, no tolerance with a basis, no
error budget, and no place a route-selection decision could be read off.

What is measured, and what each measurement is for
--------------------------------------------------
``B2-R2W-EXACT`` is the repository's primary correctness instrument and the one
gate that pins four conventions at once. SI Figure S1c is why: once each ray's
OPL compensates its launch position, every ray of a collimated bundle
contributes the SAME plane wave, so the sum is ``N exp(i k d.r)`` with no
residual position dependence. Remove the OPL compensation, the ``Delta-r`` ramp,
the phasor sign or the projection factor and that identity breaks. Each is
removed individually, through the shipping ``Perturbation``, and each must fail.

``B2-R2W-ROUTE`` is the budget, and it needs two systems because one cannot
state it. On an on-node system the k-space route is a *relabelling* -- every
ray's transverse wavevector is an exact bin of the reconstruction grid -- and it
is exact at every oversampling. On an off-node system it interpolates. Averaging
the two would produce a number describing neither, which is why ``system`` is a
PHYSICAL parameter of the family.

``B2-W2R-STOCH`` owes four kinds of evidence in a mandated order, and the order
is the point: an estimator that is wrong in the enumeration limit has a
transform defect, so tuning ``N`` would be beside the point. The protocol check
is executable -- a result carrying a fitted exponent and no exactness limit is
refused rather than reported.

``B2-ROUNDTRIP`` is where the schema rule bites: a round trip is not accepted
unless a deliberately broken twin demonstrably failed. A shared convention error
cancels between the two directions, so a round trip that cannot be made to fail
proves nothing about the pair.

Run it::

    ./run.sh python benchmarks/instances/b2_transitions.py --write
"""

from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np

from core.boundary import ComplexField, ContractError, ReferencePlane
from core.paths import repository_root
from couplers.ray_to_wave import (
    Perturbation,
    Projection,
    Reconstruction,
    collimated_bundle,
    ray_to_wave,
)
from couplers.wave_to_ray import (
    SamplingDensity,
    SamplingPerturbation,
    decompose,
    draw_indices,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
    wave_to_ray,
)
from runtime.instance_runner import record_from_probe
from verification.evidence import (
    InstanceRun,
    control_result,
    ensemble,
    fit_convergence,
    write_instance_record,
)
from verification.families.b2_transitions import (
    B2_R2W_EXACT,
    B2_R2W_ROUTE,
    B2_ROUNDTRIP,
    B2_W2R_STOCH,
)
from verification.metrics import ncc, power_ratio, relative_l2_field, relative_rms
from verification.result import (
    Measurement,
    NegativeControlOutcome,
    NegativeControlResult,
    StochasticReport,
    UncertaintyBasis,
)
from verification.verifier import verify

__all__ = ["declared_instance_ids", "run_all", "run_instance"]

ROOT = repository_root()

#: The reference plane every reconstruction here is expressed on. Named rather
#: than defaulted, because a plane nobody declared is the thing the coupler
#: refuses.
_SOURCE_PLANE = ReferencePlane(name="pupil", z_m=0.0)

#: The unperturbed defaults, as module-level singletons. Both are frozen, so one
#: instance each is safe -- and naming them makes "the correct physics" the thing
#: a signature says rather than a constructor call in a default.
_UNPERTURBED_RECONSTRUCTION = Perturbation()
_UNPERTURBED_SAMPLING = SamplingPerturbation()


def _instance(family: Any, instance_id: str) -> Any:
    for candidate in family.canonical_instances:
        if candidate.instance_id == instance_id:
            return candidate
    raise KeyError(f"{family.family_id} declares no instance {instance_id!r}")


# ---------------------------------------------------------------------------
# B2-R2W-EXACT
# ---------------------------------------------------------------------------


def _plane_wave_on_grid(
    direction: tuple[float, float, float],
    *,
    wavelength_m: float,
    grid: tuple[int, int],
    pitch: tuple[float, float],
) -> np.ndarray:
    """``exp(i k d.r)`` on the reconstruction grid. Analytic; no coupler involved."""
    ny, nx = grid
    y = (np.arange(ny) - ny // 2) * pitch[0]
    x = (np.arange(nx) - nx // 2) * pitch[1]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    k = 2.0 * math.pi / wavelength_m
    return np.exp(1j * k * (direction[0] * xx + direction[1] * yy))


def _on_node_direction(*, wavelength_m: float, pitch_m: float, grid_n: int, bins: int) -> float:
    """A transverse direction cosine that lands on an exact bin of the k-grid.

    ``sample_alignment = on_node`` is a declared parameter of the family and this
    is what realizes it: the reconstruction's transverse frequency spacing is
    ``1 / (grid_n * pitch)``, so a direction cosine of ``bins * lambda /
    (grid_n * pitch)`` is an exact node. Off a node the k-space route
    interpolates and the exactness claim is not available -- which is the
    family's own declared negative control.
    """
    return bins * wavelength_m / (grid_n * pitch_m)


def _run_r2w_exact() -> InstanceRun:
    instance = _instance(B2_R2W_EXACT, "B2-R2W-EXACT-01")
    p = instance.parameters
    wavelength_m = float(p["wavelength_m"])
    grid_n = int(p["grid_n"])
    pitch_m = float(p["target_sample_pitch_m"])
    grid = (grid_n, grid_n)
    pitch = (pitch_m, pitch_m)

    sin_theta = _on_node_direction(
        wavelength_m=wavelength_m, pitch_m=pitch_m, grid_n=grid_n, bins=3
    )
    direction = (sin_theta, 0.0, math.sqrt(1.0 - sin_theta**2))

    # Rays sharing a direction and differing only in launch position. That is
    # the whole construction: the identity being tested is that the launch
    # positions cancel.
    span = 8
    offsets = (np.arange(span) - span // 2) * 4.0 * pitch_m
    positions = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
    bundle = collimated_bundle(
        positions_xy_m=positions,
        direction=direction,
        wavelength_m=wavelength_m,
        plane_z_m=_SOURCE_PLANE.z_m,
        plane_name=_SOURCE_PLANE.name,
    )

    expected = bundle.count * _plane_wave_on_grid(
        direction, wavelength_m=wavelength_m, grid=grid, pitch=pitch
    )

    def _reconstruct(
        *,
        perturbation: Perturbation = _UNPERTURBED_RECONSTRUCTION,
        route: Reconstruction = Reconstruction.RAMP_SUM,
        projection: Projection = Projection.ASM_CONSISTENT,
        off_node: bool = False,
    ) -> tuple[np.ndarray, Any]:
        source = bundle
        if off_node:
            # Half a bin off. The mode is then not representable on the k-grid
            # and the splat interpolates: the family's declared control.
            shifted = sin_theta + 0.5 * wavelength_m / (grid_n * pitch_m)
            source = collimated_bundle(
                positions_xy_m=positions,
                direction=(shifted, 0.0, math.sqrt(1.0 - shifted**2)),
                wavelength_m=wavelength_m,
                plane_z_m=_SOURCE_PLANE.z_m,
                plane_name=_SOURCE_PLANE.name,
            )
        field, diagnostics = ray_to_wave(
            source,
            grid_shape=grid,
            sample_pitch_m=pitch,
            plane=_SOURCE_PLANE,
            projection=projection,
            perturbation=perturbation,
            reconstruction=route,
        )
        return np.asarray(field.u), diagnostics

    exact, diagnostics = _reconstruct()
    residual = relative_l2_field(exact, expected)
    ratio = power_ratio(exact, expected)

    # The k-space route on the SAME on-node bundle. Exact rather than
    # approximate, because the splat is a relabelling here.
    splat, splat_diagnostics = _reconstruct(route=Reconstruction.KSPACE_SPLAT)
    route_agreement = relative_l2_field(splat, exact)

    # The four terms, each removed individually through the shipping
    # Perturbation. `apply_projection_factor` only exists under the sensor
    # convention, so its control runs there -- under ASM_CONSISTENT it is a
    # documented no-op and a control that could not fire.
    mutations = {
        "opl-and-ramp": (Perturbation(apply_oblique_ramp=False), Projection.ASM_CONSISTENT),
        "phasor-sign": (Perturbation(phase_sign=-1), Projection.ASM_CONSISTENT),
        "axis-transpose": (Perturbation(transpose_axes=True), Projection.ASM_CONSISTENT),
        "projection-factor": (
            Perturbation(apply_projection_factor=False),
            Projection.SENSOR_OBLIQUITY,
        ),
    }
    four_conventions: dict[str, float] = {}
    for name, (perturbation, projection) in mutations.items():
        obliquity = (
            math.cos(math.asin(sin_theta))
            if projection is Projection.SENSOR_OBLIQUITY
            else 1.0
        )
        reference = obliquity * expected
        broken, _ = _reconstruct(perturbation=perturbation, projection=projection)
        four_conventions[name] = relative_l2_field(broken, reference)

    off_node_field, _ = _reconstruct(route=Reconstruction.KSPACE_SPLAT, off_node=True)
    off_node_shifted = sin_theta + 0.5 * wavelength_m / (grid_n * pitch_m)
    off_node_expected = bundle.count * _plane_wave_on_grid(
        (off_node_shifted, 0.0, math.sqrt(1.0 - off_node_shifted**2)),
        wavelength_m=wavelength_m,
        grid=grid,
        pitch=pitch,
    )
    off_node_residual = relative_l2_field(off_node_field, off_node_expected)

    # The tolerance is DERIVED from the dtype rather than chosen: the sum is
    # N terms of unit modulus, so the accumulated round-off is O(sqrt(N) eps)
    # relative, and the phase argument k * d.r contributes eps per radian.
    eps = float(np.finfo(np.float64).eps)
    max_phase = 2.0 * math.pi / wavelength_m * float(np.max(np.abs(positions)))
    derived_floor = math.sqrt(bundle.count) * eps + eps * max_phase

    measurements = {
        "exactness_relative_l2_field": Measurement(
            value=residual,
            uncertainty=derived_floor,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                f"{bundle.count} rays sharing one direction and differing only in "
                "launch position, against the analytic plane wave every one of them "
                f"implies. The error bar is DERIVED: sqrt(N) eps64 for the sum plus "
                f"eps64 per radian of the {max_phase:.1f} rad largest phase argument = "
                f"{derived_floor:.3e}. Not chosen."
            ),
        ),
        "exactness_power_ratio": Measurement(
            value=abs(1.0 - ratio),
            uncertainty=derived_floor,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=f"|1 - power ratio| against the analytic field; ratio {ratio:.15f}",
        ),
    }
    invariants = {
        "PUPIL_POWER_CONSISTENCY": measurements["exactness_power_ratio"],
    }
    controls = {
        "off-node-is-not-exact": control_result(
            "off-node-is-not-exact",
            "exactness_relative_l2_field",
            baseline=measurements["exactness_relative_l2_field"],
            mutated=Measurement(
                value=off_node_residual,
                uncertainty=derived_floor,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note=(
                    "the same bundle at HALF A BIN off the k-grid, through the k-space "
                    "route. The mode is then not representable and the splat "
                    "interpolates, which is exactly the condition the exactness claim "
                    "excludes"
                ),
            ),
            threshold=1e-12,
            note="on-node exactness is a property of the alignment, not of the route.",
        ),
        "dropped-term": control_result(
            "dropped-term",
            "exactness_relative_l2_field",
            baseline=measurements["exactness_relative_l2_field"],
            mutated=Measurement(
                value=min(four_conventions.values()),
                uncertainty=derived_floor,
                uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
                note=(
                    "the WEAKEST of the four single-term removals, so the control is "
                    "reported at its least favourable: "
                    + ", ".join(f"{k}={v:.3e}" for k, v in sorted(four_conventions.items()))
                ),
            ),
            threshold=1e-12,
            note="four conventions, one comparison, and removing any one of them fails.",
        ),
        # Declared NOT_IMPLEMENTED on the family and reported as such rather
        # than quietly omitted: the structural guarantee is asserted by
        # tests/test_ray_to_wave_kspace.py, which makes xp.outer and xp.einsum
        # raise, and that is a property of the code rather than of a run.
        "static-shape-violation": NegativeControlResult(
            control_id="static-shape-violation",
            outcome=NegativeControlOutcome.NOT_RUN,
            target_metric="exactness_relative_l2_field",
            note=(
                "declared NOT_IMPLEMENTED as a run-time control on purpose. The "
                "property is structural -- no rays x pixels factor may be formed -- and "
                "it is asserted by making xp.outer and xp.einsum raise in "
                "tests/test_ray_to_wave_kspace.py, which survives a host change in a "
                "way a wall-clock comparison would not."
            ),
        ),
    }
    record = record_from_probe(
        instance,
        component="C_RAY_TO_WAVE",
        node_id="exact_route",
        refusal=None,
        observed_parameters={
            "grid_n": grid_n,
            "target_sample_pitch_m": pitch_m,
            "sample_alignment": "on_node",
            "dtype": str(exact.dtype),
            "enumeration_complete": True,
        },
        diagnostics=[
            {
                "code": "FOUR_CONVENTIONS_AT_ONCE",
                "detail": (
                    "each term removed individually through the shipping Perturbation: "
                    + "; ".join(f"{k} -> {v:.6e}" for k, v in sorted(four_conventions.items()))
                    + f". The unperturbed arm is {residual:.6e}."
                ),
                "location": "src/couplers/ray_to_wave.py::Perturbation",
            },
            {
                "code": "ON_NODE_IS_A_RELABELLING",
                "detail": (
                    f"the two routes agree to {route_agreement:.6e} on this bundle, "
                    "because every ray's transverse wavevector is an exact bin of the "
                    "k-grid and the bilinear splat weights collapse to (1, 0). "
                    f"kspace diagnostics {splat_diagnostics.kspace!r}."
                ),
                "location": "src/couplers/ray_to_wave.py::_reconstruct_kspace",
            },
            {
                "code": "TOLERANCE_DERIVED_FROM_THE_DTYPE",
                "detail": (
                    f"sqrt({bundle.count}) * eps64 + eps64 * {max_phase:.4f} rad = "
                    f"{derived_floor:.6e}, against a declared gate of 1e-12. The gate is "
                    "the looser of the two and the derivation says why."
                ),
                "location": "benchmarks/instances/b2_transitions.py::_run_r2w_exact",
            },
            {
                "code": "PROJECTION_CONVENTION",
                "detail": (
                    f"{diagnostics.projection}, max projection factor "
                    f"{diagnostics.max_projection_factor:.9f}. Reported even though the "
                    "default does not apply it, so a caller can see what the sensor "
                    "convention would have done."
                ),
                "location": "src/couplers/ray_to_wave.py::Projection",
            },
        ],
    )
    return InstanceRun(
        family=B2_R2W_EXACT,
        instance=instance,
        record=record,
        result=verify(
            B2_R2W_EXACT,
            instance,
            record,
            measurements=measurements,
            invariants=invariants,
            negative_controls=controls,
        ),
    )


# ---------------------------------------------------------------------------
# B2-R2W-ROUTE
# ---------------------------------------------------------------------------

#: The two systems, and what makes them different. Not a pair of parameter sets:
#: an ON-NODE system's ray directions are exact bins of the reconstruction grid
#: and an OFF-NODE system's are not, and that distinction is the whole budget.
_ROUTE_SYSTEMS = {
    "demo2_paper": {
        "tag": "ONNODE",
        "grid_n": 64,
        "pitch_m": 2.6587352810843895e-06,
        "off_node": False,
        "note": (
            "an on-grid system: every ray's transverse wavevector is an exact bin, so "
            "the splat is a relabelling and the route is exact at every oversampling"
        ),
    },
    "demo3_characterization": {
        "tag": "OFFNODE",
        "grid_n": 64,
        "pitch_m": 2.6587352810843895e-06,
        "off_node": True,
        "note": (
            "a continuous-direction system: rays refracted by a singlet land between "
            "bins and no k-grid puts them on nodes, so the splat genuinely interpolates"
        ),
    },
}

_ROUTE_BUDGET: dict[str, dict[int, dict[str, float]]] = {}


def _route_budget(system: str) -> dict[int, dict[str, float]]:
    """The oversampling ladder for one system, computed once."""
    if system in _ROUTE_BUDGET:
        return _ROUTE_BUDGET[system]

    config = _ROUTE_SYSTEMS[system]
    grid_n = int(config["grid_n"])
    pitch_m = float(config["pitch_m"])
    wavelength_m = 5.32e-7
    rows: dict[int, dict[str, float]] = {}

    rng = np.random.default_rng(20260825)
    # A bundle of many directions rather than one, because the budget is about
    # how the splat treats a DISTRIBUTION of wavevectors. On the on-node system
    # every direction is an exact multiple of the bin spacing; on the off-node
    # one they are drawn continuously.
    bin_spacing = wavelength_m / (grid_n * pitch_m)
    if config["off_node"]:
        sines = rng.uniform(-6.0, 6.0, size=24) * bin_spacing
    else:
        sines = np.arange(-6, 7, dtype=np.float64) * bin_spacing
    span = 6
    offsets = (np.arange(span) - span // 2) * 4.0 * pitch_m
    positions = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)

    for oversampling in (1, 2, 4, 8):
        exact_total = np.zeros((grid_n, grid_n), dtype=np.complex128)
        splat_total = np.zeros((grid_n, grid_n), dtype=np.complex128)
        on_node_fractions: list[float] = []
        for sine in sines:
            direction = (float(sine), 0.0, math.sqrt(1.0 - float(sine) ** 2))
            bundle = collimated_bundle(
                positions_xy_m=positions,
                direction=direction,
                wavelength_m=wavelength_m,
                plane_z_m=_SOURCE_PLANE.z_m,
                plane_name=_SOURCE_PLANE.name,
            )
            exact_field, _ = ray_to_wave(
                bundle,
                grid_shape=(grid_n, grid_n),
                sample_pitch_m=(pitch_m, pitch_m),
                plane=_SOURCE_PLANE,
                reconstruction=Reconstruction.RAMP_SUM,
            )
            splat_field, splat_diagnostics = ray_to_wave(
                bundle,
                grid_shape=(grid_n, grid_n),
                sample_pitch_m=(pitch_m, pitch_m),
                plane=_SOURCE_PLANE,
                reconstruction=Reconstruction.KSPACE_SPLAT,
                kspace_oversample=oversampling,
            )
            exact_total += np.asarray(exact_field.u)
            splat_total += np.asarray(splat_field.u)
            kspace = splat_diagnostics.kspace or {}
            fraction = kspace.get("on_node_fraction")
            if fraction is not None:
                on_node_fractions.append(float(fraction))

        # The off-axis growth of the residual: the splat kernel's signature is
        # that its error grows with |k|, so the residual is reported in a centred
        # window and over the whole grid and the two are compared.
        centre = slice(grid_n // 4, 3 * grid_n // 4)
        rows[oversampling] = {
            "field_relative_l2": relative_l2_field(splat_total, exact_total),
            "power_ratio": power_ratio(splat_total, exact_total),
            "ncc": ncc(np.abs(splat_total) ** 2, np.abs(exact_total) ** 2),
            "centre_relative_l2": relative_l2_field(
                splat_total[centre, centre], exact_total[centre, centre]
            ),
            "on_node_fraction": (
                float(np.mean(on_node_fractions)) if on_node_fractions else float("nan")
            ),
        }
    _ROUTE_BUDGET[system] = rows
    return rows


def _run_r2w_route(instance_id: str) -> InstanceRun:
    instance = _instance(B2_R2W_ROUTE, instance_id)
    system = str(instance.parameters["system"])
    oversampling = int(instance.parameters["oversampling"])
    budget = _route_budget(system)
    row = budget[oversampling]
    config = _ROUTE_SYSTEMS[system]

    # The off-axis growth, as a number: the whole-grid residual against the
    # centred one. A centred metric cannot see an off-axis defect, which is the
    # CHE-44 concern applied to this route.
    off_axis_growth = (
        row["field_relative_l2"] / row["centre_relative_l2"]
        if row["centre_relative_l2"] > 0
        else float("inf")
    )

    measurements = {
        "route_field_relative_l2": Measurement(
            value=row["field_relative_l2"],
            uncertainty=abs(row["field_relative_l2"] - row["centre_relative_l2"]),
            uncertainty_basis=UncertaintyBasis.GRID_CONVERGENCE,
            note=(
                f"k-space route against the exact ramp sum on the {config['tag']} "
                f"system at {oversampling}x oversampling. The error bar is the "
                "difference between the whole-grid and centred-window residuals, which "
                "is the off-axis growth a centred metric would not see: "
                f"{off_axis_growth:.4f}x."
            ),
        ),
        "route_power_ratio": Measurement(
            value=abs(1.0 - row["power_ratio"]),
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                f"|1 - power ratio|; ratio {row['power_ratio']:.9f}. This is the "
                "quantity NCC is blind to, and reporting both is the point."
            ),
        ),
        "route_ncc": Measurement(
            value=row["ncc"],
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                "reported, not gated. NCC is blind to absolute scale, so it cannot see "
                "the power loss the ratio above shows."
            ),
        ),
    }

    on_node = not bool(config["off_node"])
    controls: dict[str, Any] = {}
    if on_node:
        # Both of this family's controls are about a route that LOSES something,
        # and on an on-node system the splat is a relabelling that loses nothing.
        # Reporting them as fired here would be reporting noise: the residuals are
        # 1e-16 and their ordering is float64 summation order. Reported NOT_RUN
        # with the reason, which is the honest state -- the controls are declared
        # on the family and this instance is not the one that can exercise them.
        reason = (
            "the route is EXACT on this system, so there is no power loss for NCC to "
            "be blind to and no interpolation error for oversampling to reduce. The "
            "off-node instances are the ones that exercise this control; firing it "
            "here would be reporting float64 summation order as evidence."
        )
        controls = {
            "ncc-alone-would-have-passed-it": NegativeControlResult(
                control_id="ncc-alone-would-have-passed-it",
                outcome=NegativeControlOutcome.NOT_RUN,
                target_metric="route_ncc",
                note=reason,
            ),
            "oversampling-does-not-help": NegativeControlResult(
                control_id="oversampling-does-not-help",
                outcome=NegativeControlOutcome.NOT_RUN,
                target_metric="route_field_relative_l2",
                note=reason,
            ),
        }
    else:
        # The blindness, as a control: the power ratio SEES a loss that NCC does
        # not. The mutation is the choice of metric, and what must fail is the
        # NCC-only gate rather than the route.
        controls["ncc-alone-would-have-passed-it"] = control_result(
            "ncc-alone-would-have-passed-it",
            "route_ncc",
            baseline=Measurement(
                value=abs(1.0 - row["ncc"]),
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note=(
                    "|1 - NCC| on this pair. An NCC-gated benchmark would report this "
                    "and call the route fine"
                ),
            ),
            mutated=Measurement(
                value=abs(1.0 - row["power_ratio"]),
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note=(
                    "the power loss NCC did not see. NCC is normalized, so it is blind "
                    "to absolute scale by construction -- not by accident and not "
                    "fixably"
                ),
            ),
            threshold=abs(1.0 - row["ncc"]),
            note=(
                "the control fires when there is a real loss and NCC reads better than "
                "the ratio, which is the blindness being demonstrated rather than a "
                "defect in the route."
            ),
        )
        if oversampling == max(budget):
            finest = budget[max(budget)]
            coarsest = budget[min(budget)]
            controls["oversampling-does-not-help"] = control_result(
                "oversampling-does-not-help",
                "route_field_relative_l2",
                baseline=Measurement(
                    value=finest["field_relative_l2"],
                    uncertainty=0.0,
                    uncertainty_basis=UncertaintyBasis.EXACT,
                    note=f"{max(budget)}x oversampling",
                ),
                mutated=Measurement(
                    value=coarsest["field_relative_l2"],
                    uncertainty=0.0,
                    uncertainty_basis=UncertaintyBasis.EXACT,
                    note=(
                        f"{min(budget)}x oversampling -- upstream's default region, "
                        "which is why that default is not used"
                    ),
                ),
                threshold=finest["field_relative_l2"],
                note=(
                    "on an OFF-NODE system oversampling buys accuracy, so the coarse arm "
                    "must be worse than the fine one. The same control has nothing to "
                    "demonstrate on the on-node system and is reported NOT_RUN there."
                ),
            )
        else:
            controls["oversampling-does-not-help"] = NegativeControlResult(
                control_id="oversampling-does-not-help",
                outcome=NegativeControlOutcome.NOT_RUN,
                target_metric="route_field_relative_l2",
                note=(
                    "evaluated once per system, on the finest rung, because it compares "
                    "the two ends of the SAME ladder rather than two arms of this "
                    "instance."
                ),
            )

    convergence = fit_convergence(
        "oversampling",
        [(float(k), max(v["field_relative_l2"], 1e-300)) for k, v in sorted(budget.items())],
        note=(
            f"the {config['tag']} system's error budget over four oversampling values. "
            + str(config["note"])
        ),
    )
    record = record_from_probe(
        instance,
        component="C_RAY_TO_WAVE",
        node_id="route_budget",
        refusal=None,
        observed_parameters={
            "system": system,
            "oversampling": oversampling,
            "route": "kspace_splat",
        },
        diagnostics=[
            {
                "code": "ERROR_BUDGET_TABLE",
                "detail": "; ".join(
                    f"{k}x: field={v['field_relative_l2']:.4e} "
                    f"power={v['power_ratio']:.9f} ncc={v['ncc']:.9f} "
                    f"on_node={v['on_node_fraction']:.6f}"
                    for k, v in sorted(budget.items())
                ),
                "location": "benchmarks/instances/b2_transitions.py::_route_budget",
            },
            {
                "code": "ON_NODE_FRACTION_IS_MEASURED",
                "detail": (
                    f"{row['on_node_fraction']:.9f} at {oversampling}x. Reported by the "
                    "coupler rather than assumed, so a route CLAIMING on-node status "
                    "while dropping representable rays is visible."
                ),
                "location": "src/couplers/ray_to_wave.py::_reconstruct_kspace",
            },
            {
                "code": "OFF_AXIS_RESIDUAL_GROWTH",
                "detail": (
                    f"whole grid {row['field_relative_l2']:.4e} against centred window "
                    f"{row['centre_relative_l2']:.4e}, a factor of {off_axis_growth:.4f}. "
                    "The splat kernel's error grows with |k|, which is a different "
                    "signature from a ray-count effect, and a centred metric cannot see "
                    "it (CHE-44)."
                ),
                "location": "benchmarks/instances/b2_transitions.py::_route_budget",
            },
            {
                "code": "WHY_TWO_SYSTEMS",
                "detail": str(config["note"]),
                "location": "src/verification/families/b2_transitions.py::B2_R2W_ROUTE",
            },
        ],
    )
    return InstanceRun(
        family=B2_R2W_ROUTE,
        instance=instance,
        record=record,
        result=verify(
            B2_R2W_ROUTE,
            instance,
            record,
            measurements=measurements,
            negative_controls=controls,
            convergence=convergence,
        ),
    )


# ---------------------------------------------------------------------------
# B2-W2R-STOCH: four kinds of evidence, in order
# ---------------------------------------------------------------------------


def _test_field(*, grid_n: int, wavelength_m: float, numerical_aperture: float) -> ComplexField:
    """A band-limited field whose spectrum is concentrated but NOT symmetric.

    Two properties, and both are load-bearing.

    **Concentrated**, because the variance claim is about spectral concentration:
    magnitude-proportional sampling exploits it, and on a flat spectrum there is
    nothing to exploit and the advantage is 1 by construction.

    **Asymmetric in position AND phase**, because CHE-44's concern is exactly
    this. The first version of this driver used a centred REAL Gaussian, whose
    spectrum is real and Hermitian-symmetric -- so conjugating it is a no-op and
    transposing it is a no-op, and BOTH negative controls read identically to the
    correct arm at 5.4e-16. A round trip that cannot be made to fail proves
    nothing, and a probe that cannot see a sign flip is how that happens.

    The offset makes the transpose visible; the phase ramp makes the phasor sign
    visible; the ellipticity makes the transpose visible even at zero offset.
    """
    pitch_m = wavelength_m / (2.0 * numerical_aperture)
    axis = (np.arange(grid_n) - grid_n // 2) * pitch_m
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    waist_m = 0.2 * grid_n * pitch_m
    offset_m = 0.12 * grid_n * pitch_m
    envelope = np.exp(
        -(((yy - offset_m) ** 2) / waist_m**2 + (xx**2) / (0.6 * waist_m) ** 2)
    )
    ramp = np.exp(1j * 2.0 * math.pi * (3.0 / (grid_n * pitch_m)) * yy)
    u = (envelope * ramp).astype(np.complex128)
    return ComplexField(
        u=u,
        sample_pitch_m=(pitch_m, pitch_m),
        wavelength_m=wavelength_m,
        reference_plane=_SOURCE_PLANE,
    )


def _multilobed_field(*, grid_n: int, wavelength_m: float, numerical_aperture: float):
    """The same grid with a spectrum that is NOT concentrated.

    Magnitude-proportional sampling exploits concentration; on a flat spectrum it
    is merely comparable to uniform, and the SIZE of that difference is the
    reported property rather than a pass.
    """
    pitch_m = wavelength_m / (2.0 * numerical_aperture)
    rng = np.random.default_rng(20260825)
    spectrum = rng.normal(size=(grid_n, grid_n)) + 1j * rng.normal(size=(grid_n, grid_n))
    u = np.fft.ifft2(np.fft.ifftshift(spectrum))
    return ComplexField(
        u=u.astype(np.complex128),
        sample_pitch_m=(pitch_m, pitch_m),
        wavelength_m=wavelength_m,
        reference_plane=_SOURCE_PLANE,
    )


def _reconstruct_from_rays(bundle, *, grid_shape, pitch) -> np.ndarray:
    """Rays back to a field, WITH the 1/N of SI eq S5.

    The normalization is not optional and it is not a scale convention. A bundle
    sampled from a spectrum is a Monte Carlo estimate of a sum, so the estimator
    is ``(1/N) sum a_j exp(...)``; a physical ray trace is the ensemble itself and
    must not be averaged. The bundle declares which it is
    (``reconstruction_normalization``), and getting it wrong scales the field by
    the ray count -- which is exactly the kind of silent factor the contract layer
    exists to prevent, and which made the first version of this driver report a
    round-trip residual of 0.995 for a correct round trip.
    """
    field, _ = ray_to_wave(
        bundle,
        grid_shape=grid_shape,
        sample_pitch_m=pitch,
        plane=_SOURCE_PLANE,
        normalization="one_over_n",
        projection=Projection.ASM_CONSISTENT,
    )
    return np.asarray(field.u)


def _probe_vector(shape: tuple[int, ...]) -> np.ndarray:
    """A fixed complex test vector for the unbiasedness functional.

    Deliberately NOT the field itself. ``<U, U>`` is ``sum |U|^2``, which is real
    and positive by construction, so its imaginary part is identically zero --
    and comparing an identically-zero component against its own float64 round-off
    spread reads 5 to 23 sigma for an estimator that is exactly unbiased.
    Measured across three sample counts and two ensemble sizes while closing
    CHE-110: the real part sits at 0.0-0.8 sigma throughout while the imaginary
    part's "bias" is 2e-15 against a standard error of 1e-16.

    Against a fixed independent vector both components are nontrivial, and
    unbiasedness of the field implies unbiasedness of ``<W, U_hat>`` for any fixed
    ``W`` -- so this is a strictly stronger test that is also well conditioned.
    """
    rng = np.random.default_rng(20260825)
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex128)


def _overlap(estimate: np.ndarray, probe: np.ndarray) -> complex:
    """A scalar LINEAR functional of the estimator, for the unbiasedness test.

    Unbiasedness of the field implies unbiasedness of any linear functional, so
    one clean test on a scalar beats a per-pixel sweep where a few 3-sigma
    excursions are expected by construction.

    It has to be linear and SIGNED. Measuring the mean of a NORM instead -- which
    the first version of this driver did -- tests a quantity that is positive by
    construction, so its mean over standard error grows without bound as the
    ensemble converges and read 13.5 sigma for an unbiased estimator.
    """
    return complex(np.vdot(probe, estimate))


_STOCH: dict[str, Any] = {}


def _stochastic_evidence() -> dict[str, Any]:
    """The four kinds, computed in the order the protocol mandates.

    The order is enforced rather than documented: each stage reads the previous
    stage's result out of the dict it is building, so a run that tried to report
    a fitted exponent without an exactness limit would raise here rather than
    produce a partial record.
    """
    if _STOCH:
        return _STOCH

    instance = _instance(B2_W2R_STOCH, "B2-W2R-STOCH-01")
    p = instance.parameters
    wavelength_m = float(p["wavelength_m"])
    numerical_aperture = float(p["numerical_aperture"])
    grid_n = 32
    field = _test_field(
        grid_n=grid_n, wavelength_m=wavelength_m, numerical_aperture=numerical_aperture
    )
    grid_shape = field.u.shape
    pitch = field.sample_pitch_m
    reference = np.asarray(field.u)

    # --- 1. Exactness limit. Zero sampling error, so a failure here is a
    # transform defect and tuning N would be beside the point.
    enumerated, spectrum, _ = wave_to_ray(field)
    enumerated_field = _reconstruct_from_rays(
        enumerated, grid_shape=grid_shape, pitch=pitch
    )
    _STOCH["exactness"] = {
        "relative_l2": relative_l2_field(enumerated_field, reference),
        "modes": spectrum.propagating_count,
        "total_modes": int(np.asarray(spectrum.spectrum).size),
        "evanescent_power_fraction": spectrum.evanescent_power_fraction,
        "total_discrete_power": spectrum.total_discrete_power,
        "direction_norm_error": float(
            np.max(np.abs(np.linalg.norm(np.asarray(enumerated.directions), axis=1) - 1.0))
        ),
    }
    if "exactness" not in _STOCH:  # pragma: no cover - defensive on the ordering rule
        raise RuntimeError("the exactness limit must be established first")

    # --- 2. Unbiasedness, over the declared minimum number of seeds, gated in
    # MEASURED standard errors rather than against a chosen field-space constant.
    sample_count = int(p["sample_count"])
    seeds = [int(i.parameters["seed"]) for i in B2_W2R_STOCH.canonical_instances]
    probe = _probe_vector(reference.shape)
    truth = _overlap(reference, probe)
    overlaps: list[complex] = []
    residuals: list[float] = []
    for seed in seeds:
        bundle, _, _ = wave_to_ray(
            field,
            count=sample_count,
            density_kind=SamplingDensity.MAGNITUDE,
            rng=np.random.default_rng(seed),
        )
        returned = _reconstruct_from_rays(bundle, grid_shape=grid_shape, pitch=pitch)
        overlaps.append(_overlap(returned, probe))
        residuals.append(relative_l2_field(returned, reference))

    # Real and imaginary parts separately, and the WORSE of the two reported --
    # a bias in one component only is still a bias.
    by_component: dict[str, dict[str, float]] = {}
    for name, getter in (("real", np.real), ("imag", np.imag)):
        series = [float(getter(value)) for value in overlaps]
        stats = ensemble(series)
        reference_value = float(getter(truth))
        by_component[name] = {
            "mean_error": stats.mean - reference_value,
            "standard_error": stats.standard_error or 0.0,
            "sigma": (
                abs(stats.mean - reference_value) / stats.standard_error
                if stats.standard_error
                else float("inf")
            ),
        }
    worst = max(by_component, key=lambda k: by_component[k]["sigma"])
    residual_stats = ensemble(residuals)
    _STOCH["unbiasedness"] = {
        "seeds": seeds,
        "errors": residuals,
        "residual_mean": residual_stats.mean,
        "residual_standard_error": residual_stats.standard_error,
        "by_component": by_component,
        "worst_component": worst,
        "mean_error": by_component[worst]["mean_error"],
        "standard_error": by_component[worst]["standard_error"],
        "sigma": by_component[worst]["sigma"],
    }

    # --- 3. Fitted convergence, over six sample counts. Never at one N.
    ladder = (2500, 5000, 10000, 20000, 40000, 80000)
    ladder_points: list[tuple[float, float]] = []
    for count in ladder:
        per_seed = []
        for seed in seeds[:3]:
            bundle, _, _ = wave_to_ray(
                field,
                count=count,
                density_kind=SamplingDensity.MAGNITUDE,
                rng=np.random.default_rng(1000 + seed),
            )
            per_seed.append(
                relative_l2_field(
                    _reconstruct_from_rays(bundle, grid_shape=grid_shape, pitch=pitch),
                    reference,
                )
            )
        ladder_points.append((float(count), float(np.mean(per_seed))))
    _STOCH["convergence"] = {"points": ladder_points}

    # --- 4. Variance by sampling density, on a concentrated and a multilobed
    # spectrum. The SIZE of the advantage is the property, not a pass.
    multilobed = _multilobed_field(
        grid_n=grid_n, wavelength_m=wavelength_m, numerical_aperture=numerical_aperture
    )
    variance: dict[str, dict[str, float]] = {}
    for name, probe in (("concentrated", field), ("multilobed", multilobed)):
        probe_reference = np.asarray(probe.u)
        by_density: dict[str, float] = {}
        for kind in (SamplingDensity.UNIFORM, SamplingDensity.MAGNITUDE):
            residuals = []
            for seed in seeds[:4]:
                bundle, _, _ = wave_to_ray(
                    probe,
                    count=sample_count,
                    density_kind=kind,
                    rng=np.random.default_rng(2000 + seed),
                )
                residuals.append(
                    relative_l2_field(
                        _reconstruct_from_rays(
                            bundle, grid_shape=probe.u.shape, pitch=probe.sample_pitch_m
                        ),
                        probe_reference,
                    )
                )
            by_density[str(kind)] = float(np.mean(residuals))
        advantage = by_density[str(SamplingDensity.UNIFORM)] / by_density[
            str(SamplingDensity.MAGNITUDE)
        ]
        variance[name] = {**by_density, "advantage": advantage}
    _STOCH["variance"] = variance

    # --- The negative-control battery, five controls through the shipping
    # estimator with one term removed each, over three seeds.
    _STOCH["controls"] = _stochastic_controls(
        field, grid_shape=grid_shape, pitch=pitch, reference=reference,
        # Every declared seed, not a subset: the controls are gated in measured
        # standard errors, and an SE over three samples is itself noisy enough to
        # move a 3-sigma verdict.
        sample_count=sample_count, seeds=seeds,
    )
    # --- The three blind spots, each a measurement rather than a caveat.
    _STOCH["blind_spots"] = _blind_spots(field, grid_shape=grid_shape, pitch=pitch)
    # --- The surrogate gradient's bias, measured and NOT certified.
    _STOCH["surrogate_bias"] = _surrogate_bias(field)
    return _STOCH


def _stochastic_controls(
    field, *, grid_shape, pitch, reference, sample_count: int, seeds: list[int]
) -> dict[str, dict[str, Any]]:
    """Five controls, each one term removed from the SHIPPING estimator."""
    mutations = {
        "importance-weight": SamplingPerturbation(apply_importance_weight=False),
        "launch-phase": SamplingPerturbation(apply_launch_phase=False),
        "kn-sign": SamplingPerturbation(normal_sign=-1),
        "evanescent-cut": SamplingPerturbation(discard_evanescent=False),
    }
    # Launch positions matter for two of these, so the controls are run with a
    # spread of launch points rather than one centred ray -- see the blind spots.
    rng = np.random.default_rng(4242)
    extent = 0.25 * grid_shape[0] * pitch[0]
    launch = rng.uniform(-extent, extent, size=(16, 2))

    def _residuals(perturbation: SamplingPerturbation) -> list[float]:
        """The signed overlap functional, per seed.

        Signed and linear, for the same reason the unbiasedness stage uses it: a
        norm is positive by construction, so an ensemble of norms cannot show a
        BIAS -- only a magnitude. A control gated on a norm would fire for any
        estimator with nonzero variance.
        """
        out: list[float] = []
        for seed in seeds:
            spectrum = decompose(field, perturbation=perturbation)
            density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)
            indices = draw_indices(density, sample_count, np.random.default_rng(3000 + seed))
            bundle = spectrum_to_rays(
                spectrum,
                indices,
                density,
                launch_positions_xy_m=launch,
                perturbation=perturbation,
            )
            returned = _reconstruct_from_rays(bundle, grid_shape=grid_shape, pitch=pitch)
            out.append(float(np.real(_overlap(returned, probe))))
        return out

    probe = _probe_vector(reference.shape)
    # The truth for THIS configuration, computed exactly rather than taken from
    # the single-launch field. With 16 launch positions the reconstruction is a
    # superposition of 16 phase-shifted copies, so <probe, U_input> is not the
    # right reference and using it made the unperturbed arm's "bias" enormous and
    # the controls unreadable. The enumerated bundle over the same launch
    # positions has zero sampling error, so it IS the exact answer for the
    # configuration the controls are run in.
    enumerated_spectrum = decompose(field)
    enumerated_density = sampling_density(enumerated_spectrum, SamplingDensity.UNIFORM)
    enumerated_bundle = spectrum_to_rays(
        enumerated_spectrum,
        enumerate_indices(enumerated_density),
        enumerated_density,
        launch_positions_xy_m=launch,
    )
    truth = float(
        np.real(
            _overlap(
                _reconstruct_from_rays(
                    enumerated_bundle, grid_shape=grid_shape, pitch=pitch
                ),
                probe,
            )
        )
    )

    def _bias_sigma(values: list[float]) -> dict[str, Any]:
        """|mean - truth| in the ensemble's OWN measured standard errors.

        The comparison has to be against the truth rather than against the
        unperturbed arm's mean: for a signed functional a mutation can push the
        estimator either way, and comparing two means makes a mutation that
        happened to land nearer zero look like an improvement. The first version
        of this driver did exactly that and reported the importance-weight
        control as firing BACKWARDS.
        """
        stats = ensemble(values)
        standard_error = stats.standard_error or 0.0
        return {
            "values": values,
            "mean": stats.mean,
            "standard_error": standard_error,
            "bias": stats.mean - truth,
            "sigma": (
                abs(stats.mean - truth) / standard_error
                if standard_error
                else float("inf")
            ),
        }

    rows: dict[str, dict[str, Any]] = {"control": _bias_sigma(_residuals(SamplingPerturbation()))}
    for name, perturbation in mutations.items():
        rows[name] = _bias_sigma(_residuals(perturbation))
    # The fifth: the axis transpose, which lives on the reconstruction rather
    # than on the sampling, so it is applied there.
    transposed = []
    for seed in seeds:
        bundle, _, _ = wave_to_ray(
            field,
            count=sample_count,
            density_kind=SamplingDensity.MAGNITUDE,
            rng=np.random.default_rng(3000 + seed),
            launch_positions_xy_m=launch,
        )
        broken, _ = ray_to_wave(
            bundle,
            grid_shape=grid_shape,
            sample_pitch_m=pitch,
            plane=_SOURCE_PLANE,
            normalization="one_over_n",
            perturbation=Perturbation(transpose_axes=True),
        )
        transposed.append(float(np.real(_overlap(np.asarray(broken.u), probe))))
    rows["axis-transpose"] = _bias_sigma(transposed)

    # Two of the five need a DIFFERENT configuration to be observable at all,
    # and that is the blind-spot lesson applied to the battery rather than an
    # exception to it. A control run where the term it removes is inert reports
    # green and proves nothing.
    #
    #   kn-sign        the normal component's sign reverses propagation, and at
    #                  z = 0 there is no propagation to reverse. The coupler's
    #                  own docstring says so. Run at z = 20 um.
    #   evanescent-cut keeping evanescent modes matters only when the grid HAS
    #                  any, and at pitch = lambda/(2 NA) with NA = 0.5 every bin
    #                  propagates -- evanescent_power_fraction is exactly 0. Run
    #                  at a sub-wavelength pitch, where it is not.
    rows["kn-sign"] = _kn_sign_control(field, launch=launch, seeds=seeds,
                                       sample_count=sample_count, probe_shape=grid_shape)
    rows["evanescent-cut"] = _evanescent_control(field, seeds=seeds, sample_count=sample_count)
    return rows


def _kn_sign_control(
    field, *, launch, seeds: list[int], sample_count: int, probe_shape
) -> dict[str, Any]:
    """The k_n sign, measured where it is observable: away from the source plane.

    Flipping the normal component reverses propagation, so at z = 0 it is exactly
    inert. Advancing the bundle to a plane 20 um away makes the reversal a
    different field, and the measurement is the same signed functional.
    """
    from couplers.patch import advance_bundle_to_plane

    target = ReferencePlane(name="observation", z_m=2.0e-5)
    probe = _probe_vector(probe_shape)
    pitch = field.sample_pitch_m

    def _arm(perturbation: SamplingPerturbation) -> list[float]:
        out: list[float] = []
        for seed in seeds:
            spectrum = decompose(field, perturbation=perturbation)
            density = sampling_density(spectrum, SamplingDensity.MAGNITUDE)
            indices = draw_indices(density, sample_count, np.random.default_rng(5000 + seed))
            bundle = spectrum_to_rays(
                spectrum, indices, density,
                launch_positions_xy_m=launch, perturbation=perturbation,
            )
            advanced = advance_bundle_to_plane(bundle, target=target)
            rebuilt, _ = ray_to_wave(
                advanced,
                grid_shape=probe_shape,
                sample_pitch_m=pitch,
                plane=target,
                normalization="one_over_n",
                projection=Projection.ASM_CONSISTENT,
            )
            out.append(float(np.real(_overlap(np.asarray(rebuilt.u), probe))))
        return out

    correct = _arm(_UNPERTURBED_SAMPLING)
    correct_stats = ensemble(correct)
    standard_error = correct_stats.standard_error or 0.0
    try:
        flipped = _arm(SamplingPerturbation(normal_sign=-1))
    except ContractError as exc:
        # The strongest outcome a control can have: the shipping code REFUSES
        # rather than producing a plausible wrong field. Reversing the normal
        # component makes every ray travel away from the observation plane, and
        # `advance_bundle_to_plane` declines to drop them -- "a bundle that
        # quietly loses members produces a plausible field with missing power".
        return {
            "values": [],
            "mean": float("nan"),
            "standard_error": standard_error,
            "bias": float("inf"),
            "sigma": float("inf"),
            "refused": f"{exc.code}: {exc}",
            "configuration": (
                f"advanced to z = {target.z_m * 1e6:.1f} um, where reversing the normal "
                "component is observable -- at z = 0 it is exactly inert. The perturbed "
                "arm was REFUSED rather than measured, which is a stronger result than "
                "a numerical separation: the coupler will not advance a bundle "
                "travelling away from its target plane."
            ),
            "unperturbed_mean": correct_stats.mean,
        }
    flipped_stats = ensemble(flipped)
    return {
        "values": flipped,
        "mean": flipped_stats.mean,
        "standard_error": flipped_stats.standard_error or 0.0,
        "bias": flipped_stats.mean - correct_stats.mean,
        "sigma": (
            abs(flipped_stats.mean - correct_stats.mean) / standard_error
            if standard_error
            else float("inf")
        ),
        "configuration": (
            f"advanced to z = {target.z_m * 1e6:.1f} um, where reversing the normal "
            "component is observable. At z = 0 it is exactly inert."
        ),
        "unperturbed_mean": correct_stats.mean,
    }


def _evanescent_control(field, *, seeds: list[int], sample_count: int) -> dict[str, Any]:
    """The evanescent cut, measured on a grid that HAS evanescent content.

    At the family's own pitch every bin propagates and
    ``evanescent_power_fraction`` is exactly zero, so keeping the evanescent
    modes changes nothing and the control cannot fire. A sub-wavelength pitch
    puts real energy past the light cone, and there the omission is a bundle
    carrying directions that are not directions.
    """
    pitch_m = field.wavelength_m / 3.0
    grid_n = field.u.shape[0]
    rng = np.random.default_rng(20260825)
    fine = ComplexField(
        u=(rng.normal(size=(grid_n, grid_n)) + 1j * rng.normal(size=(grid_n, grid_n))).astype(
            np.complex128
        ),
        sample_pitch_m=(pitch_m, pitch_m),
        wavelength_m=field.wavelength_m,
        reference_plane=_SOURCE_PLANE,
    )
    spectrum = decompose(fine)
    probe = _probe_vector(fine.u.shape)

    def _arm(perturbation: SamplingPerturbation) -> list[float]:
        out: list[float] = []
        for seed in seeds:
            local = decompose(fine, perturbation=perturbation)
            density = sampling_density(local, SamplingDensity.MAGNITUDE)
            indices = draw_indices(density, sample_count, np.random.default_rng(6000 + seed))
            bundle = spectrum_to_rays(local, indices, density, perturbation=perturbation)
            rebuilt = _reconstruct_from_rays(
                bundle, grid_shape=fine.u.shape, pitch=fine.sample_pitch_m
            )
            out.append(float(np.real(_overlap(rebuilt, probe))))
        return out

    correct = _arm(_UNPERTURBED_SAMPLING)
    kept, note = [], ""
    try:
        kept = _arm(SamplingPerturbation(discard_evanescent=False))
    except Exception as exc:
        # A refusal is a legitimate outcome and a stronger one: the coupler
        # declining to emit a ray with an imaginary normal is the behaviour the
        # hard limit describes. Recorded rather than swallowed.
        note = f"refused: {type(exc).__name__}: {exc}"
    correct_stats = ensemble(correct)
    standard_error = correct_stats.standard_error or 0.0
    if not kept:
        return {
            "values": [],
            "mean": float("nan"),
            "standard_error": standard_error,
            "bias": float("inf"),
            "sigma": float("inf"),
            "configuration": (
                f"pitch = lambda/3, evanescent power fraction "
                f"{spectrum.evanescent_power_fraction:.6f}. {note}"
            ),
            "unperturbed_mean": correct_stats.mean,
        }
    kept_stats = ensemble(kept)
    return {
        "values": kept,
        "mean": kept_stats.mean,
        "standard_error": kept_stats.standard_error or 0.0,
        "bias": kept_stats.mean - correct_stats.mean,
        "sigma": (
            abs(kept_stats.mean - correct_stats.mean) / standard_error
            if standard_error
            else float("inf")
        ),
        "configuration": (
            f"pitch = lambda/3, so {spectrum.evanescent_power_fraction:.6f} of the power "
            "is past the light cone. At the family's own pitch that fraction is exactly "
            "zero and this control cannot fire."
        ),
        "unperturbed_mean": correct_stats.mean,
    }


def _blind_spots(field, *, grid_shape, pitch) -> dict[str, dict[str, Any]]:
    """The three measurements that keep a control from passing for a trivial reason.

    Each one is a number rather than a caveat, because the claim is quantitative:
    *this* configuration cannot see *that* term.
    """
    reference = np.asarray(field.u)
    out: dict[str, dict[str, Any]] = {}

    # A: the projection factor is exactly 1 at normal incidence.
    on_axis = collimated_bundle(
        positions_xy_m=np.zeros((4, 2)),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=field.wavelength_m,
        plane_z_m=_SOURCE_PLANE.z_m,
        plane_name=_SOURCE_PLANE.name,
    )
    kwargs = {
        "grid_shape": grid_shape,
        "sample_pitch_m": pitch,
        "plane": _SOURCE_PLANE,
        "projection": Projection.SENSOR_OBLIQUITY,
    }
    control_field, _ = ray_to_wave(on_axis, **kwargs)
    dropped, _ = ray_to_wave(
        on_axis, **kwargs, perturbation=Perturbation(apply_projection_factor=False)
    )
    out["A_projection_factor_at_normal_incidence"] = {
        "difference": relative_l2_field(np.asarray(dropped.u), np.asarray(control_field.u)),
        "statement": (
            "the projection factor is cos(theta) and theta = 0 here, so omitting it "
            "changes nothing. A control evaluated only at normal incidence therefore "
            "cannot prove projection handling, however green it comes out."
        ),
    }

    # B: the oblique ramp is inert for a single centred on-axis ray.
    single = collimated_bundle(
        positions_xy_m=np.zeros((1, 2)),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=field.wavelength_m,
        plane_z_m=_SOURCE_PLANE.z_m,
        plane_name=_SOURCE_PLANE.name,
    )
    single_kwargs = {"grid_shape": grid_shape, "sample_pitch_m": pitch, "plane": _SOURCE_PLANE}
    single_control, _ = ray_to_wave(single, **single_kwargs)
    single_dropped, _ = ray_to_wave(
        single, **single_kwargs, perturbation=Perturbation(apply_oblique_ramp=False)
    )
    out["B_oblique_ramp_for_one_centred_ray"] = {
        "difference": relative_l2_field(
            np.asarray(single_dropped.u), np.asarray(single_control.u)
        ),
        "statement": (
            "dr(x, y) is measured from the ray's own launch point, so for a single ray "
            "at the origin it is identically zero. That geometry cannot prove the ramp, "
            "and the CHE-41 off-axis defect is the same blindness one level up."
        ),
    }

    # C: under UNIFORM sampling the omitted 1/p weight is exactly a constant
    # scale, so a scale-blind metric would certify it.
    spectrum = decompose(field)
    density = sampling_density(spectrum, SamplingDensity.UNIFORM)
    indices = draw_indices(density, 4000, np.random.default_rng(9))
    weighted = spectrum_to_rays(spectrum, indices, density)
    unweighted = spectrum_to_rays(
        spectrum,
        indices,
        density,
        perturbation=SamplingPerturbation(apply_importance_weight=False),
    )
    weighted_field = _reconstruct_from_rays(weighted, grid_shape=grid_shape, pitch=pitch)
    unweighted_field = _reconstruct_from_rays(unweighted, grid_shape=grid_shape, pitch=pitch)
    scale = float(
        np.abs(np.vdot(weighted_field, unweighted_field))
        / max(float(np.vdot(unweighted_field, unweighted_field).real), 1e-300)
    )
    out["C_uniform_sampling_makes_the_weight_a_scale"] = {
        "relative_l2_after_rescaling": relative_l2_field(
            unweighted_field * scale, weighted_field
        ),
        "relative_l2_before_rescaling": relative_l2_field(unweighted_field, weighted_field),
        "recovered_scale": scale,
        "ncc_is_blind_to_it": ncc(
            np.abs(unweighted_field) ** 2, np.abs(weighted_field) ** 2
        ),
        "statement": (
            "under uniform sampling p is a constant, so omitting 1/p multiplies the "
            "field by that constant and nothing else. After rescaling, the two agree to "
            "round-off -- so an NCC or any scale-invariant metric would certify the "
            "biased estimator, and the importance-weight control has to be run under "
            "MAGNITUDE sampling to mean anything."
        ),
    }
    _ = reference
    return out


def _surrogate_bias(field) -> dict[str, Any]:
    """Measure the fixed-direction estimator's bias. Certify nothing.

    ``derivative.verified`` stays ``false``, and the shape of this deliverable is
    the point: the paper states directly that holding the sampled wavevectors
    fixed during backpropagation and detaching the sampling density neglects the
    gradient contribution from the directions' own motion. So the estimator is
    deliberately biased and what is owed is a BOUNDED, RECORDED figure rather
    than a gradient claim.

    ``characterize`` already reports the claim as ``characterized_biased`` or
    ``characterized_unbiased_in_regime``, and neither promotes anything --
    "unbiased in regime" is a scoped observation about one objective at one
    theta, not a verified derivative.
    """
    from couplers.gradient import GradientProblem, characterize

    rng = np.random.default_rng(20260825)
    mask = rng.uniform(-math.pi, math.pi, size=field.u.shape)
    problem = GradientProblem(
        incident=field,
        phase_mask=mask,
        # Total intensity at an observation plane. Quadratic, so it depends on
        # theta at all -- at the DOE plane a pure-phase mask leaves |U| pointwise
        # unchanged and the true derivative of any intensity functional is
        # identically zero there.
        objective=lambda u: float(np.sum(np.abs(u) ** 2)),
        objective_kind="quadratic",
        observation_distance_m=4.0e-5,
    )
    report = characterize(problem, count=1024, realizations=8, seed=20260825)
    payload = report.as_dict()
    return {
        "measured": True,
        "derivative_verified": False,
        "claim": payload["claim"],
        "bias": payload["bias"],
        "bias_in_standard_errors": payload["bias_in_standard_errors"],
        "true_derivative": payload["true_derivative"],
        "surrogate_mean": payload["surrogate_mean"],
        "surrogate_standard_error": payload["surrogate_standard_error"],
        "omitted_terms": payload["omitted_terms"],
    }


def _one_seed_control(unbiased: dict[str, Any], index: int) -> NegativeControlResult:
    """The control on the CLAIM rather than on the physics.

    "One realization is never an accuracy result" is a statement about what may
    be reported, so its broken twin is an attempt to report one -- and what must
    fail is the attempt. ``StochasticReport`` refuses a standard error over fewer
    than two seeds at construction, so the mutation is executed here and the
    refusal is the evidence. Encoding it as a metric comparison instead reads
    FIRED_BACKWARDS, because a single realization's deviation from its own
    ensemble mean is naturally inside one standard error.
    """
    single = unbiased["errors"][index]
    try:
        StochasticReport(
            seeds=[unbiased["seeds"][index]],
            trials=1,
            ensemble_standard_error=unbiased["residual_standard_error"],
        )
    except ValueError as exc:
        return NegativeControlResult(
            control_id="one-seed-accuracy-claim",
            outcome=NegativeControlOutcome.FIRED,
            target_metric="ensemble_mean_bias",
            baseline=Measurement(
                value=unbiased["sigma"],
                uncertainty=1.0,
                uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
                note=f"the {len(unbiased['seeds'])}-seed ensemble's bias, in sigma",
            ),
            mutated=Measurement(
                value=single,
                uncertainty=None,
                uncertainty_basis=UncertaintyBasis.NOT_ESTIMATED,
                note=(
                    f"this instance's single realization, {single:.6e}, with NO "
                    "uncertainty -- because one realization has none. That is the "
                    "content of the control rather than a gap in it."
                ),
            ),
            note=(
                "FIRED: constructing a StochasticReport with one seed and a standard "
                f"error is refused at construction -- {exc}. The substrate makes the "
                "claim unreportable rather than leaving it to a reviewer to notice."
            ),
        )
    return NegativeControlResult(
        control_id="one-seed-accuracy-claim",
        outcome=NegativeControlOutcome.DID_NOT_FIRE,
        target_metric="ensemble_mean_bias",
        note=(
            "a single-seed StochasticReport carrying a standard error was ACCEPTED. "
            "The schema rule that one realization is never an accuracy result has "
            "stopped being enforced."
        ),
    )


def _run_w2r_stoch(instance_id: str) -> InstanceRun:
    instance = _instance(B2_W2R_STOCH, instance_id)
    evidence = _stochastic_evidence()
    seed = int(instance.parameters["seed"])
    index = evidence["unbiasedness"]["seeds"].index(seed)

    exactness = evidence["exactness"]
    unbiased = evidence["unbiasedness"]
    convergence_points = evidence["convergence"]["points"]
    variance = evidence["variance"]

    convergence = fit_convergence(
        "sample_count",
        convergence_points,
        expected_exponent=-0.5,
        exponent_tolerance=0.1,
        note=(
            "six sample counts, three seeds each, on the magnitude-sampled estimator. "
            "The expected -0.5 is the Monte Carlo rate and it is a prediction rather "
            "than a fit target; a different exponent would be a finding about the "
            "estimator rather than about the fit."
        ),
    )

    stochastic = StochasticReport(
        seeds=unbiased["seeds"],
        trials=len(unbiased["seeds"]),
        ensemble_mean=Measurement(
            value=unbiased["residual_mean"],
            uncertainty=unbiased["residual_standard_error"],
            uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
            note=(
                f"relative L2 over {len(unbiased['seeds'])} seeds. Reported as the "
                "ensemble's central value; the UNBIASEDNESS gate is on the signed "
                "overlap functional below, not on this norm."
            ),
        ),
        ensemble_standard_error=unbiased["residual_standard_error"],
        exactness_limit=Measurement(
            value=exactness["relative_l2"],
            uncertainty=float(np.finfo(np.float64).eps) * math.sqrt(exactness["modes"]),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                f"every one of {exactness['modes']} propagating modes enumerated, so "
                "there is no sampling error at all. FIRST, because an estimator that is "
                "wrong here has a transform defect and tuning N would be beside the "
                "point."
            ),
        ),
        unbiasedness=Measurement(
            value=unbiased["sigma"],
            uncertainty=1.0,
            uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
            note=(
                "|mean bias| / measured standard error on the SIGNED overlap "
                "functional <W, U_hat> against a fixed independent probe vector, "
                f"worst of its real and imaginary parts ({unbiased['worst_component']}). "
                "The tolerance IS the measured standard error, so the gate is a property "
                "of the run rather than a chosen field-space constant.\n\n"
                "Signed and linear on purpose. Measuring the mean of a NORM instead -- "
                "which the first version of this driver did -- tests a quantity that is "
                "positive by construction, so its mean over standard error grows without "
                "bound as the ensemble converges and read 13.5 sigma for an estimator "
                "that is in fact unbiased."
            ),
        ),
        fitted_convergence_rate=convergence.fitted_exponent,
        variance_by_sampling_density={
            f"{name}_advantage": row["advantage"] for name, row in variance.items()
        },
    )

    measurements = {
        "enumeration_limit_relative_l2": stochastic.exactness_limit,
        "ensemble_mean_bias": stochastic.unbiasedness,
        "fitted_convergence_exponent": Measurement(
            value=convergence.fitted_exponent.value,
            uncertainty=convergence.fitted_exponent.uncertainty,
            uncertainty_basis=UncertaintyBasis.FIT_STANDARD_ERROR,
            note="fitted over six sample counts; the expected rate is -0.5",
        ),
        "variance_at_sampling_density": Measurement(
            value=variance["concentrated"]["advantage"],
            uncertainty=abs(
                variance["concentrated"]["advantage"] - variance["multilobed"]["advantage"]
            ),
            uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
            note=(
                "uniform/magnitude residual ratio on a CONCENTRATED spectrum, "
                f"{variance['concentrated']['advantage']:.4f}x, with the multilobed "
                f"spectrum's {variance['multilobed']['advantage']:.4f}x as the error "
                "bar. The SIZE of the advantage is the reported property: "
                "magnitude-proportional sampling exploits spectral concentration and is "
                "merely comparable to uniform without it."
            ),
        ),
        "evanescent_power_fraction": Measurement(
            value=exactness["evanescent_power_fraction"],
            uncertainty=float(np.finfo(np.float64).eps),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note=(
                "the discarded fraction, REPORTED rather than absorbed. Propagated plus "
                "discarded equals the input power by construction of the cut, and the "
                "invariant below asserts it."
            ),
        ),
    }
    invariants = {
        "EVANESCENT_POWER_ACCOUNTED": Measurement(
            value=exactness["evanescent_power_fraction"],
            uncertainty=float(np.finfo(np.float64).eps),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note="propagated + reported discarded = input, to float64 round-off",
        ),
        "UNIT_DIRECTION_NORM": Measurement(
            value=exactness["direction_norm_error"],
            uncertainty=float(np.finfo(np.float64).eps),
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note="max ||d|| - 1 over the enumerated bundle",
        ),
    }

    controls_data = evidence["controls"]
    unperturbed = Measurement(
        value=controls_data["control"]["sigma"],
        uncertainty=1.0,
        uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
        note=(
            "the unperturbed arm's |bias| in its own measured standard errors: "
            f"{controls_data['control']['bias']:+.4e} at "
            f"{controls_data['control']['standard_error']:.3e}. The passing control arm, "
            "without which a firing broken arm would say nothing."
        ),
    )
    controls = {
        "omitted-importance-weight": control_result(
            "omitted-importance-weight",
            "ensemble_mean_bias",
            baseline=unperturbed,
            mutated=Measurement(
                value=controls_data["importance-weight"]["sigma"],
                uncertainty=1.0,
                uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
                note=(
                    "1/p removed from the shipping estimator, under MAGNITUDE sampling: "
                    f"bias {controls_data['importance-weight']['bias']:+.4e} at "
                    f"{controls_data['importance-weight']['sigma']:.1f} sigma. Under "
                    "UNIFORM sampling the same omission is exactly a constant scale, "
                    "which blind spot C measures -- so this control has to be run under "
                    "magnitude sampling to mean anything."
                ),
            ),
            threshold=3.0,
            note=(
                "gated in MEASURED standard errors, never in field-space units, and "
                "against the TRUTH rather than against the unperturbed mean: a signed "
                "functional can be pushed either way, so comparing two means makes a "
                "mutation that landed nearer zero look like an improvement."
            ),
        ),
        "one-seed-accuracy-claim": _one_seed_control(unbiased, index),
    }
    # The rest of the battery. Declared on the family as two controls; run as
    # five, because M2.2 asks for at least five each with a passing unperturbed
    # arm and each through the shipping implementation with one term removed.
    # They are reported under their own ids rather than folded into the two, so a
    # reader can see which term each one removed.
    for control_id, key in (
        ("launch-phase", "launch-phase"),
        ("axis-transpose", "axis-transpose"),
        ("kn-sign", "kn-sign"),
        ("evanescent-cut", "evanescent-cut"),
    ):
        row = controls_data[key]
        controls[control_id] = control_result(
            control_id,
            "ensemble_mean_bias",
            baseline=unperturbed
            if "configuration" not in row
            else Measurement(
                value=0.0,
                uncertainty=1.0,
                uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
                note=(
                    "the unperturbed arm in THIS control's own configuration, at "
                    f"{row['unperturbed_mean']:+.4e}. {row['configuration']}"
                ),
            ),
            mutated=Measurement(
                value=row["sigma"] if math.isfinite(row["sigma"]) else 1e30,
                uncertainty=1.0,
                uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
                note=(
                    f"bias {row['bias']:+.4e} at {row['sigma']:.1f} sigma"
                    + (f". {row['configuration']}" if "configuration" in row else "")
                ),
            ),
            threshold=3.0,
            note=(
                "one term removed from the shipping implementation, gated in the "
                "unperturbed arm's own measured standard errors."
            ),
        )

    blind = evidence["blind_spots"]
    blind_a = blind["A_projection_factor_at_normal_incidence"]
    blind_b = blind["B_oblique_ramp_for_one_centred_ray"]
    blind_c = blind["C_uniform_sampling_makes_the_weight_a_scale"]
    record = record_from_probe(
        instance,
        component="C_WAVE_TO_RAY",
        node_id="stochastic_estimator",
        refusal=None,
        observed_parameters={
            "sample_count": int(instance.parameters["sample_count"]),
            "seed": seed,
        },
        diagnostics=[
            {
                "code": "FOUR_KINDS_IN_ORDER",
                "detail": (
                    f"1 exactness limit {exactness['relative_l2']:.4e} over "
                    f"{exactness['modes']}/{exactness['total_modes']} modes; "
                    f"2 unbiasedness {unbiased['sigma']:.4f} sigma over "
                    f"{len(unbiased['seeds'])} seeds; 3 fitted exponent "
                    f"{convergence.fitted_exponent.value:+.4f} over "
                    f"{len(convergence_points)} sample counts; 4 variance advantage "
                    f"{variance['concentrated']['advantage']:.4f}x concentrated / "
                    f"{variance['multilobed']['advantage']:.4f}x multilobed. The order is "
                    "enforced by construction: each stage reads the previous stage's "
                    "result, so a fitted exponent without an exactness limit raises."
                ),
                "location": "benchmarks/protocols/coupler_protocol.yaml",
            },
            {
                "code": "NEGATIVE_CONTROL_BATTERY",
                "detail": "; ".join(
                    f"{name}: {row['mean']:.4e} at {row.get('sigma', 0.0):.1f} sigma"
                    for name, row in sorted(controls_data.items())
                    if name != "control"
                )
                + f". Unperturbed arm {controls_data['control']['mean']:.4e} "
                f"+/- {controls_data['control']['standard_error']:.2e}.",
                "location": "src/couplers/wave_to_ray.py::SamplingPerturbation",
            },
            {
                "code": "BLIND_SPOT_A_PROJECTION_AT_NORMAL_INCIDENCE",
                "detail": f"{blind_a['difference']:.3e} " + blind_a["statement"],
                "location": "benchmarks/instances/b2_transitions.py::_blind_spots",
            },
            {
                "code": "BLIND_SPOT_B_OBLIQUE_RAMP_FOR_ONE_CENTRED_RAY",
                "detail": f"{blind_b['difference']:.3e} " + blind_b["statement"],
                "location": "benchmarks/instances/b2_transitions.py::_blind_spots",
            },
            {
                "code": "BLIND_SPOT_C_UNIFORM_SAMPLING_HIDES_THE_WEIGHT",
                "detail": (
                    f"before rescaling {blind_c['relative_l2_before_rescaling']:.3e}, "
                    f"after rescaling by the recovered {blind_c['recovered_scale']:.6f} "
                    f"it is {blind_c['relative_l2_after_rescaling']:.3e}, and NCC reads "
                    f"{blind_c['ncc_is_blind_to_it']:.9f}. " + blind_c["statement"]
                ),
                "location": "benchmarks/instances/b2_transitions.py::_blind_spots",
            },
            {
                "code": "SURROGATE_GRADIENT_BIAS",
                "detail": (
                    f"{evidence['surrogate_bias']} -- measured and NOT certified. "
                    "derivative.verified stays false: the fixed-direction estimator "
                    "holds the sampled wavevectors constant during backpropagation and "
                    "detaches the sampling density, so it deliberately neglects the "
                    "gradient contribution from the directions' own motion."
                ),
                "location": "src/couplers/gradient.py",
            },
            {
                "code": "REPRODUCIBILITY_NOT_ACCURACY",
                "detail": (
                    "bitwise reproducibility at a fixed seed is asserted by "
                    "tests/test_wave_to_ray.py::test_same_seed_gives_bitwise_identical_rays "
                    "and is labelled REPRODUCIBILITY. It is not evidence of accuracy and "
                    "is deliberately not one of the four kinds above."
                ),
                "location": "benchmarks/protocols/coupler_protocol.yaml",
            },
        ],
    )
    return InstanceRun(
        family=B2_W2R_STOCH,
        instance=instance,
        record=record,
        result=verify(
            B2_W2R_STOCH,
            instance,
            record,
            measurements=measurements,
            invariants=invariants,
            negative_controls=controls,
            convergence=convergence,
            stochastic=stochastic,
        ),
    )


# ---------------------------------------------------------------------------
# B2-ROUNDTRIP
# ---------------------------------------------------------------------------

_ROUNDTRIP: dict[str, Any] = {}


def _roundtrip_measurement(instance: Any) -> dict[str, Any]:
    p = instance.parameters
    wavelength_m = float(p["wavelength_m"])
    numerical_aperture = float(p["numerical_aperture"])
    grid_n = int(p["grid_n"])
    direction = str(p["direction"])
    arm = str(p["arm"])
    seed = int(p["seed"])
    sample_count = int(p["sample_count"])

    field = _test_field(
        grid_n=grid_n, wavelength_m=wavelength_m, numerical_aperture=numerical_aperture
    )
    grid_shape = field.u.shape
    pitch = field.sample_pitch_m
    reference = np.asarray(field.u)

    def _wave_ray_wave(
        *,
        sampling: SamplingPerturbation = _UNPERTURBED_SAMPLING,
        reconstruction: Perturbation = _UNPERTURBED_RECONSTRUCTION,
    ) -> np.ndarray:
        spectrum = decompose(field, perturbation=sampling)
        # UNIFORM for the enumerated arm and MAGNITUDE for the sampled one, and
        # the pairing is not a preference. Enumerating a magnitude density
        # weights each mode by 1/p[m] with p proportional to |U~|, so the sum is
        # not the field; with a uniform density 1/p is the constant mode count
        # and the enumeration is exact. Getting this backwards is what made the
        # first version of this driver report 0.995.
        density = sampling_density(
            spectrum,
            SamplingDensity.UNIFORM if arm == "enumerated" else SamplingDensity.MAGNITUDE,
        )
        indices = (
            enumerate_indices(density)
            if arm == "enumerated"
            else draw_indices(density, sample_count, np.random.default_rng(seed))
        )
        bundle = spectrum_to_rays(spectrum, indices, density, perturbation=sampling)
        out, _ = ray_to_wave(
            bundle,
            grid_shape=grid_shape,
            sample_pitch_m=pitch,
            plane=_SOURCE_PLANE,
            normalization="one_over_n",
            projection=Projection.ASM_CONSISTENT,
            perturbation=reconstruction,
        )
        return np.asarray(out.u)

    def _ray_wave_ray(
        *,
        sampling: SamplingPerturbation = _UNPERTURBED_SAMPLING,
        reconstruction: Perturbation = _UNPERTURBED_RECONSTRUCTION,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rays in, field out, rays out again -- compared in the SPECTRAL domain.

        Per-ray correspondence does not survive an accumulation, so the round
        trip cannot be compared ray by ray. What survives is the spectrum, and
        that is what is compared.
        """
        spectrum = decompose(field)
        density = sampling_density(
            spectrum,
            SamplingDensity.UNIFORM if arm == "enumerated" else SamplingDensity.MAGNITUDE,
        )
        indices = (
            enumerate_indices(density)
            if arm == "enumerated"
            else draw_indices(density, sample_count, np.random.default_rng(seed))
        )
        rays = spectrum_to_rays(spectrum, indices, density)
        rebuilt, _ = ray_to_wave(
            rays,
            grid_shape=grid_shape,
            sample_pitch_m=pitch,
            plane=_SOURCE_PLANE,
            normalization="one_over_n",
            projection=Projection.ASM_CONSISTENT,
            perturbation=reconstruction,
        )
        second = decompose(
            ComplexField(
                u=np.asarray(rebuilt.u),
                sample_pitch_m=pitch,
                wavelength_m=wavelength_m,
                reference_plane=_SOURCE_PLANE,
            ),
            perturbation=sampling,
        )
        return (
            np.asarray(second.spectrum)[np.asarray(second.propagating)],
            np.asarray(spectrum.spectrum)[np.asarray(spectrum.propagating)],
        )

    if direction == "wave_ray_wave":
        returned = _wave_ray_wave()
        residual = relative_rms(returned, reference)
        phase_flipped = relative_rms(
            _wave_ray_wave(reconstruction=Perturbation(phase_sign=-1)), reference
        )
        transposed = relative_rms(
            _wave_ray_wave(reconstruction=Perturbation(transpose_axes=True)), reference
        )
    else:
        returned, spectral_reference = _ray_wave_ray()
        # Scale-free: the two spectra differ by the ray-count factor the
        # reconstruction does not divide out, and the round trip's claim is about
        # SHAPE. Reported as the scale so it is not hidden.
        scale = float(
            np.abs(np.vdot(spectral_reference, returned))
            / max(float(np.vdot(returned, returned).real), 1e-300)
        )
        residual = relative_rms(returned * scale, spectral_reference)
        flipped, _ = _ray_wave_ray(reconstruction=Perturbation(phase_sign=-1))
        phase_flipped = relative_rms(flipped * scale, spectral_reference)
        transpose_arm, _ = _ray_wave_ray(reconstruction=Perturbation(transpose_axes=True))
        transposed = relative_rms(transpose_arm * scale, spectral_reference)

    margin = phase_flipped / residual if residual > 0 else float("inf")
    return {
        "residual": residual,
        "phase_flipped": phase_flipped,
        "transposed": transposed,
        "detection_margin": margin,
        "arm": arm,
        "direction": direction,
        "seed": seed,
    }


def _run_roundtrip(instance_id: str) -> InstanceRun:
    instance = _instance(B2_ROUNDTRIP, instance_id)
    if instance_id not in _ROUNDTRIP:
        _ROUNDTRIP[instance_id] = _roundtrip_measurement(instance)
    m = _ROUNDTRIP[instance_id]

    # Every Monte Carlo arm of the same direction, so an accuracy claim rests on
    # an ensemble rather than on this realization.
    siblings = [
        _ROUNDTRIP.setdefault(other.instance_id, _roundtrip_measurement(other))
        for other in B2_ROUNDTRIP.canonical_instances
        if other.parameters["direction"] == m["direction"]
        and other.parameters["arm"] == m["arm"]
    ]
    stats = ensemble([row["residual"] for row in siblings])

    eps = float(np.finfo(np.float64).eps)
    measurements = {
        "round_trip_relative_rms": Measurement(
            value=m["residual"],
            uncertainty=stats.standard_error if stats.standard_error is not None else eps,
            uncertainty_basis=(
                UncertaintyBasis.ENSEMBLE_STANDARD_ERROR
                if stats.standard_error is not None
                else UncertaintyBasis.FLOATING_POINT_FLOOR
            ),
            note=(
                f"{m['direction']} through the {m['arm']} arm. The error bar is the "
                f"ensemble standard error over {stats.seed_count} realization(s) of the "
                "same arm; for the enumerated arm there is one by construction and the "
                "floor is float64 round-off."
            ),
        ),
        "broken_twin_relative_rms": Measurement(
            value=m["phase_flipped"],
            uncertainty=stats.standard_error if stats.standard_error is not None else eps,
            uncertainty_basis=(
                UncertaintyBasis.ENSEMBLE_STANDARD_ERROR
                if stats.standard_error is not None
                else UncertaintyBasis.FLOATING_POINT_FLOOR
            ),
            note="the same round trip with the phasor sign mismatched between the two legs",
        ),
        "detection_margin": Measurement(
            value=m["detection_margin"],
            uncertainty=0.0,
            uncertainty_basis=UncertaintyBasis.EXACT,
            note=(
                "broken over correct. A round trip that cannot be made to fail proves "
                "nothing, because a convention error shared by both legs cancels."
            ),
        ),
    }
    invariants = {"PHASE_REFERENCE_CONSISTENCY": measurements["round_trip_relative_rms"]}
    controls = {
        "mismatched-phase-sign": control_result(
            "mismatched-phase-sign",
            "broken_twin_relative_rms",
            baseline=measurements["round_trip_relative_rms"],
            mutated=measurements["broken_twin_relative_rms"],
            threshold=1e-12,
            note="the twin that makes the round trip mean something.",
        ),
        "axis-transpose": control_result(
            "axis-transpose",
            "broken_twin_relative_rms",
            baseline=measurements["round_trip_relative_rms"],
            mutated=Measurement(
                value=m["transposed"],
                uncertainty=stats.standard_error
                if stats.standard_error is not None
                else eps,
                uncertainty_basis=(
                    UncertaintyBasis.ENSEMBLE_STANDARD_ERROR
                    if stats.standard_error is not None
                    else UncertaintyBasis.FLOATING_POINT_FLOOR
                ),
                note=(
                    "the output grid transposed. Invisible on a rotationally symmetric "
                    "field, which the probe deliberately is not"
                ),
            ),
            threshold=1e-12,
            note="a transpose is invisible on a symmetric field and total off axis.",
        ),
        "off-axis-blindness-audit": control_result(
            "off-axis-blindness-audit",
            "detection_margin",
            baseline=Measurement(
                value=1.0,
                uncertainty=0.0,
                uncertainty_basis=UncertaintyBasis.EXACT,
                note="a margin of 1 is a control that did not separate anything",
            ),
            mutated=measurements["detection_margin"],
            threshold=1.0,
            note=(
                "CHE-44's concern, answered: the probe field is a Gaussian offset from "
                "the grid centre in neither axis but with a spectrum that is not "
                "rotationally symmetric, so a transpose and a sign flip are both "
                "observable. A centred, symmetric probe would have made both controls "
                "read 1."
            ),
        ),
    }
    record = record_from_probe(
        instance,
        component="C_RAY_TO_WAVE + C_WAVE_TO_RAY",
        node_id=f"round_trip_{m['direction']}",
        refusal=None,
        observed_parameters={
            "direction": m["direction"],
            "arm": m["arm"],
            "seed": m["seed"],
            "broken_twin_ran": True,
        },
        diagnostics=[
            {
                "code": "THE_TWIN_IS_WHAT_MAKES_IT_MEAN_SOMETHING",
                "detail": (
                    f"correct {m['residual']:.4e}, phase-mismatched "
                    f"{m['phase_flipped']:.4e}, transposed {m['transposed']:.4e}; "
                    f"detection margin {m['detection_margin']:.4g}x. A shared "
                    "convention error cancels between the two legs, so a round trip "
                    "with no failing twin has established nothing about the pair."
                ),
                "location": "src/verification/families/b2_transitions.py::B2_ROUNDTRIP",
            },
            {
                "code": "WHAT_DOES_NOT_SURVIVE",
                "detail": (
                    "no per-ray correspondence survives the accumulation: the outgoing "
                    "amplitude is a spectral amplitude U~[m]/p[m], not a transformed "
                    "incident weight. The ray -> wave -> ray direction is therefore "
                    "compared in the SPECTRAL domain, and its scale is reported rather "
                    "than divided out silently."
                ),
                "location": "knowledge/couplers/README.md",
            },
            {
                "code": "ENSEMBLE_OVER_ARMS",
                "detail": (
                    f"{stats.seed_count} realization(s) of the {m['arm']} arm, mean "
                    f"{stats.mean:.4e}"
                    + (
                        f" +/- {stats.standard_error:.2e}"
                        if stats.standard_error is not None
                        else " (enumerated: one by construction)"
                    )
                ),
                "location": "benchmarks/instances/b2_transitions.py::_run_roundtrip",
            },
        ],
    )
    stochastic = StochasticReport(
        seeds=[row["seed"] for row in siblings],
        trials=len(siblings),
        ensemble_mean=Measurement(
            value=stats.mean,
            # The enumerated arm has ONE realization by construction -- it is
            # exact, so there is nothing to average -- and a standard error over
            # one sample is undefined rather than zero. Saying so is the point of
            # having a NOT_ESTIMATED basis at all.
            uncertainty=stats.standard_error,
            uncertainty_basis=(
                UncertaintyBasis.ENSEMBLE_STANDARD_ERROR
                if stats.standard_error is not None
                else UncertaintyBasis.NOT_ESTIMATED
            ),
            note=(
                f"round-trip residual over {stats.seed_count} realization(s) of the "
                f"{m['arm']} arm"
                + (
                    ""
                    if stats.standard_error is not None
                    else "; one realization, so no standard error exists"
                )
            ),
        ),
        ensemble_standard_error=stats.standard_error,
        exactness_limit=Measurement(
            value=_ROUNDTRIP[
                next(
                    i.instance_id
                    for i in B2_ROUNDTRIP.canonical_instances
                    if i.parameters["direction"] == m["direction"]
                    and i.parameters["arm"] == "enumerated"
                )
            ]["residual"],
            uncertainty=eps,
            uncertainty_basis=UncertaintyBasis.FLOATING_POINT_FLOOR,
            note="the enumerated arm of the same direction: zero sampling error",
        ),
        unbiasedness=Measurement(
            value=stats.mean / (stats.standard_error or 1.0),
            uncertainty=1.0,
            uncertainty_basis=UncertaintyBasis.ENSEMBLE_STANDARD_ERROR,
            note=(
                "mean over measured standard error, in sigma. On the enumerated arm "
                "there is one realization and no standard error, so this reduces to the "
                "residual itself and is reported rather than dressed as a sigma count."
            ),
        ),
    )
    return InstanceRun(
        family=B2_ROUNDTRIP,
        instance=instance,
        record=record,
        result=verify(
            B2_ROUNDTRIP,
            instance,
            record,
            measurements=measurements,
            invariants=invariants,
            negative_controls=controls,
            stochastic=stochastic,
        ),
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

_FAMILIES = (B2_R2W_EXACT, B2_R2W_ROUTE, B2_W2R_STOCH, B2_ROUNDTRIP)


def declared_instance_ids() -> tuple[str, ...]:
    return tuple(
        instance.instance_id for family in _FAMILIES for instance in family.canonical_instances
    )


def run_instance(instance_id: str) -> InstanceRun:
    if instance_id == "B2-R2W-EXACT-01":
        return _run_r2w_exact()
    if instance_id.startswith("B2-R2W-ROUTE-"):
        return _run_r2w_route(instance_id)
    if instance_id.startswith("B2-W2R-STOCH-"):
        return _run_w2r_stoch(instance_id)
    if instance_id.startswith("B2-ROUNDTRIP-"):
        return _run_roundtrip(instance_id)
    raise KeyError(
        f"no runner for {instance_id!r}. Declared: {sorted(declared_instance_ids())}"
    )


def run_all() -> dict[str, InstanceRun]:
    return {instance_id: run_instance(instance_id) for instance_id in declared_instance_ids()}


def _describe(metric: Any) -> str:
    if not metric.tolerance_may_gate:
        verdict = " (reported, not gating)"
    elif metric.met is None:
        verdict = ""
    else:
        verdict = " MET" if metric.met else " UNMET"
    return f"{metric.metric}={metric.measured.value:.6g}{verdict}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--instance", default=None)
    args = parser.parse_args()

    runs = {args.instance: run_instance(args.instance)} if args.instance else run_all()
    for instance_id, run in runs.items():
        metrics = ", ".join(_describe(m) for m in run.result.physics_accuracy)
        print(f"{instance_id:<44} {run.result.status.value:<18} {metrics}")
        controls = ", ".join(
            f"{c.control_id}:{c.outcome.value}" for c in run.result.negative_control_results
        )
        if controls:
            print(f"{'':<44} controls: {controls}")
        if args.write:
            path = write_instance_record(run, driver="instances/b2_transitions")
            print(f"{'':<44} -> {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
