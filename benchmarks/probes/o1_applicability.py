"""Can O1 decide ``B3-PSF-SINGLET``'s gate at ``1e-3``? No, and by how much.

CHE-117 (M4.2), the second half. The first half
(:mod:`benchmarks.probes.singlet_residual_attribution`) established that the
``2.2072391812867093e-3`` residual is *converged* -- flat to 0.87% from 49,537 to
3,148,801 rays, invariant to ten significant figures under an 8x sensor-pitch
refinement -- and that it is not caused by the production quadrature weight,
because the uniform arm converges to the same number from below. It closed with
one question narrowed rather than answered: **can an aberration-free paraxial
Airy oracle decide a real traced singlet at the 1e-3 level at all?**

This probe answers it. Three measurements, and the answer is no.

**1. What the gate metric actually resolves.** ``[2 J1(v)/v]^2`` has one free
parameter, the Airy scale, and the gate metric is *linear* in a fractional error
in it: comparing O1 at ``NA`` against O1 at ``NA (1 + eps)`` over the same
5-Airy-radius disc gives ``rel-L2 = 1.53 eps`` across four decades of ``eps``.
So the ``1.0e-3`` gate is a statement that the Airy scale is known to
``6.5e-4`` -- 0.065%. This experiment uses O1 on both sides. No coupler, no
traced data, no propagator.

**2. How well this system defines that scale.** Not that well. ``M3-SINGLET-REF``
admits two defensible image-space ``NA`` declarations: the paraxial geometric
``a / sqrt(a^2 + R^2)`` and the largest traced transverse direction cosine, which
this singlet's residual spherical aberration makes **0.29% larger** -- the
marginal ray crosses the axis about 14 um before the declared image plane
(CHE-38 section 3, Trap 1). 0.29% of scale is ``4.4e-3`` of gate metric: **4.4x
the gate**, and twice the entire observed residual. O1's own free parameter is
less determined on this system than the threshold it is being asked to decide.

**3. Where the observed residual sits inside that.** Fitting O1's ``NA`` to the
frozen 512-ring reconstruction -- one graph run through ``GraphExecutor``, the
same path the frozen number comes off -- puts the best fit ``1.36e-3`` *below*
the declared value, which is **inside** the 0.29% interval the geometry leaves
open, and leaves a residual of ``7.0e-4`` there: inside the gate. The residual
decomposes, in quadrature, into ``2.09e-3`` of pure Airy-scale offset (94.8%)
and ``7.0e-4`` of shape (5.2%).

The conclusion, stated as narrowly as the evidence supports
----------------------------------------------------------
**O1 cannot decide this gate on this system.** 94.8% of the residual lives in the
one degree of freedom O1 leaves free and this system does not pin, and the part
O1 can speak about is inside the gate.

What that is **not**: a reason to widen ``1.0e-3``, which is unchanged; a reason
to promote O2, which is not consulted here at all; a reason to fit the oracle to
the data and call the gate met. The last one is the important one. Fitting O1's
scale to the field under test destroys the independence that makes O1 the only
admissible decider in the first place -- a scale-fitted oracle cannot reject a
wrong answer of the same shape. The best-fit number is characterization of the
oracle's resolving power, and the gate stays ``NOT_MET`` at the declared ``NA``.

Two further questions this probe closes
---------------------------------------
**M0.2's amplitude drift is not in this number, and that is checkable rather than
arguable.** CHE-103 attributed a ~20-order-of-magnitude absolute-power drift in
three committed records to CHE-47's amplitude change -- the same code change as
the quadrature weight under investigation here -- so the two could have been one
problem. They cannot be: the drift is the *global* per-ray cell-area factor, and
the gate metric peak-normalizes both of its inputs, so a global amplitude factor
divides out exactly. Experiment 3 scales the measured intensity by ``2^64``, by
``1e20`` and by the recorded uniform/weighted power ratio itself and re-measures.
Experiment 3 also shows the recorded 24-order power gap *is* that global factor,
to 0.44%: ``sqrt(P_weighted / P_uniform)`` reproduces the nominal hexapolar cell
area, and the 0.44% left over is the two boundary corrections -- which is the
non-global part, the only part that can reach this metric, and the part
``singlet_residual_attribution`` already showed vanishes on convergence.

**The replacement negative control's property, measured.** The retired
``inverted-quadrature-weight`` control asserted that the production weight
improves rel-L2 agreement with O1 by >= 1.2x. That premise is false at
convergence -- both arms converge to the same residual -- so no ray count makes
that control fire honestly. What CHE-47 actually established is absolute-power
convergence: with the weight, reconstructed power is invariant under ray
refinement; without it, power scales as ``(traced rays)^1.995`` (CHE-33).
Experiment 4 measures the ring-doubling excess ``|P(2N)/P(N) - 1|`` on both arms
through the executor: ``7.7e-4`` weighted against ``14.9`` uniform, a detection
margin near ``1.9e4``. That is the property, it is the one the weight was
introduced to fix, and it fires in the correct direction.

    ./run.sh python benchmarks/probes/o1_applicability.py
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from core.paths import repository_root  # noqa: E402
from core.provenance import RECORD_PROVENANCE_KEY, record_provenance  # noqa: E402
from runtime.instance_runner import execute  # noqa: E402
from runtime.variants import with_config_overrides  # noqa: E402
from verification.metrics import disc_relative_l2_intensity  # noqa: E402
from verification.psf_measurement import PsfNormalization, measure_psf_from_record  # noqa: E402
from verification.psf_oracles import airy_first_null_radius_m, airy_psf_on_grid  # noqa: E402

ROOT = repository_root()
RECORD = ROOT / "benchmarks/probes/records/o1_applicability.json"

#: The gate this probe is asking about, frozen since M3.2 (CHE-31) and unchanged
#: by this work.
GATE = 1.0e-3

#: Fractional Airy-scale offsets for experiment 1. Four decades, so linearity is
#: a measured property of the metric rather than a small-angle assumption.
SCALE_OFFSETS = (1e-5, 1e-4, 3e-4, 7e-4, 1e-3, 2e-3, 5e-3, 1e-2)

#: Ring counts for experiment 4. The point is a *doubling*, and 64 -> 128 is the
#: cheapest doubling that is already deep inside the weighted arm's converged
#: region (converged from 24 rings, per m3_quadrature_weight.json).
POWER_RINGS = (64, 128)

#: Global amplitude factors for experiment 3. ``2**64`` is exact in binary so it
#: tests the invariance without any rounding at all; ``1e20`` is the order of the
#: drift CHE-103 attributed; the third is the recorded ratio itself.
EXACT_FACTOR = 2.0**64
DRIFT_FACTOR = 1e20

#: ``benchmarks/probes/records/m3_quadrature_weight.json``,
#: ``absolute_power.*_power_by_ray_count`` at the frozen 787,969 rays. Quoted so
#: experiment 3's arithmetic is against the committed record rather than a rerun.
RECORDED_UNIFORM_POWER = 21.976737710816636
RECORDED_WEIGHTED_POWER = 1.3531459792934653e-24

#: The exit pupil radius and the pupil-to-image distance of ``M3-SINGLET-REF``,
#: from ``benchmarks/probes/sensor_handoff_convergence.py``. They are here for
#: one purpose: forming the *other* admissible NA declaration, the paraxial
#: geometric one, so that the span between the two conventions is computed rather
#: than quoted from a report.
PUPIL_RADIUS_M = 0.00024978414778669653


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _driver() -> Any:
    """``benchmarks/instances/b3_psf_singlet.py``.

    The frozen configuration is reached through the committed graph document and
    ``GraphExecutor``, not through a bundle this probe builds. That is what makes
    the self-check below meaningful: the residual at the declared ``NA`` must come
    back as the frozen ``0.0022072391812867093`` bit-identically, so every number
    derived from the same field is derived from the gate's own field.
    """
    return _load("b3_psf_singlet_driver", ROOT / "benchmarks/instances/b3_psf_singlet.py")


# ---------------------------------------------------------------------------
# 1. What the gate metric resolves, using O1 on both sides
# ---------------------------------------------------------------------------


def _oracle(*, shape: tuple[int, int], pitch_m: float, wavelength_m: float, na: float):
    return airy_psf_on_grid(
        shape=shape,
        sample_pitch_m=(pitch_m, pitch_m),
        wavelength_m=wavelength_m,
        numerical_aperture=na,
    )


def _scale_sensitivity(
    *, shape: tuple[int, int], pitch_m: float, wavelength_m: float, na: float, disc_radius_m: float
) -> dict[str, Any]:
    """``rel-L2(O1(NA), O1(NA (1 + eps)))`` over the gate disc.

    The disc is held at the *declared* ``NA``, exactly as the gate defines it: a
    metric whose mask moves with the thing being measured would report a smaller
    number for a bigger error.
    """
    reference = _oracle(shape=shape, pitch_m=pitch_m, wavelength_m=wavelength_m, na=na)

    def residual(offset: float) -> float:
        return disc_relative_l2_intensity(
            _oracle(
                shape=shape, pitch_m=pitch_m, wavelength_m=wavelength_m, na=na * (1.0 + offset)
            ),
            reference,
            sample_pitch_m=pitch_m,
            max_radius_m=disc_radius_m,
        )

    ladder = []
    for offset in SCALE_OFFSETS:
        value = residual(offset)
        ladder.append(
            {"scale_offset": offset, "relative_l2": value, "slope": value / offset}
        )

    # The offset the gate itself corresponds to. Bisected rather than divided by
    # a slope, so the number does not inherit the linearity it is used to argue.
    low, high = 0.0, max(SCALE_OFFSETS)
    for _ in range(80):
        middle = 0.5 * (low + high)
        if residual(middle) < GATE:
            low = middle
        else:
            high = middle
        if high - low < 1e-15:
            break
    gate_equivalent = 0.5 * (low + high)

    return {
        "ladder": ladder,
        "slope_relative_l2_per_fractional_scale": ladder[0]["slope"],
        "gate_equivalent_scale_offset": gate_equivalent,
        "gate_equivalent_check_relative_l2": residual(gate_equivalent),
        "statement": (
            "the gate metric is linear in a fractional Airy-scale error, so the "
            f"{GATE:.1e} gate is the statement that the scale is known to "
            f"{gate_equivalent:.3e}"
        ),
    }


def _na_conventions(*, na_declared: float, distance_m: float) -> dict[str, Any]:
    """The two defensible image-space ``NA`` declarations, and the span between them.

    Both are properties of the system, not of the coupler. The geometric one is
    the paraxial ``a / sqrt(a^2 + R^2)``; the declared one is the largest traced
    transverse direction cosine, which is larger because the marginal ray focuses
    short of the declared image plane. Nothing here reads a reconstruction.
    """
    na_geometric = PUPIL_RADIUS_M / math.hypot(PUPIL_RADIUS_M, distance_m)
    return {
        "na_declared_largest_traced_direction_cosine": na_declared,
        "na_geometric_paraxial_sphere": na_geometric,
        "fractional_span": na_declared / na_geometric - 1.0,
        "marginal_focus_offset_m": distance_m * (1.0 - na_geometric / na_declared),
        "statement": (
            "two admissible declarations of the same system's NA, differing by the "
            "singlet's own residual spherical aberration. Neither is a coupler input"
        ),
    }


# ---------------------------------------------------------------------------
# 2. Where the observed residual sits inside O1's own scale freedom
# ---------------------------------------------------------------------------


def _best_fit_scale(
    measured, *, pitch_m: float, wavelength_m: float, na: float, disc_radius_m: float
) -> dict[str, Any]:
    """Minimise the gate metric over O1's one free parameter.

    Golden section rather than a derivative: the metric is evaluated, not
    modelled. The bracket is +/-2%, an order of magnitude wider than the 0.29%
    the system's own NA conventions span, so the minimum is not found by the
    bracket.
    """

    def residual(offset: float) -> float:
        return disc_relative_l2_intensity(
            measured,
            _oracle(
                shape=measured.shape,
                pitch_m=pitch_m,
                wavelength_m=wavelength_m,
                na=na * (1.0 + offset),
            ),
            sample_pitch_m=pitch_m,
            max_radius_m=disc_radius_m,
        )

    golden = 0.5 * (math.sqrt(5.0) - 1.0)
    low, high = -0.02, 0.02
    left, right = high - golden * (high - low), low + golden * (high - low)
    f_left, f_right = residual(left), residual(right)
    for _ in range(80):
        if f_left < f_right:
            high, right, f_right = right, left, f_left
            left = high - golden * (high - low)
            f_left = residual(left)
        else:
            low, left, f_left = left, right, f_right
            right = low + golden * (high - low)
            f_right = residual(right)
        if high - low < 1e-12:
            break
    offset = 0.5 * (low + high)
    at_best_fit = residual(offset)
    at_declared = residual(0.0)
    scale_term = math.sqrt(max(at_declared**2 - at_best_fit**2, 0.0))
    return {
        "best_fit_scale_offset": offset,
        "best_fit_na": na * (1.0 + offset),
        "relative_l2_at_best_fit_na": at_best_fit,
        "relative_l2_at_declared_na": at_declared,
        "scale_term_in_quadrature": scale_term,
        "scale_term_fraction_of_residual": scale_term / at_declared,
        "shape_term_fraction_of_residual": at_best_fit / at_declared,
        "why_this_is_not_a_gate": (
            "fitting the oracle's scale to the field under test removes the "
            "independence that makes O1 admissible: a scale-fitted Airy pattern "
            "cannot reject a wrong answer of the same shape. This is the oracle's "
            "resolving power, measured. The gate stays at the declared NA and stays "
            "NOT_MET."
        ),
    }


# ---------------------------------------------------------------------------
# 3. M0.2's amplitude drift, and why it is not in this number
# ---------------------------------------------------------------------------


def _amplitude_normalization(
    measured, oracle, *, pitch_m: float, disc_radius_m: float
) -> dict[str, Any]:
    """Global amplitude factors divide out of the gate metric, exactly.

    ``disc_relative_l2_intensity`` peak-normalizes both inputs, so this is a
    property of the metric's definition. Measured anyway, because "it should
    cancel" is what CHE-100 believed about three records for weeks.
    """
    baseline = disc_relative_l2_intensity(
        measured, oracle, sample_pitch_m=pitch_m, max_radius_m=disc_radius_m
    )
    recorded_ratio = RECORDED_UNIFORM_POWER / RECORDED_WEIGHTED_POWER
    arms = {
        "exact_binary_2_pow_64": EXACT_FACTOR,
        "drift_scale_1e20": DRIFT_FACTOR,
        "recorded_uniform_over_weighted_power_ratio": recorded_ratio,
    }
    scaled = {
        name: disc_relative_l2_intensity(
            measured * factor, oracle, sample_pitch_m=pitch_m, max_radius_m=disc_radius_m
        )
        for name, factor in arms.items()
    }

    # The recorded 24-order power gap, against the amplitude convention CHE-103
    # named: amplitude = sqrt(intensity) * cell area in m^2, so the power ratio
    # between the arms is that area squared for every interior ray.
    nominal_cell_area_m2 = math.pi * PUPIL_RADIUS_M**2 / (3.0 * 512**2)
    implied_area_m2 = math.sqrt(RECORDED_WEIGHTED_POWER / RECORDED_UNIFORM_POWER)

    return {
        "relative_l2_unscaled": baseline,
        "relative_l2_by_global_amplitude_factor": scaled,
        "max_relative_deviation": max(
            abs(value - baseline) / baseline for value in scaled.values()
        ),
        "recorded_power_ratio_uniform_over_weighted": recorded_ratio,
        "implied_per_ray_area_m2": implied_area_m2,
        "nominal_hexapolar_cell_area_m2": nominal_cell_area_m2,
        "implied_over_nominal": implied_area_m2 / nominal_cell_area_m2,
        "statement": (
            "the 20-orders-of-magnitude drift CHE-103 attributed to CHE-47's "
            "amplitude change is the GLOBAL per-ray cell-area factor, and the gate "
            "metric peak-normalizes it away exactly. What is left of the same code "
            "change is the boundary corrections -- the 0.44% by which the implied "
            "area misses the nominal cell -- which is the part "
            "singlet_residual_attribution.json already showed vanishes on "
            "convergence. Same code change, different quantity, and it cannot reach "
            "this metric."
        ),
    }


# ---------------------------------------------------------------------------
# 4. The property the replacement negative control tests
# ---------------------------------------------------------------------------


def _power_under_ray_refinement(driver: Any, instance: Any) -> dict[str, Any]:
    """``|P(2N)/P(N) - 1|`` with the production weight and without it.

    Through the executor as graph variants, so the arm without the weight is the
    shipping code path with one declared perturbation rather than a
    reimplementation of it.
    """

    def power(rings: int, *, weighted: bool) -> dict[str, Any]:
        edges: dict[str, Any] = {}
        if not weighted:
            edges["sensor_reconstruction"] = {"perturbation": {"apply_quadrature_weight": False}}
        graph = with_config_overrides(
            driver.load_graph(),
            nodes={"lens": {"num_rays": rings}},
            edges=edges or None,
            task_id=f"B3-PSF-SINGLET-01/power_refinement/{rings}/"
            f"{'weighted' if weighted else 'uniform'}",
        )
        record = execute(graph, instance, seed=1)
        measurement = measure_psf_from_record(
            record.artifacts[driver.COUPLER_FIELD], normalization=PsfNormalization.RAW
        )
        intensity = np.asarray(measurement.intensity, dtype=np.float64)
        pitch = measurement.sample_pitch_m
        return {
            "rings": rings,
            "traced_rays": 1 + 3 * rings * (rings + 1),
            "reconstructed_power": float(intensity.sum()) * float(pitch[0]) * float(pitch[1]),
        }

    arms: dict[str, Any] = {}
    for name, weighted in (("weighted_production", True), ("uniform_mutation", False)):
        coarse, fine = (power(rings, weighted=weighted) for rings in POWER_RINGS)
        ratio = fine["reconstructed_power"] / coarse["reconstructed_power"]
        arms[name] = {
            "coarse": coarse,
            "fine": fine,
            "power_ratio": ratio,
            "doubling_excess": abs(ratio - 1.0),
        }
        print(
            f"power  {name:20s} {coarse['reconstructed_power']:.6e} -> "
            f"{fine['reconstructed_power']:.6e}  ratio {ratio:.6e}",
            flush=True,
        )

    weighted_excess = arms["weighted_production"]["doubling_excess"]
    arms["detection_margin"] = arms["uniform_mutation"]["doubling_excess"] / weighted_excess
    arms["statement"] = (
        "the production weight makes reconstructed power invariant under ray "
        "refinement; omitting it reproduces CHE-33's N^2 divergence. This is the "
        "property CHE-47 established, and it is what the replacement control tests"
    )
    return arms


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def characterize() -> dict[str, Any]:
    driver = _driver()
    instance = driver.canonical_instance()
    wavelength_m = float(instance.parameters["wavelength_m"])
    na = float(instance.parameters["numerical_aperture"])
    distance_m = driver._SENSOR_Z_M - driver._PUPIL_Z_M

    began = time.perf_counter()
    record = execute(driver.load_graph(), instance, seed=1)
    measurement = measure_psf_from_record(
        record.artifacts[driver.COUPLER_FIELD], normalization=PsfNormalization.RAW
    )
    measured = np.asarray(measurement.intensity, dtype=np.float64)
    pitch_m = float(measurement.sample_pitch_m[0])
    disc_radius_m = driver.GATE_AIRY_RADII * airy_first_null_radius_m(wavelength_m, na)
    oracle = _oracle(shape=measured.shape, pitch_m=pitch_m, wavelength_m=wavelength_m, na=na)
    frozen = disc_relative_l2_intensity(
        measured, oracle, sample_pitch_m=pitch_m, max_radius_m=disc_radius_m
    )
    print(
        f"frozen configuration: {frozen!r} in {time.perf_counter() - began:.1f}s "
        f"(driver.FROZEN_OBSERVED == {frozen == driver.FROZEN_OBSERVED})",
        flush=True,
    )

    sensitivity = _scale_sensitivity(
        shape=measured.shape,
        pitch_m=pitch_m,
        wavelength_m=wavelength_m,
        na=na,
        disc_radius_m=disc_radius_m,
    )
    conventions = _na_conventions(na_declared=na, distance_m=distance_m)
    conventions["relative_l2_across_the_span"] = sensitivity[
        "slope_relative_l2_per_fractional_scale"
    ] * abs(conventions["fractional_span"])
    conventions["gate_multiples_across_the_span"] = (
        conventions["relative_l2_across_the_span"] / GATE
    )

    best_fit = _best_fit_scale(
        measured,
        pitch_m=pitch_m,
        wavelength_m=wavelength_m,
        na=na,
        disc_radius_m=disc_radius_m,
    )
    best_fit["best_fit_na_is_inside_the_convention_span"] = bool(
        conventions["na_geometric_paraxial_sphere"]
        <= best_fit["best_fit_na"]
        <= conventions["na_declared_largest_traced_direction_cosine"]
    )

    return {
        "probe": "o1_applicability",
        "issue": "CHE-117",
        "question": (
            "can O1 -- the analytic Airy pattern, paraxial and aberration-free -- "
            "decide B3-PSF-SINGLET's 1.0e-3 gate on the real traced M3-SINGLET-REF "
            "system?"
        ),
        "oracle": (
            "O1 only, on both sides of every comparison. O2, our own float64 ASM/RS "
            "propagator, is not consulted; no second Optiland PSF route is consulted "
            "either (PB7/CHE-58 finding F2: FFTPSF and HuygensPSF share one "
            "Wavefront/OPD front end and are one oracle, not two)."
        ),
        "gate": GATE,
        "frozen_configuration": {
            "relative_l2_at_declared_na": frozen,
            "reproduces_the_frozen_number_bit_identically": frozen == driver.FROZEN_OBSERVED,
            "grid_n": int(measured.shape[0]),
            "sensor_pitch_m": pitch_m,
            "gate_disc_radius_m": disc_radius_m,
            "source": (
                "GraphExecutor over examples/graphs/psf_singlet_sensor.yaml, driven by "
                "benchmarks/instances/b3_psf_singlet.py, float64 sensor-plane "
                "reconstruction"
            ),
        },
        "oracle_scale_sensitivity": sensitivity,
        "na_conventions": conventions,
        "residual_at_o1s_best_fit_scale": best_fit,
        "amplitude_normalization": _amplitude_normalization(
            measured, oracle, pitch_m=pitch_m, disc_radius_m=disc_radius_m
        ),
        "absolute_power_under_ray_refinement": _power_under_ray_refinement(driver, instance),
    }


def _verdict(record: dict[str, Any]) -> dict[str, Any]:
    sensitivity = record["oracle_scale_sensitivity"]
    conventions = record["na_conventions"]
    best_fit = record["residual_at_o1s_best_fit_scale"]
    amplitude = record["amplitude_normalization"]
    power = record["absolute_power_under_ray_refinement"]

    slopes = [row["slope"] for row in sensitivity["ladder"]]
    return {
        "o1_cannot_decide_this_gate_at_1e-3": {
            "gate_equivalent_scale_offset": sensitivity["gate_equivalent_scale_offset"],
            "system_na_convention_span": conventions["fractional_span"],
            "span_over_gate_equivalent": abs(conventions["fractional_span"])
            / sensitivity["gate_equivalent_scale_offset"],
            "relative_l2_across_the_span": conventions["relative_l2_across_the_span"],
            "statement": (
                "the gate resolves the Airy scale to "
                f"{sensitivity['gate_equivalent_scale_offset']:.3e}; this system's two "
                "admissible NA declarations differ by "
                f"{abs(conventions['fractional_span']):.3e}, which is "
                f"{conventions['gate_multiples_across_the_span']:.1f}x the gate in "
                "metric units. The oracle's own free parameter is less determined here "
                "than the threshold it is asked to decide. SETTLED: not applicable as a "
                "1e-3 decider on this system."
            ),
        },
        "the_metric_is_linear_in_the_airy_scale": {
            "slope_min": min(slopes),
            "slope_max": max(slopes),
            "statement": (
                f"rel-L2 = {min(slopes):.4f}..{max(slopes):.4f} times the fractional "
                "scale offset over four decades, so 'how much scale error is 1e-3' has "
                "one answer rather than being a small-offset approximation"
            ),
        },
        "the_residual_is_94_percent_scale": {
            "relative_l2_at_declared_na": best_fit["relative_l2_at_declared_na"],
            "relative_l2_at_best_fit_na": best_fit["relative_l2_at_best_fit_na"],
            "scale_term_in_quadrature": best_fit["scale_term_in_quadrature"],
            "scale_term_fraction_of_residual": best_fit["scale_term_fraction_of_residual"],
            "best_fit_na_is_inside_the_convention_span": best_fit[
                "best_fit_na_is_inside_the_convention_span"
            ],
            "statement": (
                "of the observed "
                f"{best_fit['relative_l2_at_declared_na']:.4e}, "
                f"{best_fit['scale_term_in_quadrature']:.4e} is removable by O1's own "
                "scale parameter and "
                f"{best_fit['relative_l2_at_best_fit_na']:.4e} is not. The best-fit NA "
                "lands INSIDE the interval the system's geometry leaves open, and the "
                "part O1 can speak about is inside the gate. This is NOT a claim that "
                "the gate is met: see why_this_is_not_a_gate."
            ),
        },
        "m0_2_amplitude_drift_is_not_in_this_number": {
            "max_relative_deviation_under_global_rescaling": amplitude[
                "max_relative_deviation"
            ],
            "implied_over_nominal_cell_area": amplitude["implied_over_nominal"],
            "statement": amplitude["statement"],
        },
        "the_replacement_control_fires_in_the_right_direction": {
            "weighted_doubling_excess": power["weighted_production"]["doubling_excess"],
            "uniform_doubling_excess": power["uniform_mutation"]["doubling_excess"],
            "detection_margin": power["detection_margin"],
            "statement": (
                "the retired control's premise -- that the weight improves rel-L2 "
                "agreement by >= 1.2x -- is false at convergence, so no ray count makes "
                "it fire honestly. The property CHE-47 did establish is absolute-power "
                "convergence, and on that property the weight is right by a factor of "
                f"{power['detection_margin']:.3g}."
            ),
        },
        "what_this_does_not_do": (
            "widen anything (1.0e-3 unchanged), promote O2 (not consulted), use a "
            "second Optiland PSF route (not consulted), or declare the gate met at a "
            "fitted NA (that would destroy the oracle's independence). The gate stays "
            "NOT_MET at 2.2072391812867093e-3 with every term of it now named."
        ),
    }


def main() -> None:
    record = characterize()
    record["verdict"] = _verdict(record)
    record[RECORD_PROVENANCE_KEY] = record_provenance(
        probe="benchmarks/probes/o1_applicability.py", root=ROOT
    )
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record["verdict"], indent=1))
    print(f"wrote {RECORD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
