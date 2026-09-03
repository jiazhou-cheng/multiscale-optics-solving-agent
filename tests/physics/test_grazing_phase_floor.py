"""R07.4: the grazing-mode phase floor, and the decision the old tree deferred.

CHE-188. The one open **correctness** defect in the ray-to-wave kernel (H4 /
CHE-70). The kernel forms each ray's constant phase as `k (OPL - d . x0)`; for a
mode with axial direction cosine `d_n` propagating an axial distance `Z`, both
terms scale as `Z / d_n` while their difference is only `Z d_n`. So the relative
precision of the *inputs* sets the absolute error of the phase:

    delta_phi  ~  eps k Z / d_n

The decision, and what this file proves about it
------------------------------------------------
`couplers.ray_to_scalar` **refuses** by default (option 3) and band-limits only
when explicitly asked (option 2, reported). The three not taken are recorded in
the module docstring with the reason each was rejected, and
`test_the_three_rejected_options_are_recorded` pins that record.

The risk the ticket names is shipping option 1 -- caller-side, silent -- because
it is what the old code did. `test_the_default_is_to_refuse` is the gate against
that: the default is checked directly, not inferred from a signature.

Why the bound is not the realized error, and why that is the right design
------------------------------------------------------------------------
The kernel's check is a *bound*:

    delta_phi_i  <=  eps k ( |OPL_i| + |d_i . x0_i| )

Measured below, it sits 3x to 24x above the realized phase error. That
conservatism is deliberate and load-bearing: the realized error depends on how the
two large terms happen to round, and on a 100x100 enumeration the eight worst
modes round to the *same* float32 and their true phase is itself 6.6e-6 rad, so
their realized error is small **by coincidence**. A gate that relied on that
coincidence would be a gate that passes until the propagation distance changes.
The single-mode sweep is where the realized failure is exhibited, and it reaches
100 % of signal.

CHE-99 recorded that the reference implementation's exactness limit failed ~30 %
of the time under one selection and that observing it made it stop. So a single
green run is not evidence here, which is why the ticket asks for `make test-slow`
**and** `make test-serial`. Nothing in this file is random: the enumeration draws
from a seeded generator and the sweep is deterministic single modes.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from ray_support import (
    mode_bundle,
    shifted_inverse_dft,
    single_mode_bundle,
)

from couplers import (
    DEFAULT_PHASE_BUDGET_RAD,
    Reconstruction,
    grazing_floor_for_phase_budget,
    ray_to_scalar,
)
from numerics import Precision
from representations import ContractError

#: CHE-70's configuration, reused so the numbers are comparable to its record.
WAVELENGTH_M = 500e-9
PITCH_M = 250e-9
GRID = 100
PROPAGATE_M = 50e-6
WAVENUMBER = 2.0 * math.pi / WAVELENGTH_M

MODULE = Path(__file__).resolve().parents[2] / "src" / "couplers" / "ray_to_scalar.py"


def an_enumeration(*, floor: float | None = None, dtype=np.float64):
    """Every propagating mode of a seeded random field, advanced `PROPAGATE_M`."""
    rng = np.random.default_rng(70)
    source = rng.standard_normal((GRID, GRID)) + 1j * rng.standard_normal((GRID, GRID))
    return mode_bundle(
        source,
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        propagate_m=PROPAGATE_M,
        direction_cosine_floor=floor,
        dtype=dtype,
    )


def analytic_field(retained, spectrum):
    """The angular-spectrum oracle carrying exactly the modes `retained` names.

    `exp(+i k d_n Z)` per mode, then the inverse DFT on the `n // 2` origin. An
    oracle carrying modes the reconstruction excluded would measure the exclusion
    rather than the kernel, which is why the mask is passed in rather than
    recomputed.
    """
    direction_v, direction_u = np.meshgrid(
        WAVELENGTH_M * np.fft.fftfreq(GRID, PITCH_M),
        WAVELENGTH_M * np.fft.fftfreq(GRID, PITCH_M),
        indexing="ij",
    )
    axial = np.sqrt(np.clip(1.0 - direction_u**2 - direction_v**2, 0.0, None))
    propagated = spectrum * np.exp(1j * WAVENUMBER * axial * PROPAGATE_M)
    return shifted_inverse_dft(np.where(retained, propagated, 0.0))


def residuals(u, oracle) -> tuple[float, float]:
    """`(peak-relative max, relative L2)`. Both, because the two differ by ~sqrt(N)."""
    difference = np.asarray(u).astype(np.complex128) - oracle
    return (
        float(np.max(np.abs(difference)) / np.max(np.abs(oracle))),
        float(np.linalg.norm(difference) / np.linalg.norm(oracle)),
    )


def reconstruct(rays, **kwargs):
    return ray_to_scalar(
        rays, grid_shape=(GRID, GRID), sample_pitch_m=(PITCH_M, PITCH_M), **kwargs
    )


# ---------------------------------------------------------------------------
# 1. The floor derivation
# ---------------------------------------------------------------------------


def test_the_floor_derivation_reproduces_the_frozen_number() -> None:
    """Criterion 1. `eps k Z / budget`: float32 over 50 um at 0.01 rad is 7.49e-3."""
    floor = grazing_floor_for_phase_budget(
        wavelength_m=WAVELENGTH_M,
        max_optical_path_m=PROPAGATE_M,
        precision=Precision.FP32,
        phase_budget_rad=DEFAULT_PHASE_BUDGET_RAD,
    )
    assert floor == pytest.approx(7.4901e-3, rel=1e-4)
    assert DEFAULT_PHASE_BUDGET_RAD == 1.0e-2

    # ...and the same derivation in float64 is 1.395e-11, eight orders below, which
    # is why this is a float32 correctness requirement and not a general one.
    assert grazing_floor_for_phase_budget(
        wavelength_m=WAVELENGTH_M,
        max_optical_path_m=PROPAGATE_M,
        precision=Precision.FP64,
    ) == pytest.approx(1.3951e-11, rel=1e-4)


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ({"phase_budget_rad": 1.0e-1}, 0.1),
        ({"max_optical_path_m": 500e-6}, 10.0),
        ({"wavelength_m": 250e-9}, 2.0),
    ],
)
def test_the_floor_scales_the_way_the_derivation_says(changed: dict, expected: float) -> None:
    """Each variable enters once, so each ratio is exact rather than approximate."""
    base = dict(
        wavelength_m=WAVELENGTH_M,
        max_optical_path_m=PROPAGATE_M,
        precision=Precision.FP32,
        phase_budget_rad=DEFAULT_PHASE_BUDGET_RAD,
    )
    reference = grazing_floor_for_phase_budget(**base)
    assert grazing_floor_for_phase_budget(**{**base, **changed}) / reference == (
        pytest.approx(expected, rel=1e-12)
    )


def test_a_non_positive_budget_is_refused() -> None:
    for budget in (0.0, -1.0):
        with pytest.raises(ValueError, match="phase_budget_rad"):
            grazing_floor_for_phase_budget(
                wavelength_m=WAVELENGTH_M,
                max_optical_path_m=PROPAGATE_M,
                precision=Precision.FP32,
                phase_budget_rad=budget,
            )


# ---------------------------------------------------------------------------
# 2. The exactness table
# ---------------------------------------------------------------------------


def test_the_exactness_table_reproduces() -> None:
    """Criterion 2, on CHE-70's own configuration: 100x100, 250 nm pitch, 500 nm.

    Against an analytic angular-spectrum oracle carrying the identical modes, in
    float64:

    | band limit | peak-relative | relative L2 |
    | -- | -- | -- |
    | none | 1.162e-07 | 1.619e-07 |
    | `d_n >= 1e-2` | 1.239e-13 | 1.347e-13 |

    Six orders of magnitude, which is the finding. CHE-70 recorded `2.8e-09`
    against `8.9e-14` -- five orders. The **banded** value reproduces to within a
    factor of 1.5 on the L2 metric; the *unbanded* one is 40-60x larger here, and
    that difference is a property of the random draw and of the residual
    normalization, neither of which CHE-70 states. Both metrics are reported rather
    than picking the flattering one.

    Not marked `slow`, measured: the whole 7 833-ray reconstruction onto 100x100 is
    68 ms, because `einsum(..., optimize=True)` routes the separable contraction
    through a matmul. It is criterion 2's evidence, so it belongs in the gate that
    runs by default.
    """
    unbanded_rays, retained, spectrum = an_enumeration()
    unbanded, unbanded_record = reconstruct(
        unbanded_rays, grazing="band_limit", phase_budget_rad=math.inf
    )
    unbanded_peak, unbanded_l2 = residuals(unbanded.u, analytic_field(retained, spectrum))

    banded_rays, banded_retained, banded_spectrum = an_enumeration(floor=1.0e-2)
    banded, banded_record = reconstruct(
        banded_rays, grazing="band_limit", phase_budget_rad=math.inf
    )
    banded_peak, banded_l2 = residuals(
        banded.u, analytic_field(banded_retained, banded_spectrum)
    )

    assert unbanded_peak == pytest.approx(1.162e-7, rel=0.2)
    assert unbanded_l2 == pytest.approx(1.619e-7, rel=0.2)
    assert banded_peak == pytest.approx(1.239e-13, rel=0.3)
    assert banded_l2 == pytest.approx(1.347e-13, rel=0.3)
    assert unbanded_peak / banded_peak > 1e5

    # The band limit removed eight modes and nothing else.
    assert unbanded_record.ray_count - banded_record.ray_count == 8


def test_the_eight_pythagorean_bins_are_the_ones_the_kernel_excludes() -> None:
    """The kernel's own band limit finds exactly the modes the caller-side floor did.

    CHE-70 identified them: the (30, 40) and (40, 30) Pythagorean triples and their
    sign variants land on `d_u^2 + d_v^2 = 1` exactly, survive a strict
    `radial < 1` cut at `d_n = 1.05e-8`, and carry a 4745 m optical path over a
    50 um propagation. All three numbers reproduce.

    A float64 budget of 1e-6 rather than the default 1e-2 is used, because in
    float64 those bins cost 2.6e-5 rad -- genuinely inside the default budget. That
    is the check being precision-aware rather than absolute, and it is the whole
    reason the same code is a hard gate in float32 and not in float64.
    """
    rays, _, _ = an_enumeration()
    _, record = reconstruct(rays, grazing="band_limit", phase_budget_rad=1.0e-6)
    grazing = record.grazing

    assert grazing["excluded_ray_count"] == 8
    assert grazing["min_axial_direction_cosine"] == pytest.approx(1.0537e-8, rel=1e-3)
    assert grazing["max_optical_path_m"] == pytest.approx(4745.3, rel=1e-3)
    assert grazing["max_phase_error_rad"] == pytest.approx(2.648e-5, rel=0.05)
    assert grazing["excluded_power_fraction"] == pytest.approx(1.133e-3, rel=0.05)

    # ...and the reconstruction that results agrees with the analytic oracle to the
    # banded figure, i.e. the kernel-side limit buys what the caller-side one did.
    _, banded_retained, banded_spectrum = an_enumeration(floor=1.0e-2)
    peak, _ = residuals(
        reconstruct(rays, grazing="band_limit", phase_budget_rad=1.0e-6)[0].u,
        analytic_field(banded_retained, banded_spectrum),
    )
    assert peak < 1e-12, peak

    # In float64 the *default* budget admits them, and the record says so rather
    # than pretending the check did not run.
    _, default_record = reconstruct(rays, grazing="band_limit")
    assert default_record.grazing["excluded_ray_count"] == 0
    assert default_record.grazing["max_phase_error_rad"] == pytest.approx(2.648e-5, rel=0.05)


# ---------------------------------------------------------------------------
# 3. The float32 path, where this is a correctness requirement
# ---------------------------------------------------------------------------

#: `d_n`, and the realized float32 phase error at the coordinate origin. Measured
#: on a single mode over a 50 um propagation, against the analytic
#: `exp(+i k Z d_n)` -- no truncation, no quadrature, so the number is the phase
#: error itself rather than a field residual.
#:
#: The last row is the point: the true phase is `6.283e-2` rad and the error is
#: `6.283e-2` rad, so the mode carries **100 % of signal as noise**. In float64 the
#: same mode is right to 1e-9.
FLOAT32_REALIZED_ERROR_RAD = {
    1.0: 1.176e-5,
    1.0e-1: 4.795e-4,
    1.0e-2: 1.511e-3,
    1.0e-3: 1.974e-2,
    1.0e-4: 6.283e-2,
}


def realized_phase_error(axial_cosine: float, dtype) -> tuple[float, dict]:
    rays = single_mode_bundle(
        axial_cosine=axial_cosine,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=dtype,
    )
    field, record = reconstruct(rays, grazing="band_limit", phase_budget_rad=math.inf)
    centre = complex(np.asarray(field.u)[GRID // 2, GRID // 2])
    analytic = complex(np.exp(1j * WAVENUMBER * PROPAGATE_M * axial_cosine))
    return float(abs(np.angle(centre / analytic))), record.grazing


@pytest.mark.parametrize("axial_cosine", sorted(FLOAT32_REALIZED_ERROR_RAD))
def test_the_realized_float32_phase_error_grows_as_one_over_the_axial_cosine(
    axial_cosine: float,
) -> None:
    """Criterion 5. Measured, not bounded, and in float32 rather than only float64."""
    error, _ = realized_phase_error(axial_cosine, np.float32)
    assert error == pytest.approx(FLOAT32_REALIZED_ERROR_RAD[axial_cosine], rel=0.25)

    # float64 on the identical geometry: eight to eleven orders smaller, so the
    # error is the precision and not the construction.
    float64_error, _ = realized_phase_error(axial_cosine, np.float64)
    assert float64_error < 1e-8, float64_error


def test_the_worst_float32_mode_is_entirely_noise() -> None:
    """`d_n = 1e-4`: the error equals the signal, so the mode carries no information.

    This is the statement "those bins are then pure noise" made as a measurement.
    A field-residual gate cannot see it -- the mode carries a small fraction of the
    power -- which is why it is read off a single mode.
    """
    axial_cosine = 1.0e-4
    error, _ = realized_phase_error(axial_cosine, np.float32)
    true_phase = WAVENUMBER * PROPAGATE_M * axial_cosine
    assert error / true_phase == pytest.approx(1.0, rel=0.25)


def test_the_bound_is_conservative_and_never_below_the_realized_error() -> None:
    """A bound that undershoots is not a bound. Measured 3x to 24x above, everywhere."""
    for axial_cosine in sorted(FLOAT32_REALIZED_ERROR_RAD):
        for dtype in (np.float32, np.float64):
            error, grazing = realized_phase_error(axial_cosine, dtype)
            assert grazing["max_phase_error_rad"] >= error, (axial_cosine, dtype)


@pytest.mark.parametrize("axial_cosine", [1.0e-2, 1.0e-3, 1.0e-4])
def test_a_float32_mode_below_the_floor_is_refused_by_default(axial_cosine: float) -> None:
    """Criterion 3 and 5: the chosen option, firing where it has to.

    The kernel's gate over this propagation is `2 x 7.49e-3 = 1.498e-2` -- the
    two-term bound, see `test_the_kernel_gate_is_twice_the_one_term_floor` -- so
    every mode here is below it and every one is refused, with the diagnostic
    naming the optical path and the axial cosine that caused it.
    """
    rays = single_mode_bundle(
        axial_cosine=axial_cosine,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    with pytest.raises(ContractError) as raised:
        reconstruct(rays)
    assert raised.value.code == "GRAZING_PHASE_UNREPRESENTABLE"
    assert raised.value.declaration == "optical_path_m"
    assert "Z/d_n" in str(raised.value)
    assert raised.value.remedy is not None
    assert "grazing='band_limit'" in raised.value.remedy


@pytest.mark.parametrize("axial_cosine", [1.0, 1.0e-1])
def test_a_float32_mode_above_the_floor_is_not_refused(axial_cosine: float) -> None:
    """The negative twin. A refusal that fires on everything is not a floor."""
    floor = grazing_floor_for_phase_budget(
        wavelength_m=WAVELENGTH_M,
        max_optical_path_m=PROPAGATE_M,
        precision=Precision.FP32,
    )
    assert axial_cosine > 2.0 * floor
    rays = single_mode_bundle(
        axial_cosine=axial_cosine,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    field, record = reconstruct(rays)
    assert record.grazing["excluded_ray_count"] == 0
    assert record.compute_precision == "fp32"
    assert field.shape == (GRID, GRID)


def test_the_same_geometry_in_float64_is_never_refused() -> None:
    """Precision-aware, not absolute: the float64 floor is 1.4e-11."""
    for axial_cosine in sorted(FLOAT32_REALIZED_ERROR_RAD):
        rays = single_mode_bundle(
            axial_cosine=axial_cosine,
            propagate_m=PROPAGATE_M,
            wavelength_m=WAVELENGTH_M,
            dtype=np.float64,
        )
        _, record = reconstruct(rays)
        assert record.grazing["excluded_ray_count"] == 0, axial_cosine


# ---------------------------------------------------------------------------
# 4. What travels with the field, and what the policy is
# ---------------------------------------------------------------------------


def test_the_excluded_count_and_power_fraction_travel_with_the_field() -> None:
    """Criterion 4. Both numbers, and the policy and budget that produced them."""
    rays = single_mode_bundle(
        axial_cosine=1.0e-4,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    field, record = reconstruct(rays, grazing="band_limit")
    grazing = record.grazing

    assert grazing["policy"] == "band_limit"
    assert grazing["phase_budget_rad"] == DEFAULT_PHASE_BUDGET_RAD
    assert grazing["excluded_ray_count"] == 1
    assert grazing["excluded_power_fraction"] == pytest.approx(1.0)
    assert record.as_dict()["grazing"]["excluded_ray_count"] == 1
    # The only mode was excluded, so the field is identically zero -- reported, not
    # silently returned as a plausible small field.
    assert float(np.max(np.abs(np.asarray(field.u)))) == 0.0


def test_the_record_is_present_even_when_nothing_was_excluded() -> None:
    """"The check ran and found nothing" and "the check did not run" are different."""
    rays = single_mode_bundle(
        axial_cosine=1.0,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float64,
    )
    _, record = reconstruct(rays)
    assert record.grazing["excluded_ray_count"] == 0
    assert record.grazing["policy"] == "refuse"
    assert record.grazing["max_phase_error_rad"] > 0.0
    assert set(record.grazing) >= {
        "policy",
        "phase_budget_rad",
        "optical_path_epsilon",
        "geometry_epsilon",
        "max_phase_error_rad",
        "max_optical_path_m",
        "min_axial_direction_cosine",
        "excluded_ray_count",
        "excluded_power_fraction",
    }


def test_the_default_is_to_refuse() -> None:
    """The risk the ticket names: shipping the caller-side option because it was there.

    Checked as behaviour on the same bundle, not read off the signature -- a default
    that a wrapper or a keyword reorder changed would still satisfy an
    introspection test.
    """
    rays = single_mode_bundle(
        axial_cosine=1.0e-3,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    with pytest.raises(ContractError):
        reconstruct(rays)
    field, record = reconstruct(rays, grazing="band_limit")
    assert record.grazing["policy"] == "band_limit"
    assert float(np.max(np.abs(np.asarray(field.u)))) == 0.0


def test_an_infinite_budget_admits_everything_and_records_that_it_did() -> None:
    """The unbanded row has to be expressible, and it has to be visible in the record.

    It is how the exactness table's "none" line is measured. It is deliberately not
    the default, and it is deliberately not silent: a run made this way says so.
    """
    rays = single_mode_bundle(
        axial_cosine=1.0e-4,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    _, record = reconstruct(rays, phase_budget_rad=math.inf)
    assert record.grazing["excluded_ray_count"] == 0
    assert record.grazing["phase_budget_rad"] == math.inf
    assert record.grazing["max_phase_error_rad"] > 1.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"phase_budget_rad": 0.0}, "phase_budget_rad"),
        ({"phase_budget_rad": -1.0}, "phase_budget_rad"),
        ({"grazing": "warn"}, "grazing"),
    ],
)
def test_an_unusable_policy_or_budget_is_rejected(kwargs: dict, match: str) -> None:
    rays = single_mode_bundle(
        axial_cosine=1.0,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match=match):
        reconstruct(rays, **kwargs)


def test_the_k_space_route_applies_the_same_floor() -> None:
    """The floor belongs to the operation, not to one of its realizations."""
    rays = single_mode_bundle(
        axial_cosine=1.0e-3,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    with pytest.raises(ContractError) as raised:
        reconstruct(rays, reconstruction=Reconstruction.KSPACE)
    assert raised.value.code == "GRAZING_PHASE_UNREPRESENTABLE"


# ---------------------------------------------------------------------------
# 5. The decision, recorded
# ---------------------------------------------------------------------------


def test_the_three_rejected_options_are_recorded() -> None:
    """Criterion 3's second half: the options not chosen, with the reason each.

    Pinned so the record cannot be deleted as prose tidying. The reasons are the
    part a later ticket needs: option 4 in particular is rejected as *unavailable
    to this module* rather than as future work, because `OPL_i` arrives already
    rounded and no arithmetic here can recover what the input does not carry.
    """
    prose = MODULE.read_text(encoding="utf-8")
    assert "The three options not taken" in prose
    for marker in (
        "Caller-side band limit",
        "band-limits itself, silently, by default",
        "Reformulate the constant phase",
    ):
        assert marker in prose, marker
    # Each is rejected, in writing.
    assert prose.count("Rejected") >= 3
    # And the numbers the decision rests on.
    for number in ("4745 m", "1.05e-8", "d_u^2 + d_v^2 = 1"):
        assert number in prose, number


def test_the_kernel_gate_is_twice_the_one_term_floor() -> None:
    """The relation between the planning helper and the gate, pinned rather than left.

    `grazing_floor_for_phase_budget` returns the **one-term** form
    `eps k Z / budget` -- the derivation the reference implementation froze `1e-2`
    from, which is why criterion 1 asks for that number. The kernel bounds both
    terms, and for a mode of an advanced angular spectrum the launch ramp is
    `(1 - d_n^2) |OPL|`, i.e. nearly equal to the optical path. So the smallest
    `d_n` the kernel admits is about **2x** the helper's floor.

    Measured on single float32 modes over a 50 um propagation, helper floor
    `7.4901e-3`: `d_n = 2.05 x floor` is admitted and `d_n = 1.95 x floor` is
    refused. Without this test a caller could size a spectrum at exactly the floor,
    be refused, and read it as a defect in one or the other.
    """
    floor = grazing_floor_for_phase_budget(
        wavelength_m=WAVELENGTH_M,
        max_optical_path_m=PROPAGATE_M,
        precision=Precision.FP32,
    )

    admitted = single_mode_bundle(
        axial_cosine=2.05 * floor,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    _, record = reconstruct(admitted)
    assert record.grazing["excluded_ray_count"] == 0

    refused = single_mode_bundle(
        axial_cosine=1.95 * floor,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    with pytest.raises(ContractError) as raised:
        reconstruct(refused)
    assert raised.value.code == "GRAZING_PHASE_UNREPRESENTABLE"

    # ...and the one-term form is exact for a ray crossing the plane on axis, where
    # the transverse ramp term vanishes and only the optical path is left. That is
    # the case the helper is literally right for: `1.05 x floor` is admitted there
    # and refused above.
    on_axis_crossing = single_mode_bundle(
        axial_cosine=1.05 * floor,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    at_origin = dataclasses.replace(
        on_axis_crossing,
        positions_m=np.array([[0.0, 0.0, PROPAGATE_M]], dtype=np.float32),
    )
    _, origin_record = reconstruct(at_origin)
    assert origin_record.grazing["excluded_ray_count"] == 0
    assert origin_record.grazing["max_phase_error_rad"] < DEFAULT_PHASE_BUDGET_RAD


def test_the_bound_reads_the_epsilon_of_the_dtype_each_term_was_stored_in() -> None:
    """A mixed-dtype bundle is legitimate, and the bound has to see the coarser half.

    `RayBundle` deliberately does not unify dtype across an artifact, and
    `backends.optiland` emits exactly this shape: it preserves the trace dtype for
    the geometry while `declare_optical_path_m` returns float64. `_compute_precision`
    takes the *max*, so such a bundle computes in FP64 -- and a bound taken from the
    compute precision would be eight orders of magnitude too small for the float32
    half, admit the mode, and report `max_phase_error_rad` as a number that is not
    an error bound at all.

    Measured: `d_n = 1e-4` with float32 geometry and a float64 optical path has a
    realized phase error of `6.283e-2` rad against a true phase of `6.283e-2` rad --
    100 % of signal. It must be refused.
    """
    axial_cosine = 1.0e-4
    float32_bundle = single_mode_bundle(
        axial_cosine=axial_cosine,
        propagate_m=PROPAGATE_M,
        wavelength_m=WAVELENGTH_M,
        dtype=np.float32,
    )
    mixed = dataclasses.replace(
        float32_bundle,
        optical_path_m=np.asarray(float32_bundle.optical_path_m, dtype=np.float64),
    )
    _, unbanded = reconstruct(mixed, grazing="band_limit", phase_budget_rad=math.inf)
    assert unbanded.compute_precision == "fp64"
    assert unbanded.grazing["optical_path_epsilon"] < unbanded.grazing["geometry_epsilon"]
    assert unbanded.grazing["max_phase_error_rad"] > DEFAULT_PHASE_BUDGET_RAD

    with pytest.raises(ContractError) as raised:
        reconstruct(mixed)
    assert raised.value.code == "GRAZING_PHASE_UNREPRESENTABLE"


def test_a_float16_bundle_is_bounded_by_float16_round_off() -> None:
    """The declared promotion is a *compute* promotion, not a repair of the input.

    `_compute_precision` computes a float16 bundle in float32, which the module
    calls out as a promotion rather than float16 support. The stored path length is
    still float16, so the bound must be float16's -- and on axis over a 1 mm
    propagation that is enough to refuse.
    """
    on_axis = single_mode_bundle(
        axial_cosine=1.0, propagate_m=1.0e-3, wavelength_m=WAVELENGTH_M, dtype=np.float64
    )
    half = dataclasses.replace(
        on_axis,
        positions_m=np.asarray(on_axis.positions_m, dtype=np.float16),
        directions=np.asarray(on_axis.directions, dtype=np.float16),
        optical_path_m=np.asarray(on_axis.optical_path_m, dtype=np.float16),
        # There is no complex32 in any of the three namespaces, so the smallest an
        # amplitude can be is complex64 -- which is also why FP16 has no complex
        # dtype in `numerics.Precision` and why the compute floor is FP32.
        amplitude=np.asarray(on_axis.amplitude, dtype=np.complex64),
    )
    _, record = reconstruct(half, grazing="band_limit", phase_budget_rad=math.inf)
    assert record.compute_precision == "fp32"
    assert record.grazing["optical_path_epsilon"] == pytest.approx(9.77e-4, rel=0.05)
    assert record.grazing["max_phase_error_rad"] > 1.0

    with pytest.raises(ContractError) as raised:
        reconstruct(half)
    assert raised.value.code == "GRAZING_PHASE_UNREPRESENTABLE"


def test_a_float32_wide_angle_spectrum_is_refused_and_band_limits_to_the_same_eight() -> None:
    """Criterion 5, on a *spectrum* rather than a single mode.

    The enumeration is the shape the defect was found on, so the criterion is met
    literally as well as in substance: the float32 100x100 enumeration over a 50 um
    propagation is refused by default, and `band_limit` excludes exactly the same
    eight bins carrying `1.133e-3` of launch power that the float64 run at a tighter
    budget does -- at `compute_precision = fp32`.

    What this case does **not** show is a large realized field error, and the test
    does not claim one. On this configuration the eight bins' true phase is
    `k Z d_n = 6.6e-6` rad and both float32 terms round to the same value, so the
    realized error there is small by coincidence rather than by design. The realized
    failure is exhibited on single modes above, where it reaches 100 % of signal.
    """
    rays, _, _ = an_enumeration(dtype=np.float32)
    assert str(rays.state.dtype) == "float32"

    with pytest.raises(ContractError) as raised:
        reconstruct(rays)
    assert raised.value.code == "GRAZING_PHASE_UNREPRESENTABLE"

    _, record = reconstruct(rays, grazing="band_limit")
    assert record.compute_precision == "fp32"
    assert record.grazing["excluded_ray_count"] == 8
    assert record.grazing["excluded_power_fraction"] == pytest.approx(1.133e-3, rel=0.05)
    assert record.grazing["max_phase_error_rad"] > 1.0e3
