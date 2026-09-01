"""R08.1: the angular-spectrum decomposition, and the round trip that stays here.

CHE-189. `scalar_to_ray` is a quadrature scheme for an integral whose exact value
is known, not an approximation of physics -- so the first and mandatory gate is
that enumerating every propagating bin collapses onto the deterministic reference
at dtype round-off. Only after that passes is there any point discussing sampling
error, which is R08.2's file.

**The round trip lives in this file and nowhere else.** A `ScalarField ->
RayBundle -> ScalarField` conversion with no physical transformation in between
changes no state; it is a representation-consistency check. R08 criterion 2 asks
for it here and for no production operator to perform it, and
`test_no_round_trip_operation_landed` is the second half of that.

The oracle is not the code under test. `ray_support.propagating_only` writes the
centred DFT, the strict `radial < 1` cut and the centred inverse out from the
transform, so a round trip is graded against NumPy's FFT rather than against
`scalar_to_ray`'s own decomposition.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from ray_support import a_random_field, a_surface, propagating_only

from couplers import (
    SAMPLING_DENSITIES,
    Projection,
    Reconstruction,
    SamplingDiagnostics,
    ray_to_scalar,
    scalar_to_ray,
)
from operations import OperationDescriptor, OperationKind, registry, resolve
from representations import ContractError, ReferenceSurface
from sources import plane_wave

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "couplers"
MODULE = PACKAGE / "scalar_to_ray.py"

SHAPE = (24, 32)
PITCH_M = (0.40e-6, 0.35e-6)


def round_trip(field, **kwargs):
    """Decompose and reconstruct on the field's own grid. Returns both records."""
    rays, sampling = scalar_to_ray(field, **kwargs)
    reconstructed, reconstruction = ray_to_scalar(
        rays, grid_shape=field.shape, sample_pitch_m=field.sample_pitch_m
    )
    return reconstructed, sampling, reconstruction


def peak_relative_residual(u, reference) -> float:
    return float(
        np.max(np.abs(np.asarray(u) - np.asarray(reference)))
        / np.max(np.abs(np.asarray(reference)))
    )


# ---------------------------------------------------------------------------
# 1. The exactness limit
# ---------------------------------------------------------------------------


def test_exhaustive_enumeration_round_trips_to_round_off() -> None:
    """Criterion 1, the mandatory gate: the estimator collapses onto the reference.

    Every propagating bin, uniform density, `ASM_CONSISTENT` reconstruction.
    Measured **3.5e-15** of peak against the field's own propagating content, on a
    24x32 grid at `(0.40, 0.35) um` -- dtype round-off, which is the only
    tolerance this case is allowed.

    The oracle is the *propagating* field, not the source field, and that is not a
    concession. An evanescent mode has no propagation direction to give a ray, so
    the round trip cannot return it; grading against the full source field would
    be grading the decomposition for a loss that is physical. That loss is
    reported rather than hidden -- see the next test.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    reconstructed, sampling, reconstruction = round_trip(field)

    assert sampling.selection == "exhaustive"
    assert sampling.density == "uniform"
    assert peak_relative_residual(reconstructed.u, propagating_only(field)) < 1e-13
    assert reconstruction.projection == str(Projection.ASM_CONSISTENT)
    assert reconstruction.normalization == "one_over_n"


def test_the_evanescent_loss_is_real_reported_and_the_only_thing_missing() -> None:
    """What the round trip does *not* return, quantified.

    On this field 5 of 768 modes are evanescent, carrying `7.6e-3` of the modal
    power, and the round trip misses the *source* field by `5.3e-2` of peak
    amplitude for exactly that reason. Both numbers are on the record, so a caller
    can tell an unrepresentable field from a broken coupler -- a large evanescent
    fraction is the signature of a field that should not be turned into rays at
    all.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    reconstructed, sampling, _ = round_trip(field)

    assert sampling.total_modes == SHAPE[0] * SHAPE[1]
    assert sampling.evanescent_mode_count == 5
    assert sampling.propagating_modes == sampling.total_modes - 5
    assert sampling.evanescent_power_fraction == pytest.approx(7.62e-3, rel=0.05)

    against_source = peak_relative_residual(reconstructed.u, field.u)
    assert against_source == pytest.approx(5.3e-2, rel=0.15)
    # ...and the residual really is the evanescent content, not a coupler defect:
    # what is missing equals the difference between the two oracles.
    missing = np.asarray(field.u) - propagating_only(field)
    assert peak_relative_residual(
        np.asarray(reconstructed.u) - np.asarray(field.u), -missing
    ) < 1e-12


def test_the_round_trip_holds_for_the_k_space_route_too() -> None:
    """The pair is exact as an *operation*, independent of which realization runs it.

    The k-space route quantizes each ray's direction onto a k-grid, and an
    enumerated spectrum lands on a node exactly when the k-grid period matches the
    grid the modes were enumerated on -- so `kspace_grid_shape` is named here, as
    R07.2's error budget says an enumerating caller must.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rays, _ = scalar_to_ray(field)
    reconstructed, record = ray_to_scalar(
        rays,
        grid_shape=SHAPE,
        sample_pitch_m=PITCH_M,
        reconstruction=Reconstruction.KSPACE,
        kspace_grid_shape=SHAPE,
    )
    assert record.kspace["on_node_fraction"] == 1.0
    assert peak_relative_residual(reconstructed.u, propagating_only(field)) < 1e-13


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_the_round_trip_computes_at_the_fields_own_precision(dtype) -> None:
    """A complex64 field decomposes in FP32 and comes back complex64, at FP32 accuracy.

    The tolerance is the dtype's, not a widened float64 one: a float32 round trip
    through a 768-bin spectrum accumulates round-off across the sum, and holding
    it to 1e-13 would be a category error rather than strictness.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M, dtype=dtype)
    reconstructed, sampling, reconstruction = round_trip(field)

    expected = "fp32" if dtype is np.complex64 else "fp64"
    assert sampling.compute_precision == expected
    assert reconstruction.compute_precision == expected
    assert str(reconstructed.state.dtype) == np.dtype(dtype).name

    tolerance = 1e-5 if dtype is np.complex64 else 1e-13
    assert peak_relative_residual(reconstructed.u, propagating_only(field)) < tolerance


# ---------------------------------------------------------------------------
# 2. The measure, and `1/p` applied exactly once
# ---------------------------------------------------------------------------


def test_the_emitted_measure_is_the_one_the_wavelet_sum_accepts() -> None:
    """Criterion 2. The two couplers agree, or the round trip cannot be exact.

    `importance_weight` is the declaration, and it is what obliges `ray_to_scalar`
    to apply the `1/N` of SI eq S5. Settled here rather than by loosening R07:
    R07.3's table maps `importance_weight -> one_over_n`, and this asserts the
    emitted bundle lands on that row.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rays, sampling = scalar_to_ray(field)

    assert rays.measure_kind == "importance_weight"
    assert sampling.measure_kind == "importance_weight"
    assert sampling.reconstruction_normalization == "one_over_n"

    _, reconstruction = ray_to_scalar(
        rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M
    )
    assert reconstruction.normalization == "one_over_n"


def test_the_amplitude_carries_no_importance_weight() -> None:
    """Criterion 4, the structural half: `1/p` is in the measure and only there.

    The reference implementation emitted `amplitude = U~/p`, which under a uniform
    density over 763 bins is 763x the modal amplitude. The new bundle emits the
    modal amplitude itself, so the two are distinguishable by a factor of the mode
    count rather than by reading a comment.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rays, sampling = scalar_to_ray(field)
    modes = sampling.propagating_modes

    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.asarray(field.u)))) / (
        SHAPE[0] * SHAPE[1]
    )
    modal_peak = float(np.max(np.abs(spectrum)))

    assert float(np.max(np.abs(np.asarray(rays.amplitude)))) == pytest.approx(
        modal_peak, rel=1e-12
    )
    # The measure is where 1/p lives, and under the uniform density it is the mode
    # count exactly.
    assert np.allclose(np.asarray(rays.measure_weight), float(modes), rtol=1e-12)


def test_double_application_of_the_importance_weight_is_visible() -> None:
    """Criterion 4, the measurement half: a case where applying `1/p` twice shows.

    Under the uniform density `1/p = M`, so a second application scales the whole
    field by exactly `M = 763`. Constructed by perturbing the *bundle* -- the same
    idiom R07's negative twins use -- so the shipping kernel is what runs, and the
    factor is a bare integer rather than a residual someone has to interpret.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rays, sampling = scalar_to_ray(field)
    modes = sampling.propagating_modes

    correct, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    doubled = dataclasses.replace(
        rays, measure_weight=np.asarray(rays.measure_weight) * float(modes)
    )
    twice, _ = ray_to_scalar(doubled, grid_shape=SHAPE, sample_pitch_m=PITCH_M)

    ratio = float(np.max(np.abs(np.asarray(twice.u)))) / float(
        np.max(np.abs(np.asarray(correct.u)))
    )
    assert ratio == pytest.approx(float(modes), rel=1e-9)
    assert peak_relative_residual(correct.u, propagating_only(field)) < 1e-13
    assert peak_relative_residual(twice.u, propagating_only(field)) > 1e2


def test_enumerating_a_non_uniform_density_is_refused() -> None:
    """Enumeration is the zero-variance case of a *uniform* draw, and only of one.

    Every bin is selected once regardless of `p`, so the `1/p` in the measure has
    no compensating draw frequency and the reconstruction is
    `sum U~ / (M p)` rather than the field. A silent wrong answer, so the pair is
    refused rather than repaired by quietly substituting the uniform density.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(field, density="magnitude")
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "density"
    # ...and the combination that is legal does work, so this is a pairing rule
    # rather than a ban on the density.
    rays, sampling = scalar_to_ray(
        field, count=64, density="magnitude", rng=np.random.default_rng(1), seed=1
    )
    assert sampling.density == "magnitude"
    assert rays.count == 64


# ---------------------------------------------------------------------------
# 3. The evanescent cut, and the bins it keeps
# ---------------------------------------------------------------------------


def test_the_cut_is_the_strict_radial_less_than_one() -> None:
    """Criterion 3. Strict, and the count matches an independently written mask."""
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    _, sampling = scalar_to_ray(field)

    direction_v, direction_u = np.meshgrid(
        np.fft.fftshift(np.fft.fftfreq(SHAPE[0], PITCH_M[0])) * field.wavelength_m,
        np.fft.fftshift(np.fft.fftfreq(SHAPE[1], PITCH_M[1])) * field.wavelength_m,
        indexing="ij",
    )
    expected = int(np.sum(direction_u**2 + direction_v**2 < 1.0))
    assert sampling.propagating_modes == expected
    # A non-strict cut would keep the `d_n = 0` bins, which are singular for any
    # 1/d_n factor. On this grid there are none exactly on the circle, so the
    # difference shows on the grid built to have them -- below.
    assert sampling.min_axial_direction_cosine > 0.0


def test_the_grazing_bins_that_survive_the_strict_cut_are_counted() -> None:
    """Criterion 3, and the handshake with R07.4.

    On CHE-70's grid -- 100x100 at 250 nm, 500 nm light -- eight bins land on
    `d_u^2 + d_v^2 = 1` exactly (the (30, 40) and (40, 30) Pythagorean triples and
    their sign variants) and survive the strict cut at `d_n = 1.05e-8`. They are
    kept, because a bin one ulp inside the circle is a propagating mode; and they
    are **counted and reported**, because whether a *reconstruction* can carry
    them is `ray_to_scalar`'s grazing floor to decide. Two tickets, one mask, and
    the count is what lets the two be checked against each other.
    """
    field = a_random_field(
        shape=(100, 100), sample_pitch_m=(250e-9, 250e-9), wavelength_m=500e-9, seed=70
    )
    _, sampling = scalar_to_ray(field)

    assert sampling.grazing_report_floor == 1.0e-2
    assert sampling.grazing_mode_count == 8
    assert sampling.min_axial_direction_cosine == pytest.approx(1.0537e-8, rel=1e-3)
    assert sampling.evanescent_mode_count == 10000 - sampling.propagating_modes

    # Nothing is excluded by the report: the eight are in the emitted ensemble.
    rays, _ = scalar_to_ray(field)
    axial = np.asarray(rays.directions)[:, 2]
    assert int(np.sum(axial < 1.0e-2)) == 8


def test_the_propagating_set_is_never_empty_because_dc_always_propagates() -> None:
    """Why there is no "no propagating mode" refusal, stated as a fact rather than
    left as an omission.

    A pitch far below half a wavelength puts almost every bin outside the unit
    circle -- here `lambda / (2 pitch) = 13.75`, so the grid expresses direction
    cosines up to 13.75 and only a handful survive. But the DC bin has
    `radial = 0 < 1` on *every* grid, so the set is never empty and a branch for
    that case would be a declared failure nothing can reach.

    What is left is a single-mode plane wave carrying almost none of the field,
    which the evanescent fraction reports as the near-total loss it is.
    """
    wavelength_m, pitch_m = 0.55e-6, 20e-9
    assert wavelength_m / (2 * pitch_m) > 1.0
    field = a_random_field(
        shape=(8, 8), sample_pitch_m=(pitch_m, pitch_m), wavelength_m=wavelength_m
    )
    rays, sampling = scalar_to_ray(field)
    assert sampling.propagating_modes == 1
    assert sampling.min_axial_direction_cosine == pytest.approx(1.0)
    assert sampling.evanescent_power_fraction > 0.9
    assert rays.count == 1


def test_a_zero_field_has_no_magnitude_density() -> None:
    """`EMPTY_ENSEMBLE` where it *is* reachable: `p_mag` is undefined on a zero spectrum."""
    field = a_random_field(shape=(8, 8), sample_pitch_m=PITCH_M)
    zeroed = dataclasses.replace(field, u=np.zeros_like(np.asarray(field.u)))
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(zeroed, count=4, density="magnitude", rng=np.random.default_rng(0))
    assert raised.value.code == "EMPTY_ENSEMBLE"


# ---------------------------------------------------------------------------
# 4. What the coupler refuses, and what it will not decide for you
# ---------------------------------------------------------------------------


def test_a_stochastic_draw_without_a_generator_is_refused() -> None:
    """No implicit seed. A result depending on state the caller cannot see is not
    reproducible even in principle."""
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(field, count=32)
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "rng"


def test_a_surface_the_field_is_not_on_is_refused() -> None:
    """The mirror image of `ray_to_scalar`'s check.

    If either direction could relabel the surface, the pair would compose into a
    silent defocus -- which is why both refuse rather than one.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(
            field, surface=ReferenceSurface(name="sensor", z_m=1e-3, medium_index=1.0)
        )
    assert raised.value.code == "FRAME_MISMATCH"
    rays, _ = scalar_to_ray(field, surface=field.reference_surface)
    assert rays.reference_surface == field.reference_surface


@pytest.mark.parametrize("bad", [0, -4])
def test_a_non_positive_mode_count_is_refused(bad: int) -> None:
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(field, count=bad, rng=np.random.default_rng(0))
    assert raised.value.code == "EMPTY_ENSEMBLE"


def test_an_unknown_density_is_refused() -> None:
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(field, density="p_energy")  # type: ignore[arg-type]
    assert raised.value.code == "MISSING_DECLARATION"
    assert SAMPLING_DENSITIES == ("uniform", "magnitude")


def test_launch_positions_must_be_a_pair_per_point() -> None:
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    with pytest.raises(ContractError) as raised:
        scalar_to_ray(field, launch_positions_xy_m=np.zeros((3, 3)))
    assert raised.value.code == "SHAPE_MISMATCH"


# ---------------------------------------------------------------------------
# 5. Launch positions
# ---------------------------------------------------------------------------


def test_a_launch_position_contributes_the_phase_its_offset_implies() -> None:
    """`exp(i k (d_u x_p + d_v y_p))`, and `P * S` rays for `P` positions.

    The outer product is what keeps the ray count a caller's budget rather than
    something that grows multiplicatively across cascaded surfaces (SI Algorithm
    S1). The phase is checked against the analytic factor per mode, so a sign or a
    transposition in it would show.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    offsets = np.array([[0.0, 0.0], [1.3e-6, -0.7e-6]])
    centred, _ = scalar_to_ray(field)
    shifted, sampling = scalar_to_ray(field, launch_positions_xy_m=offsets)

    assert sampling.launch_position_count == 2
    assert shifted.count == 2 * centred.count

    modes = centred.count
    first = np.asarray(shifted.amplitude)[:modes]
    second = np.asarray(shifted.amplitude)[modes:]
    transverse = np.asarray(centred.directions)[:, :2]
    expected = np.exp(
        1j
        * field.wavenumber
        * (transverse[:, 0] * offsets[1, 0] + transverse[:, 1] * offsets[1, 1])
    )
    assert np.allclose(first, np.asarray(centred.amplitude), rtol=1e-12)
    assert np.allclose(second / first, expected, rtol=1e-9)


def test_a_launch_offset_moves_no_field_because_the_launch_phase_cancels_it() -> None:
    """What the launch phase is actually for, and it is not a shift.

    `scalar_to_ray` gives ray `(p, m)` the amplitude `U~[m] exp(+i k d_m . x_p)`
    and puts it at `x_p`. `ray_to_scalar` then forms `k (OPL - d . x_0)` with
    `x_0 = x_p`, so the two cancel exactly and the reconstruction is
    `sum_m U~[m] exp(i k d_m . r)` -- the same field, from any launch point.
    Measured: the best whole-sample roll matching a `(2 dx, -3 dy)` offset is
    `(0, 0)` at 3.6e-15.

    That is correct physics -- a plane wave is infinite, so where on the plane it
    is launched from cannot matter -- and it is what keeps `P` launch points a
    superposition of `P` copies of *one* field rather than of `P` shifted ones.

    Stated precisely, because the loose version is wrong: with the mode indices
    drawn **once** and reused at every position, as they are here, the `P`
    contributions are term-by-term identical, so `P` positions currently reduce
    variance by exactly zero and multiply cost by `P`. Positions buy something only
    when each draws independently, or when the rays subsequently traverse something
    that distinguishes them. R08.2 is where that is decided; nothing in this file
    should be read as evidence that it is already true.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    offset = np.array([[2 * PITCH_M[1], -3 * PITCH_M[0]]])
    rays, _ = scalar_to_ray(field, launch_positions_xy_m=offset)
    reconstructed, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    assert peak_relative_residual(reconstructed.u, propagating_only(field)) < 1e-13


def test_the_twin_dropping_the_launch_phase_shifts_the_field() -> None:
    """The negative twin, and the measurement that makes the phase load-bearing.

    Strip `exp(+i k d . x_p)` from the amplitudes -- the bundle is otherwise
    identical, so the shipping kernel is what runs -- and the kernel's `-d . x_0`
    is left uncancelled. The reconstruction becomes `u(r - x_p)`: the field slid
    by the launch offset, `(-3, +2)` samples for a `(2 dx, -3 dy)` offset,
    reproduced to 3.2e-15.

    A test that only checked "the reconstruction is a valid-looking field" would
    pass on both. Two launch points under this defect would then average two
    *differently shifted* copies of the field and blur it, which is a plausible
    result and a wrong one.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    offset = np.array([[2 * PITCH_M[1], -3 * PITCH_M[0]]])
    rays, _ = scalar_to_ray(field, launch_positions_xy_m=offset)
    centred, _ = scalar_to_ray(field)
    stripped = dataclasses.replace(rays, amplitude=np.asarray(centred.amplitude))

    reconstructed, _ = ray_to_scalar(stripped, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    shifted = np.roll(propagating_only(field), shift=(-3, 2), axis=(0, 1))
    assert peak_relative_residual(reconstructed.u, shifted) < 1e-13
    assert peak_relative_residual(reconstructed.u, propagating_only(field)) > 1e-1


def test_several_launch_points_reproduce_the_identical_estimator() -> None:
    """The consequence: `P` points, `P * S` rays, and bit-for-bit the same answer.

    `1/N` over `P * S` rays is `1/P` of the sum over positions of an estimator each
    of which is already the field. And because the indices are drawn once and
    reused, the `P` contributions are not merely equal in expectation -- they are
    the same terms, so the reconstruction matches the single-position one to
    round-off and the ray count is `P` times larger for nothing.

    That is asserted rather than glossed, because it is the fact R08.2 has to
    change if positional sampling is to buy variance reduction.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rng = np.random.default_rng(5)
    offsets = rng.uniform(-4e-6, 4e-6, size=(7, 2))
    rays, sampling = scalar_to_ray(field, launch_positions_xy_m=offsets)
    reconstructed, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)

    single, _ = ray_to_scalar(
        scalar_to_ray(field)[0], grid_shape=SHAPE, sample_pitch_m=PITCH_M
    )
    assert sampling.launch_position_count == 7
    assert sampling.ray_count == 7 * sampling.propagating_modes
    assert peak_relative_residual(reconstructed.u, propagating_only(field)) < 1e-13
    assert peak_relative_residual(reconstructed.u, single.u) < 1e-13


# ---------------------------------------------------------------------------
# 6. The record, the boundary, and what did not land
# ---------------------------------------------------------------------------


def test_the_sampling_record_is_json_shaped_and_states_how_the_rays_were_obtained() -> None:
    """Criterion 3 of R08: a record can say how its rays were obtained."""
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rays, sampling = scalar_to_ray(
        field, count=50, density="magnitude", rng=np.random.default_rng(11), seed=11
    )
    assert isinstance(sampling, SamplingDiagnostics)
    assert sampling.selection == "stochastic"
    assert sampling.density == "magnitude"
    assert sampling.seed == 11
    assert sampling.drawn_mode_count == 50
    assert sampling.ray_count == rays.count

    record = sampling.as_dict()
    assert record["grid_shape"] == list(SHAPE)
    for key, value in record.items():
        assert isinstance(value, (int, float, str, bool, list, dict, type(None))), key


def test_parseval_relates_the_two_powers_rather_than_equating_them() -> None:
    """`sum |U~|^2 = (1 / (ny nx)) sum |u|^2`, with the pitch factored out.

    Reported as two numbers because they are two numbers. `field_discrete_power`
    carries `dy dx`; `modal_power_sum` does not.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    _, sampling = scalar_to_ray(field)
    ny, nx = SHAPE
    dy, dx = PITCH_M
    assert sampling.modal_power_sum == pytest.approx(
        sampling.field_discrete_power / (dy * dx) / (ny * nx), rel=1e-12
    )


def test_the_module_imports_no_solver_and_no_backend() -> None:
    """Criterion 5. The same rule `test_ray_to_scalar.py` states for the package."""
    forbidden = {"optiland", "chromatix", "jax", "torch", "solvers", "problems"}
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & forbidden), sorted(imported & forbidden)

    probe = (
        "import sys; from couplers import scalar_to_ray; "
        "print(sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'optiland', 'chromatix', 'jax', 'torch'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip() == "[]", result.stdout


def test_no_round_trip_operation_landed() -> None:
    """R08 criterion 2, second half. The round trip is a test, and stays one.

    A ray -> wave -> ray conversion with no physical transformation between the
    couplers changes no state, so it is a representation-consistency check.
    Shipping it as an operation would advertise a physical capability that is
    really a test fixture -- and the name is the thing that would do the
    advertising, which is why this is a name check.
    """
    defined = {
        node.name
        for module in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    # Names, not prose. The package docstring *says* there is no round trip here,
    # and a substring search over the source would flag that sentence -- the same
    # trap `tests/solvers/test_optiland_boundary.py` documents when it exempts
    # docstrings: the only correct response to a gate that fails on its own
    # explanation is to stop explaining.
    for banned in ("round_trip", "roundtrip", "ray_to_wave_to_ray", "wave_to_ray_to_wave"):
        assert not any(banned in name for name in defined), banned
    assert defined, "the walk read nothing, so it cannot fail"


def test_the_avoided_sampling_classes_did_not_land() -> None:
    """Criterion 6, the part a budget cannot record: what was avoided."""
    defined = {
        node.name
        for module in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    for avoided in (
        "AngularSpectrum",
        "SamplingPerturbation",
        "PositionPlan",
        "PatchPlan",
        "Ensemble",
        "SamplingDensity",
    ):
        assert avoided not in defined, f"{avoided} landed in couplers/"
    assert defined == {
        "Projection",
        "Reconstruction",
        "ReconstructionDiagnostics",
        "SamplingDiagnostics",
    }


@pytest.fixture()
def isolated_registry() -> Iterator[None]:
    saved = dict(registry._REGISTERED)
    registry._REGISTERED.clear()
    yield
    registry._REGISTERED.clear()
    registry._REGISTERED.update(saved)


def test_the_decomposition_registers_as_a_coupler(isolated_registry: None) -> None:
    """`scalar_field -> ray_bundle`, the reverse port pair of `C_RAY_TO_SCALAR`.

    Test-side for the reason every solver, operator and coupler ticket since R05.3
    has recorded: `couplers/` may not import `operations/` and vice versa, so no
    production registration site exists.
    """
    descriptor = registry.register(
        OperationDescriptor(
            operation_id="C_SCALAR_TO_RAY",
            kind=OperationKind.COUPLER,
            input="scalar_field",
            output="ray_bundle",
            implementation="couplers.scalar_to_ray:scalar_to_ray",
            approximation=(
                "the field is decomposed into plane-wave modes on its own grid and each "
                "selected mode becomes a ray. Evanescent modes are discarded -- they have "
                "no propagation direction to give a ray -- and the discarded power is "
                "reported. A stochastic selection is a Monte-Carlo estimator of the modal "
                "sum, so the emitted measure is an importance weight and the "
                "reconstruction owes a 1/N; an exhaustive enumeration is the same "
                "estimator with zero variance"
            ),
            validity=(
                "scalar, monochromatic, fully coherent",
                "the field's grid fixes the mode set; a mode finer than the pitch is not "
                "represented",
                "exhaustive enumeration is exact only under the uniform density",
            ),
            evidence=(
                "tests/physics/test_scalar_to_ray.py",
                "tests/physics/test_scalar_to_ray_estimator.py",
            ),
            capabilities=None,
            derivative="forward_only",
        )
    )
    assert descriptor.kind is OperationKind.COUPLER
    assert descriptor.derivative == "forward_only"
    assert resolve("C_SCALAR_TO_RAY") is scalar_to_ray


def test_the_grid_nyquist_limit_and_the_evanescent_cut_are_different_conditions() -> None:
    """Two sampling conditions that fail independently, and are often confused.

    The evanescent cut is a property of the *field's* pitch -- which modes exist.
    The grid Nyquist limit is a property of the *output* grid -- which ramps it can
    represent. Reconstructing onto a coarser grid than the field was decomposed on
    is refused by `ray_to_scalar`, and refining the ray count does not help.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rays, _ = scalar_to_ray(field)
    coarse = (PITCH_M[0] * 4, PITCH_M[1] * 4)
    with pytest.raises(ContractError) as raised:
        ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=coarse)
    assert raised.value.code == "SHAPE_MISMATCH"
    assert "steepest wavelet ramp" in str(raised.value)


# ---------------------------------------------------------------------------
# 7. The mode set is a property of the grid, not of the storage dtype
# ---------------------------------------------------------------------------


def test_the_evanescent_mask_does_not_depend_on_the_fields_storage_precision() -> None:
    """The same field in complex64 and complex128 decomposes into the same modes.

    Not free: on CHE-70's grid the eight `(30, 40)` bins sit at
    `d_u^2 + d_v^2 = 1 - 1.1e-16`. float64 keeps them and float32 rounds the sum to
    `>= 1`, so computing the mask at the storage precision would make a complex64
    field emit a *different mode set* -- and would lose exactly the population
    criterion 3 exists to report. The mask is therefore computed at float64,
    because the grid pitch and the wavelength are Python floats and the mode set is
    a property of them.
    """
    grid = dict(shape=(100, 100), sample_pitch_m=(250e-9, 250e-9), wavelength_m=500e-9, seed=70)
    _, wide = scalar_to_ray(a_random_field(**grid, dtype=np.complex128))
    _, narrow = scalar_to_ray(a_random_field(**grid, dtype=np.complex64))

    assert narrow.compute_precision == "fp32"
    assert wide.compute_precision == "fp64"
    assert narrow.propagating_modes == wide.propagating_modes
    assert narrow.grazing_mode_count == wide.grazing_mode_count == 8
    assert narrow.min_axial_direction_cosine == pytest.approx(
        wide.min_axial_direction_cosine, rel=1e-6
    )


# ---------------------------------------------------------------------------
# 8. The recorded seed
# ---------------------------------------------------------------------------


def test_a_recorded_seed_regenerates_the_ensemble() -> None:
    """R08's criterion 3, the half that belongs here: bit-identical from a seed."""
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    first, first_record = scalar_to_ray(
        field, count=64, rng=np.random.default_rng(2026), seed=2026
    )
    second, _ = scalar_to_ray(
        field, count=64, rng=np.random.default_rng(2026), seed=2026
    )
    assert first_record.seed == 2026
    assert np.array_equal(np.asarray(first.directions), np.asarray(second.directions))
    assert np.array_equal(np.asarray(first.amplitude), np.asarray(second.amplitude))
    assert np.array_equal(np.asarray(first.measure_weight), np.asarray(second.measure_weight))


def test_a_seed_that_would_not_regenerate_the_ensemble_is_refused() -> None:
    """A record naming a seed that does not reproduce it reads as reproducible and
    is not, so it is refused rather than written.

    The legitimate case -- several ensembles from one advanced generator -- simply
    cannot name a single seed for the second draw, and `seed=None` says so.
    """
    field = a_random_field(shape=SHAPE, sample_pitch_m=PITCH_M)
    rng = np.random.default_rng(7)
    scalar_to_ray(field, count=8, rng=rng, seed=7)  # first draw: state matches

    with pytest.raises(ContractError) as advanced:
        scalar_to_ray(field, count=8, rng=rng, seed=7)  # ...but now it has moved
    assert advanced.value.code == "MISSING_DECLARATION"
    assert advanced.value.declaration == "seed"

    with pytest.raises(ContractError):
        scalar_to_ray(field, count=8, rng=np.random.default_rng(7), seed=8)

    # Omitting the seed is always allowed, and the record says so rather than lying.
    _, record = scalar_to_ray(field, count=8, rng=rng)
    assert record.seed is None


# ---------------------------------------------------------------------------
# 9. A round trip with no FFT in the oracle at all
# ---------------------------------------------------------------------------


def test_a_plane_wave_on_a_spectral_bin_round_trips_to_itself() -> None:
    """The strongest form of criterion 1: the *source* field, and no transform oracle.

    Every other round-trip gate here compares against `propagating_only`, which is
    NumPy's FFT -- independent of this repository, but still a transform. This one
    has none: `sources.plane_wave` builds `exp(i k_t . r)` from an analytic
    formula, the wavevector is placed exactly on a spectral bin so the
    decomposition is a single mode with no leakage, and the round trip must return
    the field itself rather than a truncation of it.

    A plane wave has no evanescent content, so unlike the random-field case there
    is nothing the round trip is allowed to lose.
    """
    shape, pitch = (16, 20), (0.40e-6, 0.35e-6)
    wavelength_m = 0.55e-6
    # Bin (2, 3): k_x = 2 pi * 3 / (nx dx), k_y = 2 pi * 2 / (ny dy). Exactly on a
    # node of the centred DFT, so the spectrum is one nonzero coefficient.
    k_y = 2.0 * math.pi * 2 / (shape[0] * pitch[0])
    k_x = 2.0 * math.pi * 3 / (shape[1] * pitch[1])
    field = plane_wave(
        shape,
        sample_pitch_m=pitch,
        wavelength_m=wavelength_m,
        reference_surface=a_surface("plane"),
        transverse_wavevector_rad_per_m=(k_y, k_x),
    )

    rays, sampling = scalar_to_ray(field)
    assert sampling.evanescent_power_fraction < 1e-12
    reconstructed, _ = ray_to_scalar(
        rays, grid_shape=shape, sample_pitch_m=pitch
    )
    assert peak_relative_residual(reconstructed.u, field.u) < 1e-5  # complex64 source
