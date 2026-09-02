"""R07.1: the wavelet-sum kernel, its frozen conventions, and the four-item checklist.

CHE-185. Every gate here compares the coupler against something written outside
this repository -- an analytic plane wave, an inverse DFT, or the stationary-phase
value of a Fresnel integral -- so none of them is this tree's numerical code
grading itself.

The negative twins perturb the *bundle*, not the kernel
-------------------------------------------------------
The reference implementation carried a `Perturbation` dataclass whose fields
switched terms off inside the kernel, so a negative test exercised the shipping
code with one factor removed. R07 lists that class as avoided, and the
replacement has the same property for a different reason: each twin here hands
the shipping kernel a bundle that is wrong in exactly one declared way -- a
negated optical path, a transposed direction, a geometry read in millimetres --
and asserts the reconstruction is wrong in the corresponding way. Nothing
parallel to the kernel is written, so nothing parallel to it can drift.

What each checklist item would miss without its twin is the point of having one:
a rotationally symmetric case never catches a `(y, x)` transposition, and normal
incidence never catches a missing obliquity factor.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ray_support import (
    FOCAL_M,
    WAVELENGTH_M,
    collimated_bundle,
    converging_bundle,
    focal_peak_oracle,
    mode_bundle,
    plateau_radius_m,
    shifted_inverse_dft,
)

from couplers import (
    SCALE_NOTE,
    Projection,
    ReconstructionDiagnostics,
    grid_nyquist_direction_limit,
)
from couplers.ray_to_scalar import ray_to_scalar
from operations import CATALOG, OperationKind, resolve
from representations import UNVERIFIED, ContractError, ReferenceSurface

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "couplers"

#: Deliberately non-square in both count and pitch. An axis-symmetric fixture
#: cannot fail on a transposition, which is checklist item 2's whole warning.
SHAPE = (32, 40)
PITCH_M = (0.30e-6, 0.25e-6)


def reconstruct(rays, *, shape=SHAPE, pitch=PITCH_M, **kwargs):
    return ray_to_scalar(rays, grid_shape=shape, sample_pitch_m=pitch, **kwargs)


def peak_relative_residual(u, reference) -> float:
    return float(np.max(np.abs(u - reference)) / np.max(np.abs(reference)))


# ---------------------------------------------------------------------------
# 1. The validated numbers: the field is preserved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theta", [0.0, 0.35])
def test_a_single_mode_reconstructs_to_dtype_round_off(theta: float) -> None:
    """Criterion 1, SI Figure S1c: `U(r) = N dA exp(+i k d_hat . r)`, exactly.

    Every ray carries the optical path its own launch point implies, so the
    ensemble is one plane-wave mode and the oracle is analytic. The reference
    implementation's validated tolerance for this case is dtype round-off, and
    that is what is asserted: measured 3.1e-15 at both angles in float64.
    """
    rays, d_hat, area = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(math.sin(theta), 0.0, math.cos(theta))
    )
    field, diagnostics = reconstruct(rays)

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    oracle = rays.count * area * np.exp(
        1j * rays.wavenumber * (d_hat[0] * grid_x + d_hat[1] * grid_y)
    )

    assert peak_relative_residual(field.u, oracle) < 1e-13
    assert field.validity == frozenset({"surface_only", "no_wavefront_curvature_term"})
    assert diagnostics.normalization == "none"
    assert diagnostics.measure_kind == "quadrature_area_m2"


def test_the_asm_convention_reproduces_a_whole_random_field() -> None:
    """Criterion 1, and the measurement the default projection was chosen by.

    Summing every propagating mode of a random field returns that field, because
    a representation change has to preserve it. CHE-25 measured 7.1e-15 on a
    16x16 grid; the same construction here measures 1.3e-15 -- same round-off
    order, and the number differs only because the grid and the random draw do.
    """
    rng = np.random.default_rng(20260831)
    source = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    pitch = (0.5e-6, 0.5e-6)
    rays, retained, spectrum = mode_bundle(source, sample_pitch_m=pitch)
    reference = shifted_inverse_dft(np.where(retained, spectrum, 0.0))

    field, diagnostics = reconstruct(rays, shape=(16, 16), pitch=pitch)

    assert peak_relative_residual(field.u, reference) < 1e-13
    # An importance weight is an estimator, so this one *does* owe the 1/N -- and
    # the weight makes that 1/N the inverse DFT's own normalization.
    assert diagnostics.normalization == "one_over_n"


@pytest.mark.parametrize("medium_index", [1.0, 1.336, 1.5168])
def test_the_transverse_ramp_carries_the_medium_index(medium_index: float) -> None:
    """The `n` R09 found missing (CHE-192): the ramp is `n k0 d_hat . dr`, not `k0 d_hat . dr`.

    In a medium of index `n` a plane wave is `exp(i n k0 s_hat . r)`, so the same
    collimated ensemble -- whose optical path is `n` times the geometric one, because
    an optical path always is -- must reconstruct to `N dA exp(i n k0 d_hat . r)`.
    Checked at three indices including air, so the `n = 1` case cannot be the only
    one that passes, and every number this tree measured in air is unchanged.
    """
    theta = 0.35
    rays, d_hat, area = collimated_bundle(
        shape=SHAPE,
        sample_pitch_m=PITCH_M,
        direction=(math.sin(theta), 0.0, math.cos(theta)),
        medium_index=medium_index,
    )
    field, diagnostics = reconstruct(rays)

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    ramp = medium_index * rays.wavenumber * (d_hat[0] * grid_x + d_hat[1] * grid_y)
    oracle = rays.count * area * np.exp(1j * ramp)
    assert peak_relative_residual(field.u, oracle) < 1e-13

    # The Nyquist limit is on the *medium* wavelength: the ramp on the grid has
    # spatial frequency `n d_t / lambda_0`, so a medium tightens it by exactly `n`.
    assert diagnostics.grid_nyquist_direction_limit == (
        pytest.approx(grid_nyquist_direction_limit(WAVELENGTH_M / medium_index, PITCH_M[0])),
        pytest.approx(grid_nyquist_direction_limit(WAVELENGTH_M / medium_index, PITCH_M[1])),
    )


def test_the_vacuum_ramp_is_the_negative_control_for_the_medium_one() -> None:
    """The half that makes the test above mean something: at `n != 1` they differ.

    A kernel that ignored `medium_index` would reconstruct `exp(i k0 d_hat . r)`
    against an optical path that already grew by `n`, and the two disagree by
    `(n - 1) k0 d_t . dr` -- unbounded in waves, and here **1.7 of peak amplitude**
    over a 10 um window at `n = 1.336`. Not a tolerance question, which is why R09
    refused rather than let it compute.
    """
    theta = 0.35
    medium_index = 1.336
    rays, d_hat, area = collimated_bundle(
        shape=SHAPE,
        sample_pitch_m=PITCH_M,
        direction=(math.sin(theta), 0.0, math.cos(theta)),
        medium_index=medium_index,
    )
    field, _ = reconstruct(rays)

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    vacuum = rays.count * area * np.exp(
        1j * rays.wavenumber * (d_hat[0] * grid_x + d_hat[1] * grid_y)
    )
    assert peak_relative_residual(field.u, vacuum) > 1.0


def test_the_medium_ramp_agrees_with_the_plane_wave_source() -> None:
    """Corroboration from a module that was already `n`-aware, and is not this one.

    `sources.plane_wave` states illumination as the transverse wavevector `k_t` and
    refuses `|k_t| > n k0`, so its `k_t` is the **medium** wavevector -- which is what
    told R09 the missing `n` was real rather than a bookkeeping choice. Building the
    source at `k_t = n k0 d_t` and the bundle at the same `d_hat` must give the same
    ramp, so the two conventions are checked against each other rather than each
    against itself.
    """
    from sources import plane_wave

    theta = 0.35
    medium_index = 1.336
    surface = ReferenceSurface(name="in water", z_m=0.0, medium_index=medium_index)
    rays, d_hat, area = collimated_bundle(
        shape=SHAPE,
        sample_pitch_m=PITCH_M,
        direction=(math.sin(theta), 0.0, math.cos(theta)),
        medium_index=medium_index,
    )
    field, _ = reconstruct(rays)

    transverse = medium_index * rays.wavenumber * np.asarray([d_hat[1], d_hat[0]])
    source = plane_wave(
        SHAPE,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=surface,
        transverse_wavevector_rad_per_m=(float(transverse[0]), float(transverse[1])),
    )
    # complex64 from the source, so the tolerance is float32's on a phase of ~30 rad.
    assert peak_relative_residual(
        np.asarray(field.u) / (rays.count * area), np.asarray(source.u)
    ) < 1e-5


def test_the_sensor_obliquity_convention_is_a_different_operator() -> None:
    """Criterion 3, checklist item 3: main-text eq 2 and SI eq S5 are not one operator.

    Measured on the same random field: `ASM_CONSISTENT` reproduces it to 1.3e-15
    and `SENSOR_OBLIQUITY` misses it by 1.4e-1 of peak amplitude, with a smallest
    `cos(theta)` of 0.63 on this grid. CHE-25 measured 2.2 % on a grid whose
    smallest `cos(theta)` was far closer to 1; what reproduces is the *finding* --
    one convention preserves the field to round-off and the other does not, by an
    amount set by how oblique the spectrum reaches -- and not the percentage,
    which is a property of the grid. Note that the miss is not `1 - min cos`
    either: the most oblique mode is not the one carrying peak amplitude, so the
    test asserts the thirteen orders of magnitude between the two rather than a
    fitted coefficient.
    """
    rng = np.random.default_rng(20260831)
    source = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
    pitch = (0.5e-6, 0.5e-6)
    rays, retained, spectrum = mode_bundle(source, sample_pitch_m=pitch)
    reference = shifted_inverse_dft(np.where(retained, spectrum, 0.0))

    exact, exact_diagnostics = reconstruct(rays, shape=(16, 16), pitch=pitch)
    sensor, diagnostics = reconstruct(
        rays, shape=(16, 16), pitch=pitch, projection=Projection.SENSOR_OBLIQUITY
    )

    assert peak_relative_residual(exact.u, reference) < 1e-13
    missed = peak_relative_residual(sensor.u, reference)
    assert 1e-2 < missed < 1.0 - diagnostics.min_projection_factor, missed
    assert missed > 1e12 * peak_relative_residual(exact.u, reference)
    # ...and the record names which of the two equations produced the field, so a
    # consumer can tell a reconstruction from a detector model without guessing.
    assert "eq S5" in exact_diagnostics.equation
    assert "eq 2" in diagnostics.equation


# ---------------------------------------------------------------------------
# 2. The oracle for the absolute scale
# ---------------------------------------------------------------------------


def test_the_lambda_r_plateau_oracle_fixes_the_launch_amplitude_scale() -> None:
    """Criterion 2. `|U(0)| = lambda R` for a unit-density pupil at `a^2 = lambda R / 3`.

    Stationary phase on the converging bundle gives
    `int dA exp(i k rho^2 / 2R) = i lambda R (1 - exp(i pi a^2 / (lambda R)))`, so
    the truncation factor is exactly 1 at this aperture and the gate is the bare
    number `lambda R = 2.660625e-09`. Measured: 2.660742e-09 at 32 rings, a ratio
    of 1.000044.

    This is the test that fails if the launch amplitude scale moves -- in
    particular if `measure_weight` stops being multiplied into the sum, which
    changes the answer by `dA` per ray while leaving every peak-normalized metric
    untouched.
    """
    radius = plateau_radius_m()
    rays, _ = converging_bundle(rings=32, radius_m=radius)
    field, _ = reconstruct(rays, shape=(9, 9), pitch=(0.2e-6, 0.2e-6))

    peak = float(abs(field.u[9 // 2, 9 // 2]))
    assert peak == pytest.approx(WAVELENGTH_M * FOCAL_M, rel=1e-3)
    assert peak == pytest.approx(focal_peak_oracle(radius_m=radius), rel=1e-3)


def test_the_focal_peak_follows_the_closed_form_at_a_second_aperture() -> None:
    """The same oracle away from its convenient point, so the agreement is not a fit.

    At `a^2 = 2 lambda R / 3` the truncation factor is `2 sin(pi / 3) = 1.732`,
    which no scale error could reproduce by coincidence.
    """
    radius = math.sqrt(2.0) * plateau_radius_m()
    rays, _ = converging_bundle(rings=32, radius_m=radius)
    field, _ = reconstruct(rays, shape=(9, 9), pitch=(0.2e-6, 0.2e-6))

    oracle = focal_peak_oracle(radius_m=radius)
    assert oracle == pytest.approx(math.sqrt(3.0) * WAVELENGTH_M * FOCAL_M, rel=1e-9)
    assert float(abs(field.u[4, 4])) == pytest.approx(oracle, rel=2e-3)


@pytest.mark.slow
def test_the_focal_peak_converges_under_ray_refinement() -> None:
    """The scale is ray-density-independent: refining the fan does not move it.

    Measured ratios against `lambda R`: 1.000044 at 32 rings (3 169 rays),
    1.000012 at 64, 1.000004 at 128. Marked slow because 128 rings is 49 537 rays
    against the 81-pixel grid.
    """
    radius = plateau_radius_m()
    oracle = WAVELENGTH_M * FOCAL_M
    ratios = []
    for rings in (32, 64, 128):
        rays, _ = converging_bundle(rings=rings, radius_m=radius)
        field, _ = reconstruct(rays, shape=(9, 9), pitch=(0.2e-6, 0.2e-6))
        ratios.append(float(abs(field.u[4, 4])) / oracle)

    assert ratios == sorted(ratios, reverse=True), ratios
    assert ratios[-1] == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# 3. The four-item sign and orientation checklist, each with its negative twin
# ---------------------------------------------------------------------------


def test_item_1_a_wavelet_travelling_plus_z_gains_phase() -> None:
    """`exp(-i omega t)` with `+z` propagation: two surfaces differ by `exp(+i k dz)`."""
    step_m = 3.0e-6
    at_zero, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0), z_m=0.0
    )
    advanced, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0), z_m=step_m
    )
    before, _ = reconstruct(at_zero)
    after, _ = reconstruct(advanced)

    ratio = complex(after.u[0, 0] / before.u[0, 0])
    assert ratio == pytest.approx(np.exp(1j * at_zero.wavenumber * step_m), rel=1e-9)


def test_item_1_twin_a_negated_optical_path_conjugates_the_wavefront() -> None:
    """The twin. A flipped phasor sign is invisible in `|U|^2` and reverses every phase."""
    step_m = 3.0e-6
    at_zero, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0), z_m=0.0,
        optical_path_sign=-1.0,
    )
    advanced, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0), z_m=step_m,
        optical_path_sign=-1.0,
    )
    before, _ = reconstruct(at_zero)
    after, _ = reconstruct(advanced)

    ratio = complex(after.u[0, 0] / before.u[0, 0])
    expected = np.exp(1j * at_zero.wavenumber * step_m)
    assert ratio == pytest.approx(np.conjugate(expected), rel=1e-9)
    assert not np.isclose(ratio, expected)
    # ...and the intensity cannot tell, which is why the sign is checked in phase.
    assert float(abs(after.u[0, 0])) == pytest.approx(float(abs(before.u[0, 0])), rel=1e-12)


def test_item_2_a_tilt_about_y_makes_the_phase_increase_with_x() -> None:
    """`d_hat = (sin theta, 0, cos theta)`, theta > 0: phase rises along `+x`.

    Run on an axis-asymmetric grid, because the `(y, x)`-versus-`(x, y)`
    transposition this catches is invisible on a square one.
    """
    theta = 0.25
    rays, d_hat, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(math.sin(theta), 0.0, math.cos(theta))
    )
    field, _ = reconstruct(rays)
    y, x = field.coordinates()

    row = np.unwrap(np.angle(np.asarray(field.u)[SHAPE[0] // 2, :]))
    slope = np.polyfit(np.asarray(x), row, 1)[0]
    assert slope == pytest.approx(rays.wavenumber * d_hat[0], rel=1e-6)
    assert slope > 0.0

    column = np.unwrap(np.angle(np.asarray(field.u)[:, SHAPE[1] // 2]))
    assert abs(np.polyfit(np.asarray(y), column, 1)[0]) < 1e-3 * abs(slope)


def test_item_2_twin_the_same_tilt_about_x_moves_the_ramp_to_y() -> None:
    """The twin: transposing the direction transposes the ramp, and nothing else changes.

    A test that only checked `|slope|` would pass on both, which is exactly how a
    transposition survives.
    """
    theta = 0.25
    rays, d_hat, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, math.sin(theta), math.cos(theta))
    )
    field, _ = reconstruct(rays)
    y, x = field.coordinates()

    row = np.unwrap(np.angle(np.asarray(field.u)[SHAPE[0] // 2, :]))
    column = np.unwrap(np.angle(np.asarray(field.u)[:, SHAPE[1] // 2]))
    assert abs(np.polyfit(np.asarray(x), row, 1)[0]) < 1e-3 * rays.wavenumber * abs(d_hat[1])
    assert np.polyfit(np.asarray(y), column, 1)[0] == pytest.approx(
        rays.wavenumber * d_hat[1], rel=1e-6
    )


def test_item_3_the_projection_factor_is_visible_only_off_axis() -> None:
    """Checklist item 3, and its twin in one place: at normal incidence there is nothing to see.

    Off-axis, `SENSOR_OBLIQUITY` scales the whole field by `<n_hat, d_hat>`; on
    axis the two conventions are bit-identical, which is why running this check
    at normal incidence would prove nothing.
    """
    theta = 0.5
    oblique, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(math.sin(theta), 0.0, math.cos(theta))
    )
    exact, _ = reconstruct(oblique)
    sensor, diagnostics = reconstruct(oblique, projection=Projection.SENSOR_OBLIQUITY)
    assert diagnostics.min_projection_factor == pytest.approx(math.cos(theta), rel=1e-12)
    assert np.allclose(np.asarray(sensor.u), math.cos(theta) * np.asarray(exact.u), rtol=1e-12)

    axial, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    axial_exact, _ = reconstruct(axial)
    axial_sensor, _ = reconstruct(axial, projection=Projection.SENSOR_OBLIQUITY)
    assert np.array_equal(np.asarray(axial_exact.u), np.asarray(axial_sensor.u))


def test_item_4_a_unit_scale_error_is_not_a_small_error() -> None:
    """Checklist item 4, on a case with a large `k * OPL`.

    The converging bundle carries `k * R = 5.5e4` rad. Reading its millimetres as
    metres scales every length by 1000, so `k * OPL` scales with it and the focal
    peak moves by the square of the aperture scale -- 1e6, not a few percent.
    """
    grid, pitch = (9, 9), (0.2e-6, 0.2e-6)
    radius = plateau_radius_m()
    correct, _ = converging_bundle(rings=16, radius_m=radius)
    assert correct.wavenumber * FOCAL_M == pytest.approx(5.526e4, rel=1e-3)

    field, right = reconstruct(correct, shape=grid, pitch=pitch)
    mistaken_rays, _ = converging_bundle(rings=16, radius_m=radius, length_scale=1000.0)
    mistaken, wrong = reconstruct(mistaken_rays, shape=grid, pitch=pitch)

    # The directions are unchanged -- a scale error does not tilt a ray -- so the
    # same grid serves both, and the whole difference is in `k * OPL` and in the
    # area element. The area alone would give 1e6; the measured peak ratio is far
    # from it, because the misread geometry also spins the pupil phase a thousand
    # times faster and the coherent sum partly cancels.
    assert wrong.measure_sum / right.measure_sum == pytest.approx(1e6, rel=1e-6)
    ratio = float(abs(mistaken.u[4, 4])) / float(abs(field.u[4, 4]))
    assert 1e3 < ratio < 1e6, ratio  # measured 3.82e5 against 1e6 from area alone


# ---------------------------------------------------------------------------
# 4. The boundary: no solver, no backend
# ---------------------------------------------------------------------------


def test_the_coupler_imports_no_solver_and_no_backend() -> None:
    """Criterion 4, the AST half.

    `scripts/check_dependencies.py` already forbids `couplers -> backends` as an
    allowlist edge, and the walks in `tests/backends/test_optiland_boundary.py` and
    `test_chromatix_boundary.py` already read every module here for native names.
    This is the direct statement of the same rule against this package's own
    imports, kept local so the reason travels with the coupler: if the coupler
    core could reach an engine, a coupler defect could be misattributed to engine
    behaviour.
    """
    forbidden = {"optiland", "chromatix", "jax", "torch", "backends", "problems"}
    offenders = []
    for module in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            for name in sorted(names & forbidden):
                offenders.append(f"{module.relative_to(ROOT)}: {name}")
    assert offenders == [], "the coupler core imports a solver or a backend:\n  " + "\n  ".join(
        offenders
    )


def test_importing_the_coupler_loads_no_backend() -> None:
    """Criterion 4, the runtime half: the failure is transitive, so ask `sys.modules`."""
    probe = (
        "import sys; import couplers; "
        "loaded = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'optiland', 'chromatix', 'jax', 'torch'}); "
        "print(loaded)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip() == "[]", result.stdout


# ---------------------------------------------------------------------------
# 5. The record, and the classes that did not land
# ---------------------------------------------------------------------------


def test_the_wavelet_sum_registers_as_a_coupler() -> None:
    """It changes representation, not state: `ray_bundle -> scalar_field` on one
    surface.

    The descriptor used to be constructed here, inside a fixture that emptied the
    registry, because `couplers/` may not import `operations/` and there was no
    production registration site anywhere. CHE-221 (R03.4) put one *inside*
    `operations/`: the catalog names the implementation as a
    `"module.path:attribute"` string, so it needs no dependency edge in either
    direction, and the allowlist is unchanged. What is read below is the shipped
    record rather than a copy this file kept in step by hand.
    """
    descriptor = next(d for d in CATALOG if d.operation_id == "C_RAY_TO_SCALAR")
    assert descriptor.kind is OperationKind.COUPLER
    assert descriptor.kind is not OperationKind.PHYSICAL_OPERATOR
    assert descriptor.inputs == ("ray_bundle",)
    assert descriptor.primary_output == "scalar_field"
    assert descriptor.returns == ("scalar_field", "reconstruction_diagnostics")
    assert descriptor.returns_auxiliary is True
    assert descriptor.requires == ("grid_shape", "sample_pitch_m")
    assert descriptor.derivative == "forward_only"
    assert descriptor.capabilities is None
    assert resolve("C_RAY_TO_SCALAR") is ray_to_scalar


def test_the_avoided_coupler_classes_did_not_land() -> None:
    """A budget records what exists; only a test can record what was avoided."""
    source = "\n".join(
        module.read_text(encoding="utf-8") for module in sorted(PACKAGE.rglob("*.py"))
    )
    defined = {
        node.name
        for module in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    for avoided in (
        "RayToWaveCoupler",
        "CoherentHandoff",
        "DeclaredHandoffPlane",
        "Perturbation",
        "HandoffPerturbation",
        "StreamingReconstruction",
        "StreamingResult",
        "PositionalAngularSampler",
        "LaunchGeometry",
        "BandLimit",
        "ChunkWorkItem",
        "CurvatureBudget",
        "Coupler",
        "CouplerRunRequest",
        "CouplerRunResult",
        "GradientProblem",
        "DifferentiabilityReport",
    ):
        assert avoided not in defined, f"{avoided} landed in couplers/"
    assert "class " in source  # the walk read real source, not an empty package


def test_the_grid_nyquist_limit_is_the_condition_it_claims() -> None:
    """`lambda / (2 * pitch)`, and a bundle past it is refused rather than aliased."""
    assert grid_nyquist_direction_limit(0.5e-6, 0.25e-6) == pytest.approx(1.0)
    assert grid_nyquist_direction_limit(WAVELENGTH_M, 2.0e-6) == pytest.approx(0.1375)

    theta = 0.5  # sin = 0.479, well past lambda / (2 * pitch) = 0.1375
    rays, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=(2.0e-6, 2.0e-6),
        direction=(math.sin(theta), 0.0, math.cos(theta)),
    )
    with pytest.raises(ContractError, match="steepest wavelet ramp") as raised:
        reconstruct(rays, pitch=(2.0e-6, 2.0e-6))
    assert raised.value.code == "SHAPE_MISMATCH"


def test_the_diagnostics_record_is_json_shaped() -> None:
    """`as_dict` is what a record or a report writes; it must contain no array."""
    rays, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    _, diagnostics = reconstruct(rays)
    assert isinstance(diagnostics, ReconstructionDiagnostics)
    record = diagnostics.as_dict()
    assert record["grid_shape"] == list(SHAPE)
    assert record["projection"] == "asm_consistent"
    for key, value in record.items():
        assert isinstance(value, (int, float, str, bool, list, dict, type(None))), key
    # The per-axis pair, reported as the pair it is enforced as.
    assert record["max_transverse_direction"] == [0.0, 0.0]
    assert record["grid_nyquist_direction_limit"] == pytest.approx(
        [WAVELENGTH_M / (2 * PITCH_M[0]), WAVELENGTH_M / (2 * PITCH_M[1])]
    )


def test_no_power_this_coupler_reports_is_a_watt() -> None:
    """Criterion 7. The scale is relative and the record has to say so, in words.

    There is no `1/(i lambda z)` Kirchhoff prefactor and no declared `A_0`, so
    every power here is `i lambda z` times an SI one -- about eighteen orders of
    magnitude out. The failure mode this guards is a downstream report that
    prints one of these numbers with a unit.
    """
    rays, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    _, diagnostics = reconstruct(rays)
    assert diagnostics.scale is SCALE_NOTE
    assert diagnostics.scale.startswith("relative")
    text = " ".join(str(value) for value in diagnostics.as_dict().values()).lower()
    for unit in ("watt", " w)", "joule"):
        assert unit not in text, unit


# ---------------------------------------------------------------------------
# 6. What the coupler refuses
# ---------------------------------------------------------------------------


def test_a_bundle_with_an_undeclared_measure_is_refused() -> None:
    """The contract R07.3 owns, landed with the kernel that would otherwise guess.

    `measure_kind` defaults to `'undeclared'` in `representations/rays.py`
    precisely so that refusing is what happens when nobody thought about it.
    Treating the weight as uniform here would invent a quadrature, and the
    invented one differs from the true one by the aperture area.
    """
    rays, _, _ = collimated_bundle(
        shape=(4, 5), sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    undeclared = dataclasses.replace(rays, measure_weight=None, measure_kind="undeclared")
    with pytest.raises(ContractError) as raised:
        reconstruct(undeclared, shape=(4, 5))
    assert raised.value.code == "MEASURE_UNDECLARED"
    assert "invent" in str(raised.value)


def test_a_bundle_without_coherent_state_is_refused_by_the_representation() -> None:
    """A real intensity weight is not a complex amplitude, and this coupler will not decide."""
    rays, _, _ = collimated_bundle(
        shape=(4, 5), sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    with pytest.raises(ContractError) as raised:
        reconstruct(dataclasses.replace(rays, amplitude=None), shape=(4, 5))
    assert raised.value.code == "COHERENT_STATE_INCOMPLETE"

    with pytest.raises(ContractError) as unverified:
        reconstruct(
            dataclasses.replace(rays, optical_path_reference=UNVERIFIED), shape=(4, 5)
        )
    assert unverified.value.code == "OPL_REFERENCE_UNVERIFIED"


def test_a_surface_the_bundle_is_not_on_is_refused_rather_than_relabelled() -> None:
    """`surface` is an expectation, not an override: this coupler does not propagate."""
    rays, _, _ = collimated_bundle(
        shape=(4, 5), sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    with pytest.raises(ContractError) as raised:
        reconstruct(
            rays,
            shape=(4, 5),
            surface=ReferenceSurface(name="sensor", z_m=1.0e-3, medium_index=1.0),
        )
    assert raised.value.code == "FRAME_MISMATCH"
    # ...and the surface it *is* on is accepted, so the check is not vacuous.
    field, _ = reconstruct(rays, shape=(4, 5), surface=rays.reference_surface)
    assert field.reference_surface == rays.reference_surface


def test_a_tilted_reference_surface_is_refused() -> None:
    """The ramp is purely transverse, so a tilted plane loses a `d_z` term silently."""
    rays, _, _ = collimated_bundle(
        shape=(4, 5), sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    tilted = ReferenceSurface(
        name="tilted", z_m=0.0, medium_index=1.0,
        normal=(0.0, math.sin(0.1), math.cos(0.1)),
    )
    with pytest.raises(ContractError) as raised:
        reconstruct(dataclasses.replace(rays, reference_surface=tilted), shape=(4, 5))
    assert raised.value.code == "FRAME_MISMATCH"
    assert "transverse" in str(raised.value)


def test_rays_that_are_not_on_the_declared_plane_are_refused() -> None:
    """A ray off the plane contributes as though it were on it, losing `k d_z dz`."""
    rays, _, _ = collimated_bundle(
        shape=(4, 5), sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    positions = np.asarray(rays.positions_m).copy()
    positions[3, 2] += 1.0e-7
    with pytest.raises(ContractError) as raised:
        reconstruct(dataclasses.replace(rays, positions_m=positions), shape=(4, 5))
    assert raised.value.code == "FRAME_MISMATCH"
    assert "axial deviation" in str(raised.value)
    # Round-off on the plane's own coordinate is not a deviation.
    nudged = np.asarray(rays.positions_m).copy()
    nudged[3, 2] += 5.0e-17
    reconstruct(dataclasses.replace(rays, positions_m=nudged), shape=(4, 5))


def test_a_non_positive_grid_is_refused() -> None:
    rays, _, _ = collimated_bundle(
        shape=(4, 5), sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    with pytest.raises(ContractError) as raised:
        reconstruct(rays, shape=(0, 5))
    assert raised.value.code == "SHAPE_MISMATCH"


def test_a_float32_bundle_computes_in_float32_and_says_so() -> None:
    """The compute precision is derived from the data and floored, never assumed.

    `numerics.PHASE_ACCUMULATION_FLOOR` is FP32, so a float32 bundle computes in
    float32 and returns complex64 on the same device -- and the record reports
    the precision it *computed* in beside the dtype it actually produced, which
    are two different facts under JAX without x64.

    The tolerance is the honest one for the dtype: `k * OPL` here is 2.9 rad and
    float32 carries it to about 1e-6 relative, so a float64 gate would fail for
    reasons that have nothing to do with the kernel.
    """
    theta = 0.05
    rays, d_hat, area = collimated_bundle(
        shape=SHAPE,
        sample_pitch_m=PITCH_M,
        direction=(math.sin(theta), 0.0, math.cos(theta)),
        dtype=np.float32,
    )
    assert str(rays.state.dtype) == "float32"

    field, diagnostics = reconstruct(rays)
    assert diagnostics.compute_precision == "fp32"
    assert diagnostics.output_state["dtype"] == "complex64"

    y, x = field.coordinates()
    grid_y, grid_x = np.meshgrid(np.asarray(y), np.asarray(x), indexing="ij")
    oracle = rays.count * area * np.exp(
        1j * rays.wavenumber * (d_hat[0] * grid_x + d_hat[1] * grid_y)
    )
    assert peak_relative_residual(np.asarray(field.u), oracle) < 1e-5
