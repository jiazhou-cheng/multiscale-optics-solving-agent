"""CHE-24 — C_RAY_TO_WAVE verified against oracles independent of it.

Three levels of evidence, in increasing order of how much they could have
caught:

1. An **exact analytic** oracle. A collimated bundle launched from many lateral
   positions must reconstruct the plane wave ``exp(+i k d.r)`` exactly, because
   SI Figure S1c says each ray's OPL compensates its launch position. The
   tolerance is derived from float64 round-off, not chosen.
2. An **independent implementation**. Advancing the rays geometrically and
   reconstructing at the new plane must agree with propagating the
   reconstructed field through the M1-verified Chromatix angular spectrum.
   Chromatix is imported by this test, never by the coupler core.
3. **Negative controls**. Each term of main-text eq 2 is removed in turn, using
   the shipping implementation rather than a parallel copy, and each removal
   must be detected.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from multiscale_optics_agent.couplers import ComplexField, ContractError, RayBundle, ReferencePlane
from multiscale_optics_agent.couplers.contracts import Frame
from multiscale_optics_agent.couplers.ray_to_wave import (
    Perturbation,
    Projection,
    collimated_bundle,
    grid_nyquist_direction_limit,
    ray_to_wave,
)

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = (
    "src/multiscale_optics_agent/couplers/contracts.py",
    "src/multiscale_optics_agent/couplers/ray_to_wave.py",
)
WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
GRID = (32, 32)

FLOAT64_EPS = np.finfo(np.float64).eps


def _roundoff_bound(bundle, shape=GRID, pitch=(PITCH_M, PITCH_M)) -> float:
    """Worst-case float64 round-off for this bundle's coherent sum.

    The argument actually evaluated per ray per output pixel is
    ``k * (OPL_i + d_t . (r - r0_i))``, so its representable resolution is
    ``eps`` times the magnitude of that whole argument -- not of the OPL alone.
    The ramp term dominates whenever the output grid is wide, which is exactly
    the case a bound derived only from the OPL would under-estimate.

    Summing N wavelets, the worst case is that the per-ray errors add
    coherently: ``N * eps * max|argument|``. Observed errors sit well inside
    this because they do not in fact add coherently, but a bound that assumed
    cancellation would not be a bound.
    """
    ny, nx = shape
    dy, dx = pitch
    half_extent = math.hypot((ny // 2) * dy, (nx // 2) * dx)
    transverse = np.linalg.norm(bundle.directions[:, :2], axis=1)

    opl_phase = np.abs(bundle.optical_path_length_m)
    offset_phase = np.abs(np.sum(bundle.directions[:, :2] * bundle.positions_m[:, :2], axis=1))
    ramp_phase = transverse * half_extent
    max_argument = bundle.wavenumber * float(np.max(opl_phase + offset_phase + ramp_phase))

    return bundle.count * FLOAT64_EPS * max(max_argument, 1.0)


def _grid_positions(n: int, pitch: float) -> np.ndarray:
    coords = (np.arange(n, dtype=np.float64) - n // 2) * pitch
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()])


def _plane_wave(direction, wavelength_m: float, shape, pitch, plane_z_m: float = 0.0):
    ny, nx = shape
    dy, dx = pitch
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * dy
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * dx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    k = 2.0 * math.pi / wavelength_m
    dxc, dyc, dzc = direction
    return np.exp(1j * k * (dxc * xx + dyc * yy + dzc * plane_z_m))


# --- The core must not import an engine --------------------------------------


@pytest.mark.parametrize("module", CORE_MODULES)
def test_coupler_core_imports_no_solver_engine(module: str) -> None:
    """Static half of the rule frozen in coupler_protocol.yaml. If the core
    could import an engine, a coupler defect could be misattributed to engine
    behaviour and M1's independence evidence would stop bounding the search."""
    tree = ast.parse((ROOT / module).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"optiland", "chromatix"}, sorted(imported)


def test_coupler_core_loads_no_engine_at_runtime() -> None:
    """Dynamic half: importing and running the core must not pull an engine in."""
    import subprocess
    import sys

    script = (
        "import sys, numpy as np\n"
        "from multiscale_optics_agent.couplers.ray_to_wave import "
        "collimated_bundle, ray_to_wave\n"
        "b = collimated_bundle(positions_xy_m=np.zeros((4, 2)), direction=(0.0, 0.0, 1.0),"
        " wavelength_m=5e-7)\n"
        "ray_to_wave(b, grid_shape=(8, 8), sample_pitch_m=(1e-6, 1e-6))\n"
        "loaded = [m for m in sys.modules if m.split('.')[0] in {'optiland', 'chromatix'}]\n"
        "print(loaded)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "[]", completed.stdout


# --- Oracle 1: exact analytic plane wave --------------------------------------


@pytest.mark.parametrize("theta_x", [0.0, 0.05, 0.2, -0.24])
def test_collimated_bundle_reconstructs_the_exact_plane_wave(theta_x: float) -> None:
    """SI Figure S1c. Rays share a direction but not a launch point; with the
    OPL each position implies, every ray contributes the SAME plane wave, so the
    sum is N * exp(i k d.r) with no residual position dependence."""
    direction = (math.sin(theta_x), 0.0, math.cos(theta_x))
    positions = _grid_positions(8, 4 * PITCH_M)
    bundle = collimated_bundle(
        positions_xy_m=positions, direction=direction, wavelength_m=WAVELENGTH_M
    )

    field, diagnostics = ray_to_wave(
        bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M)
    )

    expected = bundle.count * _plane_wave(direction, WAVELENGTH_M, GRID, (PITCH_M, PITCH_M))
    error = float(np.max(np.abs(field.u - expected)))
    bound = _roundoff_bound(bundle)
    assert error <= bound, f"error {error:.3e} exceeds round-off bound {bound:.3e}"

    # cos(theta) is reported even though the coupler default does not apply it,
    # so a caller can see what the sensor convention would have done.
    assert diagnostics.max_projection_factor == pytest.approx(math.cos(theta_x))
    assert diagnostics.projection == "asm_consistent"
    assert diagnostics.perturbation == "none"


def test_reconstruction_is_independent_of_where_the_rays_are_launched() -> None:
    """The whole content of the dr(x,y) term. Two ensembles of the same mode,
    launched from completely different lateral positions, must agree."""
    direction = (math.sin(0.15), math.sin(0.07), 0.0)
    direction = (*direction[:2], math.sqrt(1.0 - direction[0] ** 2 - direction[1] ** 2))
    rng = np.random.default_rng(20260812)

    dense = collimated_bundle(
        positions_xy_m=_grid_positions(6, 3 * PITCH_M),
        direction=direction,
        wavelength_m=WAVELENGTH_M,
    )
    scattered = collimated_bundle(
        positions_xy_m=rng.uniform(-2e-5, 2e-5, size=(dense.count, 2)),
        direction=direction,
        wavelength_m=WAVELENGTH_M,
    )

    dense_field, _ = ray_to_wave(dense, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))
    scattered_field, _ = ray_to_wave(
        scattered, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M)
    )

    error = float(np.max(np.abs(dense_field.u - scattered_field.u)))
    assert error <= _roundoff_bound(dense)


def test_superposed_modes_add_coherently() -> None:
    """Linearity: reconstructing two modes together equals summing their
    separate reconstructions. Catches any per-call normalization that depends
    on ray count when it should not."""
    positions = _grid_positions(4, 5 * PITCH_M)
    first = collimated_bundle(
        positions_xy_m=positions, direction=(0.1, 0.0, math.sqrt(1 - 0.01)),
        wavelength_m=WAVELENGTH_M,
    )
    second = collimated_bundle(
        positions_xy_m=positions, direction=(0.0, -0.08, math.sqrt(1 - 0.0064)),
        wavelength_m=WAVELENGTH_M,
    )
    combined = RayBundle(
        positions_m=np.vstack([first.positions_m, second.positions_m]),
        directions=np.vstack([first.directions, second.directions]),
        wavelength_m=WAVELENGTH_M,
        reference_plane=first.reference_plane,
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=np.concatenate([first.amplitude, second.amplitude]),
        optical_path_length_m=np.concatenate(
            [first.optical_path_length_m, second.optical_path_length_m]
        ),
        optical_path_length_reference=first.optical_path_length_reference,
    )

    a, _ = ray_to_wave(first, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))
    b, _ = ray_to_wave(second, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))
    both, _ = ray_to_wave(combined, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))

    error = float(np.max(np.abs(both.u - (a.u + b.u))))
    assert error <= _roundoff_bound(combined)


# --- Oracle 2: independent implementation (Chromatix ASM) ---------------------


def _fft_bin_direction(mode_x: int, mode_y: int, n: int, pitch: float, wavelength: float):
    """Direction of an exact FFT-bin plane-wave mode of an n x n grid.

    M1 Case 1 validated Chromatix's angular spectrum against exactly these
    modes to float32 round-off, so using them makes Chromatix an exact oracle
    rather than an approximate one.
    """
    kx = mode_x / (n * pitch)
    ky = mode_y / (n * pitch)
    dx = kx * wavelength
    dy = ky * wavelength
    dz = math.sqrt(1.0 - dx * dx - dy * dy)
    return (dx, dy, dz)


def test_geometric_advance_agrees_with_chromatix_angular_spectrum() -> None:
    """Independent-implementation oracle.

    Two routes to the field at ``z = Z``:
      A. advance the rays geometrically by Z and reconstruct there;
      B. reconstruct at z = 0 and propagate the field with Chromatix ASM.

    Route A exercises the coupler's OPL and ramp handling; route B is the
    M1-verified engine. They are independent, so agreement is evidence.
    Chromatix is imported here, in the driver -- never in the coupler core.
    """
    pytest.importorskip("chromatix")
    import chromatix.functional as cf
    import jax
    import jax.numpy as jnp

    n = 32
    pitch = 1.0e-6
    distance_m = 40e-6
    modes = [(0, 0), (2, 0), (0, 3), (-3, 2)]

    positions = _grid_positions(6, 3 * pitch)
    bundles = [
        collimated_bundle(
            positions_xy_m=positions,
            direction=_fft_bin_direction(mx, my, n, pitch, WAVELENGTH_M),
            wavelength_m=WAVELENGTH_M,
        )
        for mx, my in modes
    ]
    merged = RayBundle(
        positions_m=np.vstack([b.positions_m for b in bundles]),
        directions=np.vstack([b.directions for b in bundles]),
        wavelength_m=WAVELENGTH_M,
        reference_plane=ReferencePlane(name="z=0", z_m=0.0),
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=np.concatenate([b.amplitude for b in bundles]),
        optical_path_length_m=np.concatenate([b.optical_path_length_m for b in bundles]),
        optical_path_length_reference=bundles[0].optical_path_length_reference,
    )

    at_zero, _ = ray_to_wave(merged, grid_shape=(n, n), sample_pitch_m=(pitch, pitch))

    # Route A: advance every ray geometrically to z = Z. Path travelled is
    # Z / d_z along the ray, which adds exactly that much OPL.
    step = distance_m / merged.directions[:, 2]
    advanced = RayBundle(
        positions_m=merged.positions_m + merged.directions * step[:, None],
        directions=merged.directions,
        wavelength_m=WAVELENGTH_M,
        reference_plane=ReferencePlane(name="z=Z", z_m=distance_m),
        frame=Frame(axis_order="flat per-ray arrays"),
        amplitude=merged.amplitude,
        optical_path_length_m=merged.optical_path_length_m + step,
        optical_path_length_reference=merged.optical_path_length_reference,
    )
    route_a, _ = ray_to_wave(advanced, grid_shape=(n, n), sample_pitch_m=(pitch, pitch))

    # Route B: propagate the z = 0 reconstruction with the M1-verified engine.
    # Unpadded, because these modes are periodic on the grid -- zero-padding
    # would manufacture an aperture edge the physics does not contain, exactly
    # as M1's L1-WAVE-01 Case 1 recorded.
    jax.config.update("jax_enable_x64", False)
    field_in = cf.Field.build(
        jnp.asarray(at_zero.u, dtype=jnp.complex64),
        jnp.asarray([[pitch, pitch]]),
        WAVELENGTH_M,
    )
    propagated = cf.asm_propagate(field_in, z=distance_m, n=1.0, pad_width=0)
    route_b = np.asarray(jax.device_get(propagated.u)).reshape(n, n)

    scale = float(np.max(np.abs(route_b)))
    residual = float(np.max(np.abs(route_a.u - route_b))) / scale

    # The floor here is Chromatix's complex64, not the coupler's float64. Reuse
    # the bound M1 derived for exactly this situation in L1-WAVE-01 Case 1:
    # float32 phase round-off of 5 * eps32 per radian of accumulated phase
    # (M1 recorded 7.04e-05 at 118 rad and 1.76e-03 at 2952 rad, both 5.96e-07
    # per radian). Derived, not chosen.
    accumulated_phase_rad = 2.0 * math.pi / WAVELENGTH_M * distance_m
    bound = accumulated_phase_rad * 5.0 * float(np.finfo(np.float32).eps)
    assert residual < bound, (
        f"routes disagree by {residual:.3e}, above the float32 round-off bound {bound:.3e} "
        f"for {accumulated_phase_rad:.1f} rad of accumulated phase"
    )
    # Sanity on the attribution: a residual below one eps32 per radian cannot be
    # a physics disagreement, only accumulated rounding.
    assert residual / accumulated_phase_rad < float(np.finfo(np.float32).eps)


# --- Oracle 3: power accounting ------------------------------------------------


def test_power_is_reported_and_scales_as_the_square_of_ray_count() -> None:
    """N identical in-phase wavelets give N times the amplitude, so N^2 times
    the power. Reported rather than gated: it is bookkeeping about a chosen
    ensemble, not a conservation law about a physical aperture."""
    direction = (0.0, 0.0, 1.0)
    powers = []
    for count in (1, 4, 16):
        bundle = collimated_bundle(
            positions_xy_m=np.zeros((count, 2)),
            direction=direction,
            wavelength_m=WAVELENGTH_M,
        )
        _, diagnostics = ray_to_wave(
            bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M)
        )
        powers.append(diagnostics.reconstructed_discrete_power)

    assert powers[1] / powers[0] == pytest.approx(16.0, rel=1e-12)
    assert powers[2] / powers[0] == pytest.approx(256.0, rel=1e-12)


def test_one_over_n_normalization_is_applied_only_when_asked() -> None:
    bundle = collimated_bundle(
        positions_xy_m=_grid_positions(4, PITCH_M),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=WAVELENGTH_M,
    )
    plain, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))
    averaged, diagnostics = ray_to_wave(
        bundle,
        grid_shape=GRID,
        sample_pitch_m=(PITCH_M, PITCH_M),
        normalization="one_over_n",
    )

    np.testing.assert_allclose(averaged.u, plain.u / bundle.count, rtol=0, atol=0)
    assert diagnostics.normalization == "one_over_n"
    assert "one_over_n" in averaged.normalization


# --- Negative controls ---------------------------------------------------------


def _reference_and_perturbed(
    perturbation: Perturbation,
    *,
    theta_x: float = 0.25,
    projection: Projection = Projection.ASM_CONSISTENT,
):
    """Same bundle, same code path, one term removed."""
    direction = (math.sin(theta_x), 0.0, math.cos(theta_x))
    bundle = collimated_bundle(
        positions_xy_m=_grid_positions(6, 3 * PITCH_M),
        direction=direction,
        wavelength_m=WAVELENGTH_M,
    )
    kwargs = {
        "grid_shape": GRID,
        "sample_pitch_m": (PITCH_M, PITCH_M),
        "projection": projection,
    }
    control, _ = ray_to_wave(bundle, **kwargs)
    perturbed, _ = ray_to_wave(bundle, **kwargs, perturbation=perturbation)

    obliquity = math.cos(theta_x) if projection is Projection.SENSOR_OBLIQUITY else 1.0
    expected = (
        bundle.count
        * obliquity
        * _plane_wave(direction, WAVELENGTH_M, GRID, (PITCH_M, PITCH_M))
    )
    return control, perturbed, expected, bundle


@pytest.mark.parametrize(
    ("perturbation", "projection"),
    [
        (Perturbation(phase_sign=-1), Projection.ASM_CONSISTENT),
        (Perturbation(apply_oblique_ramp=False), Projection.ASM_CONSISTENT),
        (Perturbation(transpose_axes=True), Projection.ASM_CONSISTENT),
        # The projection factor only exists under the sensor convention, so its
        # negative control has to be run there.
        (Perturbation(apply_projection_factor=False), Projection.SENSOR_OBLIQUITY),
    ],
    ids=["phase_sign", "oblique_ramp", "axis_transpose", "projection_factor"],
)
def test_each_removed_term_is_detected(
    perturbation: Perturbation, projection: Projection
) -> None:
    control, perturbed, expected, bundle = _reference_and_perturbed(
        perturbation, projection=projection
    )

    # The unperturbed control passes...
    assert float(np.max(np.abs(control.u - expected))) <= _roundoff_bound(bundle)
    # ...and the perturbation is caught by the same comparison.
    residual = float(np.max(np.abs(perturbed.u - expected))) / bundle.count
    assert residual > 1e-3, f"{perturbation.describe()} was not detected"


def test_projection_factor_omission_is_invisible_at_normal_incidence() -> None:
    """Why the negative test must be run off-axis. At theta = 0 the factor is
    1, so omitting it changes nothing -- an on-axis smoke test would pass and
    every off-axis result would be wrong."""
    on_axis_control, on_axis_perturbed, _, _ = _reference_and_perturbed(
        Perturbation(apply_projection_factor=False),
        theta_x=0.0,
        projection=Projection.SENSOR_OBLIQUITY,
    )
    np.testing.assert_allclose(on_axis_perturbed.u, on_axis_control.u, rtol=0, atol=0)


def test_the_two_projection_conventions_differ_by_exactly_the_obliquity_factor() -> None:
    """CHE-25's finding, pinned.

    Main-text eq 2 carries <n,d>; SI eq S5, which derives the same sum as an
    estimator of the angular-spectrum integral, does not. They are different
    operators. Only the factor-free form preserves a field -- proven in
    tests/test_wave_to_ray.py by enumerating every propagating mode -- so it is
    the coupler default, and the eq-2 form is kept as an explicitly named
    sensor model.
    """
    theta = 0.25
    direction = (math.sin(theta), 0.0, math.cos(theta))
    bundle = collimated_bundle(
        positions_xy_m=_grid_positions(4, 3 * PITCH_M),
        direction=direction,
        wavelength_m=WAVELENGTH_M,
    )
    kwargs = {"grid_shape": GRID, "sample_pitch_m": (PITCH_M, PITCH_M)}

    asm, asm_diagnostics = ray_to_wave(bundle, **kwargs, projection=Projection.ASM_CONSISTENT)
    sensor, sensor_diagnostics = ray_to_wave(
        bundle, **kwargs, projection=Projection.SENSOR_OBLIQUITY
    )

    np.testing.assert_allclose(sensor.u, asm.u * math.cos(theta), rtol=1e-14)
    assert asm_diagnostics.projection == "asm_consistent"
    assert sensor_diagnostics.projection == "sensor_obliquity"
    assert "S5" in asm.provenance["equation"]
    assert "eq 2" in sensor.provenance["equation"]

    # The default is the field-preserving one; choosing silently would have cost
    # cos(theta) off-axis for a reason no test would have named.
    default, _ = ray_to_wave(bundle, **kwargs)
    np.testing.assert_array_equal(default.u, asm.u)


def test_oblique_ramp_omission_is_invisible_for_a_single_centred_ray() -> None:
    """The same trap for dr(x,y): one on-axis ray at the origin cannot detect
    a missing ramp, because there is no lateral offset and no tilt."""
    bundle = collimated_bundle(
        positions_xy_m=np.zeros((1, 2)),
        direction=(0.0, 0.0, 1.0),
        wavelength_m=WAVELENGTH_M,
    )
    control, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))
    perturbed, _ = ray_to_wave(
        bundle,
        grid_shape=GRID,
        sample_pitch_m=(PITCH_M, PITCH_M),
        perturbation=Perturbation(apply_oblique_ramp=False),
    )
    np.testing.assert_allclose(perturbed.u, control.u, rtol=0, atol=1e-18)


def test_a_millimetre_for_metre_pitch_error_is_caught_by_the_grid_condition() -> None:
    """Using millimetres where metres are meant makes the pitch 1000x too
    coarse, so the grid can no longer represent the ramp."""
    bundle = collimated_bundle(
        positions_xy_m=_grid_positions(4, PITCH_M),
        direction=(math.sin(0.2), 0.0, math.cos(0.2)),
        wavelength_m=WAVELENGTH_M,
    )
    with pytest.raises(ContractError, match="steepest wavelet ramp"):
        ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M * 1000, PITCH_M * 1000))


# --- Validity conditions --------------------------------------------------------


def test_grid_nyquist_limit_matches_lambda_over_two_pitch() -> None:
    assert grid_nyquist_direction_limit(500e-9, 1e-6) == pytest.approx(0.25)
    # A grid at exactly half a wavelength can represent any propagating mode.
    assert grid_nyquist_direction_limit(500e-9, 250e-9) == pytest.approx(1.0)


def test_a_steep_angle_is_a_grid_limitation_not_a_physics_limitation() -> None:
    """theta = 0.3 rad exceeds lambda/(2*pitch) at a 1 um pitch and is refused.
    The same bundle on a finer grid reconstructs exactly. Recorded because the
    natural misreading of that refusal is that the coupler cannot handle steep
    angles, when what it cannot do is write them onto a coarse grid."""
    theta = 0.3
    direction = (math.sin(theta), 0.0, math.cos(theta))
    bundle = collimated_bundle(
        positions_xy_m=_grid_positions(4, PITCH_M),
        direction=direction,
        wavelength_m=WAVELENGTH_M,
    )
    with pytest.raises(ContractError, match="steepest wavelet ramp"):
        ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))

    fine = PITCH_M / 4
    field, diagnostics = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=(fine, fine))
    expected = bundle.count * _plane_wave(direction, WAVELENGTH_M, GRID, (fine, fine))
    assert float(np.max(np.abs(field.u - expected))) <= _roundoff_bound(
        bundle, pitch=(fine, fine)
    )
    assert diagnostics.grid_nyquist_satisfied is True


def test_grid_condition_can_be_measured_without_being_enforced() -> None:
    """A caller studying the aliasing regime needs to reach it. The condition
    is still reported, so a result produced there is never mistaken for a valid
    one."""
    bundle = collimated_bundle(
        positions_xy_m=_grid_positions(4, PITCH_M),
        direction=(math.sin(0.6), 0.0, math.cos(0.6)),
        wavelength_m=WAVELENGTH_M,
    )
    _, diagnostics = ray_to_wave(
        bundle,
        grid_shape=GRID,
        sample_pitch_m=(PITCH_M, PITCH_M),
        enforce_grid_nyquist=False,
    )
    assert diagnostics.grid_nyquist_satisfied is False
    assert diagnostics.max_transverse_direction > diagnostics.grid_nyquist_direction_limit


def test_ray_density_diagnostic_separates_the_two_sampling_conditions() -> None:
    """Grid Nyquist and ray density are independent, and refining one does not
    fix the other. Both are reported so the failing one is identifiable."""
    positions = _grid_positions(8, 2 * PITCH_M)
    bundle = collimated_bundle(
        positions_xy_m=positions, direction=(0.1, 0.0, math.sqrt(1 - 0.01)),
        wavelength_m=WAVELENGTH_M,
    )
    _, diagnostics = ray_to_wave(
        bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M)
    )
    # A single collimated mode has no direction spread, so neighbouring ramps
    # never disagree: the wavelet approximation is exact here.
    assert diagnostics.ray_density_status == "wavelet_approximation_holds"
    assert diagnostics.max_adjacent_ray_phase_rad == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.ray_spacing_estimate_m == pytest.approx(2 * PITCH_M, rel=1e-9)


def test_ray_density_diagnostic_reports_not_computed_rather_than_guessing() -> None:
    """Above the O(N^2) scan limit the diagnostic declines to answer. A
    fabricated estimate of a sampling condition is worse than an absent one."""
    from multiscale_optics_agent.couplers import ray_to_wave as module

    positions = _grid_positions(72, PITCH_M)  # 5184 rays > the 4096 scan limit
    assert positions.shape[0] > module._NEAREST_NEIGHBOUR_SCAN_LIMIT
    bundle = collimated_bundle(
        positions_xy_m=positions, direction=(0.0, 0.0, 1.0), wavelength_m=WAVELENGTH_M
    )
    _, diagnostics = ray_to_wave(bundle, grid_shape=(8, 8), sample_pitch_m=(PITCH_M, PITCH_M))
    assert diagnostics.ray_density_status == "not_computed_above_scan_limit"
    assert diagnostics.max_adjacent_ray_phase_rad is None


# --- Refusals inherited from the contract layer ----------------------------------


def test_a_bundle_carrying_only_an_optiland_weight_is_refused() -> None:
    bundle = RayBundle(
        positions_m=np.zeros((3, 3)),
        directions=np.tile([0.0, 0.0, 1.0], (3, 1)),
        wavelength_m=WAVELENGTH_M,
        reference_plane=ReferencePlane(name="p", z_m=0.0),
        frame=Frame(axis_order="flat per-ray arrays"),
        weight=np.ones(3),
        weight_semantics="RealRays.i is a per-ray intensity, not a complex amplitude",
    )
    with pytest.raises(ContractError, match="AMPLITUDE_IS_A_WEIGHT"):
        ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))


def test_the_reconstructed_field_declares_its_own_provenance() -> None:
    bundle = collimated_bundle(
        positions_xy_m=np.zeros((2, 2)), direction=(0.0, 0.0, 1.0), wavelength_m=WAVELENGTH_M
    )
    field, _ = ray_to_wave(bundle, grid_shape=GRID, sample_pitch_m=(PITCH_M, PITCH_M))

    assert isinstance(field, ComplexField)
    assert field.provenance["coupler"] == "C_RAY_TO_WAVE"
    assert "eq S5" in field.provenance["equation"]
    assert field.provenance["projection"] == "asm_consistent"
    assert field.provenance["optical_path_length_reference"]
