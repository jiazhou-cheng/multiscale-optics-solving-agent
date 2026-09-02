"""The parity gate: reproduce the frozen ray numbers, or explain the difference.

CHE-182 (R05.4). What must be established, and where each part is:

1. ray positions and directions on the fixture systems, against the frozen
   records -- section 1;
4. quadrature weight sum -> `pi a^2` under ring refinement -- section 2;
5. clipping behaviour and alive-ray counts -- section 3;
6. analytic ray invariants, not only record agreement -- section 4.

(2 and 3, the declared OPL and its falsifiable twin, are
`test_optiland_opl_convention.py`.)

Where the numbers come from, and why a record is the weaker oracle
-----------------------------------------------------------------
`RECORD` below is transcribed from
`pre-rewrite-2026-08-30:benchmarks/probes/records/optiland/exit_pupil_handoff.json`,
which is the frozen output of the deleted adapter at `num_rings = 16`, on axis, at
550 nm, host float64. It is evidence of what the code once did.

That is deliberately not the only gate here. A record can be wrong -- several
frozen records in this project were produced before a convention change and
carried a scale differing by `dA^2` -- so section 4 checks *analytic* invariants
that hold whatever any record says: the traced effective focal length against
`R / (n - 1)`, the paraxial exit pupil against the thin-lens construction, the
transverse magnification of a collimated bundle, and the direction norms. Where
the two disagree the analytic oracle wins.

Tolerances, and the oracle that set each one
--------------------------------------------
* `abs=0.0` on every record comparison. These are the same arithmetic on the same
  host in the same precision, so agreement is exact or the gate has found
  something. It reproduced exactly, and it is recorded as `0.0` rather than as a
  small number so that a future drift is visible instead of absorbed.
* the direction-norm bound is `representations.direction_norm_tolerance(dtype)`,
  which is `max(1e-9, 64 * eps)` -- the reference implementation's legacy float64
  allowance as a floor, widened by dtype so a float32 trace is held to float32
  round-off rather than failed for arithmetic it never claimed to do.
* `EFL_RELATIVE = 2e-5` on the traced focal length of the *innermost* ray. The
  oracle is analytic -- `R / (n - 1)` for a thin plano-convex singlet -- and the
  residual is the singlet's *physical* departure from the thin-lens formula at
  0.2 mm of centre thickness, not a numerical error. Measured 1.13e-5 here; the
  reference implementation recorded 1.81e-5 for its own innermost-ray
  construction, which sampled a different ring.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fixtures.systems import (
    REVERSE_TELEPHOTO,
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM,
    SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
    singlet_ref,
    singlet_source,
)

from problems.ray_trace import OpticalSetup, SourceSpec, SurfaceSpec
from representations import ContractError, direction_norm_tolerance
from solvers.optiland import trace
from solvers.optiland.launch import launch
from solvers.optiland.rays import (
    NATIVE_LENGTH_M,
    hexapolar_area_weight_m2,
    hexapolar_ray_count,
    hexapolar_ring_index,
    to_ray_bundle,
)
from solvers.optiland.system import build_lens

WAVELENGTH_UM = 0.55
NUM_RINGS = 16
EFL_RELATIVE = 2.0e-5

#: The illumination the frozen records were taken under: on axis, one wavelength,
#: object at infinity. CHE-219 needs it explicitly because the launch is now
#: declared before the trace rather than reconstructed from it.
ON_AXIS = SourceSpec(wavelength_um=WAVELENGTH_UM)

#: Transcribed from `exit_pupil_handoff.json`: 16 rings, on axis, 550 nm, host
#: float64. Every value is in SI, as the record stores it.
RECORD: dict[str, dict[str, dict[str, float]]] = {
    "M3SingletRef": {
        "image_surface": {
            "surviving_ray_count": 817,
            "reference_plane_z_m": 0.00490560476022521,
            "x_m_max": 7.267701944603245e-07,
            "y_m_max": 7.267701944603245e-07,
            "max_direction_norm_error": 6.661338147750939e-16,
            "intensity_min": 1.0,
            "intensity_max": 1.0,
            "intensity_sum": 817.0,
        },
        "exit_pupil": {
            "surviving_ray_count": 817,
            "reference_plane_z_m": 6.814345991561232e-05,
            "x_m_max": 0.0002497841477866965,
            "y_m_max": 0.0002497841477866965,
            "max_direction_norm_error": 6.661338147750939e-16,
            "intensity_min": 1.0,
            "intensity_max": 1.0,
            "intensity_sum": 817.0,
            "exit_pupil_diameter_m": 0.0004987073505473812,
            "exit_pupil_location_from_image_m": -0.004837461300309598,
            "max_projection_step_m": 0.004843943388606104,
        },
    },
    "ReverseTelephoto": {
        "image_surface": {
            "surviving_ray_count": 817,
            "reference_plane_z_m": 0.005209361469999999,
            # Not equal, and that asymmetry is the point: at the image surface the
            # telephoto's on-axis spot is 0.11% wider in x than in y, so a
            # transposed axis would show here. It is equal at the exit pupil.
            "x_m_max": 5.695279815682992e-07,
            "y_m_max": 5.68882899539519e-07,
            "max_direction_norm_error": 1.5543122344752192e-15,
            "intensity_min": 0.9998197316455706,
            "intensity_max": 0.999831103160469,
            "intensity_sum": 816.8576191444665,
        },
        "exit_pupil": {
            "surviving_ray_count": 817,
            "reference_plane_z_m": 0.0021547825721481666,
            "x_m_max": 0.0002309860106999379,
            "y_m_max": 0.0002309860106999379,
            "max_direction_norm_error": 1.5543122344752192e-15,
            "intensity_min": 0.9998197316455706,
            "intensity_max": 0.999831103160469,
            "intensity_sum": 816.8576191444665,
            "exit_pupil_diameter_m": 0.0004605349363758163,
            "exit_pupil_location_from_image_m": -0.003054578897851833,
            "max_projection_step_m": 0.0030632778099899415,
        },
    },
}

SETUPS: dict[str, OpticalSetup] = {
    "M3SingletRef": singlet_ref(),
    "ReverseTelephoto": REVERSE_TELEPHOTO,
}


def _traced(name: str, reference_surface: str) -> tuple[object, dict[str, object]]:
    """The bundle and its diagnostics, at the record's own settings.

    CHE-218 (R05.7) split the illumination out of the setup, and this is one of
    the two places the split had to be shown not to move a number. The lens is
    built with **no source**, which declares the on-axis field and the setup's own
    reference wavelength -- exactly what the record was taken under -- and the
    trace still runs at 550 nm, which for M3-REVERSE-TELEPHOTO is outside the
    wavelength set the fixture used to carry and always was.

    CHE-219 (R05.8) is the other place the same obligation lands: the pupil
    measure and the object-space reference term are now taken from the launch
    declaration rather than reconstructed here, and the record numbers below are
    what says that moved nothing. `ON_AXIS` is the source the launch is declared
    for -- on axis at the record's own wavelength, which is the field a
    source-less `build_lens` declares.
    """
    lens = build_lens(SETUPS[name])
    _, declaration = launch(lens, ON_AXIS, num_rings=NUM_RINGS)
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=NUM_RINGS)
    return to_ray_bundle(
        lens, native, launch=declaration, reference_surface=reference_surface
    )


# ---------------------------------------------------------------------------
# 1. Positions, directions and the reference surface, against the record
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RECORD))
@pytest.mark.parametrize("reference_surface", ["image_surface", "exit_pupil"])
def test_positions_and_directions_reproduce_the_frozen_record(
    name: str, reference_surface: str
) -> None:
    """`abs=0.0`: the same arithmetic on the same host in the same precision.

    The transverse extents are the discriminating quantities. On the singlet they
    differ between the two reference surfaces by a factor of 344 -- 0.73 um at the
    focus against 250 um at the pupil -- so a bundle exported at the wrong surface
    cannot pass this by accident.
    """
    expected = RECORD[name][reference_surface]
    bundle, _ = _traced(name, reference_surface)

    assert bundle.count == expected["surviving_ray_count"]
    assert bundle.reference_surface.z_m == pytest.approx(
        expected["reference_plane_z_m"], abs=0.0
    )
    positions = np.asarray(bundle.positions_m)
    assert float(np.max(positions[:, 0])) == pytest.approx(expected["x_m_max"], abs=0.0)
    assert float(np.min(positions[:, 0])) == pytest.approx(-expected["x_m_max"], abs=0.0)
    assert float(np.max(positions[:, 1])) == pytest.approx(expected["y_m_max"], abs=0.0)
    # A hexapolar fan on axis is symmetric, and the declared surface is planar, so
    # z is single-valued at the surface it declares.
    assert float(np.ptp(positions[:, 2])) == pytest.approx(0.0, abs=0.0)
    assert float(positions[0, 2]) == pytest.approx(expected["reference_plane_z_m"], abs=0.0)

    directions = np.asarray(bundle.directions)
    norm_error = float(np.max(np.abs(np.linalg.norm(directions, axis=1) - 1.0)))
    assert norm_error == pytest.approx(expected["max_direction_norm_error"], abs=0.0)
    assert norm_error <= direction_norm_tolerance(bundle.state.dtype)


@pytest.mark.parametrize("name", sorted(RECORD))
def test_the_amplitude_is_the_square_root_of_the_recorded_weight(name: str) -> None:
    """`RealRays.i` is a weight, and the mapping to an amplitude is declared.

    The record froze the weight; the bundle carries `sqrt` of it, phase-free. Its
    sum is the discriminating number: the telephoto's 816.857... is not 817, so a
    bundle that had quietly normalized or replaced the weight would show here.
    """
    expected = RECORD[name]["image_surface"]
    bundle, diagnostics = _traced(name, "image_surface")
    amplitude = np.asarray(bundle.amplitude)
    assert amplitude.dtype == np.complex128
    assert float(np.max(np.abs(amplitude.imag))) == pytest.approx(0.0, abs=0.0), (
        "a real weight maps to a phase-free amplitude; every radian comes from the OPL"
    )
    weight = np.abs(amplitude) ** 2
    assert float(np.sum(weight)) == pytest.approx(expected["intensity_sum"], rel=1e-15)
    assert float(np.min(weight)) == pytest.approx(expected["intensity_min"], rel=1e-15)
    assert float(np.max(weight)) == pytest.approx(expected["intensity_max"], rel=1e-15)
    assert "amplitude = sqrt(intensity)" in str(diagnostics["amplitude_mapping"])


@pytest.mark.parametrize("name", sorted(RECORD))
def test_the_exit_pupil_is_read_from_the_system_not_constructed(name: str) -> None:
    """`XPL()` is signed and measured from the image surface, which is the easy
    thing to get wrong: it yields a plausible plane rather than an error."""
    expected = RECORD[name]["exit_pupil"]
    _, diagnostics = _traced(name, "exit_pupil")
    pupil = diagnostics["exit_pupil"]
    assert pupil["diameter_m"] == pytest.approx(expected["exit_pupil_diameter_m"], abs=0.0)
    assert pupil["location_from_image_m"] == pytest.approx(
        expected["exit_pupil_location_from_image_m"], abs=0.0
    )
    assert pupil["z_m"] == pytest.approx(expected["reference_plane_z_m"], abs=0.0)
    # Virtual on both fixture systems: the pupil sits before refracting surfaces
    # the rays still travel through, so a position there is an asymptote.
    assert pupil["is_virtual"] is True
    assert "ASYMPTOTE" in pupil["position_semantics"]


@pytest.mark.parametrize("name", sorted(RECORD))
def test_the_projection_changes_no_direction_and_adds_no_path(name: str) -> None:
    """The pupil projection is a reparameterization along each ray.

    Directions must be *bitwise* identical between the two reference surfaces, and
    the OPL difference must be the geometric transfer rather than anything else.
    """
    at_image, _ = _traced(name, "image_surface")
    at_pupil, diagnostics = _traced(name, "exit_pupil")
    np.testing.assert_array_equal(
        np.asarray(at_image.directions), np.asarray(at_pupil.directions)
    )
    step_m = np.abs(
        np.asarray(at_pupil.positions_m)[:, 2] - np.asarray(at_image.positions_m)[:, 2]
    )
    expected_step = abs(RECORD[name]["exit_pupil"]["exit_pupil_location_from_image_m"])
    # `rel=1e-15` rather than `abs=0.0`, and the reason is stated because it is the
    # only relaxed record comparison in this file: the step is a difference of two
    # already-scaled metre coordinates while the record's value is `XPL * 1e-3`, so
    # the two are the same quantity through a different order of operations and
    # differ by one ulp. Widening it to a physical tolerance would hide a real
    # plane error; one ulp cannot.
    assert float(np.max(step_m)) == pytest.approx(expected_step, rel=1e-15)
    assert diagnostics["image_space_refractive_index"] == 1.0, (
        "read from the prescription; 'it is air' is a property of these two systems"
    )


# ---------------------------------------------------------------------------
# 2. The quadrature measure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_rings", [1, 2, 4, 8, 16, 64, 256])
def test_the_weight_sum_converges_on_the_aperture_area(num_rings: int) -> None:
    """`sum dA_i = pi a^2 (1 + 1/(4 n^2))` exactly, so it converges under refinement.

    This is what makes a reconstructed field's discrete power converge under ray
    refinement instead of growing as `(ray count)^2`: the weight is an *absolute*
    area element rather than a relative correction factor. The identity is worked
    from the ring counts `6j`, so `abs=0.0` is the right tolerance -- it is
    arithmetic, not a limit.
    """
    aperture_radius_m = 0.25e-3
    pupil_x, pupil_y = _hexapolar_disk(num_rings)
    ring_index = hexapolar_ring_index(pupil_x, pupil_y, num_rings)
    weight = hexapolar_area_weight_m2(ring_index, num_rings, aperture_radius_m)

    aperture_area = math.pi * aperture_radius_m**2
    assert float(np.sum(weight)) / aperture_area == pytest.approx(
        1.0 + 1.0 / (4.0 * num_rings**2), rel=1e-14
    )
    assert weight.size == hexapolar_ray_count(num_rings)
    assert float(np.min(weight)) > 0.0


def test_the_boundary_corrections_are_the_measured_ones() -> None:
    """Centre 3/4, rim 1/2, interior 1x the nominal cell `pi a^2 / (3 n^2)`.

    The rim correction is the one with a physical reason worth restating: the
    outermost ring sits exactly on `rho = a`, so it represents only the inner half
    of its annulus -- there is no ray beyond the rim to average with.
    """
    num_rings, aperture_radius_m = 8, 1.0
    pupil_x, pupil_y = _hexapolar_disk(num_rings)
    ring_index = hexapolar_ring_index(pupil_x, pupil_y, num_rings)
    weight = hexapolar_area_weight_m2(ring_index, num_rings, aperture_radius_m)

    nominal = math.pi * aperture_radius_m**2 / (3.0 * num_rings**2)
    assert float(weight[ring_index == 0][0]) == pytest.approx(0.75 * nominal, abs=0.0)
    assert float(weight[ring_index == num_rings][0]) == pytest.approx(0.5 * nominal, abs=0.0)
    interior = weight[(ring_index > 0) & (ring_index < num_rings)]
    assert float(np.ptp(interior)) == pytest.approx(0.0, abs=0.0)
    assert float(interior[0]) == pytest.approx(nominal, abs=0.0)
    # Ring j carries 6j rays, which is what makes the interior cell exact.
    for j in range(1, num_rings + 1):
        assert int(np.count_nonzero(ring_index == j)) == 6 * j
    assert int(np.count_nonzero(ring_index == 0)) == 1


@pytest.mark.parametrize("name", sorted(RECORD))
def test_the_emitted_measure_is_declared_and_scaled_to_the_real_aperture(name: str) -> None:
    """A declared measure, with its kind, scaled to the entrance pupil the system has."""
    bundle, diagnostics = _traced(name, "exit_pupil")
    assert bundle.measure_kind == "quadrature_area_m2"
    aperture_radius_m = (
        SETUPS[name].entrance_pupil_diameter_mm / 2.0
    ) * NATIVE_LENGTH_M
    assert float(np.sum(bundle.measure_weight)) == pytest.approx(
        math.pi * aperture_radius_m**2 * (1.0 + 1.0 / (4.0 * NUM_RINGS**2)), rel=1e-14
    )
    assert "quadrature_area_m2" in str(diagnostics["measure"])


def test_a_non_hexapolar_pupil_cannot_be_given_a_quadrature_weight() -> None:
    """A vignetted or hand-built fan gets no area element rather than a guessed one.

    Dropping a ray shifts which rows correspond to which ring, so the weight would
    be silently mis-binned rather than absent.
    """
    pupil_x, pupil_y = _hexapolar_disk(4)
    perturbed = pupil_x.copy()
    perturbed[5] += 0.01
    with pytest.raises(ContractError) as excinfo:
        hexapolar_ring_index(perturbed, pupil_y, 4)
    assert excinfo.value.code == "MEASURE_UNDECLARED"
    assert "vignetted" in str(excinfo.value) or "invent" in str(excinfo.value)


def test_a_launch_declaration_that_does_not_describe_the_trace_is_refused() -> None:
    """CHE-219: what replaced "guess the quadrature from the traced row count".

    Before R05.8 a trace whose row count did not match an un-vignetted fan left
    the measure `undeclared`, because the measure was being *reconstructed* from
    the traced output and a mismatch meant no ring index could be assigned. The
    measure is now declared at launch, so a count mismatch is a different fact
    entirely: the declaration in hand describes a different launch than the one
    that was traced, and every per-ray array in it -- the measure, the
    object-space term -- would be applied to the wrong rows.

    That is refused rather than degraded. Optiland keeps a clipped ray's row and
    zeroes its intensity rather than removing it, so a generated trace's row count
    always equals the launch count; a mismatch cannot be a vignetted fan.
    """
    lens = build_lens(singlet_ref())
    _, declaration = launch(lens, ON_AXIS, num_rings=NUM_RINGS)
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=NUM_RINGS - 1)
    with pytest.raises(ContractError) as excinfo:
        to_ray_bundle(lens, native, launch=declaration, reference_surface="image_surface")
    assert excinfo.value.code == "SHAPE_MISMATCH"
    assert str(hexapolar_ray_count(NUM_RINGS)) in str(excinfo.value)
    assert str(hexapolar_ray_count(NUM_RINGS - 1)) in str(excinfo.value)


def _hexapolar_disk(num_rings: int) -> tuple[np.ndarray, np.ndarray]:
    """The solver's own hexapolar unit disk, regenerated.

    Read from the solver rather than reimplemented: the ring assignment is a
    tolerance test against `j / num_rings`, and a hand-built fan would be testing
    this test's arithmetic instead of the solver's sampling.
    """
    import optiland.backend.utils as be_utils
    from optiland.distribution import create_distribution

    distribution = create_distribution("hexapolar")
    distribution.generate_points(num_rings)
    return (
        np.asarray(be_utils.to_numpy(distribution.x), dtype=np.float64),
        np.asarray(be_utils.to_numpy(distribution.y), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# 3. Clipping and alive-ray counts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RECORD))
def test_nothing_is_clipped_on_the_fixture_systems(name: str) -> None:
    """The honest state, recorded rather than assumed.

    `Optic.trace` exposes survivors only, and neither fixture clips at any field
    the setup declares -- which is why the mask logic below is exercised against
    a stub instead. Pinning this means a future prescription change that starts
    clipping is visible rather than silently reducing the ray count.
    """
    bundle, diagnostics = _traced(name, "image_surface")
    assert diagnostics["traced_ray_count"] == hexapolar_ray_count(NUM_RINGS)
    assert diagnostics["alive_ray_count"] == diagnostics["traced_ray_count"]
    assert diagnostics["clipped_ray_count"] == 0
    assert bundle.count == diagnostics["alive_ray_count"]


def test_a_clipped_ray_is_dropped_and_counted() -> None:
    """A zeroed weight or a non-finite position is not read as physics.

    Optiland clips by zeroing `RealRays.i` and keeping the row with its state
    frozen at the clip, so a clipped ray still carries a plausible-looking
    position. A stub trace, because no fixture system clips: the assertion is about
    the mask, and a system that happened to clip would also be testing its own
    prescription.
    """
    lens = build_lens(singlet_ref())
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=NUM_RINGS)
    clipped = _StubTrace(native, zero_weight_at=(3, 7), non_finite_at=(11,))

    _, declaration = launch(lens, ON_AXIS, num_rings=NUM_RINGS)
    bundle, diagnostics = to_ray_bundle(
        lens, clipped, launch=declaration, reference_surface="exit_pupil"
    )
    assert diagnostics["traced_ray_count"] == hexapolar_ray_count(NUM_RINGS)
    assert diagnostics["clipped_ray_count"] == 3
    assert bundle.count == hexapolar_ray_count(NUM_RINGS) - 3
    assert np.all(np.isfinite(np.asarray(bundle.positions_m)))
    assert float(np.min(np.abs(bundle.amplitude))) > 0.0
    # The measure is still declared: the *fan* was un-vignetted, so every ring
    # index was assignable, and the dropped rows take their own weights with them.
    assert bundle.measure_kind == "quadrature_area_m2"
    assert bundle.measure_weight.shape == (bundle.count,)


# --- A really vignetted fan, and what its measure is worth (CHE-220 / R05.9) ---
#
# The decision this section records, because R05.8 and R05.9 had to agree about it:
# **a vignetted fan still has a declared quadrature measure.** Each surviving ray
# keeps the entrance-pupil cell area it was *launched* with, and the clipped cells
# are power the system genuinely did not collect -- which is the physically right
# answer for a quadrature over the pupil, not a degradation.
#
# It is R05.8 that makes it true rather than a claim: `launch._declare_measure`
# assigns the area element from the pupil coordinates the fan was **generated**
# from, while the complete sample and every ring identity are still known, so the
# declaration does not depend on the traced output at all. Had the measure still
# been reconstructed after the trace -- from a traced row count, or by recovering a
# ring index from surviving rays -- a clipped ring would have made it `undeclared`
# and every downstream reconstruction would have refused. That would have looked
# like an R05.9 regression and been an R05.8 artifact.
#
# The alternative -- refusing, on the grounds that a vignetted fan is no longer the
# sample the quadrature was derived for -- was rejected: the cells that survive are
# unchanged, and a refusal would make physical vignetting unreconstructable rather
# than merely lossy.

#: A purpose-built two-surface system for the vignetting case, so no fixture's
#: frozen numbers are involved. Surface 0 is the stop with a rim wide enough to
#: satisfy the stop consistency rule; surface 1 is a plane whose rim falls *between*
#: the two outer rings of a 2-ring fan.
#:
#: Both surfaces are planes in air, so a collimated on-axis ray keeps its launch
#: radius all the way to surface 1 and the clipped set is analytic: ring `j` of a
#: 2-ring fan sits at `rho = j / 2`, i.e. at 0, 0.25 and 0.5 mm for a 1 mm entrance
#: pupil, and a 0.4 mm rim removes ring 2 and nothing else.
VIGNETTE_ENTRANCE_PUPIL_MM = 1.0
VIGNETTE_RIM_MM = 0.4
VIGNETTE_RINGS = 2


def _vignetting_setup() -> OpticalSetup:
    return OpticalSetup(
        name="VignettingProbe",
        description=(
            "CHE-220: two planes in air, the second with a rim between the two outer "
            "rings of a 2-ring hexapolar fan."
        ),
        surfaces=(
            SurfaceSpec(thickness_mm=1.0, comment="the stop, unapertured"),
            SurfaceSpec(
                thickness_mm=5.0,
                clear_semi_diameter_mm=VIGNETTE_RIM_MM,
                comment="the vignetting rim",
            ),
        ),
        stop_index=0,
        entrance_pupil_diameter_mm=VIGNETTE_ENTRANCE_PUPIL_MM,
        reference_wavelength_um=WAVELENGTH_UM,
    )


def test_a_real_aperture_vignettes_the_fan_by_exactly_the_outer_ring() -> None:
    """The setup for the measure claim below: the clipped set is the one intended.

    Established first and separately, because "the measure survived vignetting" is
    only evidence if the vignetting was the analytic one. A 2-ring fan is 19 rays --
    1 + 6 + 12 -- so removing ring 2 leaves 7.
    """
    lens = build_lens(_vignetting_setup())
    _, declaration = launch(lens, ON_AXIS, num_rings=VIGNETTE_RINGS)
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=VIGNETTE_RINGS)
    bundle, diagnostics = to_ray_bundle(
        lens, native, launch=declaration, reference_surface="image_surface"
    )

    assert diagnostics["traced_ray_count"] == hexapolar_ray_count(VIGNETTE_RINGS) == 19
    assert diagnostics["clipped_ray_count"] == 12, "ring 2 of a 2-ring fan is 12 rays"
    assert diagnostics["alive_ray_count"] == 7
    assert bundle.count == 7
    # Every survivor is inside the rim, in metres, which is the geometric statement
    # that the aperture is what removed them.
    radius_m = np.hypot(*np.asarray(bundle.positions_m)[:, :2].T)
    assert float(np.max(radius_m)) <= VIGNETTE_RIM_MM * NATIVE_LENGTH_M


def test_a_vignetted_fan_keeps_the_cell_areas_it_was_launched_with() -> None:
    """Acceptance criterion 6: a stated measure, and the missing power stays missing.

    The survivors' weights are the *launch* cells for rings 0 and 1 -- `3/4` and `1`
    of the nominal `pi a^2 / (3 n^2)` -- not a rescaled quadrature over 7 rays, and
    not the rim correction ring 1 would have been given had it become the outermost
    surviving ring. That last distinction is the one worth testing: a measure
    recomputed from the survivors would have called ring 1 the rim and halved it.
    """
    lens = build_lens(_vignetting_setup())
    _, declaration = launch(lens, ON_AXIS, num_rings=VIGNETTE_RINGS)
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=VIGNETTE_RINGS)
    bundle, diagnostics = to_ray_bundle(
        lens, native, launch=declaration, reference_surface="image_surface"
    )

    assert bundle.measure_kind == "quadrature_area_m2", (
        "a vignetted fan is unweighted only if the measure is reconstructed from the "
        "trace; it is declared at launch, so it survives"
    )
    weight = np.asarray(bundle.measure_weight)
    assert weight.shape == (7,)

    aperture_radius_m = (VIGNETTE_ENTRANCE_PUPIL_MM / 2.0) * NATIVE_LENGTH_M
    nominal_m2 = math.pi * aperture_radius_m**2 / (3.0 * VIGNETTE_RINGS**2)
    assert float(np.sum(weight)) == pytest.approx(
        (0.75 + 6.0) * nominal_m2, rel=1e-14
    ), "one centre cell at 3/4 and six ring-1 cells at 1x, unchanged by the clipping"
    assert float(np.min(weight)) == pytest.approx(0.75 * nominal_m2, abs=0.0)
    assert float(np.max(weight)) == pytest.approx(nominal_m2, abs=0.0)
    # `np.any(array == pytest.approx(scalar))` would be a scalar all-comparison and
    # could not fail here, so the elementwise form is the one that discriminates.
    assert not bool(np.isclose(weight, 0.5 * nominal_m2).any()), (
        "ring 1 must not be given the RIM correction: it is the outermost SURVIVING "
        "ring, not the outermost ring of the fan the pupil was sampled with"
    )

    # The missing cells are missing power, which is the physical content of the
    # decision. The whole fan sums to `pi a^2 (1 + 1/(4n^2))`; ring 2 carried 12
    # half-cells, and the survivors are short by exactly that.
    whole_fan_m2 = math.pi * aperture_radius_m**2 * (1.0 + 1.0 / (4.0 * VIGNETTE_RINGS**2))
    assert float(np.sum(weight)) == pytest.approx(
        whole_fan_m2 - 12.0 * 0.5 * nominal_m2, rel=1e-14
    )
    # And the caller is told, in the same declaration the measure travels in. The
    # launch note quotes the sum over the WHOLE fan and its exact `1 + 1/(4 n^2)`
    # ratio, which is true of the sampling and false of this bundle, so the traced
    # note has to say which of the two it is quoting.
    note = str(diagnostics["measure"])
    assert "quadrature_area_m2" in note
    assert "VIGNETTED" in note
    assert "12 of 19" in note
    assert f"{float(np.sum(weight)):.9e}" in note, (
        "the surviving sum has to be in the note, or the only number a caller reads "
        "is the launch fan's"
    )
    assert diagnostics["clipped_ray_count"] == 12


def test_a_fully_clipped_trace_is_refused_rather_than_returned_empty() -> None:
    lens = build_lens(singlet_ref())
    _, declaration = launch(lens, ON_AXIS, num_rings=2)
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=2)
    everything = tuple(range(int(np.asarray(_as_host(native.x)).size)))
    with pytest.raises(ContractError) as excinfo:
        to_ray_bundle(
            lens,
            _StubTrace(native, zero_weight_at=everything, non_finite_at=()),
            launch=declaration,
            reference_surface="image_surface",
        )
    assert excinfo.value.code == "EMPTY_ENSEMBLE"


def _as_host(value: object) -> np.ndarray:
    import optiland.backend.utils as be_utils

    return np.asarray(be_utils.to_numpy(value))


class _StubTrace:
    """A traced ray set with chosen rows clipped. Test-only.

    Built by copying a real trace rather than by fabricating one, so every column
    except the clipped entries is a genuine traced value and the test cannot pass
    because the geometry was invented.
    """

    def __init__(
        self,
        native: object,
        *,
        zero_weight_at: tuple[int, ...],
        non_finite_at: tuple[int, ...],
    ) -> None:
        columns = {
            name: _as_host(getattr(native, name)).copy()
            for name in ("x", "y", "z", "L", "M", "N", "i", "w", "opd")
        }
        for index in zero_weight_at:
            columns["i"][index] = 0.0
        for index in non_finite_at:
            columns["x"][index] = np.nan
        for name, column in columns.items():
            setattr(self, name, column)


# ---------------------------------------------------------------------------
# 4. Analytic invariants: what is true, rather than what the code once did
# ---------------------------------------------------------------------------


def test_the_traced_focal_length_matches_the_thin_lens_formula() -> None:
    """`R / (n - 1)`, from the *innermost* traced ray's angle.

    An analytic oracle, and it is the primary one: the record says what the code
    once produced, this says what a plano-convex singlet does.

    Innermost rather than marginal, because the thin-lens formula is a paraxial
    statement. The marginal ray of this singlet crosses the axis 2.9e-3 short of
    the paraxial focus, and that is its *physical* spherical aberration -- the same
    aberration `test_optiland_opl_convention.py` measures as 0.017 waves. Holding a
    marginal ray to a paraxial formula would be a test of the wrong thing, and
    widening the tolerance to 3e-3 to make it pass would be the wrong repair. The
    innermost ring agrees to 1.13e-5, which is the singlet's departure from the
    thin-lens formula at 0.2 mm of centre thickness.
    """
    bundle = trace(
        singlet_ref(),
        singlet_source(wavelength_um=WAVELENGTH_UM),
        sampling={"num_rings": NUM_RINGS, "reference_surface": "exit_pupil"},
        execution={"device": "cpu", "precision": "fp64"},
    )
    positions = np.asarray(bundle.positions_m)
    directions = np.asarray(bundle.directions)
    radius_m = np.hypot(positions[:, 0], positions[:, 1])
    innermost = int(np.argmin(np.where(radius_m > 0.0, radius_m, np.inf)))
    # A collimated bundle crosses the axis height / tan(angle) beyond the pupil.
    transverse = math.hypot(float(directions[innermost, 0]), float(directions[innermost, 1]))
    traced_focal_m = float(radius_m[innermost]) * float(directions[innermost, 2]) / transverse
    analytic_m = SINGLET_EFFECTIVE_FOCAL_LENGTH_MM * NATIVE_LENGTH_M
    assert traced_focal_m == pytest.approx(analytic_m, rel=EFL_RELATIVE)

    marginal = int(np.argmax(radius_m))
    marginal_transverse = math.hypot(
        float(directions[marginal, 0]), float(directions[marginal, 1])
    )
    marginal_focal_m = (
        float(radius_m[marginal]) * float(directions[marginal, 2]) / marginal_transverse
    )
    assert marginal_focal_m < analytic_m, "spherical aberration pulls the marginal focus in"
    assert abs(marginal_focal_m - analytic_m) / analytic_m > 100.0 * EFL_RELATIVE, (
        "the paraxial claim would be vacuous if the marginal ray also satisfied it"
    )


def test_the_pupil_semi_extent_matches_the_paraxial_semi_diameter() -> None:
    """Sampling density pulls the measured extent inward; pupil aberration pushes
    real marginal rays outward. They agree to 0.2% here, and the *measured* extent
    is a property of the traced set rather than of the aperture -- which is why a
    consumer needing the aperture uses the paraxial diameter."""
    _, diagnostics = _traced("M3SingletRef", "exit_pupil")
    bundle, _ = _traced("M3SingletRef", "exit_pupil")
    positions = np.asarray(bundle.positions_m)
    measured = float(np.max(np.hypot(positions[:, 0], positions[:, 1])))
    paraxial = diagnostics["exit_pupil"]["diameter_m"] / 2.0
    assert measured == pytest.approx(paraxial, rel=2.0e-3)
    assert measured > paraxial, (
        "measured above paraxial on both fixture systems: real marginal rays land "
        "outside the paraxial pupil"
    )


def test_the_entrance_pupil_is_the_prescription_it_was_given() -> None:
    """The aperture is what the setup declared, not what the solver defaulted to."""
    lens = build_lens(singlet_ref())
    import optiland.backend.utils as be_utils

    epd_mm = float(np.asarray(be_utils.to_numpy(lens.paraxial.EPD())).ravel()[0])
    assert epd_mm == pytest.approx(SINGLET_ENTRANCE_PUPIL_DIAMETER_MM, abs=0.0)


def test_an_on_axis_collimated_bundle_is_centred_on_the_axis() -> None:
    """A ray invariant that no record is needed for.

    A hexapolar fan on axis has 6-fold symmetry, so its centroid is the axis:
    `sum x = sum y = 0` at either reference surface, on either system. A tilted,
    decentred or mirrored frame breaks it.

    The *extents* are only equal at the exit pupil. At the image surface the
    telephoto's spot is 0.11% wider in x than in y, because aberration acts on the
    hexapolar fan's own six-fold geometry rather than on a circle -- which is why
    the extents are compared against the record above and only the centroid is
    asserted as an invariant here.
    """
    for name in RECORD:
        for reference_surface in ("image_surface", "exit_pupil"):
            bundle, _ = _traced(name, reference_surface)
            positions = np.asarray(bundle.positions_m)
            extent = float(np.max(np.abs(positions[:, :2])))
            assert float(abs(np.sum(positions[:, 0]))) < 1e-9 * extent
            assert float(abs(np.sum(positions[:, 1]))) < 1e-9 * extent
        pupil, _ = _traced(name, "exit_pupil")
        pupil_positions = np.asarray(pupil.positions_m)
        assert float(np.max(pupil_positions[:, 0])) == pytest.approx(
            float(np.max(pupil_positions[:, 1])), abs=0.0
        )
