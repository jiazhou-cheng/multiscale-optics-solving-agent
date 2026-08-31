"""The point-source launch state, measured, and the OPL reference it justifies.

CHE-207 (R05.5), and the acceptance criteria the canceled CHE-46 carried:

* measure Optiland's launch state for a finite-object system **directly**, as
  CHE-41 did for the infinite case, rather than inferring it from the
  infinite-object result -- section 1;
* state whether the correction term is zero, and if so why, **with the
  measurement rather than the argument** -- section 1;
* lift the refusal with a verified declaration -- sections 2 and 3;
* reuse CHE-41's geometric oracle where it applies -- section 3.

The physical claim, and why it is not "obviously zero"
-----------------------------------------------------
For an object at infinity the accumulator is seeded on a plane perpendicular to z,
which is a wavefront only for a bundle travelling along z, so the reference has to
be moved onto the incoming plane wavefront by `n * (d0 . r_launch)`.

For a point source the launch state is a **single point**. A point is trivially a
common wavefront -- the degenerate sphere of zero radius centred on it -- so the
optical path from that wavefront to each launch point is zero for every ray, with
no arithmetic to round.

That argument is sound and it is still not evidence, which is the whole reason
CHE-46 refused to accept it: it assumes the launch state *is* one point, and
nothing had checked. Section 1 checks it. It also assumes the seeded accumulator
starts at zero rather than at some object-space value, and section 1 checks that
too -- if Optiland pre-loaded the accumulator with the object-to-vertex distance,
the correct term would be non-zero and the argument would have reached the right
answer for the wrong reason.

The two launch geometries are mirror images, and section 1 shows it as a
measurement: at infinity the directions are common and the origins spread; for a
point source the origin is common and the directions spread.

Tolerances
----------
Section 1 is `abs=0.0` throughout -- these are exact structural facts, not
approximations, and stating them as exact is what makes the zero term defensible.

Every other tolerance cites its oracle inline, and the three worth naming here are
the ones that are *aberration* rather than numerical error -- each identified, so
none of them is a loose number absorbing something unexplained:

* `CONJUGATE_RELATIVE = 3e-4` on the traced axial image location against the
  closed-form paraxial conjugate. The innermost traced ray is at finite height, so
  this is the singlet's spherical aberration: measured 1.8e-4 at 2f and falling to
  5.6e-5 at 10f. The paraxial magnification is checked separately at `rel=1e-12`,
  where there is no aberration to allow for.
* the per-field chief-ray tolerances, `1e-6 / 3e-6 / 1e-5` at 0.5 / 1.0 / 2.0
  degrees. Measured 2.9e-7, 1.2e-6, 4.7e-6 -- a field-squared law, i.e. third-order
  distortion. They tighten with the field rather than being one loose number,
  which is what keeps a constant error from hiding inside them.
* the spot centroid sits 2.4e-3 *relatively* below the paraxial image point at
  **every** field, so the absolute displacement is linear in field height: coma.
  That is asserted as its own fact rather than tolerated inside the chief-ray
  check, which is why the chief ray -- CHE-41's actual oracle -- is what the image
  location is compared against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from fixtures.systems import (
    FINITE_CONJUGATE_MAGNIFICATION,
    FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM,
    finite_conjugate_image_distance_mm,
    finite_conjugate_singlet,
    singlet_ref,
)

from representations import RayBundle
from solvers.optiland import trace
from solvers.optiland.rays import (
    LAUNCH_PLANE_WAVEFRONT,
    LAUNCH_POINT_SOURCE,
    NATIVE_LENGTH_M,
    OPL_REFERENCE_VERSION,
    _object_space_reference,
    require_declared_optical_path,
    to_ray_bundle,
)
from solvers.optiland.solver import _normalized_field
from solvers.optiland.system import build_lens

WAVELENGTH_UM = 0.55
WAVELENGTH_M = WAVELENGTH_UM * 1.0e-6
NUM_RINGS = 8
CPU64 = {"device": "cpu", "precision": "fp64"}

#: The traced image location against the closed-form paraxial conjugate. See the
#: module docstring: this is the singlet's spherical aberration, not a numerical
#: tolerance.
CONJUGATE_RELATIVE = 3.0e-4


def _host(value: object) -> np.ndarray:
    import optiland.backend.utils as be_utils

    return np.asarray(be_utils.to_numpy(value))


def _scalar(value: object) -> float:
    return float(np.asarray(_host(value)).ravel()[0])


def _launch_state(problem: object, *, field_deg: tuple[float, float]) -> dict[str, np.ndarray]:
    """Regenerate the launch state the way the adapter does, and hand back the columns.

    Through the solver's own public generator over the same hexapolar distribution
    `Optic.trace` builds, because `Optic.trace` retains no launch state. Section 1
    checks that the regeneration is faithful before anything is concluded from it.
    """
    import optiland.backend as be
    from optiland.distribution import create_distribution

    lens = build_lens(problem)
    field = _normalized_field(lens, field_deg)
    distribution = create_distribution("hexapolar")
    distribution.generate_points(NUM_RINGS)
    points = int(_host(distribution.x).size)
    launch = lens.ray_tracer.ray_generator.generate_rays(
        be.repeat(be.atleast_1d(be.array(float(field[0]))), points),
        be.repeat(be.atleast_1d(be.array(float(field[1]))), points),
        distribution.x,
        distribution.y,
        WAVELENGTH_UM,
    )
    return {
        "lens": lens,
        "field": field,
        **{
            name: np.asarray(_host(getattr(launch, attribute)), dtype=np.float64)
            for name, attribute in (
                ("x", "x"),
                ("y", "y"),
                ("z", "z"),
                ("L", "L"),
                ("M", "M"),
                ("N", "N"),
                ("accumulator", "opd"),
            )
        },
        "launch": launch,
    }


# ---------------------------------------------------------------------------
# 1. The launch state, measured directly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_deg", [(0.0, 0.0), (0.0, 2.0), (1.0, 1.0)])
def test_the_finite_object_launch_state_is_a_single_point(
    field_deg: tuple[float, float],
) -> None:
    """`abs=0.0`: every ray of one field leaves the same point, exactly.

    This is the premise the zero correction term rests on, so it is asserted
    exactly rather than within a tolerance. A spread that is merely small would
    mean the source is an extended one and the reference is not a single sphere.
    """
    state = _launch_state(finite_conjugate_singlet(), field_deg=field_deg)
    for axis in ("x", "y", "z"):
        assert float(np.ptp(state[axis])) == 0.0, f"{axis} origin spread must be exactly zero"
    assert state["x"].size == 1 + 3 * NUM_RINGS * (NUM_RINGS + 1)

    # Where the point is: the schema's declared convention, to the digit.
    distance = FINITE_CONJUGATE_OBJECT_DISTANCE_MM
    assert float(state["z"][0]) == pytest.approx(-distance, abs=0.0)
    assert float(state["x"][0]) == pytest.approx(
        -math.tan(math.radians(field_deg[0])) * distance, abs=0.0
    )
    assert float(state["y"][0]) == pytest.approx(
        -math.tan(math.radians(field_deg[1])) * distance, abs=0.0
    )


def test_the_finite_object_launch_accumulator_is_seeded_at_zero() -> None:
    """The second premise: nothing object-space is pre-loaded into the accumulator.

    If the solver seeded the path with the object-to-vertex distance, the correct
    reference term would not be zero and the structural argument would have been
    right by luck. Measured: exactly 0.0, with zero spread.
    """
    state = _launch_state(finite_conjugate_singlet(), field_deg=(0.0, 2.0))
    assert float(np.max(np.abs(state["accumulator"]))) == 0.0
    assert float(np.ptp(state["accumulator"])) == 0.0


def test_the_finite_object_bundle_diverges_rather_than_being_collimated() -> None:
    """The mirror image of the infinite case, shown as a measurement.

    At infinity: common directions, origins spread over a plane. For a point
    source: common origin, directions spread. Both halves are asserted here on the
    two fixture systems, because "the two cases are different geometries" is the
    reason they cannot share one arithmetic.
    """
    point = _launch_state(finite_conjugate_singlet(), field_deg=(0.0, 0.0))
    direction_spread = max(float(np.ptp(point[axis])) for axis in ("L", "M", "N"))
    origin_spread = max(float(np.ptp(point[axis])) for axis in ("x", "y", "z"))
    assert origin_spread == 0.0
    assert direction_spread > 1.0e-3, "a point source radiates; its directions must spread"

    collimated = _launch_state(singlet_ref(), field_deg=(0.0, 0.0))
    assert float(np.ptp(collimated["z"])) == 0.0, "launched on one plane"
    assert max(float(np.ptp(collimated[axis])) for axis in ("L", "M", "N")) < 1.0e-12
    assert max(float(np.ptp(collimated[axis])) for axis in ("x", "y")) > 1.0e-3


@pytest.mark.parametrize("field_deg", [(0.0, 0.0), (0.0, 2.0), (1.0, 1.0)])
def test_the_regenerated_launch_state_reproduces_the_trace(
    field_deg: tuple[float, float],
) -> None:
    """Bit-identical, which is what lets a per-ray term measured from it be used.

    The same property CHE-41 established for the infinite case, re-established
    here for the finite one rather than assumed to carry over.
    """
    state = _launch_state(finite_conjugate_singlet(), field_deg=field_deg)
    lens = state["lens"]
    traced = lens.surfaces.trace(state["launch"], skip=1)
    reference = lens.trace(
        Hx=state["field"][0],
        Hy=state["field"][1],
        wavelength=WAVELENGTH_UM,
        num_rays=NUM_RINGS,
    )
    for attribute in ("x", "y", "z", "opd"):
        np.testing.assert_array_equal(
            _host(getattr(traced, attribute)), _host(getattr(reference, attribute))
        )


# ---------------------------------------------------------------------------
# 2. The reference term, and the refusal that is now a declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_deg", [(0.0, 0.0), (0.0, 2.0), (1.0, 1.0)])
def test_the_object_space_term_is_available_and_exactly_zero(
    field_deg: tuple[float, float],
) -> None:
    """The refusal is lifted, and what replaces it is an exact zero per ray."""
    problem = finite_conjugate_singlet()
    lens = build_lens(problem)
    field = _normalized_field(lens, field_deg)
    term = _object_space_reference(
        lens,
        field=field,
        wavelength_um=WAVELENGTH_UM,
        num_rings=NUM_RINGS,
        traced_count=1 + 3 * NUM_RINGS * (NUM_RINGS + 1),
    )
    assert term["available"] is True, term["reason"]
    assert term["reason"] is None
    assert term["launch_geometry"] == LAUNCH_POINT_SOURCE
    assert float(np.max(np.abs(term["offset_native"]))) == 0.0
    assert term["span_native"] == 0.0
    assert term["object_space_refractive_index"] == 1.0
    assert term["launch_point_native"][2] == pytest.approx(
        -FINITE_CONJUGATE_OBJECT_DISTANCE_MM, abs=0.0
    )


def test_the_infinite_object_term_is_unchanged_and_still_names_its_own_geometry() -> None:
    """The control: adding the finite branch must not touch the collimated one."""
    lens = build_lens(singlet_ref())
    term = _object_space_reference(
        lens,
        field=(0.0, 0.0),
        wavelength_um=WAVELENGTH_UM,
        num_rings=NUM_RINGS,
        traced_count=1 + 3 * NUM_RINGS * (NUM_RINGS + 1),
    )
    assert term["available"] is True
    assert term["launch_geometry"] == LAUNCH_PLANE_WAVEFRONT
    assert "launch_direction" in term and "launch_plane_z_native" in term
    # On axis the collimated term is a constant, which is the pre-existing fact
    # this branch is not allowed to have changed.
    assert term["span_native"] == 0.0


def test_an_extended_finite_source_is_refused_rather_than_approximated() -> None:
    """The zero holds only because the origins coincide, so a spread is refused.

    Driven through the shipping helper with a manufactured launch state, because
    no problem this schema can state produces an extended source -- and a refusal
    with no reachable case is a claim about a path that does not exist.
    """
    from solvers.optiland.rays import _point_source_reference

    coincident = np.zeros(5)
    good = _point_source_reference(coincident, coincident, coincident - 9.0, index=1.0)
    assert good["available"] is True
    assert float(np.max(np.abs(good["offset_native"]))) == 0.0

    spread = np.array([0.0, 0.0, 1.0e-9, 0.0, 0.0])
    refused = _point_source_reference(coincident, spread, coincident - 9.0, index=1.0)
    assert refused["available"] is False
    assert refused["offset_native"] is None
    assert "is not a POINT" in refused["reason"]
    assert "extended finite source" in refused["reason"]


# ---------------------------------------------------------------------------
# 3. The declared path, and the bundle that comes out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reference_surface", ["image_surface", "exit_pupil"])
@pytest.mark.parametrize("field_deg", [(0.0, 0.0), (0.0, 2.0)])
def test_a_point_source_trace_yields_a_coherent_bundle(
    reference_surface: str, field_deg: tuple[float, float]
) -> None:
    """The ticket's headline: on axis and off axis, a valid coherent bundle.

    `require_coherent()` is the gate, not a formality: it refuses a bundle whose
    optical path reference is undeclared or unverified, which is exactly what the
    finite-conjugate path used to produce.
    """
    bundle = trace(
        finite_conjugate_singlet(),
        sampling={
            "num_rings": NUM_RINGS,
            "reference_surface": reference_surface,
            "wavelength_um": WAVELENGTH_UM,
            "field_deg": field_deg,
        },
        execution=CPU64,
    )
    assert isinstance(bundle, RayBundle)
    assert bundle.count == 1 + 3 * NUM_RINGS * (NUM_RINGS + 1)

    amplitude, optical_path = bundle.require_coherent()
    assert np.all(np.isfinite(np.asarray(amplitude)))
    assert np.all(np.isfinite(np.asarray(optical_path)))

    require_declared_optical_path(bundle)
    reference = str(bundle.optical_path_reference)
    assert reference.startswith(OPL_REFERENCE_VERSION)
    # The declaration names the SPHERE, not the plane. A finite conjugate declared
    # against a plane wavefront would be a false statement about the reference
    # surface that no downstream check reads the object distance to catch.
    assert LAUNCH_POINT_SOURCE in reference
    assert LAUNCH_PLANE_WAVEFRONT not in reference
    assert "exactly zero for every ray" in reference

    assert bundle.measure_kind == "quadrature_area_m2"
    assert bundle.measure_weight is not None
    assert float(np.min(np.asarray(bundle.measure_weight))) > 0.0
    assert bundle.reference_surface.name == reference_surface
    assert bundle.reference_surface.medium_index == 1.0


def test_the_quadrature_measure_is_the_same_area_element_as_the_collimated_case() -> None:
    """The measure is a property of the pupil sampling, not of the source.

    Same aperture and same ring count as `M3-SINGLET-REF`, so the weights must be
    identical -- if the finite conjugate changed them, the measure would be
    carrying something about the light.
    """
    common = {
        "num_rings": NUM_RINGS,
        "reference_surface": "exit_pupil",
        "wavelength_um": WAVELENGTH_UM,
    }
    finite = trace(finite_conjugate_singlet(), sampling=common, execution=CPU64)
    collimated = trace(singlet_ref(), sampling=common, execution=CPU64)
    np.testing.assert_array_equal(
        np.asarray(finite.measure_weight), np.asarray(collimated.measure_weight)
    )
    aperture_radius_m = _scalar(build_lens(singlet_ref()).paraxial.EPD()) / 2.0 * NATIVE_LENGTH_M
    assert float(np.sum(np.asarray(finite.measure_weight))) == pytest.approx(
        math.pi * aperture_radius_m**2 * (1.0 + 1.0 / (4.0 * NUM_RINGS**2)), rel=1e-14
    )


def test_the_chief_ray_piston_is_removed_for_a_finite_conjugate_too() -> None:
    """Step 4 works unchanged: the chief ray reads exactly zero.

    And the piston it removed is large -- the absolute accumulated path of this
    system is ~2e4 waves -- so the conditioning is doing the same work here as it
    does for the collimated case rather than being a no-op.
    """
    lens = build_lens(finite_conjugate_singlet())
    native = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=NUM_RINGS)
    bundle, diagnostics = to_ray_bundle(
        lens,
        native,
        field=(0.0, 0.0),
        wavelength_um=WAVELENGTH_UM,
        num_rings=NUM_RINGS,
        reference_surface="exit_pupil",
    )
    optical_path = np.asarray(bundle.optical_path_m)
    assert float(np.min(np.abs(optical_path))) == 0.0, "the chief ray is the zero"
    accumulated_waves = float(np.min(_host(native.opd))) * NATIVE_LENGTH_M / WAVELENGTH_M
    assert accumulated_waves > 1.0e4, (
        "the absolute path must be large for the piston removal to be load-bearing"
    )
    signal_waves = float(np.ptp(optical_path)) / WAVELENGTH_M
    assert accumulated_waves / signal_waves > 100.0
    assert diagnostics["object_space_reference_available"] is True
    assert diagnostics["object_space_reference_span_m"] == 0.0


def test_the_exit_pupil_transfer_is_a_reparameterization_for_a_finite_conjugate() -> None:
    """Directions bitwise identical between the two reference surfaces.

    The same property the collimated parity gate asserts, re-asserted here because
    the exit pupil of this system is on the *other* side of the image surface
    (XPL is negative and small), so the transfer runs with a different sign of
    lever arm than the collimated case.
    """
    common = {
        "num_rings": NUM_RINGS,
        "wavelength_um": WAVELENGTH_UM,
    }
    at_image = trace(
        finite_conjugate_singlet(),
        sampling={**common, "reference_surface": "image_surface"},
        execution=CPU64,
    )
    at_pupil = trace(
        finite_conjugate_singlet(),
        sampling={**common, "reference_surface": "exit_pupil"},
        execution=CPU64,
    )
    np.testing.assert_array_equal(
        np.asarray(at_image.directions), np.asarray(at_pupil.directions)
    )
    assert at_pupil.reference_surface.z_m != at_image.reference_surface.z_m
    # Both are coherent, and the declared paths differ by the plane transfer.
    at_image.require_coherent()
    at_pupil.require_coherent()
    assert float(
        np.max(np.abs(np.asarray(at_image.optical_path_m) - np.asarray(at_pupil.optical_path_m)))
    ) > 0.0


# ---------------------------------------------------------------------------
# 4. The conjugate itself, against the closed form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("multiple", [2.0, 3.0, 10.0])
def test_the_traced_image_lands_at_the_closed_form_paraxial_conjugate(
    multiple: float,
) -> None:
    """An analytic oracle for the fixture's own spacing.

    The closed form is a paraxial ray trace written out in
    `fixtures.systems.finite_conjugate_image_distance_mm`, independent of the
    solver. Three conjugates, so the agreement is a property of the formula rather
    than of one lucky distance -- and the residual *falls* as the object recedes
    (1.8e-4 at 2f, 5.6e-5 at 10f), which is aberration behaving like aberration.
    """
    distance = multiple * SINGLET_EFFECTIVE_FOCAL_LENGTH_MM
    problem = finite_conjugate_singlet(distance)
    lens = build_lens(problem)
    traced = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=NUM_RINGS)

    position = np.stack([_host(traced.x), _host(traced.y), _host(traced.z)], axis=1)
    directions = np.stack([_host(traced.L), _host(traced.M), _host(traced.N)], axis=1)
    radial = np.hypot(position[:, 0], position[:, 1])
    innermost = int(np.argmin(np.where(radial > 0.0, radial, np.inf)))
    transverse = math.hypot(float(directions[innermost, 0]), float(directions[innermost, 1]))
    crossing_mm = float(position[innermost, 2]) + float(radial[innermost]) * float(
        directions[innermost, 2]
    ) / transverse

    expected_mm = problem.surfaces[0].thickness_mm + finite_conjugate_image_distance_mm(distance)
    assert crossing_mm == pytest.approx(expected_mm, rel=CONJUGATE_RELATIVE)
    assert problem.surfaces[-1].thickness_mm == finite_conjugate_image_distance_mm(distance)


def test_the_fixture_sits_at_unit_magnification() -> None:
    """The exact half of the conjugate check: an object at 2f images at 2f.

    The solver's paraxial magnification is a scalar with no aberration in it, so it
    is held to `rel=1e-12` where the ray crossing above is held to 3e-4.
    """
    lens = build_lens(finite_conjugate_singlet())
    assert _scalar(lens.paraxial.magnification()) == pytest.approx(
        FINITE_CONJUGATE_MAGNIFICATION, rel=1e-12
    )
    assert FINITE_CONJUGATE_OBJECT_DISTANCE_MM == 2.0 * SINGLET_EFFECTIVE_FOCAL_LENGTH_MM


@pytest.mark.parametrize(("field_deg", "chief_relative"), [(0.5, 1e-6), (1.0, 3e-6), (2.0, 1e-5)])
def test_the_off_axis_chief_ray_lands_where_the_magnification_says(
    field_deg: float, chief_relative: float
) -> None:
    """CHE-41's geometric oracle, in the form a finite conjugate admits.

    At magnification -1 the image of a source at height `h` is at `-h`, and the
    oracle is the **traced chief ray** rather than the spot centroid -- which is
    what CHE-41's version says too ("the PSF lands at the traced chief-ray
    intersection"). The hexapolar fan's central ray is row 0.

    The per-field tolerances are measured and they *scale*: the chief ray departs
    from the paraxial prediction by 2.9e-7, 1.2e-6 and 4.7e-6 at 0.5, 1.0 and 2.0
    degrees. That is a field-squared law, i.e. third-order distortion of an
    uncorrected singlet, so the tolerance tightens as the field shrinks instead of
    being one loose number that would hide a constant error at every field.
    """
    bundle = trace(
        finite_conjugate_singlet(),
        sampling={
            "num_rings": NUM_RINGS,
            "reference_surface": "image_surface",
            "wavelength_um": WAVELENGTH_UM,
            "field_deg": (0.0, field_deg),
        },
        execution=CPU64,
    )
    positions = np.asarray(bundle.positions_m)
    source_y_mm = -math.tan(math.radians(field_deg)) * FINITE_CONJUGATE_OBJECT_DISTANCE_MM
    predicted_y_m = FINITE_CONJUGATE_MAGNIFICATION * source_y_mm * NATIVE_LENGTH_M

    assert predicted_y_m > 0.0, "a source below the axis images above it"
    assert float(positions[0, 1]) == pytest.approx(predicted_y_m, rel=chief_relative)
    # A y-only field must produce a y-only image point.
    assert abs(float(positions[0, 0])) < 1.0e-15


def test_the_spot_centroid_is_displaced_from_the_chief_ray_by_coma() -> None:
    """Why the previous test uses the chief ray and not the centroid.

    The centroid sits 2.4e-3 *relatively* below the paraxial image point, and that
    ratio is the **same at every field** -- so the absolute displacement grows
    linearly with field height, which is coma rather than distortion. Asserting it
    means the 2.4e-3 is identified physics rather than an unexplained residual
    that a looser tolerance would have swallowed.
    """
    offsets = []
    for field_deg in (0.5, 1.0, 2.0):
        bundle = trace(
            finite_conjugate_singlet(),
            sampling={
                "num_rings": NUM_RINGS,
                "reference_surface": "image_surface",
                "wavelength_um": WAVELENGTH_UM,
                "field_deg": (0.0, field_deg),
            },
            execution=CPU64,
        )
        positions = np.asarray(bundle.positions_m)
        source_y_mm = -math.tan(math.radians(field_deg)) * FINITE_CONJUGATE_OBJECT_DISTANCE_MM
        predicted_y_m = FINITE_CONJUGATE_MAGNIFICATION * source_y_mm * NATIVE_LENGTH_M
        centroid_y_m = float(np.mean(positions[:, 1]))
        offsets.append((centroid_y_m - predicted_y_m) / predicted_y_m)
        # The centroid is displaced, and the chief ray is not: the two disagree by
        # far more than the chief ray's own departure from the prediction.
        assert abs(centroid_y_m - float(positions[0, 1])) > 0.5 * abs(
            centroid_y_m - predicted_y_m
        )

    assert all(offset < 0.0 for offset in offsets), "the centroid trails the chief ray"
    assert max(offsets) - min(offsets) < 1.0e-4, (
        f"a field-independent RELATIVE offset is coma; measured {offsets!r}. A "
        "field-dependent one would be distortion and would mean the conjugate is wrong."
    )
    assert all(abs(offset) == pytest.approx(2.41e-3, rel=5e-2) for offset in offsets)
