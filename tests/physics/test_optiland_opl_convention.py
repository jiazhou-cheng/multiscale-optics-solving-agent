"""The OPD convention, re-established against manufactured closed-form geometries.

CHE-182 (R05.4). What must be established, and where each part is:

2. the declared OPL on axis **and** off axis, to float64 round-off on the
   manufactured closed-form geometries -- sections 1 and 3 here;
3. the negative control still fails when the sign is inverted -- section 4. A
   parity gate with no falsifiable twin proves nothing.

This is the ticket's highest-risk surface and it is why the risk is asymmetric: a
wrong OPL *reference* is a constant piston and mostly harmless, while a wrong
*sign* conjugates the wavefront -- a converging beam reconstructs as a diverging
one -- and the two are indistinguishable in any intensity, because `|U|^2` is
identical under conjugation.

Tolerances, and the oracle that set each one
--------------------------------------------
Every tolerance below is `0.0` -- exact equality, or agreement to float64
round-off stated as such -- except two, and both cite what set them:

* `SPHERE_RESIDUAL_WAVES = 0.017` on `M3-SINGLET-REF`. The oracle is
  **analytic**: for a diffraction-limited system every ray reaches the focus with
  equal total optical path, so the pupil OPL must satisfy
  `OPL(rho) - OPL(0) = R - sqrt(rho^2 + R^2)`. The residual is the singlet's own
  physical spherical aberration and not a numerical error, which is why it is a
  measured value rather than a bound: 0.016999 waves peak-to-valley, agreeing with
  the reference implementation's independently frozen 0.016996 and with the
  solver's own `Wavefront` value of 0.016999 -- three routes, one number.
* `TILT_SLOPE_RELATIVE = 3e-3` on the off-axis case. The oracle is geometric: the
  slope the declared OPL must carry to converge on the traced chief-ray
  intersection is `0.06839171464575525`, from
  `pre-rewrite-2026-08-30:benchmarks/probes/records/optiland/off_axis_opd_reference.json`.
  The tolerance is on the agreement between a least-squares fit and that
  requirement, and it is loose because the fit is over a pupil carrying real
  aberration; the *discriminating* number is three orders of magnitude away.

Nothing here uses this repository's own numerics as the oracle. The manufactured
geometries have closed-form answers, and the analytic sphere condition and the
geometric slope requirement are both statements about optics rather than about
this code.

CHE-217 (R05.6) adds section 3c: the *composed* path a supplied bundle leaves
with, which is a second declared quantity rather than a variation on the first.
It is here rather than beside the rest of R05.6's tests because this is the file
that owns the accumulator convention and the manufactured plate that gives it a
closed-form answer -- and because the composition is the place R05.6 could be
wrong without any intensity check noticing. Same discipline: an exact closed form,
`n * L` rather than `L`, and the geometric distance kept as the falsifiable twin.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fixtures.systems import REVERSE_TELEPHOTO, singlet_ref

from problems.ray_trace import Material, RayTraceProblem, SurfaceSpec
from representations import UNVERIFIED, ContractError, Frame, RayBundle, ReferenceSurface
from solvers.optiland import rays as rays_module
from solvers.optiland import trace_rays
from solvers.optiland.rays import (
    COMPOSED_OPL_REFERENCE_VERSION,
    LAUNCH_PLANE_WAVEFRONT,
    LAUNCH_POINT_SOURCE,
    NATIVE_LENGTH_M,
    OPL_REFERENCE_VERSION,
    declare_optical_path_m,
    exit_pupil,
    require_declared_optical_path,
    to_ray_bundle,
)
from solvers.optiland.system import build_lens

WAVELENGTH_UM = 0.55
WAVELENGTH_M = WAVELENGTH_UM * 1.0e-6

#: Measured, not bounded: the singlet's own spherical aberration at its paraxial
#: focus. See the module docstring for the three independent routes to it.
SPHERE_RESIDUAL_WAVES = 0.017

#: The geometric requirement, from `off_axis_opd_reference.json`.
REQUIRED_TILT_SLOPE = 0.06839171464575525
TILT_SLOPE_RELATIVE = 3.0e-3

#: What the launch-plane reference retains of that slope when the object-space
#: term is omitted: 0.13% of it, measured. The negative control.
OMITTED_TERM_SLOPE_FRACTION = 1.3e-3


def _host(value: object) -> np.ndarray:
    import optiland.backend.utils as be_utils

    return np.asarray(be_utils.to_numpy(value))


def _plate(*, thickness_mm: float, index: float, epd_mm: float = 2.0) -> RayTraceProblem:
    """A manufactured geometry: one axial gap, then a plane, then an image plane.

    Everything about it is chosen so the accumulated optical path has a
    closed-form answer: plane surfaces, so there is no refraction to compute; an
    ideal constant index, so there is no catalog to consult; and an object at
    infinity, so the launch geometry is the one the convention is stated for.
    """
    material: Material = (
        {"kind": "air"} if index == 1.0 else {"kind": "ideal", "refractive_index": index}
    )
    return RayTraceProblem(
        name=f"plate-t{thickness_mm}-n{index}",
        surfaces=(
            SurfaceSpec(thickness_mm=thickness_mm, material=material),
            SurfaceSpec(thickness_mm=0.0),
        ),
        entrance_pupil_diameter_mm=epd_mm,
        field_angles_deg=((0.0, 0.0),),
        wavelengths_um=(WAVELENGTH_UM,),
        stop_index=0,
    )


# ---------------------------------------------------------------------------
# 1. The native accumulator: sign, unit, physical meaning, reference
# ---------------------------------------------------------------------------


def test_the_accumulator_is_index_weighted_optical_path_not_geometric() -> None:
    """4 mm of air then 6 mm of n = 1.7 glass is 14.2 mm of optical path, not 10.

    The competing hypothesis is a *geometric* path accumulator, which would be
    short by 4.2 mm. Exact to float64: there is no arithmetic here beyond a sum of
    products.
    """
    lens = build_lens(
        RayTraceProblem(
            name="two-media",
            surfaces=(
                SurfaceSpec(thickness_mm=6.0, material={"kind": "ideal", "refractive_index": 1.7}),
                SurfaceSpec(thickness_mm=0.0),
            ),
            entrance_pupil_diameter_mm=2.0,
            field_angles_deg=((0.0, 0.0),),
            wavelengths_um=(WAVELENGTH_UM,),
            stop_index=0,
        )
    )
    # The launch plane sits an EPD before the first surface, so the air leg is
    # 2.0 mm and the glass leg is 6.0 * 1.7 = 10.2 mm.
    traced = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=2)
    accumulated = _host(traced.opd)
    predicted_optical = 2.0 + 6.0 * 1.7
    predicted_geometric = 2.0 + 6.0
    assert float(accumulated.min()) == pytest.approx(predicted_optical, abs=0.0)
    # The geometric hypothesis is short by 4.2 mm -- 7600 waves, so the two are not
    # a tolerance apart. A case whose falsifier sits inside round-off discriminates
    # nothing.
    assert abs(predicted_optical - predicted_geometric) == pytest.approx(4.2, abs=1e-12)
    assert abs(predicted_optical - predicted_geometric) / (WAVELENGTH_UM * 1e-3) > 1.0e3


def test_the_accumulator_zero_moves_with_the_aperture() -> None:
    """The reference is an aimed launch plane, not the first surface.

    Three apertures over the same 10 mm gap. If the zero were the first surface,
    every accumulated value would be 10.0; instead each is `10 + EPD`, so the
    reference *moves* -- which is exactly why an absolute accumulated path is not
    a declared physical quantity and why `declare_optical_path_m` removes a piston
    from the same trace rather than trusting the zero.
    """
    for epd_mm in (2.0, 4.0, 7.5):
        lens = build_lens(_plate(thickness_mm=10.0, index=1.0, epd_mm=epd_mm))
        traced = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=3)
        accumulated = _host(traced.opd)
        assert float(accumulated.min()) == pytest.approx(10.0 + epd_mm, abs=0.0)
        assert float(np.ptp(accumulated)) == pytest.approx(0.0, abs=0.0), (
            "a collimated on-axis bundle through plane surfaces has one path length"
        )


def test_the_accumulator_scales_with_the_geometry_unit() -> None:
    """A 10x geometry gives a 10x path, so the unit is the prescription's own.

    `NATIVE_LENGTH_M` is the whole content of the metre boundary, and this is the
    measurement that fixes it: nothing in the solver declares millimetres, the
    geometry does.
    """
    small = build_lens(_plate(thickness_mm=1.0, index=1.5, epd_mm=0.5))
    large = build_lens(_plate(thickness_mm=10.0, index=1.5, epd_mm=5.0))
    ratios = [
        float(_host(large.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=n).opd).min())
        / float(_host(small.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=n).opd).min())
        for n in (2, 3, 4)
    ]
    assert ratios == pytest.approx([10.0, 10.0, 10.0], abs=0.0)
    assert NATIVE_LENGTH_M == 1.0e-3


def test_a_paraxial_surface_would_not_have_served_as_an_oracle() -> None:
    """Why the reference system is built from real refractive surfaces.

    Recorded rather than re-derived, because the *consequence* is what matters
    here: through an ideal paraxial surface the OPL at the focus is not flat, so a
    reconstructed wavefront would carry a spurious quadratic term -- a defocus
    indistinguishable from a real one. This test asserts the fixture is built the
    way that finding requires: real curved refractive surfaces and a
    catalog-independent ideal index.
    """
    problem = singlet_ref()
    curved = [s for s in problem.surfaces if not s.is_plane]
    assert len(curved) == 1 and curved[0].radius_mm is not None
    assert curved[0].material["kind"] == "ideal"
    source = Path(str(rays_module.__file__)).read_text(encoding="utf-8")
    assert "paraxial" in source, (
        "the module reads the paraxial solver for the pupil, which is a different "
        "thing from tracing through a paraxial surface"
    )


# ---------------------------------------------------------------------------
# 2. The declared path, on axis, against the analytic sphere
# ---------------------------------------------------------------------------


def _declared_pupil_opl(
    problem: RayTraceProblem,
    *,
    field_deg: tuple[float, float],
    num_rings: int,
    sign: int = 1,
    apply_object_space: bool = True,
) -> dict[str, object]:
    """The declared pupil OPL, with the two terms individually switchable.

    Deliberately exercises the shipping functions with one term altered, rather
    than a hand-written parallel copy that could drift away from what ships. That
    is what makes the negative controls below controls on *this* code.
    """
    lens = build_lens(problem)
    max_field = float(lens.fields.max_field)
    field = (
        (0.0, 0.0)
        if max_field == 0.0
        else (field_deg[0] / max_field, field_deg[1] / max_field)
    )
    traced = lens.trace(
        Hx=field[0], Hy=field[1], wavelength=WAVELENGTH_UM, num_rays=num_rings
    )
    x, y, z = _host(traced.x), _host(traced.y), _host(traced.z)
    direction_z = _host(traced.N)
    image_z_mm = float(_host(lens.surfaces.surfaces[-1].geometry.cs.z))
    pupil = exit_pupil(lens, image_plane_z_mm=image_z_mm)

    step_mm = (pupil["z_mm"] - z) / direction_z
    pupil_x_mm = x + _host(traced.L) * step_mm
    pupil_y_mm = y + _host(traced.M) * step_mm

    object_space = rays_module._object_space_reference(
        lens,
        field=field,
        wavelength_um=WAVELENGTH_UM,
        num_rings=num_rings,
        traced_count=int(x.size),
    )
    if not apply_object_space:
        # The negative control for the CHE-41 term: present, measured, and *not*
        # added. Spelled as a zero span rather than as `available=False`, because
        # an unavailable term off axis is a refusal and this control needs the
        # arithmetic to run.
        object_space = {**object_space, "span_native": 0.0}

    declared, reference = declare_optical_path_m(
        _host(traced.opd),
        direction_z=direction_z,
        pupil_radius_m=np.hypot(pupil_x_mm, pupil_y_mm) * NATIVE_LENGTH_M,
        image_z_mm=image_z_mm,
        plane_z_mm=pupil["z_mm"],
        image_space_index=1.0,
        object_space=object_space,
        on_axis=field == (0.0, 0.0),
    )
    return {
        "opl_m": float(sign) * declared,
        "reference": reference,
        "pupil_x_m": pupil_x_mm * NATIVE_LENGTH_M,
        "pupil_y_m": pupil_y_mm * NATIVE_LENGTH_M,
        # Pupil to image surface, which is the radius of the reference sphere for a
        # bundle converging on the image point.
        "sphere_radius_m": (image_z_mm - pupil["z_mm"]) * NATIVE_LENGTH_M,
    }


def _sphere_residual_waves(declared: dict[str, object]) -> np.ndarray:
    """`OPL(rho) - OPL(0)` against `R - sqrt(rho^2 + R^2)`, in waves.

    The oracle uses nothing from this repository: for a diffraction-limited system
    every ray reaches the focus with equal total optical path, so the pupil OPL of
    a perfect bundle *is* the sag of the sphere centred on the focus.
    """
    radius_m = np.hypot(declared["pupil_x_m"], declared["pupil_y_m"])
    sphere_radius = declared["sphere_radius_m"]
    oracle = sphere_radius - np.sqrt(radius_m**2 + sphere_radius**2)
    return (np.asarray(declared["opl_m"]) - oracle) / WAVELENGTH_M


def test_declared_pupil_opl_matches_the_analytic_sphere_on_axis() -> None:
    """The positive gate: 0.017 waves P-V, which is the singlet's own aberration."""
    declared = _declared_pupil_opl(singlet_ref(), field_deg=(0.0, 0.0), num_rings=16)
    residual = _sphere_residual_waves(declared)
    peak_to_valley = float(np.ptp(residual))
    assert peak_to_valley == pytest.approx(SPHERE_RESIDUAL_WAVES, abs=5.0e-4), (
        f"declared pupil OPL is {peak_to_valley:.6f} waves P-V from the analytic "
        f"diffraction-limited sphere; the measured value is 0.016999"
    )


def test_the_declared_reference_names_its_version_and_its_terms() -> None:
    """A declaration a reviewer can check, not a label."""
    declared = _declared_pupil_opl(singlet_ref(), field_deg=(0.0, 0.0), num_rings=8)
    reference = str(declared["reference"])
    assert reference.startswith("optiland-declared-opl/v2")
    assert "ray minus chief" in reference
    assert "Removed piston" in reference
    # CHE-207 made the reference *surface* a named value rather than a fixed
    # sentence, because a finite conjugate is referenced to a sphere and not to a
    # plane. This system is collimated, so it must name the plane.
    assert LAUNCH_PLANE_WAVEFRONT in reference
    assert LAUNCH_POINT_SOURCE not in reference


def test_the_removed_piston_is_large_and_the_signal_is_not() -> None:
    """Why the conditioning is required rather than stylistic.

    On this system the absolute accumulated path is ~1.0e4 waves and the piston
    removed is ~1.2e3 waves, against ~12 waves of actual pupil wavefront. Forming
    a phase from the absolute value spends the float budget on a quantity no
    single-path PSF can see.
    """
    declared = _declared_pupil_opl(singlet_ref(), field_deg=(0.0, 0.0), num_rings=16)
    piston_m = float(str(declared["reference"]).split("Removed piston ")[1].split(" m")[0])
    signal_waves = float(np.ptp(np.asarray(declared["opl_m"]))) / WAVELENGTH_M
    assert abs(piston_m) / WAVELENGTH_M > 1.0e3
    assert 10.0 < signal_waves < 20.0
    assert abs(piston_m) / WAVELENGTH_M / signal_waves > 50.0


# ---------------------------------------------------------------------------
# 3. Off axis: the field-tilt term, which is undetectable on axis
# ---------------------------------------------------------------------------


def _tilt_slope(declared: dict[str, object]) -> float:
    """Least-squares `d(OPL)/d(pupil y)`, dimensionless."""
    y = np.asarray(declared["pupil_y_m"])
    design = np.vstack([y, np.ones_like(y)]).T
    return float(np.linalg.lstsq(design, np.asarray(declared["opl_m"]), rcond=None)[0][0])


def test_the_field_tilt_term_is_present_and_is_the_convergence_tilt() -> None:
    """Off axis the CHE-41 term IS the tilt that aims the wave at the image point.

    `Hy = 0.2` on `M3-REVERSE-TELEPHOTO`, i.e. 6 deg of a 30 deg system -- the
    field the off-axis evidence was taken at.
    """
    declared = _declared_pupil_opl(REVERSE_TELEPHOTO, field_deg=(0.0, 6.0), num_rings=32)
    slope = _tilt_slope(declared)
    assert slope == pytest.approx(REQUIRED_TILT_SLOPE, rel=TILT_SLOPE_RELATIVE), (
        f"the declared off-axis OPL carries slope {slope:.9e} against the geometric "
        f"requirement {REQUIRED_TILT_SLOPE:.9e}"
    )
    assert "applied: the term varies across the bundle" in str(declared["reference"])


def test_omitting_the_field_tilt_term_loses_almost_all_of_it() -> None:
    """The negative control for CHE-41, and the reason it survived three tickets.

    Without the term the declared OPL retains 0.13% of the required tilt -- a
    slope of 8.7e-5 against 0.0684 -- and what is left looks like a perfectly
    healthy converging sphere, just aimed at the axis instead of at the image
    point.
    """
    omitted = _declared_pupil_opl(
        REVERSE_TELEPHOTO, field_deg=(0.0, 6.0), num_rings=32, apply_object_space=False
    )
    slope = _tilt_slope(omitted)
    assert abs(slope / REQUIRED_TILT_SLOPE) < OMITTED_TERM_SLOPE_FRACTION * 1.1
    assert abs(slope / REQUIRED_TILT_SLOPE) > OMITTED_TERM_SLOPE_FRACTION * 0.9, (
        "the control must reproduce the measured 0.13%, not merely be small"
    )


def test_the_term_is_exactly_invisible_on_axis() -> None:
    """Bit-identical with and without, which is the whole reason it was missed.

    Not "close": the term is constant across an on-axis bundle, so
    `declare_optical_path_m` does not add it at all and the declared path is the
    same object. That is a deliberate policy, not an accident -- adding a piston
    that step 4 removes would only spend float precision.
    """
    applied = _declared_pupil_opl(REVERSE_TELEPHOTO, field_deg=(0.0, 0.0), num_rings=16)
    omitted = _declared_pupil_opl(
        REVERSE_TELEPHOTO, field_deg=(0.0, 0.0), num_rings=16, apply_object_space=False
    )
    np.testing.assert_array_equal(
        np.asarray(applied["opl_m"]), np.asarray(omitted["opl_m"])
    )
    assert "pure piston, not applied" in str(applied["reference"])


# ---------------------------------------------------------------------------
# 3b. The native accumulator is not an optical path
# ---------------------------------------------------------------------------


def _bundle_with_optical_path(
    optical_path_m: np.ndarray, reference: str | None
) -> RayBundle:
    """A minimal bundle carrying a given optical path and reference. Two rays."""
    return RayBundle(
        positions_m=np.array([[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0]]),
        directions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="exit_pupil", z_m=0.0, medium_index=1.0),
        frame=Frame(),
        amplitude=np.array([1.0, 1.0]),
        optical_path_m=optical_path_m,
        optical_path_reference=reference,
    )


def test_the_native_accumulator_scaled_to_metres_is_refused_as_an_optical_path() -> None:
    """R05.2 acceptance criterion 5, with the real accumulator values.

    The refused array is the singlet's own `opd` in metres -- correct units,
    plausible magnitude, and a spread that looks like a wavefront. It carries no
    declared reference, and that is the whole difference between a physical
    quantity and a solver-internal accumulator whose zero moves with the aperture.
    """
    lens = build_lens(singlet_ref())
    traced = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=1)
    accumulator_m = _host(traced.opd)[:2] * NATIVE_LENGTH_M
    assert accumulator_m.min() > 0.0, "the values are a real accumulated path"

    with pytest.raises(ContractError) as excinfo:
        require_declared_optical_path(
            _bundle_with_optical_path(accumulator_m, "opd_native")
        )
    assert excinfo.value.code == "OPL_REFERENCE_UNVERIFIED"
    assert "native accumulator" in str(excinfo.value)
    assert "moves with the aperture" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 3c. The composed path of a SUPPLIED bundle. CHE-217 (R05.6).
#
# The convention above is for a path this solver declared from end to end.
# `trace_rays` declares a different quantity -- the incoming bundle's own path
# plus this trace's increment -- and it is checked here, in the file that owns the
# accumulator convention, against the same manufactured plate that gave the
# original convention a closed-form answer. The composition is where a
# plausible-looking implementation of R05.6 is wrong in a way no intensity check
# can see, so it is measured rather than reasoned about.
# ---------------------------------------------------------------------------

EXECUTION = {"device": "cpu", "precision": "fp64"}


def _supplied_bundle(
    *, axial_cosines: tuple[float, ...], offsets_m: tuple[float, ...]
) -> RayBundle:
    """Rays on the plate's front surface, each with its own tilt and its own path.

    The tilt is in `x` only, so `d = (sqrt(1 - dz^2), 0, dz)` and the arc to a
    plane at axial offset `L` is exactly `L / dz`. `offsets_m` is each ray's
    incoming optical path, distinct per ray so an implementation that replaced it
    with a constant -- or dropped it -- cannot pass.
    """
    count = len(axial_cosines)
    assert len(offsets_m) == count
    axial = np.asarray(axial_cosines, dtype=np.float64)
    return RayBundle(
        positions_m=np.zeros((count, 3)),
        directions=np.column_stack([np.sqrt(1.0 - axial**2), np.zeros(count), axial]),
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(
            name="plate front", z_m=0.0, medium_index=1.0
        ),
        frame=Frame(),
        amplitude=np.linspace(0.3, 1.7, count) * np.exp(1j * np.linspace(0.1, 1.1, count)),
        optical_path_m=np.asarray(offsets_m, dtype=np.float64),
        optical_path_reference="zero at the plate front",
        measure_weight=np.linspace(1.0, 3.0, count),
        measure_kind="importance_weight",
    )


def test_the_composed_path_is_the_incoming_path_plus_the_closed_form_arc() -> None:
    """An all-air plate: no refraction anywhere, so the arc is `L / dz` exactly.

    Every ingredient of the composition is exercised at once and each has a
    closed form: the incoming path is distinct per ray, the increment is a
    per-ray arc length rather than a constant, and the unit conversion is the one
    that scales `k * OPL` by a thousand when it is wrong.
    """
    thickness = 3.0
    axial = (1.0, 0.99, 0.95, 0.8)
    offsets = (0.0, 1.0e-3, -2.0e-4, 5.0e-3)
    bundle = _supplied_bundle(axial_cosines=axial, offsets_m=offsets)

    traced = trace_rays(_plate(thickness_mm=thickness, index=1.0), bundle, execution=EXECUTION)

    length_m = thickness * NATIVE_LENGTH_M
    expected = np.asarray(offsets) + length_m / np.asarray(axial)
    assert traced.optical_path_m == pytest.approx(expected, rel=0.0, abs=1.0e-18)
    # And the increment alone is the arc, so the addition is an addition.
    assert traced.optical_path_m - bundle.optical_path_m == pytest.approx(
        length_m / np.asarray(axial), rel=0.0, abs=1.0e-18
    )


def test_the_composed_increment_is_index_weighted_not_geometric() -> None:
    """The plate in glass, axial rays: the increment is `n * L`, and `n != 1` matters.

    The falsifiable twin of the test above. An implementation that added the
    geometric distance instead of the optical one is right in air and wrong
    everywhere else -- including inside a lens, which is where a sequential trace
    actually uses this. On this plate the two differ by 52%, so the assertion
    discriminates rather than merely passes.
    """
    thickness = 3.0
    index = 1.5168
    offsets = (0.0, 4.0e-4, -1.0e-3)
    bundle = _supplied_bundle(axial_cosines=(1.0, 1.0, 1.0), offsets_m=offsets)

    traced = trace_rays(
        _plate(thickness_mm=thickness, index=index), bundle, execution=EXECUTION
    )

    length_m = thickness * NATIVE_LENGTH_M
    increment = traced.optical_path_m - bundle.optical_path_m
    assert increment == pytest.approx(index * length_m, rel=0.0, abs=1.0e-18)
    # The negative control: the geometric distance is not the answer.
    assert not np.allclose(increment, length_m, atol=1.0e-6)


def test_the_composed_reference_is_its_own_version_and_names_the_incoming_one() -> None:
    """A composition is not the absolute declaration, and must not be labelled as one.

    `declare_optical_path_m`'s reference promises a chief-ray-zeroed absolute path
    measured from a named object-space wavefront. A composed path promises the
    incoming bundle's zero plus an increment. Carrying the first label on the
    second would tell a consumer something false about where the zero is, and
    `require_coherent()` cannot tell them apart -- so the two prefixes are
    distinct, and the composed one quotes what it was composed onto.
    """
    bundle = _supplied_bundle(axial_cosines=(1.0, 0.98), offsets_m=(0.0, 1.0e-4))
    traced = trace_rays(_plate(thickness_mm=2.0, index=1.0), bundle, execution=EXECUTION)

    reference = traced.optical_path_reference
    assert reference is not None
    assert reference.startswith(COMPOSED_OPL_REFERENCE_VERSION)
    assert not reference.startswith(OPL_REFERENCE_VERSION)
    assert bundle.optical_path_reference in reference
    # Admissible, because the vocabulary was extended by enumeration.
    require_declared_optical_path(traced)
    traced.require_coherent()


@pytest.mark.parametrize(
    "reference",
    [UNVERIFIED, "the launch plane", "accumulated optical path, mm -> m", "v2"],
)
def test_only_this_solvers_own_declaration_is_admissible(reference: str) -> None:
    """A plausible label is not a declaration. The prefix is the whole check."""
    with pytest.raises(ContractError) as excinfo:
        require_declared_optical_path(
            _bundle_with_optical_path(np.array([0.0, 1.0e-6]), reference)
        )
    assert excinfo.value.code == "OPL_REFERENCE_UNVERIFIED"


def test_a_bundle_with_no_optical_path_at_all_passes() -> None:
    """Carrying no phase is honest; `require_coherent()` is what refuses to read one."""
    bundle = RayBundle(
        positions_m=np.array([[0.0, 0.0, 0.0]]),
        directions=np.array([[0.0, 0.0, 1.0]]),
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="exit_pupil", z_m=0.0, medium_index=1.0),
    )
    require_declared_optical_path(bundle)
    with pytest.raises(ContractError) as excinfo:
        bundle.require_coherent()
    assert excinfo.value.code == "COHERENT_STATE_INCOMPLETE"


def test_every_emitted_bundle_satisfies_the_post_condition() -> None:
    """The check `to_ray_bundle` runs on its own output, run again from outside."""
    lens = build_lens(singlet_ref())
    traced = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=8)
    bundle, _ = to_ray_bundle(
        lens,
        traced,
        field=(0.0, 0.0),
        wavelength_um=WAVELENGTH_UM,
        num_rings=8,
        reference_surface="exit_pupil",
    )
    require_declared_optical_path(bundle)
    assert str(bundle.optical_path_reference).startswith(OPL_REFERENCE_VERSION)
    # And it is genuinely not the accumulator: the piston is gone, so the chief
    # ray reads exactly zero where the accumulator reads 1.0e4 waves.
    assert float(np.min(np.abs(bundle.optical_path_m))) == 0.0
    accumulator_m = _host(traced.opd) * NATIVE_LENGTH_M
    assert float(np.min(accumulator_m)) / WAVELENGTH_M > 1.0e3


def test_off_axis_without_the_term_is_refused_not_defaulted() -> None:
    """The term cannot be reconstructed downstream, so its absence is a refusal.

    The exported pupil arrays carry no object-space coordinate: the missing
    quantity is linear in the *launch* coordinate. Declaring the path without it
    produces a clean converging sphere aimed at the axis, 209 um from the traced
    chief-ray intersection, with a 0.072-wave-P-V residual that looks healthy.
    """
    from representations import ContractError

    with pytest.raises(ContractError) as excinfo:
        declare_optical_path_m(
            np.array([1.0, 2.0]),
            direction_z=np.array([1.0, 1.0]),
            pupil_radius_m=np.array([0.0, 1.0e-3]),
            image_z_mm=5.0,
            plane_z_mm=5.0,
            image_space_index=1.0,
            object_space={
                "available": False,
                "reason": "manufactured for this test",
                "offset_native": None,
                "span_native": None,
            },
            on_axis=False,
        )
    assert excinfo.value.code == "MISSING_DECLARATION"
    assert "off axis" in str(excinfo.value)


def test_on_axis_without_the_term_is_accepted_and_says_why() -> None:
    """An on-axis bundle travels along z, so the launch plane IS a wavefront of it."""
    declared, reference = declare_optical_path_m(
        np.array([1.0, 2.0]),
        direction_z=np.array([1.0, 1.0]),
        pupil_radius_m=np.array([0.0, 1.0e-3]),
        image_z_mm=5.0,
        plane_z_mm=5.0,
        image_space_index=1.0,
        object_space={
            "available": False,
            "reason": "manufactured for this test",
            "offset_native": None,
            "span_native": None,
        },
        on_axis=True,
    )
    assert declared.tolist() == [0.0, 1.0e-3]
    assert "Accepted for this field only" in reference


# ---------------------------------------------------------------------------
# 4. The negative control on the sign
# ---------------------------------------------------------------------------


def test_the_inverted_sign_fails_the_analytic_gate_by_three_orders() -> None:
    """A parity gate with no falsifiable twin proves nothing.

    Inverting the sign conjugates the wavefront: the residual against the analytic
    sphere goes from 0.017 waves to 23.5, a factor of about 1380. Every intensity
    check in the project would pass either way, which is the whole argument for
    declaring the sign rather than defaulting it.
    """
    correct = _sphere_residual_waves(
        _declared_pupil_opl(singlet_ref(), field_deg=(0.0, 0.0), num_rings=16)
    )
    inverted = _sphere_residual_waves(
        _declared_pupil_opl(singlet_ref(), field_deg=(0.0, 0.0), num_rings=16, sign=-1)
    )
    correct_pv, inverted_pv = float(np.ptp(correct)), float(np.ptp(inverted))
    assert correct_pv == pytest.approx(SPHERE_RESIDUAL_WAVES, abs=5.0e-4)
    assert inverted_pv == pytest.approx(23.45, abs=0.1)
    assert inverted_pv / correct_pv > 1.0e3


def test_the_conjugate_is_invisible_in_an_intensity() -> None:
    """Why the sign has to be declared: `|U|^2` cannot see it.

    Stated as a property of the amplitudes the declared path produces, over a real
    trace, rather than as a general remark. If an intensity could distinguish them
    the refusal in `require_coherent` would be unnecessary.
    """
    declared = _declared_pupil_opl(singlet_ref(), field_deg=(0.0, 0.0), num_rings=8)
    phase = 2.0 * math.pi / WAVELENGTH_M * np.asarray(declared["opl_m"])
    forward = np.exp(1j * phase)
    conjugate = np.exp(-1j * phase)
    np.testing.assert_allclose(np.abs(forward) ** 2, np.abs(conjugate) ** 2, rtol=0.0, atol=0.0)
    assert float(np.max(np.abs(forward - conjugate))) > 0.5, (
        "the two fields are genuinely different; only the intensity is blind to it"
    )
