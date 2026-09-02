"""One generic construction path, and the surface table it produces.

CHE-179 (R05.1), acceptance criteria:

1. one generic construction function -- adding a system means handing it a
   different `OpticalSetup`, never writing a builder;
2. surface tables built from this path match the reference **exactly** on the
   fixture systems, compared as executed output rather than as source;
3. no Optiland type, unit or API concept in the signature;
4. class delta 0 -- `test_class_budget.py` owns the count; the absence of the
   named old classes is asserted here.

The oracle for criterion 2
--------------------------
`optiland.samples.objectives.ReverseTelephoto` -- the solver's *own* bundled
sample, which is where the fixture's prescription was transcribed from and which
this repository validated in M1. `Optic.to_dict()` is a usable structural oracle:
it is stable across independent builds of the same prescription
(`pre-rewrite-2026-08-30:benchmarks/probes/records/optiland/system_construction_probe.json`,
case `to_dict_is_a_usable_structural_oracle`). Comparing against the bundled
sample rather than against the deleted builder is the stronger check of the two:
it is a genuinely independent construction, whereas the old builder shares this
one's reasoning.

The mm boundary
---------------
The ticket's named risk is metre-for-millimetre, which scales `k * OPL` by 1000.
It is cheap to detect if any case has a large optical path and cheap to miss if
none does, so `test_geometry_scale_is_millimetres` pins a case whose accumulated
path is 1.0e4 waves.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from fixtures.systems import (
    REVERSE_TELEPHOTO,
    SINGLET_BACK_FOCAL_LENGTH_MM,
    SINGLET_CENTER_THICKNESS_MM,
    SINGLET_RADIUS_MM,
    SINGLET_REFRACTIVE_INDEX,
    singlet_ref,
)

from problems.ray_trace import UNAPERTURED, Material, OpticalSetup, SourceSpec, SurfaceSpec
from solvers.optiland import system as system_module
from solvers.optiland.system import NATIVE_UNITS, build_lens

ROOT = Path(__file__).resolve().parents[2]

#: Class names the reference implementation used for this job. None of them may
#: exist in the new package: a budget records what landed and cannot record what
#: was avoided.
AVOIDED_NAMES = (
    "OptilandBuilder",
    "SystemBuilder",
    "SingletBuilder",
    "ReverseTelephotoBuilder",
    "PrescriptionError",
    "_resolve_lens",
)


def _host(value: object) -> np.ndarray:
    import optiland.backend.utils as be_utils

    return np.asarray(be_utils.to_numpy(value))


# ---------------------------------------------------------------------------
# 1. One generic path
# ---------------------------------------------------------------------------


def test_one_generic_construction_function() -> None:
    """`build_lens` is the whole construction surface, and it takes a setup."""
    public = [
        name
        for name in dir(system_module)
        if not name.startswith("_")
        and callable(getattr(system_module, name))
        # Defined here, not imported here: `OpticalSetup` is callable and is the
        # argument type, which is the opposite of a second construction path.
        and getattr(getattr(system_module, name), "__module__", None) == system_module.__name__
    ]
    assert public == ["build_lens"], (
        "system.py must expose exactly one callable: the generic construction path. "
        f"Found {public}."
    )


def test_no_per_system_builder_and_no_name_resolution() -> None:
    """No builder class, and nothing turns a name into a lens."""
    source = (ROOT / "src" / "solvers" / "optiland" / "system.py").read_text(encoding="utf-8")
    for name in AVOIDED_NAMES:
        assert f"class {name}" not in source
        assert f"def {name}" not in source


def test_two_systems_one_function() -> None:
    """The same function builds both fixtures. That is the whole of criterion 1."""
    singlet = build_lens(singlet_ref())
    telephoto = build_lens(REVERSE_TELEPHOTO)
    # object + listed + image
    assert len(singlet.surfaces.surfaces) == 2 + 2
    assert len(telephoto.surfaces.surfaces) == 13 + 2


def test_refuses_something_that_is_not_a_setup() -> None:
    with pytest.raises(TypeError, match="OpticalSetup"):
        build_lens({"name": "M3SingletRef"})  # type: ignore[arg-type]


def test_refuses_a_second_argument_that_is_not_a_source() -> None:
    """CHE-218: the second argument is a declaration, never physical state.

    A `RayBundle` is a source at the *trace's* argument position, not at this one.
    Passing one here would mean construction had something to derive from it, and
    it does not: an already-materialized bundle needs no object surface placed and
    no field aimed at.
    """
    with pytest.raises(TypeError, match="SourceSpec or None"):
        build_lens(singlet_ref(), {"wavelength_um": 0.55})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. The surface table, against an independent oracle
# ---------------------------------------------------------------------------


def _without_comments(state: dict) -> dict:
    """The surface table and the aperture, with surface comments dropped.

    A comment is prescription annotation and changes no geometry, no material and
    no traced ray. The fixture carries `"plane aperture stop"` where the bundled
    sample carries `""`, which is the only difference between the two tables and
    the only one this helper hides -- everything else is compared verbatim.

    CHE-218 (R05.7) narrowed this from the whole `to_dict()` to the surface group
    and the aperture, and the narrowing is the point rather than a concession: the
    field set and the wavelength set are no longer system properties, so a bundled
    sample that declares three of each and a built lens that declares the one being
    traced *cannot* agree on those keys and should not be asked to.
    `test_the_declared_field_and_wavelength_are_the_construction_argument` asserts
    that difference deliberately, and the geometric oracle below is unweakened:
    every surface, radius, thickness, conic, material and stop flag is still
    compared element for element.
    """
    return {
        "surface_group": {
            **state["surface_group"],
            "surfaces": [
                {**surface, "comment": ""} for surface in state["surface_group"]["surfaces"]
            ],
        },
        "aperture": state["aperture"],
    }


def test_surface_table_matches_the_bundled_sample_exactly() -> None:
    """Executed output, element for element, against the solver's own sample."""
    from optiland.samples.objectives import ReverseTelephoto

    oracle = _without_comments(ReverseTelephoto().to_dict())
    built = _without_comments(build_lens(REVERSE_TELEPHOTO).to_dict())
    assert built == oracle


def test_the_declared_field_and_wavelength_are_the_construction_argument() -> None:
    """The difference `_without_comments` excludes, asserted rather than hidden.

    The bundled sample declares three fields and three wavelengths because that is
    how a prescription file was written. This builder declares exactly one of each
    -- the field being traced, and the setup's reference wavelength -- and the two
    therefore differ on those keys by design. Measured for CHE-218: the difference
    leaves `EPD`, `XPL` and `XPD` bitwise identical, so nothing about the *system*
    moved.
    """
    from optiland.samples.objectives import ReverseTelephoto

    oracle = ReverseTelephoto().to_dict()
    built = build_lens(
        REVERSE_TELEPHOTO, SourceSpec(wavelength_um=0.55, field_angle_deg=(0.0, 30.0))
    ).to_dict()
    assert len(oracle["fields"]["fields"]) == 3
    assert len(built["fields"]["fields"]) == 1
    assert len(oracle["wavelengths"]["wavelengths"]) == 3
    assert len(built["wavelengths"]["wavelengths"]) == 1

    sample = ReverseTelephoto()
    for read in ("EPD", "XPL", "XPD"):
        assert float(_host(getattr(sample.paraxial, read)()).ravel()[0]) == pytest.approx(
            float(
                _host(
                    getattr(
                        build_lens(
                            REVERSE_TELEPHOTO,
                            SourceSpec(wavelength_um=0.55, field_angle_deg=(0.0, 30.0)),
                        ).paraxial,
                        read,
                    )()
                ).ravel()[0]
            ),
            abs=0.0,
        ), f"{read} moved, so the declared sets were a system property after all"


def test_surface_table_is_deterministic_across_independent_builds() -> None:
    """No set iteration, no dict ordering, no RNG, no clock."""
    assert build_lens(REVERSE_TELEPHOTO).to_dict() == build_lens(REVERSE_TELEPHOTO).to_dict()


def test_singlet_surface_table_is_the_prescription() -> None:
    """The fixture's numbers arrive in the solver unconverted and in order."""
    lens = build_lens(singlet_ref())
    positions = _host(lens.surfaces.positions).ravel()
    assert math.isinf(positions[0]) and positions[0] < 0.0, "object at infinity sits at -inf"
    assert positions[1] == pytest.approx(0.0, abs=0.0)
    assert positions[2] == pytest.approx(SINGLET_CENTER_THICKNESS_MM, abs=0.0)
    assert positions[3] == pytest.approx(
        SINGLET_CENTER_THICKNESS_MM + SINGLET_BACK_FOCAL_LENGTH_MM, abs=0.0
    )
    front = lens.surfaces.surfaces[1]
    assert float(_host(front.geometry.radius).ravel()[0]) == SINGLET_RADIUS_MM
    assert front.is_stop is True
    rear = lens.surfaces.surfaces[2]
    assert math.isinf(float(_host(rear.geometry.radius).ravel()[0])), "a plane is radius = inf"
    assert float(_host(front.material_post.n(0.55)).ravel()[0]) == SINGLET_REFRACTIVE_INDEX


def test_aperture_field_and_wavelength_are_set_not_inherited() -> None:
    """CHE-218: exactly one field and one wavelength, and each from its own source.

    The field comes from the `SourceSpec` -- what the caller asked to trace -- and
    the wavelength from the setup's reference, which is what the backend takes as
    primary and evaluates the exit pupil at. Before R05.7 both were lists on the
    system record, and `max_field` was the largest of a set the caller had to
    declare in advance.
    """
    lens = build_lens(
        REVERSE_TELEPHOTO, SourceSpec(wavelength_um=0.55, field_angle_deg=(0.0, 21.0))
    )
    state = lens.to_dict()
    assert state["aperture"] == {
        "type": "EPD",
        "value": REVERSE_TELEPHOTO.entrance_pupil_diameter_mm,
    }
    assert tuple((f["x"], f["y"]) for f in state["fields"]["fields"]) == ((0.0, 21.0),)
    assert lens.fields.max_field == 21.0, "the declared field IS the maximum field"
    assert all(
        f["vx"] == 0.0 and f["vy"] == 0.0 and f["weight"] == 1.0
        for f in state["fields"]["fields"]
    )
    wavelengths = state["wavelengths"]["wavelengths"]
    assert len(wavelengths) == 1
    assert wavelengths[0]["is_primary"] is True
    # The SETUP's reference, not the source's 0.55: the primary is what
    # `paraxial.XPL()`/`XPD()` are evaluated at, and that is a property of the
    # system's characterization rather than of the light being traced.
    assert wavelengths[0]["value"] == REVERSE_TELEPHOTO.reference_wavelength_um
    assert {w["unit"] for w in wavelengths} == {"um"}


def test_with_no_source_the_axis_is_declared_and_the_object_is_at_infinity() -> None:
    """`source=None` is an absence, not a defaulted illumination.

    It is the R05.6 supplied-bundle path: the object surface is skipped and no
    field is aimed at, so what is built here is only what the backend requires
    before an `Optic` exists at all.
    """
    lens = build_lens(REVERSE_TELEPHOTO)
    state = lens.to_dict()
    assert tuple((f["x"], f["y"]) for f in state["fields"]["fields"]) == ((0.0, 0.0),)
    assert lens.fields.max_field == 0.0
    assert math.isinf(float(_host(lens.surfaces.positions).ravel()[0]))


# ---------------------------------------------------------------------------
# 3. Units, and the metre-for-millimetre risk
# ---------------------------------------------------------------------------


def test_declared_units_are_checked_not_assumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A schema whose radius stopped being millimetres fails before construction."""
    monkeypatch.setitem(system_module.UNITS, "radius", "m")
    with pytest.raises(ValueError, match="no longer match Optiland's native units"):
        build_lens(singlet_ref())


def test_native_units_table_matches_the_problem_schema() -> None:
    """The table this module checks against is the schema's, entry for entry."""
    from problems.ray_trace import UNITS

    assert {name: UNITS[name] for name in NATIVE_UNITS} == NATIVE_UNITS


def test_geometry_scale_is_millimetres() -> None:
    """A case with a large optical path, which is where mm-for-m is detectable.

    The singlet's accumulated path at the image surface is 5.5077 mm, i.e. 1.0e4
    waves at 550 nm. Read as metres it would be 5.5 m, 1.0e7 waves; read as
    millimetres-that-are-really-metres the reconstructed phase would wrap a
    thousand times more often than the geometry allows. The assertion is on the
    native value, because that is the number the metre boundary in `rays.py`
    scales.
    """
    lens = build_lens(singlet_ref())
    traced = lens.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=16)
    opd_native = _host(traced.opd)
    assert float(opd_native.min()) == pytest.approx(5.5076721107725914, abs=0.0)
    waves = float(opd_native.min()) * 1.0e-3 / 0.55e-6
    assert pytest.approx(waves, rel=0.01) == 1.0e4, (
        "the case must carry a large k*OPL for the unit error to be visible at all"
    )


# ---------------------------------------------------------------------------
# 4. Catalog materials: the guards, and what they refuse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("glass", "catalog", "expected_file"),
    [
        ("N-SK10", None, "glass/schott/N-SK10.yml"),
        ("SK15", None, "glass/hikari/SK15.yml"),
        ("BASF2", None, "glass/hikari/BASF2.yml"),
        ("FK3", None, "glass/schott/FK3.yml"),
        ("SF15", "hikari", "glass/hikari/SF15.yml"),
        ("N-LAK12", None, "glass/schott/N-LAK12.yml"),
    ],
)
def test_catalog_glasses_resolve_to_the_recorded_row(
    glass: str, catalog: str | None, expected_file: str
) -> None:
    """The winning manufacturer is not implied by the name, so it is recorded.

    `SK15` survives the solver's substring filter as seven rows and resolves to
    HIKARI while `N-SK10` resolves to SCHOTT
    (`system_construction_probe.json`, case
    `catalog_names_resolve_to_one_exact_match`).
    """
    problem = _one_glass_setup(glass, catalog, expected_file)
    lens = build_lens(problem)
    row = lens.surfaces.surfaces[1].material_post.material_data
    assert str(row["filename"]) == expected_file
    assert float(row["similarity_score"]) == 0.0


def test_a_glass_that_resolves_to_a_different_file_is_refused() -> None:
    """A material-database change becomes an error, not a different trace."""
    problem = _one_glass_setup("N-SK10", None, "glass/hikari/N-SK10.yml")
    with pytest.raises(ValueError, match="not the recorded"):
        build_lens(problem)


def test_an_inexact_glass_name_is_refused() -> None:
    """The pinned lookup ranks near-misses; this refuses them.

    `N-SK1` is a plausible typo of the fixture's `N-SK10` and is not a catalog
    entry. The lookup nonetheless returns the best surviving row -- measured:
    `glass/schott/N-SK10.yml` at similarity 1, with only a `Warning:` printed to
    stdout -- so without this guard the trace would run on a real but *different*
    glass and report success.
    """
    problem = _one_glass_setup("N-SK1", None, None)
    with pytest.raises(ValueError, match="by similarity"):
        build_lens(problem)


def test_an_unknown_glass_is_refused_as_a_problem_not_a_crash() -> None:
    """No substring survives, so the solver raises a bare ValueError. It reaches
    the caller as a statement about the problem rather than as a solver crash."""
    problem = _one_glass_setup("NOT-A-GLASS-AT-ALL", None, None)
    with pytest.raises(ValueError, match="could not be resolved"):
        build_lens(problem)


def _one_glass_setup(
    glass: str, catalog: str | None, expected_file: str | None
) -> OpticalSetup:
    material: Material = {
        "kind": "catalog",
        "name": glass,
        "catalog": catalog,
        "expected_catalog_file": expected_file,
    }
    return OpticalSetup(
        name=f"one-glass-{glass}",
        surfaces=(
            SurfaceSpec(radius_mm=50.0, thickness_mm=5.0, material=material),
            SurfaceSpec(thickness_mm=45.0),
        ),
        entrance_pupil_diameter_mm=10.0,
        stop_index=0,
        reference_wavelength_um=0.5876,
    )


# ---------------------------------------------------------------------------
# 5. Even aspheres (CHE-207)
# ---------------------------------------------------------------------------

#: The manufactured aspheric surface the reference probe characterized, and the
#: sag it measured. From
#: `pre-rewrite-2026-08-30:benchmarks/probes/records/optiland/system_construction_probe.json`,
#: case `even_asphere_sag_matches_analytic`.
ASPHERE_RADIUS_MM = 10.0
ASPHERE_CONIC = -0.5
ASPHERE_COEFFICIENTS = (0.001, -2.5e-05, 4e-07)
ASPHERE_RADIAL_POSITIONS_MM = (0.0, 0.5, 1.0, 1.5, 2.0)
RECORDED_SAG_MM = (
    0.0,
    0.012752352443315268,
    0.051038056739996666,
    0.1149461923986582,
    0.20463572677666927,
)


def _analytic_sag_mm(
    radial_mm: np.ndarray, *, radius_mm: float, conic: float, coefficients: tuple[float, ...]
) -> np.ndarray:
    """`conic_sag(r) + sum_i c[i] r**(2(i+1))`, written out.

    The oracle is closed-form and owes nothing to this repository: the conic sag is
    the standard expression and the polynomial is the schema's declared series.
    """
    curvature = 1.0 / radius_mm
    squared = radial_mm**2
    conic_sag = (curvature * squared) / (
        1.0 + np.sqrt(1.0 - (1.0 + conic) * curvature**2 * squared)
    )
    polynomial = sum(
        value * radial_mm ** (2 * (index + 1)) for index, value in enumerate(coefficients)
    )
    return conic_sag + polynomial


def _asphere_setup(
    coefficients: tuple[float, ...], *, radius_mm: float | None = ASPHERE_RADIUS_MM
) -> OpticalSetup:
    """A one-surface aspheric problem. Radius `None` is an aspheric plate."""
    return OpticalSetup(
        name=f"asphere-{len(coefficients)}",
        surfaces=(
            SurfaceSpec(
                radius_mm=radius_mm,
                thickness_mm=5.0,
                conic=ASPHERE_CONIC,
                aspheric_coefficients=coefficients,
                material={"kind": "ideal", "refractive_index": 1.5},
            ),
            SurfaceSpec(thickness_mm=40.0),
        ),
        entrance_pupil_diameter_mm=4.0,
        reference_wavelength_um=0.55,
        stop_index=0,
    )


def test_an_aspheric_surface_builds_the_aspheric_geometry() -> None:
    """Criterion: the right Optiland type, selected from the schema alone."""
    lens = build_lens(_asphere_setup(ASPHERE_COEFFICIENTS))
    geometry = lens.surfaces.surfaces[1].geometry
    assert type(geometry).__name__ == "EvenAsphere"
    assert list(_host(geometry.coefficients).ravel()) == list(ASPHERE_COEFFICIENTS)


def test_the_aspheric_sag_matches_the_closed_form_and_the_recorded_values() -> None:
    """Both oracles: the analytic series, and the frozen probe's own numbers.

    The falsifier is the reason this is worth two assertions rather than one --
    reading the series as starting at `r**4` is wrong by 1.1e-2 mm here, four
    orders of magnitude above the 5.6e-17 agreement, so the case discriminates the
    convention rather than merely the arithmetic.
    """
    lens = build_lens(_asphere_setup(ASPHERE_COEFFICIENTS))
    radial = np.array(ASPHERE_RADIAL_POSITIONS_MM)
    observed = _host(lens.surfaces.surfaces[1].geometry.sag(radial, np.zeros_like(radial)))

    analytic = _analytic_sag_mm(
        radial,
        radius_mm=ASPHERE_RADIUS_MM,
        conic=ASPHERE_CONIC,
        coefficients=ASPHERE_COEFFICIENTS,
    )
    assert float(np.max(np.abs(observed - analytic))) < 1e-15
    np.testing.assert_allclose(observed, np.array(RECORDED_SAG_MM), rtol=0.0, atol=0.0)

    shifted = sum(
        value * radial ** (2 * (index + 2)) for index, value in enumerate(ASPHERE_COEFFICIENTS)
    )
    conic_only = _analytic_sag_mm(
        radial, radius_mm=ASPHERE_RADIUS_MM, conic=ASPHERE_CONIC, coefficients=()
    )
    assert float(np.max(np.abs(observed - (conic_only + shifted)))) > 1e-2, (
        "the r**4-start reading must be far from the claim, or the case proves nothing"
    )


def test_an_aspheric_plate_is_the_polynomial_alone() -> None:
    """A planar base plus aspheric terms, which `is_plane` would have got wrong."""
    coefficients = (1e-3, -2.5e-5)
    lens = build_lens(_asphere_setup(coefficients, radius_mm=None))
    geometry = lens.surfaces.surfaces[1].geometry
    assert type(geometry).__name__ == "EvenAsphere"
    radial = np.array(ASPHERE_RADIAL_POSITIONS_MM)
    observed = _host(geometry.sag(radial, np.zeros_like(radial)))
    polynomial = sum(
        value * radial ** (2 * (index + 1)) for index, value in enumerate(coefficients)
    )
    np.testing.assert_allclose(observed, polynomial, rtol=0.0, atol=0.0)


def test_a_zero_coefficient_polynomial_builds_the_standard_surface() -> None:
    """The choice that keeps the frozen collimated benchmarks bit-identical.

    And the measurement that says the choice is safe either way: the aspheric
    geometry with zeroed coefficients agrees with the standard surface **bitwise**
    in sag, so selecting `standard` is a guarantee rather than a compromise.
    """
    plain = build_lens(_asphere_setup(()))
    padded = build_lens(_asphere_setup((0.0, 0.0, 0.0)))
    assert type(plain.surfaces.surfaces[1].geometry).__name__ == "StandardGeometry"
    assert type(padded.surfaces.surfaces[1].geometry).__name__ == "StandardGeometry"
    # `_without_comments` drops the problem name, which is the only thing the two
    # helpers below differ in -- the surface tables have to be identical.
    assert _without_comments(plain.to_dict()) == _without_comments(padded.to_dict())

    radial = np.array(ASPHERE_RADIAL_POSITIONS_MM)
    forced = build_lens(_asphere_setup((0.0, 0.0, 1e-300)))
    np.testing.assert_array_equal(
        _host(forced.surfaces.surfaces[1].geometry.sag(radial, np.zeros_like(radial))),
        _host(plain.surfaces.surfaces[1].geometry.sag(radial, np.zeros_like(radial))),
    )
    assert type(forced.surfaces.surfaces[1].geometry).__name__ == "EvenAsphere", (
        "a non-zero coefficient selects the aspheric geometry however small it is; the "
        "selection is on the declaration, not on whether the term happens to matter"
    )


def test_the_surface_type_and_its_keywords_are_decided_together() -> None:
    """Why `_geometry_arguments` returns both: `coefficients` is silently dropped.

    Measured against the pinned install: `surface_type='standard'` with
    `coefficients=[...]` raises nothing and builds a `StandardGeometry` with no
    `coefficients` attribute at all. So a builder that chose the type and the
    keywords in two places could pass an asphere to a sphere and trace a different
    optical system with no error -- which is the recorded
    `surface_kwargs_are_silently_filtered` hazard, reached from the other side.
    """
    import optiland.backend as be
    from optiland.physical_apertures.radial import RadialAperture

    from solvers.optiland.system import _geometry_arguments

    aspheric = SurfaceSpec(
        radius_mm=ASPHERE_RADIUS_MM,
        thickness_mm=1.0,
        aspheric_coefficients=ASPHERE_COEFFICIENTS,
    )
    surface_type, kwargs = _geometry_arguments(aspheric, be, RadialAperture)
    assert surface_type == "even_asphere"
    assert set(kwargs) == {"radius", "conic", "coefficients"}
    assert kwargs["coefficients"] == list(ASPHERE_COEFFICIENTS)
    assert isinstance(kwargs["coefficients"], list), "the pinned geometry indexes a list"

    conic = SurfaceSpec(radius_mm=ASPHERE_RADIUS_MM, thickness_mm=1.0)
    surface_type, kwargs = _geometry_arguments(conic, be, RadialAperture)
    assert surface_type == "standard"
    assert set(kwargs) == {"radius", "conic"}, (
        "a conic surface must not be handed `coefficients`; the solver would discard it "
        "silently, so the guard is that it is never assembled"
    )

    # CHE-220: `aperture` joins the same named set, and an unapertured surface
    # contributes no key at all rather than an infinite rim.
    assert "aperture" not in kwargs, (
        "an unapertured surface must not assemble an `aperture` key: the backend's "
        "clipping branch is `if self.aperture`, so the declared absence has to reach it "
        "as an absence"
    )
    apertured = SurfaceSpec(
        radius_mm=ASPHERE_RADIUS_MM, thickness_mm=1.0, clear_semi_diameter_mm=3.0
    )
    surface_type, kwargs = _geometry_arguments(apertured, be, RadialAperture)
    assert surface_type == "standard"
    assert set(kwargs) == {"radius", "conic", "aperture"}
    assert isinstance(kwargs["aperture"], RadialAperture), (
        "the aperture must be constructed here, not left as a scalar for "
        "`configure_aperture` to read as a DIAMETER and halve"
    )
    assert kwargs["aperture"].r_max == 3.0


def test_an_aspheric_system_traces() -> None:
    """It is not enough that it builds: the surface has to be reachable by rays."""
    lens = build_lens(_asphere_setup(ASPHERE_COEFFICIENTS))
    traced = lens.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=6)
    x = _host(traced.x)
    assert x.size == 1 + 3 * 6 * 7
    assert bool(np.all(np.isfinite(x)))
    assert bool(np.all(np.isfinite(_host(traced.opd))))

    # And the asphere changes the trace: a strong r**4 term must move the rays.
    strong = build_lens(_asphere_setup((0.0, -5e-3)))
    moved = _host(strong.trace(Hx=0.0, Hy=0.0, wavelength=0.55, num_rays=6).x)
    assert float(np.max(np.abs(moved - x))) > 1e-6, (
        "if the polynomial were being filtered out, the two traces would agree"
    )


# ---------------------------------------------------------------------------
# 6. The object surface, for both source geometries (CHE-207)
# ---------------------------------------------------------------------------


def test_the_object_surface_carries_the_declared_object_distance() -> None:
    """Index 0 is `thickness = object_distance_mm`, unconverted, or `inf`.

    The whole of the source geometry is this one number, so it is asserted on the
    built system's own surface positions rather than on the call that set it: the
    object surface lands at `-d`, and the first optical surface stays at 0.
    """
    from fixtures.systems import (
        FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
        finite_conjugate_singlet,
        finite_conjugate_source,
    )

    finite = build_lens(finite_conjugate_singlet(), finite_conjugate_source())
    positions = _host(finite.surfaces.positions).ravel()
    assert positions[0] == pytest.approx(-FINITE_CONJUGATE_OBJECT_DISTANCE_MM, abs=0.0)
    assert positions[1] == pytest.approx(0.0, abs=0.0)
    # `bool(...)` because the pinned solver returns `np.False_` here, which is not
    # the Python singleton -- `rays.py` coerces it for the same reason.
    assert bool(finite.object_surface.is_infinite) is False

    from fixtures.systems import singlet_source

    collimated = build_lens(singlet_ref(), singlet_source())
    collimated_positions = _host(collimated.surfaces.positions).ravel()
    assert math.isinf(collimated_positions[0]) and collimated_positions[0] < 0.0
    assert bool(collimated.object_surface.is_infinite) is True


def test_the_field_of_a_finite_conjugate_is_a_position() -> None:
    """The schema's declared point-source convention, verified against the solver.

    `problems.SourceSpec` documents the source at
    `(-tan(x_deg) * d, -tan(y_deg) * d, -d)`. That is a claim about the *solver's*
    field convention, so it is checked here rather than only in the schema tests --
    a schema that documented a convention the adapter did not produce would be
    worse than one that documented nothing.
    """
    import optiland.backend as be
    from fixtures.systems import (
        FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
        finite_conjugate_singlet,
        finite_conjugate_source,
    )
    from optiland.distribution import create_distribution

    from solvers.optiland.launch import normalized_field as _normalized_field

    distribution = create_distribution("hexapolar")
    distribution.generate_points(2)
    points = int(_host(distribution.x).size)
    for field_deg in ((0.0, 0.0), (0.0, 2.0)):
        # A lens per field, because CHE-218 made the declared field the one being
        # traced. That is the capability the split bought: the second field is not
        # in any record, and nothing had to be edited to reach it.
        lens = build_lens(
            finite_conjugate_singlet(),
            finite_conjugate_source(field_angle_deg=field_deg),
        )
        normalized = _normalized_field(lens, field_deg)
        launch = lens.ray_tracer.ray_generator.generate_rays(
            be.repeat(be.atleast_1d(be.array(normalized[0])), points),
            be.repeat(be.atleast_1d(be.array(normalized[1])), points),
            distribution.x,
            distribution.y,
            0.55,
        )
        distance = FINITE_CONJUGATE_OBJECT_DISTANCE_MM
        for axis, angle_deg in (("x", field_deg[0]), ("y", field_deg[1])):
            column = _host(getattr(launch, axis))
            assert float(column[0]) == pytest.approx(
                -math.tan(math.radians(angle_deg)) * distance, abs=0.0
            )
            assert float(np.ptp(column)) == 0.0, "a point source launches from one point"


def test_ideal_material_is_dispersionless_and_lossless() -> None:
    """An ideal index is independent of any glass catalog, which is what admits
    the analytic oracle `tests/physics/` uses."""
    lens = build_lens(singlet_ref())
    glass = lens.surfaces.surfaces[1].material_post
    for wavelength in (0.4, 0.55, 0.7):
        assert float(_host(glass.n(wavelength)).ravel()[0]) == SINGLET_REFRACTIVE_INDEX
    assert float(_host(glass.k(0.55)).ravel()[0]) == 0.0


# ---------------------------------------------------------------------------
# 7. The clear aperture (CHE-220 / R05.9)
# ---------------------------------------------------------------------------
#
# A purpose-built system, not a fixture. The clipping radius has to be
# *analytically* known for the geometry predicate below to mean anything, and the
# fixtures' rims are chosen to clip nothing precisely so that CHE-182's frozen ray
# numbers keep their meaning (`tests/fixtures/systems.py` states the measurement).
#
# TWO rays, not a fan: this is a predicate about where a boundary is, not a
# convergence study. And both sides of the boundary, because a test that only
# showed a clipped ray could not tell an aperture from an unrelated failure.

#: The declared clear semi-diameter of the purpose-built system's one surface, in
#: mm. A plane surface, so the clipping radius is the declared radius exactly --
#: there is no sag to displace the intersection off the launch radius, which is
#: what makes the boundary analytic rather than measured.
CLIP_SEMI_DIAMETER_MM = 1.0

#: The relative offset the boundary is probed at from both sides. 1e-9 is nine
#: orders of magnitude inside the factor of two that a diameter-for-radius error
#: would produce, so this is a test of *the* radius rather than of roughly it, and
#: it is still five orders above float64 round-off on a millimetre.
CLIP_PROBE_RELATIVE = 1.0e-9


def _clip_setup(
    *,
    clear_semi_diameter_mm: float | str = CLIP_SEMI_DIAMETER_MM,
    entrance_pupil_diameter_mm: float = 1.0,
) -> OpticalSetup:
    """One plane surface with a declared rim, then the image plane.

    The default 1 mm entrance pupil fills this stop to a paraxial marginal ray
    height of 0.5 mm, half its 1 mm rim, so the system is comfortably on the
    accepted side of the stop consistency rule and the only thing under test is
    where the clipping boundary is. The rays the probe injects are placed on the
    surface directly and do not come from the pupil at all.
    """
    return OpticalSetup(
        name="ClearApertureProbe",
        description="CHE-220: one plane surface whose clipping radius is its declared rim.",
        surfaces=(
            SurfaceSpec(
                thickness_mm=10.0,
                clear_semi_diameter_mm=clear_semi_diameter_mm,
                comment="the apertured plane, and the stop",
            ),
        ),
        stop_index=0,
        entrance_pupil_diameter_mm=entrance_pupil_diameter_mm,
        reference_wavelength_um=0.55,
    )


def _survives(lens: object, radius_mm: float) -> bool:
    """Whether one ray launched parallel to the axis at `radius_mm` survives surface 1.

    Native rays on purpose: this file is `NATIVE_EXEMPT` because the claim is about
    what the *backend* does to a row it clipped, and no neutral type carries it.
    Optiland clips by zeroing `RealRays.i` and keeping the row, which is the rule
    `rays._ALIVE_RULE` states.
    """
    import optiland.backend as be
    from optiland.rays import RealRays

    one = be.array([radius_mm])
    zero = be.array([0.0])
    rays = RealRays(
        x=one,
        y=be.array([0.0]),
        z=be.array([-1.0]),
        L=zero,
        M=zero,
        N=be.array([1.0]),
        intensity=be.array([1.0]),
        wavelength=be.array([0.55]),
    )
    traced = lens.surfaces.surfaces[1].trace(rays)  # type: ignore[attr-defined]
    return float(_host(traced.i).ravel()[0]) > 0.0


def test_a_declared_clear_aperture_clips_and_an_undeclared_one_does_not() -> None:
    """Acceptance criterion 2: both sides of the boundary, on the same surface.

    The negative arm is the same system with the rim removed. Without it, a test
    that only showed the outside ray dying could not distinguish an aperture from
    the surface being unreachable for some other reason -- which is exactly the
    state the whole repository was in before this ticket, where every surface was
    unapertured and nothing clipped anywhere.
    """
    apertured = build_lens(_clip_setup())
    outside_mm = CLIP_SEMI_DIAMETER_MM * 2.0
    inside_mm = CLIP_SEMI_DIAMETER_MM * 0.5
    assert _survives(apertured, inside_mm), "a ray inside the declared rim must survive"
    assert not _survives(apertured, outside_mm), "a ray outside the declared rim must be clipped"

    unapertured = build_lens(_clip_setup(clear_semi_diameter_mm=UNAPERTURED))
    assert _survives(unapertured, outside_mm), (
        "with no rim declared the same ray must survive: the aperture is what clips it, "
        "not the geometry. `Surface._trace_real` only clips `if self.aperture`, so a "
        "declared absence has to reach the backend as `None`"
    )


def test_the_clipping_radius_is_the_declared_semi_diameter_not_its_double() -> None:
    """Acceptance criterion 3, and the factor of two it exists to catch.

    `configure_aperture` reads a bare number as a **diameter** and stores
    `r_max = number / 2`. Had the schema's semi-diameter been passed as that
    scalar, the boundary would sit at `a / 2` and a ray at `0.9 a` would die. The
    probe is at `(1 -+ 1e-9) a`, so nothing but the declared radius passes it.
    """
    lens = build_lens(_clip_setup())
    a = CLIP_SEMI_DIAMETER_MM
    assert _survives(lens, a * (1.0 - CLIP_PROBE_RELATIVE))
    assert not _survives(lens, a * (1.0 + CLIP_PROBE_RELATIVE))
    # The falsifier stated as a test rather than as a comment: a diameter-read rim
    # would clip here, and it must not.
    assert _survives(lens, a * 0.9), (
        "a ray at 0.9 of the declared SEMI-diameter must survive; if it does not, the "
        "value reached the backend as a diameter and the rim is half what was declared"
    )


def test_the_surface_holds_the_aperture_that_was_set_not_a_layout_attribute() -> None:
    """Acceptance criterion 4, asserted on the constructed surface.

    `Surface.set_semi_aperture` sets `semi_aperture`, which is used for layout and
    **does not clip**, and the two are indistinguishable from outside the package.
    So the attribute is read by name, its type is checked, and `r_max` is checked
    against the declaration -- the three things that together say the clipping
    aperture is the one the problem asked for.
    """
    from optiland.physical_apertures.radial import RadialAperture

    lens = build_lens(_clip_setup())
    surface = lens.surfaces.surfaces[1]
    assert isinstance(surface.aperture, RadialAperture)
    assert surface.aperture.r_max == CLIP_SEMI_DIAMETER_MM
    assert surface.aperture.r_min == 0.0, "an annulus was not declared"

    unapertured = build_lens(_clip_setup(clear_semi_diameter_mm=UNAPERTURED))
    assert unapertured.surfaces.surfaces[1].aperture is None


def test_an_aperture_the_backend_did_not_keep_is_refused() -> None:
    """The readback guard, on both shapes of the silent failure.

    Two things can happen through the pinned factory's `**kwargs` without an error:
    the key is filtered away, so the surface never clips, or the value is
    reinterpreted, so it clips at a radius nobody declared. Neither is visible in a
    trace. The guard is called directly on a lens whose surface has been put into
    each of those states, which is the only way to reach them -- the pinned backend
    cannot be made to do either from outside.
    """
    from optiland.physical_apertures.radial import RadialAperture

    setup = _clip_setup()
    surface_spec = setup.surfaces[0]

    for damaged in (None, RadialAperture(r_max=CLIP_SEMI_DIAMETER_MM / 2.0)):
        lens = build_lens(setup)
        surface = lens.surfaces.surfaces[1]
        declared = surface.aperture
        assert declared is not None
        plans = [(surface_spec, "standard", {"aperture": declared}, None)]
        # A clean lens passes, so the failure below is the damage and not the check.
        system_module._require_apertures_were_set(lens, setup, plans)
        surface.aperture = damaged
        with pytest.raises(ValueError, match="declared a clear aperture"):
            system_module._require_apertures_were_set(lens, setup, plans)


# --- The stop / EPD / clear-aperture consistency rule ----------------------


def test_an_entrance_pupil_that_overfills_the_stop_is_refused() -> None:
    """Acceptance criterion 5: an inconsistent system is refused, not traced.

    The stop's rim is half the pupil the EPD declares, so the fan would be clipped
    at the stop and the "entrance pupil diameter" would describe an aperture the
    system does not have. That is a contradiction between two declarations rather
    than a numerical problem, so it is refused at construction.
    """
    with pytest.raises(ValueError, match="fills the stop surface"):
        build_lens(_clip_setup(clear_semi_diameter_mm=0.25, entrance_pupil_diameter_mm=1.0))


def test_a_stop_filled_exactly_to_its_rim_is_accepted() -> None:
    """The rule is an inequality, and its boundary is where the pupil meets the rim.

    A stop at the front vertex with the object at infinity has a paraxial marginal
    ray height of exactly `EPD / 2`, so this is the exact-equality case and it must
    pass: a system whose stop rim *is* its pupil is the ordinary matched design, not
    an error.
    """
    lens = build_lens(_clip_setup(clear_semi_diameter_mm=0.5, entrance_pupil_diameter_mm=1.0))
    assert lens.surfaces.surfaces[1].aperture.r_max == 0.5


def test_a_stop_rim_wider_than_the_pupil_is_accepted() -> None:
    """Under-filling is not a contradiction, and the rule says so deliberately.

    A rim wider than the marginal ray is the case where the declared EPD is an
    analysis aperture: the system is being traced at less than its full opening.
    Refusing it would force every setup to state a rim equal to its pupil, which is
    a different and stronger claim than the one the schema makes.
    """
    lens = build_lens(_clip_setup(clear_semi_diameter_mm=5.0, entrance_pupil_diameter_mm=1.0))
    assert lens.surfaces.surfaces[1].aperture.r_max == 5.0


def test_an_unapertured_stop_is_not_checked_and_is_not_a_failure() -> None:
    """The declared idealization: the EPD is the only aperture in the system.

    This is the state M3-REVERSE-TELEPHOTO is in, and the state every system in the
    repository was in before this ticket. It has to stay constructible, or the
    consistency rule would be a requirement that every surface carry a rim.
    """
    lens = build_lens(
        _clip_setup(clear_semi_diameter_mm=UNAPERTURED, entrance_pupil_diameter_mm=1.0)
    )
    assert lens.surfaces.surfaces[1].aperture is None


#: A two-surface system whose stop is the *second* surface, built to discriminate
#: the one convention the rule depends on: `paraxial.marginal_ray()` returns one
#: height per **optic** surface with the object surface at index 0, so the setup's
#: surface `i` is optic surface `i + 1`.
#:
#: On a one-surface system with the object at infinity `heights[0] == heights[1] ==
#: EPD/2`, so an off-by-one there is invisible -- which is the same class of silent
#: error as the factor of two this ticket exists to catch. Here the first surface
#: has power, so the marginal ray height at the stop is well below its height at
#: the surface before it, and a rim placed between the two is accepted under the
#: right index and refused under the wrong one.
INTERIOR_STOP_RADIUS_MM = 2.0
INTERIOR_STOP_THICKNESS_MM = 2.0
INTERIOR_STOP_INDEX_MM = 1.5
INTERIOR_STOP_RIM_MM = 0.4
INTERIOR_STOP_EPD_MM = 1.0


def _interior_stop_setup(*, clear_semi_diameter_mm: float = INTERIOR_STOP_RIM_MM) -> OpticalSetup:
    return OpticalSetup(
        name="InteriorStopProbe",
        description="CHE-220: a powered first surface, and the stop behind it.",
        surfaces=(
            SurfaceSpec(
                radius_mm=INTERIOR_STOP_RADIUS_MM,
                thickness_mm=INTERIOR_STOP_THICKNESS_MM,
                material={"kind": "ideal", "refractive_index": INTERIOR_STOP_INDEX_MM},
                comment="powered front face, not the stop",
            ),
            SurfaceSpec(
                thickness_mm=5.0,
                clear_semi_diameter_mm=clear_semi_diameter_mm,
                comment="the stop, one surface in",
            ),
        ),
        stop_index=1,
        entrance_pupil_diameter_mm=INTERIOR_STOP_EPD_MM,
        reference_wavelength_um=0.55,
    )


def test_the_stop_rule_reads_the_marginal_height_at_the_stop_not_before_it() -> None:
    """The `+1` optic-surface offset, pinned by a system where it is observable.

    Both arms are needed, and the first is what makes the second mean anything: the
    marginal height at the setup's stop must be *strictly below* the height at the
    surface before it, with the declared rim between them. Then a build that read
    the height one surface too early would refuse a system this one accepts.
    """
    setup = _interior_stop_setup()
    lens = build_lens(setup)
    heights = _host(lens.paraxial.marginal_ray()[0]).ravel()
    before_mm = abs(float(heights[setup.stop_index]))
    at_stop_mm = abs(float(heights[setup.stop_index + 1]))

    assert at_stop_mm < INTERIOR_STOP_RIM_MM < before_mm, (
        "the probe is only a discriminator if the rim separates the two heights; got "
        f"at_stop={at_stop_mm!r}, rim={INTERIOR_STOP_RIM_MM!r}, before={before_mm!r}"
    )
    # The accepted arm is the whole point: `build_lens` returned above rather than
    # refusing, which it would have done had it compared the rim against
    # `before_mm`. The refused arm confirms the same system is rejected once the rim
    # falls below the height at the stop, so acceptance is not vacuous.
    with pytest.raises(ValueError, match="fills the stop surface"):
        build_lens(_interior_stop_setup(clear_semi_diameter_mm=at_stop_mm * 0.9))


def test_the_fixture_stops_satisfy_the_consistency_rule() -> None:
    """The rule holds on the systems the repository actually measures.

    Not a tautology: `build_lens` enforces it, so what this adds is that the
    *fixtures* are on the accepted side of it with room to spare, which is the same
    fact `tests/fixtures/systems.py` states as a comment.
    """
    for setup in (singlet_ref(), REVERSE_TELEPHOTO):
        lens = build_lens(setup)
        stop = setup.surfaces[setup.stop_index]
        heights, _slopes = lens.paraxial.marginal_ray()
        filled_mm = abs(float(_host(heights).ravel()[setup.stop_index + 1]))
        if stop.has_clear_aperture:
            assert filled_mm < float(stop.clear_semi_diameter_mm)
        else:
            assert stop.clear_semi_diameter_mm == UNAPERTURED
