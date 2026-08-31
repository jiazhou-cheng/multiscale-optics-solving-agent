"""One generic construction path, and the surface table it produces.

CHE-179 (R05.1), acceptance criteria:

1. one generic construction function -- adding a system means handing it a
   different `RayTraceProblem`, never writing a builder;
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

from problems.ray_trace import Material, RayTraceProblem, SurfaceSpec
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
    """`build_lens` is the whole construction surface, and it takes a problem."""
    public = [
        name
        for name in dir(system_module)
        if not name.startswith("_")
        and callable(getattr(system_module, name))
        # Defined here, not imported here: `RayTraceProblem` is callable and is
        # the argument type, which is the opposite of a second construction path.
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


def test_refuses_something_that_is_not_a_problem() -> None:
    with pytest.raises(TypeError, match="RayTraceProblem"):
        build_lens({"name": "M3SingletRef"})  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. The surface table, against an independent oracle
# ---------------------------------------------------------------------------


def _without_comments(state: dict) -> dict:
    """Drop surface comments before a structural comparison.

    A comment is prescription annotation and changes no geometry, no material and
    no traced ray. The fixture carries `"plane aperture stop"` where the bundled
    sample carries `""`, which is the only difference between the two tables and
    the only one this helper hides -- everything else is compared verbatim.
    """
    surfaces = state["surface_group"]["surfaces"]
    state["surface_group"]["surfaces"] = [{**s, "comment": ""} for s in surfaces]
    state.pop("name", None)
    return state


def test_surface_table_matches_the_bundled_sample_exactly() -> None:
    """Executed output, element for element, against the solver's own sample."""
    from optiland.samples.objectives import ReverseTelephoto

    oracle = _without_comments(ReverseTelephoto().to_dict())
    built = _without_comments(build_lens(REVERSE_TELEPHOTO).to_dict())
    assert built == oracle


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


def test_aperture_fields_and_wavelengths_are_set_not_inherited() -> None:
    lens = build_lens(REVERSE_TELEPHOTO)
    state = lens.to_dict()
    assert state["aperture"] == {
        "type": "EPD",
        "value": REVERSE_TELEPHOTO.entrance_pupil_diameter_mm,
    }
    assert lens.fields.max_field == 30.0
    assert tuple(
        (f["x"], f["y"]) for f in state["fields"]["fields"]
    ) == REVERSE_TELEPHOTO.field_angles_deg
    assert all(
        f["vx"] == 0.0 and f["vy"] == 0.0 and f["weight"] == 1.0
        for f in state["fields"]["fields"]
    )
    primary = [w for w in state["wavelengths"]["wavelengths"] if w["is_primary"]]
    assert len(primary) == 1
    assert primary[0]["value"] == REVERSE_TELEPHOTO.primary_wavelength_um
    assert {w["unit"] for w in state["wavelengths"]["wavelengths"]} == {"um"}


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
    problem = _one_glass_problem(glass, catalog, expected_file)
    lens = build_lens(problem)
    row = lens.surfaces.surfaces[1].material_post.material_data
    assert str(row["filename"]) == expected_file
    assert float(row["similarity_score"]) == 0.0


def test_a_glass_that_resolves_to_a_different_file_is_refused() -> None:
    """A material-database change becomes an error, not a different trace."""
    problem = _one_glass_problem("N-SK10", None, "glass/hikari/N-SK10.yml")
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
    problem = _one_glass_problem("N-SK1", None, None)
    with pytest.raises(ValueError, match="by similarity"):
        build_lens(problem)


def test_an_unknown_glass_is_refused_as_a_problem_not_a_crash() -> None:
    """No substring survives, so the solver raises a bare ValueError. It reaches
    the caller as a statement about the problem rather than as a solver crash."""
    problem = _one_glass_problem("NOT-A-GLASS-AT-ALL", None, None)
    with pytest.raises(ValueError, match="could not be resolved"):
        build_lens(problem)


def _one_glass_problem(
    glass: str, catalog: str | None, expected_file: str | None
) -> RayTraceProblem:
    material: Material = {
        "kind": "catalog",
        "name": glass,
        "catalog": catalog,
        "expected_catalog_file": expected_file,
    }
    return RayTraceProblem(
        name=f"one-glass-{glass}",
        surfaces=(
            SurfaceSpec(radius_mm=50.0, thickness_mm=5.0, material=material),
            SurfaceSpec(thickness_mm=45.0),
        ),
        entrance_pupil_diameter_mm=10.0,
        field_angles_deg=((0.0, 0.0),),
        wavelengths_um=(0.5876,),
        stop_index=0,
    )


def test_ideal_material_is_dispersionless_and_lossless() -> None:
    """An ideal index is independent of any glass catalog, which is what admits
    the analytic oracle `tests/physics/` uses."""
    lens = build_lens(singlet_ref())
    glass = lens.surfaces.surfaces[1].material_post
    for wavelength in (0.4, 0.55, 0.7):
        assert float(_host(glass.n(wavelength)).ravel()[0]) == SINGLET_REFRACTIVE_INDEX
    assert float(_host(glass.k(0.55)).ravel()[0]) == 0.0
