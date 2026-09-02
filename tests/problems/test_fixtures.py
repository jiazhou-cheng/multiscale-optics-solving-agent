"""The fixtures build real lenses with no solver, and production knows no lens by name.

CHE-156 (R04), acceptance criteria 1, 2 and 3:

1. test fixtures construct neutral ray problems without importing Optiland;
2. `solve("M3_SINGLET_REF")` is not expressible -- no name-to-system resolution
   in production;
3. no concrete benchmark lens prescription exists under `src/`.

Criteria 2 and 3 are absence claims about the whole production tree, so they are
checked against the tree rather than against an import: a symbol nothing exports
can still be defined, and a prescription can be reintroduced in any module.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from fixtures.systems import (
    REVERSE_TELEPHOTO,
    SINGLET_BACK_FOCAL_LENGTH_MM,
    SINGLET_CENTER_THICKNESS_MM,
    SINGLET_CLEAR_SEMI_DIAMETER_MM,
    SINGLET_EDGE_ZERO_SEMI_DIAMETER_MM,
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM,
    SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
    SINGLET_RADIUS_MM,
    SINGLET_REFRACTIVE_INDEX,
    singlet_ref,
)

from problems.ray_trace import UNAPERTURED, OpticalSetup, SourceSpec

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
SOURCES = sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in str(p))


# ---------------------------------------------------------------------------
# Criterion 1 -- fixtures need no solver
# ---------------------------------------------------------------------------


def test_building_both_fixtures_imports_no_solver() -> None:
    """Checked in a fresh interpreter, because the failure is transitive."""
    source = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(ROOT / 'tests')!r})\n"
        "from fixtures.systems import REVERSE_TELEPHOTO, singlet_ref\n"
        "assert len(singlet_ref().surfaces) == 2\n"
        "assert len(REVERSE_TELEPHOTO.surfaces) == 13\n"
        "print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=True
    )
    loaded = set(json.loads(completed.stdout))
    forbidden = {"optiland", "chromatix", "jax", "torch"}
    assert not loaded & forbidden, (
        f"describing a lens loaded {sorted(loaded & forbidden)}. A problem that needs the "
        "solver to state it has not separated intent from construction."
    )


def test_the_fixtures_are_neutral_setups_and_neutral_sources() -> None:
    """CHE-218 (R05.7): each fixture is a setup, and each has a companion source."""
    from fixtures.systems import (
        finite_conjugate_singlet,
        finite_conjugate_source,
        reverse_telephoto_source,
        singlet_source,
    )

    for setup in (singlet_ref(), REVERSE_TELEPHOTO, finite_conjugate_singlet()):
        assert isinstance(setup, OpticalSetup)
    for source in (singlet_source(), reverse_telephoto_source(), finite_conjugate_source()):
        assert isinstance(source, SourceSpec)


def test_the_conjugate_pair_is_matched_by_the_fixture_not_by_a_field() -> None:
    """The one place a setup and a source are coupled, and it is arithmetic.

    `finite_conjugate_singlet`'s last spacing is derived from the object distance,
    so a pair is conjugate only when both factories were given the same distance.
    That coupling lives in the fixture rather than as a field on either record --
    which is what keeps the two records independently constructible.
    """
    from fixtures.systems import (
        FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
        finite_conjugate_image_distance_mm,
        finite_conjugate_singlet,
        finite_conjugate_source,
    )

    distance = 3.0 * FINITE_CONJUGATE_OBJECT_DISTANCE_MM
    setup = finite_conjugate_singlet(distance)
    source = finite_conjugate_source(object_distance_mm=distance)
    assert source.object_distance_mm == distance
    assert setup.surfaces[-1].thickness_mm == finite_conjugate_image_distance_mm(distance)
    # The defaults agree, so the common case needs no argument at all.
    assert (
        finite_conjugate_source().object_distance_mm == FINITE_CONJUGATE_OBJECT_DISTANCE_MM
    )


def test_the_singlet_derived_quantities_stay_derived() -> None:
    """A stale spacing beside a changed index is a lens nobody notices is wrong."""
    assert SINGLET_EFFECTIVE_FOCAL_LENGTH_MM == SINGLET_RADIUS_MM / (
        SINGLET_REFRACTIVE_INDEX - 1.0
    )
    singlet = singlet_ref()
    assert singlet.surfaces[1].thickness_mm == SINGLET_BACK_FOCAL_LENGTH_MM
    assert singlet.entrance_pupil_diameter_mm == SINGLET_ENTRANCE_PUPIL_DIAMETER_MM


def test_the_singlet_rim_is_inside_the_radius_where_the_element_stops_existing() -> None:
    """CHE-220: the upper half of the rim's justification, made load-bearing.

    `SINGLET_CLEAR_SEMI_DIAMETER_MM` is a chosen number -- the frozen protocol never
    stated a rim -- and its justification has two sides. The lower one is measured
    and is asserted where it belongs, by
    `tests/physics/test_optiland_rays.py::test_nothing_is_clipped_on_the_fixture_systems`.
    The upper one is *derived*: past
    `SINGLET_EDGE_ZERO_SEMI_DIAMETER_MM = sqrt(R^2 - (R - t)^2)` the plano-convex
    element's edge thickness has gone negative and there is no glass to have a rim
    on. Asserted rather than left in a comment, so a change to the radius or the
    centre thickness that invalidates the choice fails here instead of leaving a
    derivation that only looks derived.
    """
    # Checked in the other direction, so this is not the same arithmetic twice: at
    # that radius the convex face's sag `R - sqrt(R^2 - r^2)` has consumed the whole
    # centre thickness, which is what "the element stops existing" means.
    sag_at_edge_zero_mm = SINGLET_RADIUS_MM - math.sqrt(
        SINGLET_RADIUS_MM**2 - SINGLET_EDGE_ZERO_SEMI_DIAMETER_MM**2
    )
    assert sag_at_edge_zero_mm == pytest.approx(SINGLET_CENTER_THICKNESS_MM, rel=1e-12)
    assert 0.0 < SINGLET_CLEAR_SEMI_DIAMETER_MM < SINGLET_EDGE_ZERO_SEMI_DIAMETER_MM
    # Both faces of one element carry the same rim, because they are one element.
    singlet = singlet_ref()
    assert singlet.surfaces[0].clear_semi_diameter_mm == SINGLET_CLEAR_SEMI_DIAMETER_MM
    assert singlet.surfaces[1].clear_semi_diameter_mm == SINGLET_CLEAR_SEMI_DIAMETER_MM


def test_the_reverse_telephoto_declares_its_absent_rims_rather_than_defaulting_them() -> None:
    """The other CHE-220 decision, recorded where a reader would look for it.

    The bundled `optiland.samples.objectives.ReverseTelephoto` this prescription was
    transcribed from declares no aperture on any of its surfaces, so there is no rim
    to transcribe from the source the radii came from. Every surface therefore
    carries `UNAPERTURED` -- the declared idealization -- rather than a chosen
    number on the one system CHE-182's frozen ray records are gated against.
    """
    for index, surface in enumerate(REVERSE_TELEPHOTO.surfaces):
        assert surface.clear_semi_diameter_mm == UNAPERTURED, f"surfaces[{index}]"
        assert surface.has_clear_aperture is False


def test_the_transcribed_systems_are_the_measured_ones() -> None:
    """A spot check on the transcription that the R04 probe verified exhaustively.

    `tmp_probes/r04/compare_transcription.py` (recorded on CHE-156) executed the
    reference construction and this one and compared the full surface tables;
    they were identical. These few assertions are what remains committed, so a
    later edit to the numbers fails here rather than silently in R05's parity.
    """
    singlet = singlet_ref()
    assert singlet.surfaces[0].resolved_radius_mm == 2.5
    assert singlet.surfaces[0].material == {"kind": "ideal", "refractive_index": 1.5168}
    assert singlet.stop_index == 0
    assert singlet.reference_wavelength_um == 0.55
    from fixtures.systems import singlet_source

    assert singlet_source().object_at_infinity

    assert REVERSE_TELEPHOTO.surfaces[0].resolved_radius_mm == 1.69111096
    assert REVERSE_TELEPHOTO.surfaces[8].is_plane
    assert REVERSE_TELEPHOTO.stop_index == 8
    assert REVERSE_TELEPHOTO.entrance_pupil_diameter_mm == 0.3
    # The sample's own primary wavelength, which is what the exit pupil is located
    # at. Its other two wavelengths and its three field angles were dropped by
    # CHE-218: neither list ever reached a trace as a list.
    assert REVERSE_TELEPHOTO.reference_wavelength_um == 0.5876
    assert REVERSE_TELEPHOTO.surfaces[2].material["expected_catalog_file"] == (
        "glass/hikari/SK15.yml"
    )


# ---------------------------------------------------------------------------
# Criterion 2 -- no name-to-system resolution in production
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbol",
    [
        "resolve_prescription",
        "prescription_names",
        "PRESCRIPTION_NAMES",
        "_CANONICAL_PRESCRIPTIONS",
        "PRESCRIPTIONS",
        "canonical_optical_systems",
    ],
)
def test_no_name_to_system_resolution_exists_under_src(symbol: str) -> None:
    offenders = [
        path.relative_to(ROOT)
        for path in SOURCES
        if symbol in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{symbol!r} is back in {offenders}. `solve('M3SingletRef')` -- a solver call whose "
        "argument is a name this repository invented -- must not be expressible; a caller "
        "states the problem it wants."
    )


def test_the_problems_package_exposes_no_lens_by_name() -> None:
    import problems

    exported = {name: getattr(problems, name) for name in problems.__all__}
    assert not any(
        isinstance(value, OpticalSetup | SourceSpec) for value in exported.values()
    )
    # Nor a callable that returns one from a name.
    assert not {"resolve", "get", "load", "by_name"} & set(exported)


# ---------------------------------------------------------------------------
# Criterion 3 -- no concrete lens under src/
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        # Names.
        "SingletRef",
        "M3_SINGLET",
        "ReverseTelephoto",
        "REVERSE_TELEPHOTO",
        "CookeTriplet",
        "Phase3FeatureDemo",
        # Numbers that only a specific lens has. A prescription reintroduced
        # under a different name still carries these.
        "1.5168",
        "1.69111096",
        "2.35130547",
        "N-SK10",
        "BASF2",
    ],
)
def test_no_concrete_lens_prescription_exists_under_src(marker: str) -> None:
    offenders = [
        path.relative_to(ROOT)
        for path in SOURCES
        if marker in path.read_text(encoding="utf-8")
    ]
    assert not offenders, (
        f"{marker!r} appears in {offenders}. A benchmark lens is measured evidence and "
        "belongs in tests/fixtures/systems.py; in production it becomes a catalog, and a "
        "catalog becomes the shortest path for every caller."
    )
