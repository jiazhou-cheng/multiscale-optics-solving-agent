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
import subprocess
import sys
from pathlib import Path

import pytest
from fixtures.systems import (
    REVERSE_TELEPHOTO,
    SINGLET_BACK_FOCAL_LENGTH_MM,
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM,
    SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
    SINGLET_RADIUS_MM,
    SINGLET_REFRACTIVE_INDEX,
    singlet_ref,
)

from problems.ray_trace import RayTraceProblem

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


def test_the_fixtures_are_neutral_problems() -> None:
    for problem in (singlet_ref(), REVERSE_TELEPHOTO):
        assert isinstance(problem, RayTraceProblem)


def test_the_singlet_derived_quantities_stay_derived() -> None:
    """A stale spacing beside a changed index is a lens nobody notices is wrong."""
    assert SINGLET_EFFECTIVE_FOCAL_LENGTH_MM == SINGLET_RADIUS_MM / (
        SINGLET_REFRACTIVE_INDEX - 1.0
    )
    singlet = singlet_ref()
    assert singlet.surfaces[1].thickness_mm == SINGLET_BACK_FOCAL_LENGTH_MM
    assert singlet.entrance_pupil_diameter_mm == SINGLET_ENTRANCE_PUPIL_DIAMETER_MM


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
    assert singlet.wavelengths_um == (0.55,)
    assert singlet.object_at_infinity

    assert REVERSE_TELEPHOTO.surfaces[0].resolved_radius_mm == 1.69111096
    assert REVERSE_TELEPHOTO.surfaces[8].is_plane
    assert REVERSE_TELEPHOTO.stop_index == 8
    assert REVERSE_TELEPHOTO.entrance_pupil_diameter_mm == 0.3
    assert REVERSE_TELEPHOTO.primary_wavelength_um == 0.5876
    assert REVERSE_TELEPHOTO.field_angles_deg == ((0.0, 0.0), (0.0, 21.0), (0.0, 30.0))
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
    assert not any(isinstance(value, RayTraceProblem) for value in exported.values())
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
