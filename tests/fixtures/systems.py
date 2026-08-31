"""The measured benchmark systems, as neutral ray-trace problems.

CHE-156 (R04). These left production with `registry/prescriptions.py`. They are
here because they are *evidence*: two optical systems this repository has
measured, kept so a test can trace something real. A production catalog of them
made `solve("M3SingletRef")` the shortest path for every caller, and a name a
solver resolves into a lens is not a problem statement.

Importing this module imports no solver. `tests/problems/test_fixtures.py`
asserts that in a fresh interpreter, which is the point of the file: a fixture
that needed Optiland to describe a lens would mean the schema had not actually
separated intent from construction.

Transcription
-------------
Both systems were transcribed by **executing** the reference construction and the
new one and comparing the surface tables element by element, not by reading the
numbers across. Every radius, thickness, index and glass below matched
`pre-rewrite-2026-08-30:src/registry/prescriptions.py` exactly, including the
derived singlet quantities. The comparison is a one-off probe rather than a
committed test, because a committed one would have to import the deleted tree;
the run is recorded on CHE-156.

The singlet's derived quantities stay **derived**. `SINGLET_BACK_FOCAL_LENGTH_MM`
is computed from the index and the radius, exactly as the reference had it, so a
change to one of the frozen protocol inputs cannot leave a stale spacing behind.
"""

from __future__ import annotations

from problems.ray_trace import Material, RayTraceProblem, SurfaceSpec

__all__ = [
    "REVERSE_TELEPHOTO",
    "SINGLET_BACK_FOCAL_LENGTH_MM",
    "SINGLET_CENTER_THICKNESS_MM",
    "SINGLET_EFFECTIVE_FOCAL_LENGTH_MM",
    "SINGLET_ENTRANCE_PUPIL_DIAMETER_MM",
    "SINGLET_F_NUMBER",
    "SINGLET_RADIUS_MM",
    "SINGLET_REFRACTIVE_INDEX",
    "SINGLET_WAVELENGTH_UM",
    "singlet_ref",
]

# --- M3-SINGLET-REF ---------------------------------------------------------
#
# Frozen by M3.2 in `benchmarks/protocols/slice_protocol.yaml` at
# `pre-rewrite-2026-08-30`. Plano-convex, convex toward the collimated side --
# the low-aberration orientation -- with real refractive surfaces, because
# CHE-30 ruled out a paraxial surface as an OPL source. The ideal constant index
# keeps it independent of any glass catalog, which is what admits the analytic
# Airy oracle this system exists for.

SINGLET_REFRACTIVE_INDEX = 1.5168
SINGLET_RADIUS_MM = 2.5
SINGLET_CENTER_THICKNESS_MM = 0.2
SINGLET_F_NUMBER = 9.7
SINGLET_WAVELENGTH_UM = 0.55  # verified by CHE-12

SINGLET_EFFECTIVE_FOCAL_LENGTH_MM = SINGLET_RADIUS_MM / (SINGLET_REFRACTIVE_INDEX - 1.0)
SINGLET_BACK_FOCAL_LENGTH_MM = (
    SINGLET_EFFECTIVE_FOCAL_LENGTH_MM - SINGLET_CENTER_THICKNESS_MM / SINGLET_REFRACTIVE_INDEX
)
SINGLET_ENTRANCE_PUPIL_DIAMETER_MM = SINGLET_EFFECTIVE_FOCAL_LENGTH_MM / SINGLET_F_NUMBER


def singlet_ref() -> RayTraceProblem:
    """M3-SINGLET-REF: plano-convex singlet with an analytic Airy oracle.

    A function rather than a module-level constant so a test that wants to vary
    one thing -- a wavelength, a field -- has something to start from without
    mutating a shared value. `RayTraceProblem` is frozen, so the shared-value
    hazard is small; the function is for the variation, not the safety.
    """
    return RayTraceProblem(
        name="M3SingletRef",
        description=(
            "M3-SINGLET-REF: plano-convex singlet, convex toward the collimated side, "
            "admitting an analytic Airy oracle. Frozen by benchmarks/protocols/"
            "slice_protocol.yaml at pre-rewrite-2026-08-30."
        ),
        surfaces=(
            SurfaceSpec(
                radius_mm=SINGLET_RADIUS_MM,
                thickness_mm=SINGLET_CENTER_THICKNESS_MM,
                material={"kind": "ideal", "refractive_index": SINGLET_REFRACTIVE_INDEX},
                comment="convex front face, and the aperture stop",
            ),
            # Rear vertex: glass -> air, then the image plane one back focal
            # length on.
            SurfaceSpec(thickness_mm=SINGLET_BACK_FOCAL_LENGTH_MM, comment="plane rear face"),
        ),
        stop_index=0,
        entrance_pupil_diameter_mm=SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
        field_angles_deg=((0.0, 0.0),),
        wavelengths_um=(SINGLET_WAVELENGTH_UM,),
        object_distance_mm=None,  # object at infinity
    )


# --- M3-REVERSE-TELEPHOTO ---------------------------------------------------
#
# The prescription of `optiland.samples.objectives.ReverseTelephoto`, already
# validated in M1 (L1-RAY-01). Every radius, thickness, glass, field and
# wavelength is the sample's own value.
#
# `expected_catalog_file` on each glass is material provenance, not decoration:
# the pinned solver resolves a bare name by substring filter plus similarity
# ranking, so `SK15` selects the HIKARI row while `N-SK10` selects the SCHOTT
# one. Recording which row this prescription was measured against
# (`pre-rewrite-2026-08-30:benchmarks/probes/optiland/system_construction_probe.py`)
# is what turns a catalog change into an error rather than a different trace.


def _glass(name: str, catalog_file: str, catalog: str | None = None) -> Material:
    return {
        "kind": "catalog",
        "name": name,
        "catalog": catalog,
        "expected_catalog_file": catalog_file,
    }


REVERSE_TELEPHOTO = RayTraceProblem(
    name="ReverseTelephoto",
    description=(
        "M3-REVERSE-TELEPHOTO: the bundled Optiland reverse-telephoto objective "
        "validated in M1 (L1-RAY-01), transcribed into the neutral schema."
    ),
    surfaces=(
        SurfaceSpec(
            radius_mm=1.69111096,
            thickness_mm=0.08259680,
            material=_glass("N-SK10", "glass/schott/N-SK10.yml"),
        ),
        SurfaceSpec(radius_mm=0.94414496, thickness_mm=0.8),
        SurfaceSpec(
            radius_mm=4.32100401,
            thickness_mm=0.080256,
            material=_glass("SK15", "glass/hikari/SK15.yml"),
        ),
        SurfaceSpec(radius_mm=1.78117621, thickness_mm=0.5),
        SurfaceSpec(
            radius_mm=2.64050282,
            thickness_mm=0.27638160,
            material=_glass("BASF2", "glass/hikari/BASF2.yml"),
        ),
        SurfaceSpec(radius_mm=-3.86177348, thickness_mm=0.1),
        SurfaceSpec(
            radius_mm=1.05627661,
            thickness_mm=0.2,
            material=_glass("FK3", "glass/schott/FK3.yml"),
        ),
        SurfaceSpec(radius_mm=-4.06933311, thickness_mm=0.2001384),
        SurfaceSpec(thickness_mm=0.06688, comment="plane aperture stop"),
        SurfaceSpec(
            radius_mm=-2.61246583,
            thickness_mm=0.064372,
            material=_glass("SF15", "glass/hikari/SF15.yml", catalog="hikari"),
        ),
        SurfaceSpec(radius_mm=0.99117409, thickness_mm=0.3),
        SurfaceSpec(
            radius_mm=9.03045960,
            thickness_mm=0.18743120,
            material=_glass("N-LAK12", "glass/schott/N-LAK12.yml"),
        ),
        SurfaceSpec(radius_mm=-1.35680743, thickness_mm=2.35130547),
    ),
    stop_index=8,
    entrance_pupil_diameter_mm=0.3,
    field_angles_deg=((0.0, 0.0), (0.0, 21.0), (0.0, 30.0)),
    wavelengths_um=(0.4861, 0.5876, 0.6563),
    primary_wavelength_index=1,
    object_distance_mm=None,  # object at infinity
)
