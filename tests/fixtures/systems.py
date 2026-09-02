"""The measured benchmark systems, as neutral optical setups and their sources.

CHE-156 (R04), split by CHE-218 (R05.7). These left production with
`registry/prescriptions.py`. They are here because they are *evidence*: three
optical systems this repository has measured, kept so a test can trace something
real. A production catalog of them made `solve("M3SingletRef")` the shortest path
for every caller, and a name a solver resolves into a lens is not a problem
statement.

Setups and sources are separate, and one pair is matched
--------------------------------------------------------
Each system is an `OpticalSetup` with no illumination, and each has a companion
factory returning the `SourceSpec` the frozen measurements were taken under. The
two are independently constructible -- that is R05.7's whole point, and a test
that wants a different field angle or wavelength builds its own source rather
than editing a setup.

`finite_conjugate_singlet` is the exception worth naming: its last surface
spacing is *derived* from the object distance, so the setup and the source of a
conjugate pair are coupled by arithmetic. Both factories take the same
`object_distance_mm` and default to the same constant, so varying one varies
both. That coupling lives here, in the fixture, rather than as a field on either
record -- a conjugate pair is the caller's arithmetic.

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

from problems.ray_trace import Material, OpticalSetup, SourceSpec, SurfaceSpec

__all__ = [
    "FINITE_CONJUGATE_MAGNIFICATION",
    "FINITE_CONJUGATE_OBJECT_DISTANCE_MM",
    "REVERSE_TELEPHOTO",
    "REVERSE_TELEPHOTO_REFERENCE_WAVELENGTH_UM",
    "SINGLET_BACK_FOCAL_LENGTH_MM",
    "SINGLET_CENTER_THICKNESS_MM",
    "SINGLET_EFFECTIVE_FOCAL_LENGTH_MM",
    "SINGLET_ENTRANCE_PUPIL_DIAMETER_MM",
    "SINGLET_F_NUMBER",
    "SINGLET_RADIUS_MM",
    "SINGLET_REFRACTIVE_INDEX",
    "SINGLET_WAVELENGTH_UM",
    "finite_conjugate_image_distance_mm",
    "finite_conjugate_singlet",
    "finite_conjugate_source",
    "reverse_telephoto_source",
    "singlet_ref",
    "singlet_source",
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


def singlet_ref() -> OpticalSetup:
    """M3-SINGLET-REF: plano-convex singlet with an analytic Airy oracle.

    A function rather than a module-level constant so a test that wants to vary
    one thing has something to start from without mutating a shared value.
    `OpticalSetup` is frozen, so the shared-value hazard is small; the function is
    for the variation, not the safety.

    No illumination. `singlet_source` is the light the frozen measurements were
    taken under.
    """
    return OpticalSetup(
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
        reference_wavelength_um=SINGLET_WAVELENGTH_UM,
    )


def singlet_source(
    *,
    field_angle_deg: tuple[float, float] = (0.0, 0.0),
    wavelength_um: float = SINGLET_WAVELENGTH_UM,
) -> SourceSpec:
    """The collimated illumination M3-SINGLET-REF's frozen numbers were taken under.

    On axis at 550 nm, object at infinity. Both parameters are open so a test can
    vary the light **without touching the setup**, which is the capability R05.7
    exists for.
    """
    return SourceSpec(
        wavelength_um=wavelength_um,
        field_angle_deg=field_angle_deg,
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


#: The wavelength M3-REVERSE-TELEPHOTO's paraxial characterization is defined at:
#: the sample's own primary, `wavelengths_um[1]` of (0.4861, 0.5876, 0.6563).
#:
#: Named rather than inlined because it is load-bearing for the frozen exit-pupil
#: numbers. CHE-218 measured that the pinned solver evaluates `paraxial.XPL()` and
#: `XPD()` at the primary wavelength: at 0.5876 um they are -3.0545788978518327 mm
#: and 0.46053493637581633 mm, and at the 0.55 um the frozen ray records trace at
#: they are -3.0550180932891653 mm and 0.4607610620693788 mm. The record holds the
#: first pair, so this constant is what keeps it reproducible -- and the trace
#: still runs at 550 nm, which is outside the sample's declared set and always was.
REVERSE_TELEPHOTO_REFERENCE_WAVELENGTH_UM = 0.5876

REVERSE_TELEPHOTO = OpticalSetup(
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
    reference_wavelength_um=REVERSE_TELEPHOTO_REFERENCE_WAVELENGTH_UM,
)


def reverse_telephoto_source(
    *,
    field_angle_deg: tuple[float, float] = (0.0, 0.0),
    wavelength_um: float = 0.55,
) -> SourceSpec:
    """The illumination M3-REVERSE-TELEPHOTO's frozen numbers were taken under.

    On axis at **550 nm**, which is not one of the three wavelengths the bundled
    sample declares and never was: the wavelength a trace is evaluated at has
    always been free, and the frozen ray records deliberately exercise that. The
    sample's own 587.6 nm lives on the setup as its reference wavelength, which is
    what the exit pupil is located at.

    The three-wavelength list the sample carried is gone. It never reached a trace
    as a list -- one solve is one wavelength -- and its only effect was to select
    the primary. Two of the three are recoverable as
    `reverse_telephoto_source(wavelength_um=0.4861)` and `0.6563` when a test wants
    them; none is in the repository today.
    """
    return SourceSpec(
        wavelength_um=wavelength_um,
        field_angle_deg=field_angle_deg,
        object_distance_mm=None,  # object at infinity
    )


# --- M3-SINGLET-FINITE ------------------------------------------------------
#
# CHE-207 (R05.5). The **one** finite-object system this repository owns, added
# because the finite-conjugate launch state and the point-source OPL reference
# cannot be tested without one -- which is the single exception CHE-46's non-goal
# allowed ("no new prescription unless a finite-object system is required to test
# this, in which case add exactly one and freeze it").
#
# It is deliberately the same *lens* as M3-SINGLET-REF -- same radius, same ideal
# index, same centre thickness, same entrance pupil -- with only the conjugate
# changed. So a difference between the two traces is a difference in the source
# geometry and nothing else, which is what makes the collimated system a control
# for the finite one rather than merely a neighbour.
#
# The object sits at 2f, so the magnification is exactly -1 and the image lands
# one object distance beyond the lens. Optiland's paraxial solver reports
# -1.000000000000 for it, which is the arithmetic confirmation that the spacing
# below is the conjugate rather than approximately the conjugate.

#: Object distance for unit magnification: 2f from the front vertex. Derived, so a
#: change to the index or the radius moves it rather than leaving it stale.
FINITE_CONJUGATE_OBJECT_DISTANCE_MM = 2.0 * SINGLET_EFFECTIVE_FOCAL_LENGTH_MM

#: The paraxial magnification the object distance above implies. Exact, not
#: measured: an object at 2f images at 2f.
FINITE_CONJUGATE_MAGNIFICATION = -1.0


def finite_conjugate_image_distance_mm(
    object_distance_mm: float = FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
) -> float:
    """Rear-vertex-to-image distance for the singlet, from a closed-form paraxial trace.

    Derived rather than measured, for the same reason
    `SINGLET_BACK_FOCAL_LENGTH_MM` is: a spacing frozen as a literal beside an
    index that later changes is a lens nobody notices is wrong.

    A paraxial ray leaves the axial object point and reaches the front vertex at
    height `h`, so its angle there is `u = h / d`. Then, with
    `n' u' = n u - y (n' - n) / R` at each surface and a transfer through the glass:

        u1' = (u - h (n - 1) / R) / n         refraction at the convex face
        y2  = h + t u1'                      transfer through the centre thickness
        u2' = n u1'                          the plane rear face, index change only
        v   = -y2 / u2'                      where the ray crosses the axis

    `h` cancels, so the result is independent of the ray chosen -- which is what
    makes it paraxial rather than a particular ray's crossing. Verified against the
    trace by `tests/physics/test_optiland_finite_conjugate.py`.
    """
    if not object_distance_mm > 0.0:
        raise ValueError(
            f"object_distance_mm={object_distance_mm!r} must be positive; a point source "
            "sits before the first surface"
        )
    height = 1.0
    angle = height / object_distance_mm
    inside = (angle - height * (SINGLET_REFRACTIVE_INDEX - 1.0) / SINGLET_RADIUS_MM) / (
        SINGLET_REFRACTIVE_INDEX
    )
    rear_height = height + SINGLET_CENTER_THICKNESS_MM * inside
    rear_angle = SINGLET_REFRACTIVE_INDEX * inside
    return -rear_height / rear_angle


def finite_conjugate_singlet(
    object_distance_mm: float = FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
) -> OpticalSetup:
    """M3-SINGLET-FINITE: the M3 singlet configured for a finite conjugate.

    A function rather than a constant for the same reason `singlet_ref` is one: a
    test that wants to vary the conjugate has something to start from. The image
    spacing follows the object distance through the closed form above, so varying
    one keeps the system at its conjugate -- which is why this setup takes an
    object distance even though it declares no source. The distance is not
    illumination here; it is what places the last surface. Pass the same value to
    `finite_conjugate_source` and the pair stays conjugate.
    """
    return OpticalSetup(
        name="M3SingletFinite",
        description=(
            "M3-SINGLET-FINITE: the M3-SINGLET-REF lens configured for a POINT SOURCE at "
            "2f, so the magnification is -1. Added by CHE-207 as the one finite-object "
            "system needed to verify the point-source OPL reference."
        ),
        surfaces=(
            SurfaceSpec(
                radius_mm=SINGLET_RADIUS_MM,
                thickness_mm=SINGLET_CENTER_THICKNESS_MM,
                material={"kind": "ideal", "refractive_index": SINGLET_REFRACTIVE_INDEX},
                comment="convex front face, and the aperture stop",
            ),
            SurfaceSpec(
                thickness_mm=finite_conjugate_image_distance_mm(object_distance_mm),
                comment="plane rear face",
            ),
        ),
        stop_index=0,
        entrance_pupil_diameter_mm=SINGLET_ENTRANCE_PUPIL_DIAMETER_MM,
        reference_wavelength_um=SINGLET_WAVELENGTH_UM,
    )


def finite_conjugate_source(
    *,
    object_distance_mm: float = FINITE_CONJUGATE_OBJECT_DISTANCE_MM,
    field_angle_deg: tuple[float, float] = (0.0, 0.0),
    wavelength_um: float = SINGLET_WAVELENGTH_UM,
) -> SourceSpec:
    """The point source M3-SINGLET-FINITE images, at 2f before the front vertex.

    Pair it with `finite_conjugate_singlet(object_distance_mm)` at the same
    distance: the setup's image spacing is derived from it, so the two are
    conjugate only when they agree. The default is the unit-magnification 2f case.

    `field_angle_deg` is what makes the off-axis point-source path testable at all:
    for a finite object a field angle is a *position*, so at `(0, 2)` the source
    moves to `(0, -tan(2 deg) * d, -d)` and the launch state has to stay a single
    point for the CHE-207 reference to hold.
    """
    return SourceSpec(
        wavelength_um=wavelength_um,
        field_angle_deg=field_angle_deg,
        object_distance_mm=object_distance_mm,
    )
