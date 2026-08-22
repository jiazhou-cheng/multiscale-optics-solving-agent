"""Canonical prescriptions for the adapter-owned optical systems (CHE-56 / PB5).

This module is the authoritative representation of every optical system the
Optiland adapter supports by name. Each entry is a plain
:class:`~core.optical_system.OpticalSystemSpec` value,
built by the one generic builder in
:mod:`adapters.optiland_builder`; none of them carries
its own construction code, and this module imports no solver.

Adding a system means adding a prescription here (or handing one to the adapter
inline through ``config['prescription']``) -- not writing a new builder. See
``docs/prescriptions/canonical_optical_systems.md``.
"""

from __future__ import annotations

from core.optical_system import (
    ApertureSpec,
    CatalogMaterialSpec,
    EvenAsphereGeometrySpec,
    FieldSpec,
    GratingInteractionSpec,
    IdealMaterialSpec,
    OpticalSystemSpec,
    PlaneGeometrySpec,
    PrescriptionError,
    SphericalGeometrySpec,
    SurfaceSpec,
    WavelengthSpec,
)

# --- M3-SINGLET-REF ---------------------------------------------------------
#
# Frozen by M3.2 in benchmarks/slice_protocol.yaml. Plano-convex, convex toward
# the collimated side (the low-aberration orientation), real refractive surfaces
# because CHE-30 ruled out surface_type='paraxial' as an OPL source. An ideal
# constant-index material keeps it independent of any glass catalog. Scaled to
# 1/10 of a 25 mm-radius prescription -- since CHE-40 that is a cost choice
# rather than a numerical necessity, but the frozen protocol names these numbers.
#
# The derived quantities stay derived, exactly as the superseded hand-written
# builder had them, so the prescription and the protocol cannot drift apart.

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

M3_SINGLET_REF = OpticalSystemSpec(
    name="M3SingletRef",
    description=(
        "M3-SINGLET-REF (benchmarks/slice_protocol.yaml): plano-convex singlet, "
        "convex toward the collimated side, admitting an analytic Airy oracle."
    ),
    object_distance_mm=None,  # object at infinity
    surfaces=(
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=SINGLET_RADIUS_MM),
            thickness_mm=SINGLET_CENTER_THICKNESS_MM,
            material=IdealMaterialSpec(refractive_index=SINGLET_REFRACTIVE_INDEX),
            is_stop=True,
        ),
        # Rear vertex: glass -> air, then the image plane one back focal length on.
        SurfaceSpec(
            geometry=PlaneGeometrySpec(),
            thickness_mm=SINGLET_BACK_FOCAL_LENGTH_MM,
        ),
    ),
    aperture=ApertureSpec(value_mm=SINGLET_ENTRANCE_PUPIL_DIAMETER_MM),
    fields=(FieldSpec(y_deg=0.0),),
    wavelengths=(WavelengthSpec(value_um=SINGLET_WAVELENGTH_UM, is_primary=True),),
)


# --- M3-REVERSE-TELEPHOTO ---------------------------------------------------
#
# The prescription of `optiland.samples.objectives.ReverseTelephoto`, transcribed
# into the canonical schema so the adapter no longer depends on the bundled
# sample class to build it. Every radius, thickness, glass, field and wavelength
# below is the sample's own value; the bundled class remains in the tests as an
# independent structural and numerical oracle for this transcription.
#
# The `expected_catalog_file` on each glass is not decoration. Optiland resolves
# a bare name by substring filter plus Levenshtein ranking, so `SK15` selects
# HIKARI while `N-SK10` selects SCHOTT, and up to seven rows survive the filter
# before scoring. Recording the winning file (measured by
# knowledge/solvers/optiland/probes/system_construction_probe.py) turns a future
# catalog change into a structured error rather than a quietly different trace.


def _glass(name: str, catalog_file: str, catalog: str | None = None) -> CatalogMaterialSpec:
    return CatalogMaterialSpec(
        name=name,
        catalog=catalog,
        expected_catalog_file=catalog_file,
    )


REVERSE_TELEPHOTO = OpticalSystemSpec(
    name="ReverseTelephoto",
    description=(
        "M3-REVERSE-TELEPHOTO: the bundled Optiland reverse-telephoto objective "
        "already validated in M1 (L1-RAY-01), transcribed into the canonical schema."
    ),
    object_distance_mm=None,  # object at infinity
    surfaces=(
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=1.69111096),
            thickness_mm=0.08259680,
            material=_glass("N-SK10", "glass/schott/N-SK10.yml"),
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=0.94414496),
            thickness_mm=0.8,
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=4.32100401),
            thickness_mm=0.080256,
            material=_glass("SK15", "glass/hikari/SK15.yml"),
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=1.78117621),
            thickness_mm=0.5,
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=2.64050282),
            thickness_mm=0.27638160,
            material=_glass("BASF2", "glass/hikari/BASF2.yml"),
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=-3.86177348),
            thickness_mm=0.1,
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=1.05627661),
            thickness_mm=0.2,
            material=_glass("FK3", "glass/schott/FK3.yml"),
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=-4.06933311),
            thickness_mm=0.2001384,
        ),
        SurfaceSpec(
            geometry=PlaneGeometrySpec(),
            thickness_mm=0.06688,
            is_stop=True,
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=-2.61246583),
            thickness_mm=0.064372,
            material=_glass("SF15", "glass/hikari/SF15.yml", catalog="hikari"),
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=0.99117409),
            thickness_mm=0.3,
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=9.03045960),
            thickness_mm=0.18743120,
            material=_glass("N-LAK12", "glass/schott/N-LAK12.yml"),
        ),
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=-1.35680743),
            thickness_mm=2.35130547,
        ),
    ),
    aperture=ApertureSpec(value_mm=0.3),
    fields=(FieldSpec(y_deg=0.0), FieldSpec(y_deg=21.0), FieldSpec(y_deg=30.0)),
    wavelengths=(
        WavelengthSpec(value_um=0.4861),
        WavelengthSpec(value_um=0.5876, is_primary=True),
        WavelengthSpec(value_um=0.6563),
    ),
)


# --- Phase 3 feature-coverage demonstration ------------------------------------
#
# One prescription that exercises every geometry, interaction and material
# category the Phase 3 schema admits: an even-aspheric catalog-glass element, a
# spherical exit face, and a plane transmission grating on an ideal-index
# substrate. It exists so the generic builder's full supported surface is
# demonstrated by a system defined *directly* in the canonical schema rather
# than transcribed from a bundled sample.
#
# It is deliberately NOT registered as a named adapter prescription. It is a
# feature demonstration, not a validated optical design: no analytic oracle, no
# frozen protocol, no diffraction-limited claim. A caller reaches it the way any
# caller reaches an inline prescription -- through `config['prescription']`.

PHASE3_FEATURE_DEMO = OpticalSystemSpec(
    name="Phase3FeatureDemo",
    description=(
        "Feature-coverage demonstration for the Phase 3 canonical schema: "
        "even-aspheric and spherical geometry, refractive and grating "
        "interactions, and catalog, ideal and air materials in one system. "
        "Not a validated optical design."
    ),
    object_distance_mm=None,  # object at infinity
    surfaces=(
        # Even asphere on catalog glass, and the aperture stop.
        SurfaceSpec(
            geometry=EvenAsphereGeometrySpec(
                radius_mm=20.0,
                conic=-0.5,
                coefficients=(1.0e-4, -2.0e-6),
            ),
            thickness_mm=4.0,
            material=CatalogMaterialSpec(
                name="N-BK7",
                expected_catalog_file="glass/schott/N-BK7.yml",
            ),
            is_stop=True,
            comment="even-aspheric front face on catalog glass",
        ),
        # Spherical exit face into air.
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=-30.0),
            thickness_mm=2.0,
            comment="spherical exit face",
        ),
        # Plane transmission grating ruled on an ideal-index substrate. The
        # period is micrometres and the groove orientation radians; at 0 the
        # grating vector is +y, so the first order is deviated in y.
        SurfaceSpec(
            geometry=PlaneGeometrySpec(),
            interaction=GratingInteractionSpec(
                order=1,
                period_um=10.0,
                groove_orientation_rad=0.0,
            ),
            thickness_mm=1.5,
            material=IdealMaterialSpec(refractive_index=1.5),
            comment="plane transmission grating, first order",
        ),
        # Plane exit face back into air, then the image plane.
        SurfaceSpec(
            geometry=PlaneGeometrySpec(),
            thickness_mm=36.0,
            comment="plane exit face",
        ),
    ),
    aperture=ApertureSpec(value_mm=10.0),
    fields=(FieldSpec(y_deg=0.0), FieldSpec(y_deg=2.0)),
    wavelengths=(
        WavelengthSpec(value_um=0.486),
        WavelengthSpec(value_um=0.55, is_primary=True),
        WavelengthSpec(value_um=0.656),
    ),
)


#: Every optical system this repository's Optiland adapter supports by name.
#: A tuple of pairs rather than a dict literal so the declaration order is the
#: iteration order, and construction can never depend on mapping insertion
#: details.
_CANONICAL_PRESCRIPTIONS: tuple[tuple[str, OpticalSystemSpec], ...] = (
    ("ReverseTelephoto", REVERSE_TELEPHOTO),
    ("M3SingletRef", M3_SINGLET_REF),
)

#: Names in declaration order.
PRESCRIPTION_NAMES: tuple[str, ...] = tuple(name for name, _ in _CANONICAL_PRESCRIPTIONS)


def prescription_names() -> tuple[str, ...]:
    """The supported prescription names, in declaration order."""
    return PRESCRIPTION_NAMES


def resolve_prescription(name: str) -> OpticalSystemSpec:
    """Look up a canonical prescription by name.

    Raises:
        PrescriptionError: the name is not one of the canonical prescriptions.
    """
    for candidate, spec in _CANONICAL_PRESCRIPTIONS:
        if candidate == name:
            return spec
    raise PrescriptionError(
        "PRESCRIPTION_NAME_UNKNOWN",
        f"{name!r} is not a canonical prescription",
        path="name",
        expected="one of the canonical prescription names",
        supported=PRESCRIPTION_NAMES,
    )


__all__ = [
    "M3_SINGLET_REF",
    "PHASE3_FEATURE_DEMO",
    "PRESCRIPTION_NAMES",
    "REVERSE_TELEPHOTO",
    "prescription_names",
    "resolve_prescription",
]
