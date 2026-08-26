"""Commercial optical components, as published by their manufacturer (CHE-139).

This module is the *input* half of the M1.1.5 benchmark: what the manufacturer
says, where it says it, and how each published number becomes a field of
``optical-component-spec/1``. It imports no solver and performs no trace.

The rule the whole benchmark rests on
-------------------------------------
**Every number below is traceable to a manufacturer document, or it is absent.**
A radius, a thickness, a glass, an element count or a spacing that a vendor does
not publish is not estimated, not inferred from a focal length, and not copied
from a similar part. It is recorded as a :class:`ConstructionRefusal` and the
component is not built. ``M-10X`` is in this file precisely because it fails that
test, and it is kept rather than quietly dropped: a benchmark that only contains
the components that worked cannot demonstrate that it would have refused one that
did not.

Two source kinds, and why both
------------------------------
``product_page`` is the vendor's current published specification table.
``vendor_zemax_file`` is the vendor's own optical-design file, linked from the
same product page's Resources & Downloads section. The second is not decoration:
it independently confirms the radii, thicknesses, glasses, surface *order* and
entrance-pupil diameter read off the table, and where the two disagree the
disagreement is recorded as a :class:`SourceDisagreement` rather than silently
resolved in favour of whichever was more convenient. Two such disagreements
exist and both are real -- see ``_LEGACY_GLASS_NAMING`` and
``_IMAGE_DISTANCE_REFERENCE``.

Aperture, and what this schema cannot say
-----------------------------------------
Every component here publishes a clear aperture. ``optical-system-spec/1`` has
no per-surface aperture field, so the built system has no physical rim and no
ray is ever vignetted by one. That gap is recorded per component as an
:class:`UnrepresentableParameter` with its consequence stated, and the benchmark
keeps the entrance pupil inside the clear aperture instead of pretending the rim
is modelled. It is not closed by inventing a schema field: doing so would move
every recorded prescription fingerprint in the repository.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_SRC) not in sys.path:  # pragma: no cover - import-path bootstrap
    sys.path.insert(0, str(_SRC))

from core.optical_assembly import ComponentSpec  # noqa: E402
from core.optical_system import (  # noqa: E402
    CatalogMaterialSpec,
    PlaneGeometrySpec,
    PrescriptionError,
    SphericalGeometrySpec,
)

#: The date every value in this file was read from its source document. One date
#: for the whole file rather than one per value: they were all read in the same
#: session, and a per-value date that is really a copy of one date is a lie with
#: extra structure.
RETRIEVED_UTC = "2026-08-26"

#: Newport publishes clear aperture as a *fraction of the diameter* rather than
#: as a length, so the length is derived. The factor is the published one; it is
#: named here so the derivation appears once and every component cites it.
_CLEAR_APERTURE_FRACTION = 0.90


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceDocument(_Frozen):
    """One manufacturer document, and what it is."""

    key: str = Field(min_length=1)
    kind: Literal["product_page", "vendor_zemax_file"]
    url: str = Field(min_length=1)
    retrieved_utc: str = RETRIEVED_UTC
    note: str = ""


class PublishedValue(_Frozen):
    """One value a manufacturer published, with the words it published it in.

    ``verbatim`` is the source text, so a reviewer can check the transcription
    without re-fetching the page. ``derivation`` is empty when ``value`` *is* the
    published number and non-empty when it was computed from the published
    statement -- which is the only arithmetic allowed on a source value here, and
    it must be stated.
    """

    value: float | str
    unit: str
    verbatim: str = Field(min_length=1)
    source_key: str = Field(min_length=1)
    derivation: str = ""


class SourceDisagreement(_Frozen):
    """Two manufacturer documents that do not say the same thing.

    Recorded, not resolved away. ``resolution`` states which reading the
    benchmark uses and ``basis`` states why, so the choice is reviewable.
    """

    parameter: str = Field(min_length=1)
    readings: tuple[str, ...]
    resolution: str = Field(min_length=1)
    basis: str = Field(min_length=1)


class UnrepresentableParameter(_Frozen):
    """A published parameter the canonical schema cannot express.

    Distinct from a refusal: the component still builds, but something the
    manufacturer stated did not survive into the model, and ``consequence`` says
    what that costs the result.
    """

    parameter: str = Field(min_length=1)
    published: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    consequence: str = Field(min_length=1)


class ConstructionRefusal(_Frozen):
    """Why a component could not be built from what the manufacturer publishes."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    missing_parameters: tuple[str, ...]
    published_but_insufficient: tuple[str, ...]


class CatalogComponent(_Frozen):
    """A commercial part: its sources, its published values, and its model.

    ``component`` is ``None`` exactly when ``refusal`` is set. That pairing is
    enforced rather than trusted, because "unsupported" quietly carrying a
    half-built component is the failure mode this whole file exists to prevent.
    """

    part_number: str = Field(min_length=1)
    vendor: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    sources: tuple[SourceDocument, ...]
    published: dict[str, PublishedValue]
    disagreements: tuple[SourceDisagreement, ...] = ()
    unrepresentable: tuple[UnrepresentableParameter, ...] = ()
    component: ComponentSpec | None = None
    refusal: ConstructionRefusal | None = None
    #: For a supported component: which ``published`` key each surface radius and
    #: each internal thickness of ``component`` was transcribed from, in surface
    #: order. ``""`` in ``radius_keys`` means "this surface is a plane and the
    #: manufacturer publishes no radius for it". These exist so the transcription
    #: is *checked* rather than trusted -- see :meth:`_check_transcription`.
    radius_keys: tuple[str, ...] = ()
    thickness_keys: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_component(self) -> CatalogComponent:
        if (self.component is None) == (self.refusal is None):
            raise PrescriptionError(
                "CATALOG_COMPONENT_STATE_AMBIGUOUS",
                f"{self.part_number!r} sets "
                f"{'both' if self.component is not None else 'neither'} of "
                "component/refusal",
                path="component",
                expected="exactly one of a built ComponentSpec or a ConstructionRefusal",
            )
        source_keys = {source.key for source in self.sources}
        for name, value in self.published.items():
            if value.source_key not in source_keys:
                raise PrescriptionError(
                    "CATALOG_PUBLISHED_VALUE_UNSOURCED",
                    f"published[{name!r}].source_key={value.source_key!r} names no "
                    f"source document of {self.part_number!r}",
                    path=f"published.{name}",
                    expected=f"one of {sorted(source_keys)}",
                )
        return self._check_transcription()

    def _check_transcription(self) -> CatalogComponent:
        """Every built radius and thickness must equal the published value it cites.

        ``published`` and ``component`` are written separately in this file -- the
        first as the manufacturer's words, the second as the model -- and until
        this check existed nothing tied them together. A digit transposed between
        the two would have produced a component that traced perfectly and
        represented a lens nobody sells, with the correct value sitting three
        lines above it in the same literal. That is the single most likely way a
        fabricated optical parameter enters this benchmark, so it is refused at
        import time rather than reviewed by eye.

        Exact float equality is the right comparison here and not a strict one:
        both sides are the same decimal literal transcribed twice, so any
        difference at all is a transcription error rather than round-off.
        """
        if self.component is None:
            if self.radius_keys or self.thickness_keys:
                raise PrescriptionError(
                    "CATALOG_TRANSCRIPTION_KEYS_ON_REFUSED_COMPONENT",
                    f"{self.part_number!r} is refused but declares transcription keys",
                    path="radius_keys",
                    expected="no transcription keys on a component that was not built",
                )
            return self
        geometries = self.component.geometries
        thicknesses = self.component.internal_thicknesses_mm
        if len(self.radius_keys) != len(geometries):
            raise PrescriptionError(
                "CATALOG_TRANSCRIPTION_RADIUS_KEYS_MISSING",
                f"{self.part_number!r} has {len(geometries)} surfaces but "
                f"{len(self.radius_keys)} radius_keys",
                path="radius_keys",
                expected=(
                    "one published-value key per surface, in surface order; \"\" for a "
                    "plane the manufacturer publishes no radius for"
                ),
            )
        if len(self.thickness_keys) != len(thicknesses):
            raise PrescriptionError(
                "CATALOG_TRANSCRIPTION_THICKNESS_KEYS_MISSING",
                f"{self.part_number!r} has {len(thicknesses)} internal thicknesses but "
                f"{len(self.thickness_keys)} thickness_keys",
                path="thickness_keys",
                expected="one published-value key per internal thickness, in order",
            )
        for index, (key, geometry) in enumerate(zip(self.radius_keys, geometries, strict=True)):
            if key == "":
                if not isinstance(geometry, PlaneGeometrySpec):
                    raise PrescriptionError(
                        "CATALOG_TRANSCRIPTION_PLANE_EXPECTED",
                        f"{self.part_number!r} surface {index} cites no published "
                        f"radius but is not a plane ({geometry.kind.value})",
                        path=f"radius_keys[{index}]",
                        expected="a plane surface, or a published radius key",
                    )
                continue
            if isinstance(geometry, PlaneGeometrySpec):
                raise PrescriptionError(
                    "CATALOG_TRANSCRIPTION_RADIUS_ON_PLANE",
                    f"{self.part_number!r} surface {index} is a plane but cites "
                    f"published radius {key!r}",
                    path=f"radius_keys[{index}]",
                    expected='"" for a plane surface',
                )
            published = self.number(key)
            built = float(geometry.resolved_radius_mm)  # type: ignore[union-attr]
            if built != published:
                raise PrescriptionError(
                    "CATALOG_TRANSCRIPTION_RADIUS_MISMATCH",
                    f"{self.part_number!r} surface {index} is built with radius "
                    f"{built!r} mm but published[{key!r}] says {published!r} mm",
                    path=f"component.geometries[{index}]",
                    expected="the built radius to equal the published value it cites",
                )
        material_key = "material"
        if material_key in self.published:
            published_materials = str(self.published[material_key].value).upper()
            for index, material in enumerate(self.component.internal_materials):
                name = getattr(material, "name", None)
                if name is None:
                    # An air gap or an ideal index inside a part is legitimate and
                    # has no catalog name to check against.
                    continue
                if str(name).upper() not in published_materials:
                    raise PrescriptionError(
                        "CATALOG_TRANSCRIPTION_MATERIAL_MISMATCH",
                        f"{self.part_number!r} internal medium {index} is built as "
                        f"{name!r}, which does not appear in the published "
                        f"material {self.published[material_key].value!r}",
                        path=f"component.internal_materials[{index}]",
                        expected=(
                            "a catalog glass whose name the manufacturer's published "
                            "material string actually contains"
                        ),
                    )
        for index, (key, thickness) in enumerate(
            zip(self.thickness_keys, thicknesses, strict=True)
        ):
            published = self.number(key)
            if float(thickness) != published:
                raise PrescriptionError(
                    "CATALOG_TRANSCRIPTION_THICKNESS_MISMATCH",
                    f"{self.part_number!r} internal thickness {index} is built as "
                    f"{thickness!r} mm but published[{key!r}] says {published!r} mm",
                    path=f"component.internal_thicknesses_mm[{index}]",
                    expected="the built thickness to equal the published value it cites",
                )
        return self

    @property
    def supported(self) -> bool:
        return self.component is not None

    def number(self, key: str) -> float:
        """A published numeric value, by key. Raises if it is absent or textual."""
        try:
            entry = self.published[key]
        except KeyError as exc:
            raise PrescriptionError(
                "CATALOG_PUBLISHED_VALUE_MISSING",
                f"{self.part_number!r} publishes no {key!r}",
                path=f"published.{key}",
                expected=f"one of {sorted(self.published)}",
            ) from exc
        if not isinstance(entry.value, float | int):
            raise PrescriptionError(
                "CATALOG_PUBLISHED_VALUE_NOT_NUMERIC",
                f"{self.part_number!r} publishes {key!r} as text ({entry.value!r})",
                path=f"published.{key}",
                expected="a numeric published value",
            )
        return float(entry.value)

    def source_urls(self) -> tuple[str, ...]:
        return tuple(source.url for source in self.sources)

    def as_record(self) -> dict[str, Any]:
        """JSON-safe provenance block, key-sorted where order carries no meaning."""
        return {
            "part_number": self.part_number,
            "vendor": self.vendor,
            "summary": self.summary,
            "supported": self.supported,
            "sources": [source.model_dump(mode="json") for source in self.sources],
            "published": {
                key: self.published[key].model_dump(mode="json")
                for key in sorted(self.published)
            },
            "source_disagreements": [d.model_dump(mode="json") for d in self.disagreements],
            "unrepresentable_parameters": [
                u.model_dump(mode="json") for u in self.unrepresentable
            ],
            "normalized_component": (
                self.component.model_dump(mode="json") if self.component is not None else None
            ),
            "transcription_check": {
                "radius_keys": list(self.radius_keys),
                "thickness_keys": list(self.thickness_keys),
                "enforced": (
                    "every built radius and internal thickness is required at import "
                    "time to equal the published value it cites, by exact float "
                    "equality; see CatalogComponent._check_transcription"
                ),
            },
            "construction_refusal": (
                self.refusal.model_dump(mode="json") if self.refusal is not None else None
            ),
        }


# --- shared source-level findings -------------------------------------------
#
# Both disagreements below hold for more than one component, so they are written
# once and referenced, rather than restated per part where they could drift.


def _legacy_glass_naming(page_names: str, file_names: str) -> SourceDisagreement:
    """The product page names modern Schott glasses; the Zemax file names legacy ones.

    Not a transcription error on either side: ``BK7`` and ``N-BK7`` are genuinely
    different melts, and a vendor design file written years ago legitimately
    names the older one. The benchmark takes the product page, which is the
    vendor's *current* statement about the part being sold.

    The choice happens to be inconsequential on the pinned install, and that is
    a measurement rather than an assumption: ``Material(name=...)`` resolves
    'BK7' and 'N-BK7' both to ``glass/schott/N-BK7.yml``, and 'SF5' and 'N-SF5'
    both to ``glass/schott/N-SF5.yml``, each with Optiland's own
    ``similarity_score`` of 0 -- the pinned catalog carries no legacy entry to
    select instead. So either reading builds the same material here. The
    disagreement is still recorded, because on a catalog that *did* carry both
    the two readings would diverge and this note is what would explain it.
    """
    return SourceDisagreement(
        parameter="glass_names",
        readings=(
            f"product_page: {page_names}",
            f"vendor_zemax_file: {file_names}",
        ),
        resolution=f"product page ({page_names})",
        basis=(
            "the product page is the vendor's current statement about the part on "
            "sale; measured on the pinned Optiland 0.6.0 catalog, both spellings "
            "resolve to the same glass/schott/*.yml file with similarity_score 0, "
            "so the choice does not change the built material on this install"
        ),
    )


def _image_distance_reference(page_bfl_mm: float, file_disz_mm: float) -> SourceDisagreement:
    """The page's BFL and the Zemax file's image distance are at different wavelengths.

    Newport's specification table quotes BFL at the part's 589 nm design
    wavelength; the vendor Zemax file's last air distance is the image plane it
    ships, whose primary wavelength is 546.1 nm. Since ``BFL`` for a positive
    singlet in a normally dispersive glass shortens as the wavelength shortens,
    the file's value being the smaller of the two is the expected direction, not
    a contradiction. The benchmark uses the page value, and the wavelength it
    traces is the page's design wavelength, so the two agree with each other.
    """
    return SourceDisagreement(
        parameter="image_distance_from_last_vertex_mm",
        readings=(
            f"product_page BFL: {page_bfl_mm} mm at the stated 589 nm design wavelength",
            f"vendor_zemax_file last air distance: {file_disz_mm} mm, whose primary "
            "wavelength (PWAV) is 546.1 nm",
        ),
        resolution=f"product page BFL ({page_bfl_mm} mm)",
        basis=(
            "the benchmark traces the product page's stated design wavelength, so "
            "the image plane and the trace wavelength come from one document; the "
            "file's shorter distance is the expected direction for a shorter "
            "primary wavelength in a normally dispersive glass, not a conflict"
        ),
    )


#: The clear-aperture gap, identical for every Newport component here.
_CLEAR_APERTURE_NOT_MODELLED = UnrepresentableParameter(
    parameter="clear_aperture",
    published="Clear Aperture: >= central 90% of diameter",
    reason=(
        "optical-system-spec/1 has no per-surface aperture field, so the built "
        "system carries no physical rim; adding one would change the canonical "
        "normalization and move every recorded prescription fingerprint in the "
        "repository, which is far outside this benchmark's scope"
    ),
    consequence=(
        "no ray is vignetted by a component rim, so a reported clipped-ray count "
        "reflects trace failure (a missed or totally internally reflected surface) "
        "only, never aperture vignetting. The benchmark instead keeps the entrance "
        "pupil inside the clear aperture and measures the ray footprint at each "
        "downstream component vertex, so the constraint is checked rather than "
        "assumed -- see the aperture_clearance block of each multi-lens record."
    ),
)


def _n_bk7() -> CatalogMaterialSpec:
    """Schott N-BK7, pinned to the catalog file it must resolve to."""
    return CatalogMaterialSpec(name="N-BK7", expected_catalog_file="glass/schott/N-BK7.yml")


def _n_sf5() -> CatalogMaterialSpec:
    return CatalogMaterialSpec(name="N-SF5", expected_catalog_file="glass/schott/N-SF5.yml")


def _clear_aperture(diameter_mm: float) -> float:
    return _CLEAR_APERTURE_FRACTION * diameter_mm


# --- KPX094: N-BK7 plano-convex singlet -------------------------------------

_KPX094_PAGE = SourceDocument(
    key="product_page",
    kind="product_page",
    url="https://www.newport.com/p/KPX094",
    note="Newport (MKS) specification table: 'Technical Specs' section",
)
_KPX094_ZEMAX = SourceDocument(
    key="vendor_zemax_file",
    kind="vendor_zemax_file",
    url=(
        "https://api.p1.mks.com/medias/sys_master/images/images/hb8/h05/"
        "8797167124510/KPX094-ZEMAX.zip"
    ),
    note=(
        "'KPX094_ZEMAX' under Resources & Downloads on the product page; "
        "KPX094.ZMX, UNIT MM, surface 1 CURV 1.934984520123839800E-002 "
        "(radius 51.68000 mm) DISZ 4.585 GLAS BK7, surface 2 plane, ENPD 2.286E+1, "
        "STOP on surface 1"
    ),
)

KPX094 = CatalogComponent(
    part_number="KPX094",
    vendor="Newport (MKS Instruments)",
    summary="Plano-convex singlet, N-BK7, 25.4 mm diameter, 100 mm EFL, uncoated",
    sources=(_KPX094_PAGE, _KPX094_ZEMAX),
    published={
        "lens_shape": PublishedValue(
            value="plano-convex", unit="", verbatim="Lens Shape | Plano-Convex",
            source_key="product_page",
        ),
        "diameter_mm": PublishedValue(
            value=25.4, unit="mm", verbatim="Diameter | 25.4 mm", source_key="product_page",
        ),
        "material": PublishedValue(
            value="N-BK7", unit="", verbatim="Lens Material | N-BK7",
            source_key="product_page",
        ),
        "radius_1_mm": PublishedValue(
            value=51.680, unit="mm", verbatim="Radius of Curvature (R) | 51.680 mm",
            source_key="product_page",
        ),
        "center_thickness_mm": PublishedValue(
            value=4.585, unit="mm", verbatim="Center Thickness (Tc) | 4.585 mm",
            source_key="product_page",
        ),
        "edge_thickness_mm": PublishedValue(
            value=3.0, unit="mm", verbatim="Edge Thickness (Te) | 3.0 mm",
            source_key="product_page",
        ),
        "efl_mm": PublishedValue(
            value=100.0, unit="mm", verbatim="Effective Focal Length (EFL) | 100 mm",
            source_key="product_page",
        ),
        "bfl_mm": PublishedValue(
            value=96.97, unit="mm", verbatim="Back Focal Length (BFL) | 96.97 mm",
            source_key="product_page",
        ),
        "f_number": PublishedValue(
            value=3.9, unit="", verbatim="F/# | 3.9", source_key="product_page",
            derivation=(
                "the vendor's F/# is EFL / full diameter (100 / 25.4 = 3.94), not "
                "EFL / entrance pupil diameter; recorded so a simulated F-number is "
                "compared on a stated basis rather than assumed comparable"
            ),
        ),
        "design_wavelength_um": PublishedValue(
            value=0.589, unit="um", verbatim="Design Wavelength | 589 nm",
            source_key="product_page",
            derivation="589 nm expressed in the schema's micrometre wavelength unit",
        ),
        "clear_aperture_mm": PublishedValue(
            value=_clear_aperture(25.4), unit="mm",
            verbatim="Clear Aperture | >=central 90% of diameter",
            source_key="product_page",
            derivation=(
                "0.90 x the published 25.4 mm diameter = 22.86 mm; independently "
                "equal to the vendor Zemax file's own ENPD of 2.286E+1 mm"
            ),
        ),
        "entrance_pupil_diameter_mm": PublishedValue(
            value=22.86, unit="mm", verbatim="ENPD 2.286E+1",
            source_key="vendor_zemax_file",
            derivation=(
                "the vendor's own design file traces this part at 22.86 mm, which is "
                "exactly the 90% clear aperture the product page states; the "
                "benchmark uses the vendor's value rather than choosing its own pupil"
            ),
        ),
        "principal_plane_2_mm": PublishedValue(
            value=-3.02, unit="mm", verbatim="Principal Plane 2 (P2) | -3.02 mm",
            source_key="product_page",
        ),
    },
    disagreements=(
        _legacy_glass_naming("N-BK7", "BK7"),
        _image_distance_reference(96.97, 96.61074751007),
    ),
    unrepresentable=(_CLEAR_APERTURE_NOT_MODELLED,),
    component=ComponentSpec(
        name="KPX094",
        description=(
            "Newport KPX094 plano-convex singlet. Surface order is the vendor's own: "
            "the curved face (R = +51.680 mm, convex toward the object) first, the "
            "plane second -- confirmed by KPX094.ZMX, whose surface 1 carries the "
            "curvature and surface 2 is flat. This is also the low-aberration "
            "orientation for an infinite conjugate."
        ),
        geometries=(
            SphericalGeometrySpec(radius_mm=51.680),
            PlaneGeometrySpec(),
        ),
        internal_thicknesses_mm=(4.585,),
        internal_materials=(_n_bk7(),),
        clear_aperture_mm=_clear_aperture(25.4),
        # Orientation- and sign-neutral on purpose: a comment travels with its
        # surface through a reversal, so it may only name WHICH face of the part
        # this is. The radius and the installed orientation are reported beside
        # it, from the assembled prescription, where a reversal updates them.
        surface_comments=(
            "KPX094 curved face",
            "KPX094 plane face",
        ),
    ),
    radius_keys=("radius_1_mm", ""),
    thickness_keys=("center_thickness_mm",),
)


# --- KBX058: N-BK7 equiconvex singlet ---------------------------------------

_KBX058_PAGE = SourceDocument(
    key="product_page",
    kind="product_page",
    url="https://www.newport.com/p/KBX058",
    note="Newport (MKS) specification table: 'Technical Specs' section",
)
_KBX058_ZEMAX = SourceDocument(
    key="vendor_zemax_file",
    kind="vendor_zemax_file",
    url=(
        "https://api.p1.mks.com/medias/sys_master/images/images/h2a/hcb/"
        "8797131145246/KBX058-ZEMAX.zip"
    ),
    note=(
        "'KBX058_ZEMAX' under Resources & Downloads on the product page; "
        "KBX058.ZMX, UNIT MM, surface 1 CURV 1.294247071766000200E-002 "
        "(radius +77.26500 mm) DISZ 5.102 GLAS N-BK7, surface 2 CURV "
        "-1.294247071766000200E-002 (radius -77.265 mm, held by a scale -1 pickup "
        "solve on surface 1), ENPD 1.143E+1, STOP on surface 1"
    ),
)

KBX058 = CatalogComponent(
    part_number="KBX058",
    vendor="Newport (MKS Instruments)",
    summary="Equiconvex singlet, N-BK7, 25.4 mm diameter, 75.6 mm EFL, uncoated",
    sources=(_KBX058_PAGE, _KBX058_ZEMAX),
    published={
        "lens_shape": PublishedValue(
            value="bi-convex", unit="", verbatim="Lens Shape | Bi-Convex",
            source_key="product_page",
        ),
        "diameter_mm": PublishedValue(
            value=25.4, unit="mm", verbatim="Diameter | 25.4 mm", source_key="product_page",
        ),
        "material": PublishedValue(
            value="N-BK7", unit="", verbatim="Lens Material | N-BK7",
            source_key="product_page",
        ),
        "radius_1_mm": PublishedValue(
            value=77.265, unit="mm", verbatim="Radius of Curvature (R) | 77.265 mm",
            source_key="product_page",
        ),
        "radius_2_mm": PublishedValue(
            value=-77.265, unit="mm",
            verbatim="CURV -1.294247071766000200E-002 (surface 2, pickup solve '4 1 -1.')",
            source_key="vendor_zemax_file",
            derivation=(
                "the product page quotes ONE radius for a bi-convex part, which does "
                "not by itself say the second surface is its exact negative. The "
                "vendor Zemax file does: surface 2's curvature is held equal to "
                "surface 1's with scale -1, i.e. R2 = -R1 = -77.265 mm exactly. This "
                "value is read from the manufacturer's file, not inferred from the "
                "word 'bi-convex'."
            ),
        ),
        "center_thickness_mm": PublishedValue(
            value=5.102, unit="mm", verbatim="Center Thickness (Tc) | 5.102 mm",
            source_key="product_page",
        ),
        "edge_thickness_mm": PublishedValue(
            value=3.0, unit="mm", verbatim="Edge Thickness (Te) | 3.0 mm",
            source_key="product_page",
        ),
        "efl_mm": PublishedValue(
            value=75.6, unit="mm", verbatim="Effective Focal Length (EFL) | 75.6 mm",
            source_key="product_page",
        ),
        "bfl_mm": PublishedValue(
            value=73.89, unit="mm", verbatim="Back Focal Length (BFL) | 73.89 mm",
            source_key="product_page",
        ),
        "f_number": PublishedValue(
            value=2.9, unit="", verbatim="F/# | 2.9", source_key="product_page",
            derivation=(
                "EFL / full diameter (75.6 / 25.4 = 2.98); the vendor Zemax file "
                "instead traces this part at ENPD 11.43 mm, i.e. f/6.6"
            ),
        ),
        "design_wavelength_um": PublishedValue(
            value=0.589, unit="um", verbatim="Design Wavelength | 589 nm",
            source_key="product_page",
            derivation="589 nm expressed in the schema's micrometre wavelength unit",
        ),
        "clear_aperture_mm": PublishedValue(
            value=_clear_aperture(25.4), unit="mm",
            verbatim="Clear Aperture | >=central 90% of diameter",
            source_key="product_page",
            derivation="0.90 x the published 25.4 mm diameter = 22.86 mm",
        ),
        "entrance_pupil_diameter_mm": PublishedValue(
            value=11.43, unit="mm", verbatim="ENPD 1.143E+1",
            source_key="vendor_zemax_file",
            derivation=(
                "the vendor's own design file traces this part at 11.43 mm, half its "
                "clear aperture; the benchmark uses the vendor's value rather than "
                "choosing its own pupil for this component"
            ),
        ),
        "principal_plane_1_mm": PublishedValue(
            value=1.70, unit="mm", verbatim="Principal Plane 1 (P1) | 1.70 mm",
            source_key="product_page",
        ),
        "principal_plane_2_mm": PublishedValue(
            value=-1.70, unit="mm", verbatim="Principal Plane 2 (P2) | -1.70 mm",
            source_key="product_page",
        ),
    },
    disagreements=(_image_distance_reference(73.89, 72.94560765611),),
    unrepresentable=(_CLEAR_APERTURE_NOT_MODELLED,),
    component=ComponentSpec(
        name="KBX058",
        description=(
            "Newport KBX058 equiconvex singlet, R1 = +77.265 mm and R2 = -77.265 mm "
            "(the second radius taken from KBX058.ZMX's pickup solve, not inferred). "
            "Equiconvex, so the part is symmetric under reversal up to floating-point "
            "sign -- which makes it a useful second element in an assembly where only "
            "the FIRST component's orientation is meant to matter."
        ),
        geometries=(
            SphericalGeometrySpec(radius_mm=77.265),
            SphericalGeometrySpec(radius_mm=-77.265),
        ),
        internal_thicknesses_mm=(5.102,),
        internal_materials=(_n_bk7(),),
        clear_aperture_mm=_clear_aperture(25.4),
        surface_comments=(
            "KBX058 first convex face",
            "KBX058 second convex face",
        ),
    ),
    radius_keys=("radius_1_mm", "radius_2_mm"),
    thickness_keys=("center_thickness_mm",),
)


# --- PAC052: cemented achromatic doublet ------------------------------------

_PAC052_PAGE = SourceDocument(
    key="product_page",
    kind="product_page",
    url="https://www.newport.com/p/PAC052",
    note="Newport (MKS) specification table: 'Technical Specs' section",
)
_PAC052_ZEMAX = SourceDocument(
    key="vendor_zemax_file",
    kind="vendor_zemax_file",
    url=(
        "https://api.p1.mks.com/medias/sys_master/images/images/h56/h66/"
        "8797231284254/PAC052-ZEMAX.zip"
    ),
    note=(
        "'PAC052_ZEMAX' under Resources & Downloads on the product page; "
        "PAC052.ZMX, UNIT MM, surface 1 CURV 1.646334436377409900E-002 "
        "(radius 60.74100 mm) DISZ 5.0 GLAS BK7, surface 2 CURV "
        "-2.236636099306639800E-002 (radius -44.71000 mm) DISZ 2.17 GLAS SF5, "
        "surface 3 CURV -7.512922226229099700E-003 (radius -133.10400 mm), "
        "WAVM 4.861E-1 / 5.461E-1 / 6.563E-1 with PWAV 2, ENPD 2.286E+1, "
        "STOP on surface 1"
    ),
)

PAC052 = CatalogComponent(
    part_number="PAC052",
    vendor="Newport (MKS Instruments)",
    summary=(
        "Cemented achromatic doublet, N-BK7 / N-SF5, 25.4 mm diameter, 100 mm EFL, "
        "400-700 nm AR coated"
    ),
    sources=(_PAC052_PAGE, _PAC052_ZEMAX),
    published={
        "lens_type": PublishedValue(
            value="achromatic doublet", unit="",
            verbatim="Lens Type | Achromatic Doublet", source_key="product_page",
        ),
        "diameter_mm": PublishedValue(
            value=25.4, unit="mm", verbatim="Diameter | 25.4 mm", source_key="product_page",
        ),
        "material": PublishedValue(
            value="N-BK7/N-SF5", unit="", verbatim="Lens Material | N-BK7/N-SF5",
            source_key="product_page",
        ),
        "radius_1_mm": PublishedValue(
            value=60.741, unit="mm", verbatim="Radius of Curvature (R) | 60.741 mm",
            source_key="product_page",
        ),
        "radius_2_mm": PublishedValue(
            value=-44.710, unit="mm", verbatim="Radius 2 | -44.710 mm",
            source_key="product_page",
        ),
        "radius_3_mm": PublishedValue(
            value=-133.104, unit="mm", verbatim="Radius 3 | -133.104 mm",
            source_key="product_page",
        ),
        "center_thickness_1_mm": PublishedValue(
            value=5.0, unit="mm", verbatim="Thickness, Center 1 | 5.0 mm",
            source_key="product_page",
        ),
        "center_thickness_2_mm": PublishedValue(
            value=2.17, unit="mm", verbatim="Thickness, Center 2 | 2.17 mm",
            source_key="product_page",
        ),
        "center_thickness_total_mm": PublishedValue(
            value=7.17, unit="mm", verbatim="Center Thickness (Tc) | 7.17 mm",
            source_key="product_page",
            derivation=(
                "published independently of the two element thicknesses; it equals "
                "their sum (5.0 + 2.17), which is a consistency check on the "
                "transcription rather than an input"
            ),
        ),
        "edge_thickness_mm": PublishedValue(
            value=5.2, unit="mm", verbatim="Edge Thickness (Te) | 5.2 mm",
            source_key="product_page",
        ),
        "efl_mm": PublishedValue(
            value=100.0, unit="mm", verbatim="Effective Focal Length (EFL) | 100 mm",
            source_key="product_page",
        ),
        "bfl_mm": PublishedValue(
            value=96.5, unit="mm", verbatim="Back Focal Length (BFL) | 96.5 mm",
            source_key="product_page",
        ),
        "f_number": PublishedValue(
            value=3.9, unit="", verbatim="F/# | 3.9", source_key="product_page",
            derivation="EFL / full diameter (100 / 25.4 = 3.94)",
        ),
        "orientation": PublishedValue(
            value="steepest convex surface faces the infinite conjugate", unit="",
            verbatim="Orientation | Steepest convex surface should face the infinite conjugate",
            source_key="product_page",
            derivation=(
                "of the two outer surfaces, R1 = +60.741 mm and R3 = -133.104 mm are "
                "both convex outward and |R1| < |R3|, so the crown face R1 is the "
                "steepest convex surface and faces infinity. The vendor Zemax file "
                "lists the surfaces in exactly that order, so the reading is "
                "confirmed by a second manufacturer document rather than argued from "
                "the wording alone."
            ),
        ),
        "design_wavelength_um": PublishedValue(
            value=0.589, unit="um", verbatim="Design Wavelengths | 589 nm",
            source_key="product_page",
            derivation="589 nm expressed in the schema's micrometre wavelength unit",
        ),
        "chromatic_wavelength_short_um": PublishedValue(
            value=0.4861, unit="um", verbatim="WAVM 1 4.861E-1",
            source_key="vendor_zemax_file",
            derivation=(
                "the vendor's own design file evaluates this achromat at 486.1 / "
                "546.1 / 656.3 nm; the benchmark's chromatic characterization uses "
                "the vendor's short and long lines rather than lines of its own"
            ),
        ),
        "chromatic_wavelength_long_um": PublishedValue(
            value=0.6563, unit="um", verbatim="WAVM 3 6.563E-1",
            source_key="vendor_zemax_file",
        ),
        "entrance_pupil_diameter_mm": PublishedValue(
            value=22.86, unit="mm", verbatim="ENPD 2.286E+1",
            source_key="vendor_zemax_file",
        ),
        "clear_aperture_mm": PublishedValue(
            value=_clear_aperture(25.4), unit="mm",
            verbatim="Diameter | 25.4 mm (PAC-series page states no separate clear aperture)",
            source_key="product_page",
            derivation=(
                "the PAC052 page does NOT publish a clear-aperture line. The 90% "
                "figure is carried over from Newport's singlet pages, so it is an "
                "assumption about this part rather than a published value for it -- "
                "which is why the benchmark uses the vendor Zemax file's ENPD of "
                "22.86 mm as the pupil and treats this number only as the "
                "footprint-clearance reference it is compared against."
            ),
        ),
    },
    disagreements=(_legacy_glass_naming("N-BK7/N-SF5", "BK7/SF5"),),
    unrepresentable=(_CLEAR_APERTURE_NOT_MODELLED,),
    component=ComponentSpec(
        name="PAC052",
        description=(
            "Newport PAC052 cemented achromatic doublet: N-BK7 crown (R1 = +60.741, "
            "5.0 mm) cemented to an N-SF5 flint (R2 = -44.710, 2.17 mm) with an outer "
            "flint face R3 = -133.104. Written in the manufacturer's stated "
            "orientation -- steepest convex surface (R1) facing the infinite "
            "conjugate. There is no air inside the part: the cemented interface is "
            "one surface with glass on both sides."
        ),
        geometries=(
            SphericalGeometrySpec(radius_mm=60.741),
            SphericalGeometrySpec(radius_mm=-44.710),
            SphericalGeometrySpec(radius_mm=-133.104),
        ),
        internal_thicknesses_mm=(5.0, 2.17),
        internal_materials=(_n_bk7(), _n_sf5()),
        clear_aperture_mm=_clear_aperture(25.4),
        surface_comments=(
            "PAC052 crown outer face",
            "PAC052 cemented crown/flint interface",
            "PAC052 flint outer face",
        ),
    ),
    radius_keys=("radius_1_mm", "radius_2_mm", "radius_3_mm"),
    thickness_keys=("center_thickness_1_mm", "center_thickness_2_mm"),
)


# --- M-10X: published, and deliberately not built ---------------------------

_M10X_PAGE = SourceDocument(
    key="product_page",
    kind="product_page",
    url="https://www.newport.com/p/M-10X",
    note=(
        "Newport (MKS) specification table. Resources & Downloads offers 3D CAD, a "
        "2D dimension drawing and RoHS certificates -- and NO optical-design file, "
        "unlike every KPX/KBX/PAC part in this benchmark."
    ),
)

M10X = CatalogComponent(
    part_number="M-10X",
    vendor="Newport (MKS Instruments)",
    summary="10x, 0.25 NA microscope objective -- prescription not published",
    sources=(_M10X_PAGE,),
    published={
        "objective_type": PublishedValue(
            value="microscope", unit="", verbatim="Objective Type | Microscope",
            source_key="product_page",
        ),
        "magnification": PublishedValue(
            value="10x", unit="", verbatim="Magnification | 10x", source_key="product_page",
        ),
        "numerical_aperture": PublishedValue(
            value=0.25, unit="", verbatim="Numerical Aperture | 0.25",
            source_key="product_page",
        ),
        "wavelength_range_nm": PublishedValue(
            value="400-700", unit="nm", verbatim="Wavelength Range | 400-700 nm",
            source_key="product_page",
        ),
        "efl_mm": PublishedValue(
            value=16.5, unit="mm", verbatim="Effective Focal Length (EFL) | 16.5 mm",
            source_key="product_page",
        ),
        "working_distance_mm": PublishedValue(
            value=5.5, unit="mm", verbatim="Working Distance | 5.5 mm",
            source_key="product_page",
        ),
        "tube_length_mm": PublishedValue(
            value=160.0, unit="mm", verbatim="Tube Length | 160 mm", source_key="product_page",
        ),
        "clear_aperture_mm": PublishedValue(
            value=7.5, unit="mm", verbatim="Clear Aperture | 7.5 mm",
            source_key="product_page",
        ),
    },
    refusal=ConstructionRefusal(
        code="CATALOG_PRESCRIPTION_NOT_PUBLISHED",
        message=(
            "M-10X is a multi-element microscope objective whose internal "
            "prescription Newport does not publish, and whose product page links no "
            "optical-design file. Every parameter optical-component-spec/1 requires "
            "is absent: how many elements there are, where their surfaces are, how "
            "curved they are, what they are made of, and how far apart they sit. A "
            "10x/0.25 NA objective with a 16.5 mm EFL could be modelled as a thin "
            "lens or as an arbitrary invented doublet that happens to hit those "
            "numbers, and either would trace and produce a plausible-looking spot "
            "diagram while representing an optical system that does not exist. This "
            "component is therefore refused rather than approximated. Two of its "
            "published numbers -- EFL and NA -- would have been enough to fabricate "
            "something, which is exactly why the refusal is keyed on the missing "
            "geometry and not on the absence of a focal length."
        ),
        missing_parameters=(
            "element_count",
            "surface_radii",
            "surface_conics",
            "element_center_thicknesses",
            "element_glasses",
            "internal_air_spacings",
            "surface_order",
        ),
        published_but_insufficient=(
            "magnification 10x",
            "numerical_aperture 0.25",
            "effective_focal_length 16.5 mm",
            "working_distance 5.5 mm",
            "tube_length 160 mm",
            "clear_aperture 7.5 mm",
            "wavelength_range 400-700 nm",
        ),
    ),
)


#: Every catalog component this benchmark considered, in declaration order --
#: including the one it refused. A tuple of pairs rather than a dict so the
#: declaration order is the iteration order.
_CATALOG: tuple[tuple[str, CatalogComponent], ...] = (
    ("KPX094", KPX094),
    ("KBX058", KBX058),
    ("PAC052", PAC052),
    ("M-10X", M10X),
)

CATALOG_PART_NUMBERS: tuple[str, ...] = tuple(part for part, _ in _CATALOG)


def catalog_components() -> tuple[CatalogComponent, ...]:
    """Every considered component, supported or refused, in declaration order."""
    return tuple(component for _, component in _CATALOG)


def resolve_catalog_component(part_number: str) -> CatalogComponent:
    """Look up a catalog component by part number."""
    for candidate, component in _CATALOG:
        if candidate == part_number:
            return component
    raise PrescriptionError(
        "CATALOG_PART_UNKNOWN",
        f"{part_number!r} is not a catalog component of this benchmark",
        path="part_number",
        supported=CATALOG_PART_NUMBERS,
    )


def require_supported(part_number: str) -> CatalogComponent:
    """A catalog component that was actually built, or a structured refusal."""
    component = resolve_catalog_component(part_number)
    if component.component is None:
        assert component.refusal is not None  # enforced by the model validator
        raise PrescriptionError(
            component.refusal.code,
            component.refusal.message,
            path=f"catalog.{part_number}",
            expected="a component whose manufacturer publishes a full prescription",
            supported=tuple(
                candidate for candidate, entry in _CATALOG if entry.component is not None
            ),
        )
    return component


__all__ = [
    "CATALOG_PART_NUMBERS",
    "KBX058",
    "KPX094",
    "M10X",
    "PAC052",
    "RETRIEVED_UTC",
    "CatalogComponent",
    "ConstructionRefusal",
    "PublishedValue",
    "SourceDisagreement",
    "SourceDocument",
    "UnrepresentableParameter",
    "catalog_components",
    "require_supported",
    "resolve_catalog_component",
]
