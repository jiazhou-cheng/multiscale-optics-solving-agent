"""The five optical systems this benchmark builds, and where each number came from.

CHE-139 (M1.1.5). ``catalog_sources`` says what the manufacturer publishes about
each *part*; this module says how those parts are *installed* -- ordering,
orientation, air gaps, entrance pupil, object and image configuration -- and
nothing else. It imports no solver, builds no ``Optic``, and traces nothing.

Where the values come from
--------------------------
Every parameter of every system carries a :class:`ParameterSource`:

``catalog``
    read from a manufacturer document via ``catalog_sources``. Radii,
    thicknesses, glasses, design wavelengths, per-component entrance pupils and
    published BFLs are all of this kind, and none of them is chosen here.

``assembly_choice``
    a decision this benchmark makes because no manufacturer publishes it -- the
    50 mm air gap between the two components of the tandem, the field angles, the
    pupil ray count. These are legitimately the benchmark's to declare; what is
    not legitimate is declaring one without a basis, so each states one.

``derived``
    computed from the assembled system by the solver's own paraxial analysis --
    only the image-plane distance of a multi-component system, where no catalog
    BFL exists. It is a closed-form paraxial solve on the assembled
    prescription, not a search over spot size: nothing in this benchmark is
    optimized, and the image plane of the two single-component systems is the
    manufacturer's published BFL used verbatim.

The negative control
--------------------
``S5`` is ``S4`` with the achromat installed backwards, and it is the reason the
orientation machinery is trustworthy rather than merely present. Newport's PAC052
page states the orientation explicitly ("Steepest convex surface should face the
infinite conjugate"), so the control violates a manufacturer instruction rather
than an opinion, and everything else -- mechanical layout, air gap, pupil, field
set, wavelength, ray count, image-plane rule -- is held identical. If the
assembled spot size does not degrade, the orientation plumbing is not doing what
it claims, and the benchmark reports that rather than the reverse.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[2] / "src"
for _path in (str(_SRC), str(_HERE)):
    if _path not in sys.path:  # pragma: no cover - import-path bootstrap
        sys.path.insert(0, _path)

from catalog_sources import require_supported, resolve_catalog_component  # noqa: E402

from core.optical_assembly import (  # noqa: E402
    ComponentPlacement,
    Orientation,
    assemble_optical_system,
    component_surface_span,
)
from core.optical_system import (  # noqa: E402
    ApertureSpec,
    FieldSpec,
    OpticalSystemSpec,
    PrescriptionError,
    WavelengthSpec,
)


class ParameterSource(StrEnum):
    CATALOG = "catalog"
    ASSEMBLY_CHOICE = "assembly_choice"
    DERIVED = "derived"


class ImagePlaneRule(StrEnum):
    #: The last air gap is the manufacturer's published back focal length, used
    #: verbatim. Only available where a catalog publishes one for the whole system.
    CATALOG_BFL = "catalog_bfl"
    #: The last air gap is the assembled system's paraxial back focus, obtained
    #: from the solver's paraxial analysis of the built prescription. Not a
    #: search, not a best-focus fit, and not a function of any spot size.
    PARAXIAL_FOCUS = "paraxial_focus"


class SystemRole(StrEnum):
    CASE = "case"
    NEGATIVE_CONTROL = "negative_control"


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Parameter(_Frozen):
    """One system parameter, its provenance, and why it has the value it has."""

    value: float | int | str
    unit: str
    source: ParameterSource
    basis: str = Field(min_length=1)


class PlacementPlan(_Frozen):
    """One component installed in a system.

    ``air_gap_after_mm`` is ``None`` on -- and only on -- the last placement,
    where the distance to the image plane is set by the system's
    :class:`ImagePlaneRule` instead of being declared here.
    """

    part_number: str = Field(min_length=1)
    orientation: Orientation = Orientation.AS_SPECIFIED
    air_gap_after_mm: float | None = None
    air_gap_source: ParameterSource | None = None
    air_gap_basis: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _check_gap(self) -> PlacementPlan:
        if self.air_gap_after_mm is not None and (
            self.air_gap_source is None or not self.air_gap_basis
        ):
            raise PrescriptionError(
                "PLACEMENT_GAP_UNSOURCED",
                f"{self.part_number!r} declares air_gap_after_mm="
                f"{self.air_gap_after_mm!r} with no source and/or no basis",
                path="air_gap_after_mm",
                expected="every declared spacing carries a ParameterSource and a basis",
            )
        return self


class CatalogComparison(_Frozen):
    """A manufacturer-published quantity the simulation can be compared against.

    ``basis`` is not optional prose: EFL and BFL are only comparable if the
    simulated quantity is measured the same way and at the same wavelength as the
    published one, and stating how closes the gap between "the numbers are close"
    and "the numbers mean the same thing".
    """

    quantity: Literal["efl_mm", "bfl_mm"]
    catalog_value: float
    catalog_source_url: str
    basis: str = Field(min_length=1)


class SystemDefinition(_Frozen):
    """One benchmark system: which parts, installed how, traced where."""

    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    question: str = Field(min_length=1)
    placements: tuple[PlacementPlan, ...]
    entrance_pupil_diameter: Parameter
    field_angles_deg: tuple[float, ...]
    field_basis: str = Field(min_length=1)
    wavelengths_um: tuple[float, ...]
    primary_wavelength_um: float
    wavelength_basis: str = Field(min_length=1)
    image_plane_rule: ImagePlaneRule
    catalog_comparisons: tuple[CatalogComparison, ...] = ()
    pupil_rings: Parameter
    stop_component_index: int = 0
    stop_surface_index: int = 0
    role: SystemRole = SystemRole.CASE
    control_of: str | None = None
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_definition(self) -> SystemDefinition:
        if not self.placements:
            raise PrescriptionError(
                "SYSTEM_NO_PLACEMENTS",
                f"{self.key!r} places no component",
                path="placements",
                expected="one or more placements",
            )
        for index, placement in enumerate(self.placements[:-1]):
            if placement.air_gap_after_mm is None:
                raise PrescriptionError(
                    "SYSTEM_INTERIOR_GAP_MISSING",
                    f"{self.key!r} placement {index} ({placement.part_number}) is not "
                    "last but declares no air gap after it",
                    path=f"placements[{index}].air_gap_after_mm",
                    expected="every placement but the last declares its air gap",
                )
        if self.placements[-1].air_gap_after_mm is not None:
            raise PrescriptionError(
                "SYSTEM_TRAILING_GAP_DECLARED",
                f"{self.key!r} declares an air gap after its last component; the "
                "distance to the image plane is set by image_plane_rule="
                f"{self.image_plane_rule.value!r}",
                path="placements[-1].air_gap_after_mm",
                expected="null on the last placement",
            )
        if self.primary_wavelength_um not in self.wavelengths_um:
            raise PrescriptionError(
                "SYSTEM_PRIMARY_WAVELENGTH_NOT_DECLARED",
                f"{self.key!r} names primary wavelength "
                f"{self.primary_wavelength_um} um, which is not in "
                f"{self.wavelengths_um}",
                path="primary_wavelength_um",
                expected="a wavelength that is also declared in wavelengths_um",
            )
        if not self.field_angles_deg:
            raise PrescriptionError(
                "SYSTEM_NO_FIELDS",
                f"{self.key!r} declares no field angle",
                path="field_angles_deg",
                expected="one or more field angles in degrees",
            )
        if min(self.field_angles_deg) < 0.0:
            raise PrescriptionError(
                "SYSTEM_NEGATIVE_FIELD",
                f"{self.key!r} declares a negative field angle; normalized field "
                "coordinates are taken as angle / max(angle), which needs a "
                "non-negative, ascending set",
                path="field_angles_deg",
                expected="non-negative field angles",
            )
        if list(self.field_angles_deg) != sorted(set(self.field_angles_deg)):
            raise PrescriptionError(
                "SYSTEM_FIELD_SET_NOT_ASCENDING",
                f"{self.key!r} field angles {self.field_angles_deg} are not a "
                "strictly ascending set",
                path="field_angles_deg",
                expected="strictly ascending, unique field angles",
            )
        if (self.role is SystemRole.NEGATIVE_CONTROL) != (self.control_of is not None):
            raise PrescriptionError(
                "SYSTEM_CONTROL_TARGET_AMBIGUOUS",
                f"{self.key!r} sets role={self.role.value!r} and "
                f"control_of={self.control_of!r}",
                path="control_of",
                expected="control_of set exactly when role is negative_control",
            )
        if self.image_plane_rule is ImagePlaneRule.CATALOG_BFL and not any(
            comparison.quantity == "bfl_mm" for comparison in self.catalog_comparisons
        ):
            raise PrescriptionError(
                "SYSTEM_CATALOG_BFL_RULE_WITHOUT_CATALOG_BFL",
                f"{self.key!r} places its image plane at the catalog BFL but declares "
                "no catalog BFL to place it at",
                path="image_plane_rule",
                expected="a bfl_mm catalog comparison alongside the catalog_bfl rule",
            )
        return self

    # -- what the runner needs -------------------------------------------------

    @property
    def part_numbers(self) -> tuple[str, ...]:
        return tuple(placement.part_number for placement in self.placements)

    @property
    def max_field_deg(self) -> float:
        return float(max(self.field_angles_deg))

    def normalized_field(self, angle_deg: float) -> float:
        """``Hy`` for a declared field angle.

        Optiland's field coordinates are normalized to the largest declared
        field, verified on the built KPX094 system: with fields (0, 1, 2) deg,
        ``Hy=0.5`` puts the traced centroid at 1.741 mm against a paraxial
        ``f*tan(1 deg)`` of 1.746 mm -- the 0.24% shortfall is the singlet's own
        distortion, not a scaling error.
        """
        if angle_deg not in self.field_angles_deg:
            raise PrescriptionError(
                "SYSTEM_FIELD_NOT_DECLARED",
                f"{angle_deg} deg is not a declared field of {self.key!r}",
                path="field_angles_deg",
                expected=f"one of {self.field_angles_deg}",
            )
        maximum = self.max_field_deg
        if maximum == 0.0:
            return 0.0
        return float(angle_deg) / maximum

    def catalog_value(self, quantity: str) -> CatalogComparison | None:
        for comparison in self.catalog_comparisons:
            if comparison.quantity == quantity:
                return comparison
        return None

    def placements_with_image_distance(
        self, image_distance_mm: float
    ) -> tuple[ComponentPlacement, ...]:
        """Resolve this definition's placements against a concrete image distance.

        The catalog component is fetched through
        :func:`catalog_sources.require_supported`, so a system that named a
        refused part fails here with that part's structured refusal rather than
        building something else.
        """
        resolved: list[ComponentPlacement] = []
        for index, placement in enumerate(self.placements):
            catalog = require_supported(placement.part_number)
            assert catalog.component is not None  # require_supported guarantees it
            is_last = index == len(self.placements) - 1
            gap = image_distance_mm if is_last else placement.air_gap_after_mm
            assert gap is not None  # enforced by _check_definition
            resolved.append(
                ComponentPlacement(
                    component=catalog.component,
                    orientation=placement.orientation,
                    air_gap_after_mm=gap,
                    note=placement.note,
                )
            )
        return tuple(resolved)

    def assemble(self, image_distance_mm: float) -> OpticalSystemSpec:
        """The canonical prescription for this system, at a given image distance.

        The result goes to Optiland through ``build_optiland_system`` like every
        other prescription in the repository. This benchmark constructs no
        ``Optic`` of its own.
        """
        placements = self.placements_with_image_distance(image_distance_mm)
        return assemble_optical_system(
            name=self.key,
            description=self.description,
            placements=placements,
            aperture=ApertureSpec(value_mm=float(self.entrance_pupil_diameter.value)),
            fields=tuple(FieldSpec(y_deg=angle) for angle in self.field_angles_deg),
            wavelengths=tuple(
                WavelengthSpec(
                    value_um=wavelength,
                    is_primary=wavelength == self.primary_wavelength_um,
                )
                for wavelength in self.wavelengths_um
            ),
            object_distance_mm=None,  # every system here images an infinite conjugate
            stop_component_index=self.stop_component_index,
            stop_surface_index=self.stop_surface_index,
        )

    def assembly_record(self, image_distance_mm: float) -> dict[str, Any]:
        """The machine-readable assembly definition, with every source stated."""
        placements = self.placements_with_image_distance(image_distance_mm)
        return {
            "object_configuration": {
                "object_distance_mm": None,
                "meaning": "object at infinity (collimated input)",
                "source": ParameterSource.ASSEMBLY_CHOICE.value,
                "basis": (
                    "every catalog EFL/BFL in this benchmark is published for an "
                    "infinite conjugate, so an infinite object is what makes the "
                    "comparison meaningful"
                ),
            },
            "image_configuration": {
                "rule": self.image_plane_rule.value,
                "image_distance_from_last_vertex_mm": image_distance_mm,
                "source": (
                    ParameterSource.CATALOG.value
                    if self.image_plane_rule is ImagePlaneRule.CATALOG_BFL
                    else ParameterSource.DERIVED.value
                ),
                "basis": (
                    "the manufacturer's published back focal length, used verbatim"
                    if self.image_plane_rule is ImagePlaneRule.CATALOG_BFL
                    else (
                        "paraxial back focus of the assembled prescription, from the "
                        "solver's own paraxial analysis; no catalog publishes a "
                        "system-level BFL for an assembly the customer makes. It is a "
                        "closed-form paraxial quantity, not a spot-size search."
                    )
                ),
            },
            "entrance_pupil": self.entrance_pupil_diameter.model_dump(mode="json"),
            "pupil_rings": self.pupil_rings.model_dump(mode="json"),
            "stop": {
                "component_index": self.stop_component_index,
                "surface_index_within_component": self.stop_surface_index,
                "source": ParameterSource.CATALOG.value,
                "basis": (
                    "every vendor Zemax file in this benchmark marks STOP on its "
                    "first surface, so the assembly puts the stop on the first "
                    "surface of the first component"
                ),
            },
            "field_angles_deg": list(self.field_angles_deg),
            "field_basis": self.field_basis,
            "wavelengths_um": list(self.wavelengths_um),
            "primary_wavelength_um": self.primary_wavelength_um,
            "wavelength_basis": self.wavelength_basis,
            "placement_plan": [
                {
                    **plan.model_dump(mode="json"),
                    "resolved_air_gap_after_mm": placed.air_gap_after_mm,
                }
                for plan, placed in zip(self.placements, placements, strict=True)
            ],
            "component_spans": list(component_surface_span(placements)),
        }


# --- the shared declarations -------------------------------------------------

#: Hexapolar ring count. 16 rings is Optiland's own ``num_rays`` default in this
#: adapter and produces 1 + 3*16*17 = 817 rays per field per wavelength.
_PUPIL_RINGS = Parameter(
    value=16,
    unit="hexapolar rings",
    source=ParameterSource.ASSEMBLY_CHOICE,
    basis=(
        "the Optiland adapter's own default ring count, giving 817 rays per field "
        "per wavelength (1 + 3N(N+1)); dense enough for a readable spot diagram "
        "and an RMS estimate, cheap enough to run the whole matrix on one GPU"
    ),
)

#: The air gap of the two-component tandem. The only genuinely free geometric
#: parameter in this benchmark, and it is deliberately a round number that was
#: not adjusted after seeing any result.
_TANDEM_AIR_GAP_MM = 50.0
_TANDEM_AIR_GAP_BASIS = (
    "no manufacturer publishes a spacing for a system the customer assembles, so "
    "this is the benchmark's declared choice: a round 50.0 mm, roughly half the "
    "achromat's 96.5 mm back focal length, chosen so the converging beam reaches "
    "the second component at about half the entrance pupil radius and stays well "
    "inside its clear aperture. It was fixed before any trace ran and was NOT "
    "tuned against a spot size -- this is a forward-modelling benchmark, and the "
    "aperture_clearance block of the record reports the resulting footprint so "
    "the claim is measured rather than asserted."
)


def _catalog_number(part_number: str, key: str) -> float:
    """A published value, fetched from ``catalog_sources`` rather than retyped.

    Every number a system parameter takes from a manufacturer is read through
    here. Retyping it would create a second copy that can silently drift from
    the provenance record it claims to come from -- which is the same failure the
    catalog's own import-time transcription check exists to prevent, one level
    up.
    """
    return resolve_catalog_component(part_number).number(key)


def _epd_from_vendor_file(part_number: str) -> Parameter:
    return Parameter(
        value=_catalog_number(part_number, "entrance_pupil_diameter_mm"),
        unit="mm",
        source=ParameterSource.CATALOG,
        basis=(
            f"the ENPD of {part_number}'s own vendor Zemax file, so the pupil this "
            "benchmark traces is the pupil the manufacturer designed the part at "
            "rather than one chosen here"
        ),
    )


def _catalog_comparison(part_number: str, quantity: str, basis: str) -> CatalogComparison:
    """A catalog EFL/BFL comparison whose reference value comes from the catalog."""
    component = resolve_catalog_component(part_number)
    product_page = next(
        source.url for source in component.sources if source.kind == "product_page"
    )
    return CatalogComparison(
        quantity=quantity,  # type: ignore[arg-type]
        catalog_value=component.number(quantity),
        catalog_source_url=product_page,
        basis=basis,
    )


S1_KPX094_SINGLET = SystemDefinition(
    key="S1_KPX094_SINGLET",
    title="KPX094 plano-convex singlet, infinite conjugate",
    question=(
        "Can a single commercial singlet be reconstructed from its product page and "
        "traced, and does its simulated EFL/BFL match the catalog?"
    ),
    description=(
        "Newport KPX094 plano-convex singlet alone, collimated input, image plane at "
        "the catalog back focal length. The simplest case in the progression: one "
        "component, one wavelength, three fields."
    ),
    placements=(PlacementPlan(part_number="KPX094"),),
    entrance_pupil_diameter=_epd_from_vendor_file("KPX094"),
    field_angles_deg=(0.0, 1.0, 2.0),
    field_basis=(
        "benchmark choice: on-axis plus two small off-axis angles. A 100 mm singlet "
        "at 2 deg puts the image 3.49 mm off axis, inside the 22.86 mm clear "
        "aperture, so the field set does not itself push the model outside the "
        "geometry the catalog describes"
    ),
    wavelengths_um=(0.589,),
    primary_wavelength_um=0.589,
    wavelength_basis="the product page's stated 589 nm design wavelength",
    image_plane_rule=ImagePlaneRule.CATALOG_BFL,
    catalog_comparisons=(
        _catalog_comparison(
            "KPX094",
            "efl_mm",
            "the catalog EFL is an image-space effective focal length at the 589 nm "
            "design wavelength; the simulated value is Optiland's paraxial f2 on the "
            "built system at the same wavelength, so both are image-space EFLs of the "
            "same prescription",
        ),
        _catalog_comparison(
            "KPX094",
            "bfl_mm",
            "the catalog BFL is the distance from the last vertex to the paraxial "
            "focus at 589 nm; the simulated value is the declared image distance plus "
            "Optiland's paraxial F2, which is signed and measured from the image "
            "surface, so it is the same distance from the same vertex",
        ),
    ),
    pupil_rings=_PUPIL_RINGS,
)


S2_PAC052_ACHROMAT = SystemDefinition(
    key="S2_PAC052_ACHROMAT",
    title="PAC052 cemented achromatic doublet, infinite conjugate",
    question=(
        "Can a multi-element packaged commercial lens -- three surfaces, two glasses, "
        "a cemented interface -- be reconstructed and traced, and does it show the "
        "achromatic behaviour it is sold for?"
    ),
    description=(
        "Newport PAC052 achromatic doublet alone, in the manufacturer's stated "
        "orientation, collimated input, image plane at the catalog back focal length. "
        "Traced at the vendor Zemax file's own short and long lines as well as the "
        "page's design wavelength, so the chromatic spread is measured rather than "
        "assumed."
    ),
    placements=(
        PlacementPlan(
            part_number="PAC052",
            orientation=Orientation.AS_SPECIFIED,
            note=(
                "as specified = steepest convex surface (R1 = +60.741) toward the "
                "infinite conjugate, which is the orientation the product page "
                "instructs"
            ),
        ),
    ),
    entrance_pupil_diameter=_epd_from_vendor_file("PAC052"),
    field_angles_deg=(0.0, 1.0, 2.0),
    field_basis="the same field set as S1, so the singlet and the doublet are compared on one grid",
    wavelengths_um=(0.4861, 0.589, 0.6563),
    primary_wavelength_um=0.589,
    wavelength_basis=(
        "the product page's 589 nm design wavelength as primary, plus the 486.1 nm "
        "and 656.3 nm lines the vendor's own Zemax file evaluates this achromat at "
        "(WAVM 1 and WAVM 3). All three come from manufacturer documents."
    ),
    image_plane_rule=ImagePlaneRule.CATALOG_BFL,
    catalog_comparisons=(
        _catalog_comparison(
            "PAC052",
            "efl_mm",
            "image-space paraxial EFL at the 589 nm design wavelength, as for S1",
        ),
        _catalog_comparison(
            "PAC052",
            "bfl_mm",
            "last-vertex-to-paraxial-focus distance at 589 nm, as for S1. The vendor "
            "Zemax file independently ships an image distance of 96.53204485085 mm at "
            "its own 546.1 nm primary, so the two manufacturer documents agree here "
            "to 0.032 mm",
        ),
    ),
    pupil_rings=_PUPIL_RINGS,
)


S3_KBX058_BICONVEX = SystemDefinition(
    key="S3_KBX058_BICONVEX",
    title="KBX058 equiconvex singlet, infinite conjugate",
    question=(
        "Third commercial component: does a two-curved-surface singlet, whose second "
        "radius had to come from the vendor design file rather than the spec table, "
        "reconstruct and trace correctly?"
    ),
    description=(
        "Newport KBX058 equiconvex singlet alone, collimated input at the vendor's own "
        "11.43 mm entrance pupil, image plane at the catalog back focal length."
    ),
    placements=(PlacementPlan(part_number="KBX058"),),
    entrance_pupil_diameter=_epd_from_vendor_file("KBX058"),
    field_angles_deg=(0.0, 1.0),
    field_basis=(
        "on-axis plus one off-axis angle; this component exists in the benchmark to "
        "be the third reconstructed part and the second element of the tandem, so its "
        "standalone field sweep is deliberately the smallest that still includes a "
        "non-zero field"
    ),
    wavelengths_um=(0.589,),
    primary_wavelength_um=0.589,
    wavelength_basis="the product page's stated 589 nm design wavelength",
    image_plane_rule=ImagePlaneRule.CATALOG_BFL,
    catalog_comparisons=(
        _catalog_comparison(
            "KBX058",
            "efl_mm",
            "image-space paraxial EFL at the 589 nm design wavelength, as for S1",
        ),
        _catalog_comparison(
            "KBX058",
            "bfl_mm",
            "last-vertex-to-paraxial-focus distance at 589 nm, as for S1",
        ),
    ),
    pupil_rings=_PUPIL_RINGS,
)


#: Shared by S4 and its control, so the two cannot drift apart in anything but
#: the one parameter the control is about.
_TANDEM_FIELDS = (0.0, 1.0, 2.0, 3.0)
_TANDEM_FIELD_BASIS = (
    "four fields out to 3 deg. The tandem's paraxial focal length is near 60 mm, so "
    "3 deg lands the image about 3.1 mm off axis -- enough field dependence to see "
    "in a spot diagram, small enough that the beam stays inside both components' "
    "clear apertures (measured, see aperture_clearance)"
)
_TANDEM_WAVELENGTHS = (0.589,)
_TANDEM_WAVELENGTH_BASIS = (
    "the 589 nm design wavelength shared by both components' product pages. The "
    "chromatic characterization lives in S2, where the achromat is alone and the "
    "dispersion is attributable; holding the tandem monochromatic is what keeps S5 "
    "a single-variable control"
)


S4_PAC052_KBX058_TANDEM = SystemDefinition(
    key="S4_PAC052_KBX058_TANDEM",
    title="PAC052 achromat + 50 mm air gap + KBX058 singlet",
    question=(
        "Can two independently specified commercial components be assembled into one "
        "optical system with explicit ordering, orientation and spacing, and does the "
        "assembly behave as a system rather than as two lenses?"
    ),
    description=(
        "The multi-component case. Newport PAC052 achromat in its manufacturer-stated "
        "orientation, a declared 50.0 mm air gap, then the Newport KBX058 equiconvex "
        "singlet, collimated input, image plane at the assembly's paraxial back focus. "
        "The aperture stop is the achromat's first surface, as both vendor design "
        "files place it."
    ),
    placements=(
        PlacementPlan(
            part_number="PAC052",
            orientation=Orientation.AS_SPECIFIED,
            air_gap_after_mm=_TANDEM_AIR_GAP_MM,
            air_gap_source=ParameterSource.ASSEMBLY_CHOICE,
            air_gap_basis=_TANDEM_AIR_GAP_BASIS,
            note=(
                "steepest convex surface toward the infinite conjugate, per the "
                "PAC052 product page"
            ),
        ),
        PlacementPlan(
            part_number="KBX058",
            orientation=Orientation.AS_SPECIFIED,
            note=(
                "equiconvex, so its own orientation is physically immaterial; it is "
                "declared explicitly anyway so the assembled sequence states it"
            ),
        ),
    ),
    entrance_pupil_diameter=Parameter(
        value=_catalog_number("PAC052", "entrance_pupil_diameter_mm"),
        unit="mm",
        source=ParameterSource.CATALOG,
        basis=(
            "the ENPD of the FIRST component's vendor Zemax file (PAC052), "
            "which is what the entrance pupil of the assembly physically is: the stop "
            "sits on that component's first surface. The second component's own "
            "11.43 mm ENPD is not the assembly's pupil -- it is a constraint on the "
            "converging beam that reaches it, and that is checked by measurement in "
            "aperture_clearance rather than by shrinking the pupil to be safe."
        ),
    ),
    field_angles_deg=_TANDEM_FIELDS,
    field_basis=_TANDEM_FIELD_BASIS,
    wavelengths_um=_TANDEM_WAVELENGTHS,
    primary_wavelength_um=0.589,
    wavelength_basis=_TANDEM_WAVELENGTH_BASIS,
    image_plane_rule=ImagePlaneRule.PARAXIAL_FOCUS,
    pupil_rings=_PUPIL_RINGS,
)


S5_TANDEM_REVERSED_ACHROMAT = SystemDefinition(
    key="S5_TANDEM_REVERSED_ACHROMAT",
    title="S4 with the achromat installed backwards (negative control)",
    question=(
        "Does the orientation the benchmark claims to represent actually reach the "
        "built system? Installing the achromat against its manufacturer instruction "
        "must degrade the spot; if it does not, the orientation plumbing is inert."
    ),
    description=(
        "Identical to S4 -- same components, same 50.0 mm air gap, same pupil, same "
        "fields, same wavelength, same ray count, same image-plane rule -- except that "
        "the PAC052 achromat is reversed, putting its shallow flint face "
        "(R = +133.104 after reversal) toward the infinite conjugate instead of the "
        "steep crown face the product page instructs. Its own paraxial focus is used, "
        "so the control is defocus-free and isolates orientation alone."
    ),
    placements=(
        PlacementPlan(
            part_number="PAC052",
            orientation=Orientation.REVERSED,
            air_gap_after_mm=_TANDEM_AIR_GAP_MM,
            air_gap_source=ParameterSource.ASSEMBLY_CHOICE,
            air_gap_basis=_TANDEM_AIR_GAP_BASIS,
            note=(
                "DELIBERATELY WRONG: violates the PAC052 product page's stated "
                "orientation. This is the negative control, not a modelling choice."
            ),
        ),
        PlacementPlan(part_number="KBX058", orientation=Orientation.AS_SPECIFIED),
    ),
    entrance_pupil_diameter=Parameter(
        value=_catalog_number("PAC052", "entrance_pupil_diameter_mm"),
        unit="mm",
        source=ParameterSource.CATALOG,
        basis="identical to S4 by construction, from the same catalog value",
    ),
    field_angles_deg=_TANDEM_FIELDS,
    field_basis=_TANDEM_FIELD_BASIS,
    wavelengths_um=_TANDEM_WAVELENGTHS,
    primary_wavelength_um=0.589,
    wavelength_basis=_TANDEM_WAVELENGTH_BASIS,
    image_plane_rule=ImagePlaneRule.PARAXIAL_FOCUS,
    role=SystemRole.NEGATIVE_CONTROL,
    control_of="S4_PAC052_KBX058_TANDEM",
    pupil_rings=_PUPIL_RINGS,
)


_SYSTEMS: tuple[tuple[str, SystemDefinition], ...] = (
    ("S1_KPX094_SINGLET", S1_KPX094_SINGLET),
    ("S2_PAC052_ACHROMAT", S2_PAC052_ACHROMAT),
    ("S3_KBX058_BICONVEX", S3_KBX058_BICONVEX),
    ("S4_PAC052_KBX058_TANDEM", S4_PAC052_KBX058_TANDEM),
    ("S5_TANDEM_REVERSED_ACHROMAT", S5_TANDEM_REVERSED_ACHROMAT),
)

SYSTEM_KEYS: tuple[str, ...] = tuple(key for key, _ in _SYSTEMS)

#: The scalar metrics the GPU/CPU behaviour-equivalence check compares. Declared
#: here, once, so the comparison cannot be widened after seeing its result.
GPU_CPU_COMPARISON_METRICS: tuple[str, ...] = (
    "surviving_ray_count",
    "centroid_y_mm",
    "rms_spot_radius_mm",
    "max_spot_radius_mm",
    "focal_plane_z_mm",
)


def benchmark_systems() -> tuple[SystemDefinition, ...]:
    return tuple(definition for _, definition in _SYSTEMS)


def resolve_system(key: str) -> SystemDefinition:
    for candidate, definition in _SYSTEMS:
        if candidate == key:
            return definition
    raise PrescriptionError(
        "SYSTEM_KEY_UNKNOWN",
        f"{key!r} is not a system of this benchmark",
        path="key",
        supported=SYSTEM_KEYS,
    )


__all__ = [
    "GPU_CPU_COMPARISON_METRICS",
    "SYSTEM_KEYS",
    "CatalogComparison",
    "ImagePlaneRule",
    "Parameter",
    "ParameterSource",
    "PlacementPlan",
    "SystemDefinition",
    "SystemRole",
    "benchmark_systems",
    "resolve_system",
]
