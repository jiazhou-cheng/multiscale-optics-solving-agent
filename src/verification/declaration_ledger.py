"""Every declaration the registry makes, and the evidence that it is not just prose.

CHE-108 (M1.3) / CHE-112 (M2.4). The registry declares 52 things across six
components -- assumptions, warnings, hard limits and invariants -- and until this
module each of them was covered wherever the ticket that introduced it happened
to be working. That is coverage by memory, and it fails the same way twice: a new
declaration arrives with no test, or a test is deleted and the declaration
quietly becomes prose again.

Both M1.3 and M2.4 asked for the same thing in different words -- "no silent
omissions" over `assumption`/`warning`/`hard_limit`, and "every invariant has an
executable assertion or is removed" -- so this is one ledger rather than two ad
hoc test lists. The generated refusal catalogue in ``verification/refusals.py``
is the design precedent: keyed by the thing it describes, asserted complete in
both directions, and a missing entry is a test failure rather than a gap nobody
sees.

How a declaration is identified
-------------------------------
**Not by index.** An index-keyed ledger silently re-points every entry below an
insertion, which is the one failure mode a completeness ledger must not have.
Each entry instead names its ``component``, its ``kind``, and an ``anchor`` --
a distinctive substring of the declaration's own text (or, for a hard limit or
an invariant, its exact key). Resolution requires exactly one match in each
direction:

* a declaration no entry anchors is a **gap**;
* an entry that anchors no declaration is **orphaned** -- the declaration was
  edited or deleted and the evidence claim is now about nothing;
* an anchor matching two declarations, or two entries matching one declaration,
  is **ambiguous** and is an error rather than a resolution order.

All three fail ``tests/test_declaration_ledger.py``. Editing a declaration's
text past its anchor therefore forces the coverage claim to be re-examined, which
is the intended cost: the claim is about what the sentence says.

What counts as coverage
-----------------------
``EXECUTABLE_TEST``
    A test in the default suite asserts the declared behaviour. The evidence
    reference names the file and the test.
``BENCHMARK_CASE``
    A benchmark family instance exercises it. The reference names the family or
    instance id, which the family registry can resolve.
``EXPLICIT_NON_EXECUTABLE``
    It cannot be executed as a check, and the reason says *why* rather than
    "not tested". These are legitimate and there are few of them: a statement
    about upstream thread-safety, a scope note about what a version claim
    covers, a cost-model asymptote. A blank or boilerplate reason is rejected at
    construction, so this classification cannot be used as a shrug.

An invariant additionally owes a ``tolerance_basis``: an invariant asserted with
no tolerance, or with a tolerance nobody derived, is a number somebody liked.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from registry.loader import Registry

__all__ = [
    "COVERAGE_LEDGER",
    "Coverage",
    "CoverageKind",
    "DeclarationKind",
    "LedgerReport",
    "RegistryDeclaration",
    "coverage_for",
    "registry_declarations",
    "resolve_ledger",
]


class DeclarationKind(StrEnum):
    """Which registry field a declaration came from."""

    #: ``validity.assumptions``. Something the component takes to be true; if it
    #: is false the component still runs and the answer is wrong.
    ASSUMPTION = "assumption"
    #: ``validity.warnings``. A hazard the caller must know about, usually one
    #: that produces a plausible wrong answer rather than an error.
    WARNING = "warning"
    #: ``validity.hard_limits``. A parameter region excluded outright.
    HARD_LIMIT = "hard_limit"
    #: A coupler's ``invariants`` list. Something that must hold for every
    #: parameter value, checkable without an oracle.
    INVARIANT = "invariant"


class CoverageKind(StrEnum):
    """How a declaration is backed. See the module docstring."""

    EXECUTABLE_TEST = "executable_test"
    BENCHMARK_CASE = "benchmark_case"
    EXPLICIT_NON_EXECUTABLE = "explicit_non_executable"

    @property
    def is_executable(self) -> bool:
        return self is not CoverageKind.EXPLICIT_NON_EXECUTABLE


#: Rejected as evidence or as a reason. A ledger that accepts these is a ledger
#: that reports full coverage while covering nothing.
_PLACEHOLDERS = frozenset(
    {"", "-", "--", "n/a", "na", "none", "tbd", "todo", "todo.", "xxx", "?", "pending"}
)

#: An evidence reference has to be resolvable by a reader: a path, a
#: ``file::test`` pair, or a benchmark family/instance id.
_EVIDENCE_SHAPE = re.compile(r"(::|/|\.py|^B[0-4]-[A-Z0-9-]+$)")

#: Long enough that a reason has to contain a reason. Chosen rather than derived:
#: every legitimate entry below is well past it, and the shortest thing anybody
#: writes to get past a coverage check is shorter.
_MIN_REASON_CHARS = 60


def _normalize(text: str) -> str:
    """Collapse YAML block-scalar wrapping so an anchor can be one sentence.

    Registry text is folded across lines at arbitrary columns, so a substring
    that reads as contiguous prose in the file is not contiguous in the loaded
    string. Every comparison in this module is against the collapsed form.
    """
    return " ".join(text.split())


@dataclass(frozen=True, slots=True)
class RegistryDeclaration:
    """One declaration, as the registry states it."""

    component: str
    kind: DeclarationKind
    #: The collapsed declaration text. For a hard limit, the limit's own text;
    #: for an invariant, the invariant name.
    text: str
    #: Where to read it. ``file#component.field`` or ``...field[key]``.
    source_location: str
    #: For a hard limit, its YAML key. For an invariant, its name. Otherwise
    #: ``None`` -- an assumption or warning is identified by what it says.
    key: str | None = None

    @property
    def declaration_id(self) -> str:
        """A stable, readable id. The key where there is one, else a short slug.

        Used for reporting rather than for resolution: resolution goes through
        anchors, for the reason the module docstring gives.
        """
        if self.key is not None:
            return f"{self.component}:{self.kind.value}:{self.key}"
        slug = re.sub(r"[^a-z0-9]+", "-", self.text.lower())[:48].strip("-")
        return f"{self.component}:{self.kind.value}:{slug}"


def registry_declarations(registry: Registry | None = None) -> tuple[RegistryDeclaration, ...]:
    """Enumerate every assumption, warning, hard limit and invariant declared.

    Reads the typed registry rather than the YAML text, so it sees exactly what
    a consumer of ``Registry`` sees. A declaration the loader drops is not a
    declaration this repository makes.
    """
    registry = registry or Registry.from_package()
    out: list[RegistryDeclaration] = []

    for source, specs in (
        ("src/registry/models.yaml", registry.models),
        ("src/registry/couplers.yaml", registry.couplers),
    ):
        for component_id, spec in sorted(specs.items()):
            where = f"{source}#{component_id}"
            for text in spec.validity.assumptions:
                out.append(
                    RegistryDeclaration(
                        component=component_id,
                        kind=DeclarationKind.ASSUMPTION,
                        text=_normalize(text),
                        source_location=f"{where}.validity.assumptions",
                    )
                )
            for text in spec.validity.warnings:
                out.append(
                    RegistryDeclaration(
                        component=component_id,
                        kind=DeclarationKind.WARNING,
                        text=_normalize(text),
                        source_location=f"{where}.validity.warnings",
                    )
                )
            for limit_key, text in sorted(spec.validity.hard_limits.items()):
                out.append(
                    RegistryDeclaration(
                        component=component_id,
                        kind=DeclarationKind.HARD_LIMIT,
                        text=_normalize(text),
                        source_location=f"{where}.validity.hard_limits[{limit_key}]",
                        key=limit_key,
                    )
                )
            for invariant in getattr(spec, "invariants", ()) or ():
                out.append(
                    RegistryDeclaration(
                        component=component_id,
                        kind=DeclarationKind.INVARIANT,
                        text=_normalize(invariant),
                        source_location=f"{where}.invariants",
                        key=invariant,
                    )
                )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Coverage:
    """One coverage claim about one registry declaration."""

    component: str
    kind: DeclarationKind
    #: A distinctive substring of the declaration text, or -- for a hard limit
    #: or an invariant -- its exact key. Matched against the collapsed text.
    anchor: str
    coverage_kind: CoverageKind
    #: Where the evidence is. Required unless the classification is
    #: ``EXPLICIT_NON_EXECUTABLE``, in which case it is optional supporting
    #: material rather than the claim.
    evidence: tuple[str, ...] = ()
    #: Why this classification. Required for ``EXPLICIT_NON_EXECUTABLE``;
    #: everywhere else it says what the evidence actually establishes, which is
    #: not always the same as what the declaration says.
    reason: str = ""
    #: Required for an invariant: the basis of the tolerance the assertion uses.
    tolerance_basis: str | None = None

    def __post_init__(self) -> None:
        if not _normalize(self.anchor):
            raise ValueError(f"{self.component}/{self.kind}: an entry needs an anchor")

        if self.coverage_kind.is_executable:
            if not self.evidence:
                raise ValueError(
                    f"{self.component}/{self.anchor[:40]!r}: {self.coverage_kind} claims "
                    "evidence and names none. An unreferenced claim is the gap it is "
                    "supposed to close."
                )
            for reference in self.evidence:
                if _normalize(reference).lower() in _PLACEHOLDERS:
                    raise ValueError(
                        f"{self.component}/{self.anchor[:40]!r}: {reference!r} is a "
                        "placeholder, not a reference"
                    )
                if not _EVIDENCE_SHAPE.search(reference):
                    raise ValueError(
                        f"{self.component}/{self.anchor[:40]!r}: {reference!r} is not "
                        "resolvable -- name a path, a file::test, or a family id"
                    )
        else:
            reason = _normalize(self.reason)
            if reason.lower() in _PLACEHOLDERS or len(reason) < _MIN_REASON_CHARS:
                raise ValueError(
                    f"{self.component}/{self.anchor[:40]!r}: an "
                    "explicit_non_executable entry needs a real reason saying why no "
                    f"check can exist, not {self.reason!r}"
                )

        if self.kind is DeclarationKind.INVARIANT:
            if not self.tolerance_basis or _normalize(self.tolerance_basis).lower() in (
                _PLACEHOLDERS
            ):
                raise ValueError(
                    f"{self.component}/{self.anchor}: an invariant owes a tolerance "
                    "basis. An invariant asserted at a tolerance nobody derived is a "
                    "number somebody liked."
                )
            if not self.coverage_kind.is_executable:
                raise ValueError(
                    f"{self.component}/{self.anchor}: an invariant cannot be "
                    "explicit_non_executable. If nothing can assert it, the honest "
                    "move is to remove the declaration from the registry -- M2.4 says "
                    "so explicitly."
                )

    def matches(self, declaration: RegistryDeclaration) -> bool:
        if declaration.component != self.component or declaration.kind is not self.kind:
            return False
        if declaration.key is not None:
            return declaration.key == self.anchor
        return _normalize(self.anchor) in declaration.text


@dataclass(frozen=True, slots=True)
class LedgerReport:
    """The result of matching the ledger against the registry."""

    #: declaration_id -> the entry covering it.
    covered: Mapping[str, Coverage]
    #: Declarations no entry anchors.
    gaps: tuple[RegistryDeclaration, ...] = ()
    #: Entries anchoring no declaration.
    orphaned: tuple[Coverage, ...] = ()
    #: Anchors that resolved to more than one declaration, or declarations more
    #: than one entry claimed.
    ambiguous: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_complete(self) -> bool:
        return not (self.gaps or self.orphaned or self.ambiguous)

    def counts_by_coverage_kind(self) -> dict[str, int]:
        out: dict[str, int] = {kind.value: 0 for kind in CoverageKind}
        for entry in self.covered.values():
            out[entry.coverage_kind.value] += 1
        return out


def resolve_ledger(
    ledger: Sequence[Coverage] | None = None,
    registry: Registry | None = None,
) -> LedgerReport:
    """Match every declaration to exactly one entry, in both directions."""
    ledger = list(ledger if ledger is not None else COVERAGE_LEDGER)
    declarations = registry_declarations(registry)

    covered: dict[str, Coverage] = {}
    gaps: list[RegistryDeclaration] = []
    ambiguous: list[str] = []
    hits: dict[int, list[RegistryDeclaration]] = {index: [] for index in range(len(ledger))}

    for declaration in declarations:
        matching = [index for index, entry in enumerate(ledger) if entry.matches(declaration)]
        for index in matching:
            hits[index].append(declaration)
        if not matching:
            gaps.append(declaration)
        elif len(matching) > 1:
            ambiguous.append(
                f"{declaration.declaration_id} is claimed by {len(matching)} entries "
                f"(anchors: {[ledger[i].anchor[:40] for i in matching]})"
            )
        else:
            covered[declaration.declaration_id] = ledger[matching[0]]

    orphaned = tuple(entry for index, entry in enumerate(ledger) if not hits[index])
    for index, entry in enumerate(ledger):
        if len(hits[index]) > 1:
            ambiguous.append(
                f"anchor {entry.anchor[:40]!r} on {entry.component} matches "
                f"{len(hits[index])} declarations; make it distinctive"
            )

    return LedgerReport(
        covered=covered,
        gaps=tuple(gaps),
        orphaned=orphaned,
        ambiguous=tuple(ambiguous),
    )


def coverage_for(declaration_id: str) -> Coverage:
    """The entry covering a declaration, or a KeyError naming the gap."""
    report = resolve_ledger()
    try:
        return report.covered[declaration_id]
    except KeyError:
        raise KeyError(
            f"no coverage entry resolves {declaration_id!r}. A declaration with no "
            "evidence is prose, and this ledger exists so that reads as a failure."
        ) from None


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------
#
# Ordered by component, then by kind, so a reader can hold one component at a
# time. Anchors are the shortest distinctive phrase from the declaration itself.

_MODEL_OPTILAND: tuple[Coverage, ...] = (
    Coverage(
        component="M_RAY_OPTILAND",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Geometric optics is adequate inside the traced lens system",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=("B1-RAY-EFL", "B1-RAY-PLATE", "B1-RAY-SNELL", "B1-RAY-LAGRANGE"),
        reason=(
            "the ray families gate the traced result against closed forms that are "
            "only correct if geometric optics is adequate; each carries a paraxial "
            "validity predicate that reports where the assumption stops holding"
        ),
    ),
    Coverage(
        component="M_RAY_OPTILAND",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Diffractive effects are introduced only after an explicit ray-to-wave coupler",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_architecture_invariants.py",
            "tests/test_graph_validation.py",
        ),
        reason=(
            "structural rather than numerical: the ray adapter emits ray artifacts "
            "only, and a field can be produced from them only through a registered "
            "coupler edge, which the graph validator enforces"
        ),
    ),
    Coverage(
        component="M_RAY_OPTILAND",
        kind=DeclarationKind.WARNING,
        anchor="Ray caustics and hard apertures can create non-smooth derivatives",
        coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
        reason=(
            "this is a statement about derivative smoothness, and no derivative "
            "through this model is claimed or computed: derivative.verified is false "
            "and the executor refuses a graph requiring verified gradients. There is "
            "no quantity in the shipping forward path whose non-smoothness a check "
            "could observe, so an executable assertion would have to first build the "
            "gradient path this repository deliberately does not have."
        ),
        evidence=("tests/test_registry_matches_capabilities.py",),
    ),
    Coverage(
        component="M_RAY_OPTILAND",
        kind=DeclarationKind.WARNING,
        anchor="set_backend mutates global module state and is not thread-safe",
        coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
        reason=(
            "an upstream thread-safety property. Asserting it would mean racing "
            "Optiland's global backend state from two threads and observing "
            "corruption, which is a nondeterministic test of somebody else's "
            "library. What this repository can and does check instead is its own "
            "response: the adapter sets backend, precision and device explicitly on "
            "every run, at the defaults included, so no previous run's choice can "
            "leak -- and that is asserted."
        ),
        evidence=("tests/test_solver_adapter_characterization.py",),
    ),
    Coverage(
        component="M_RAY_OPTILAND",
        kind=DeclarationKind.WARNING,
        anchor="geometry in millimetres and wavelength in micrometres",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_optiland_opd_convention.py",
            "tests/test_coupler_contracts.py::test_ray_bundle_rejects_a_non_si_optiland_artifact",
        ),
        reason=(
            "the unit conversion is asserted at the artifact boundary and a non-SI "
            "artifact is refused rather than accepted; the opd_native part is covered "
            "by the OPL-reference tests, which refuse the raw value as a phase"
        ),
    ),
    Coverage(
        component="M_RAY_OPTILAND",
        kind=DeclarationKind.WARNING,
        anchor="that launch plane is perpendicular to Z",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B1-RAY-OFFAXIS-OPL",
            "benchmarks/instances/b1_ray.py",
        ),
        reason=(
            "the family measures the fraction of the required off-axis convergence "
            "tilt that survives, and its declared negative control omits "
            "n_object*(d0.r_launch) through the shipping adapter and must fail"
        ),
    ),
    Coverage(
        component="M_RAY_OPTILAND",
        kind=DeclarationKind.WARNING,
        anchor="exposes only x, y, z, L",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_coupler_contracts.py::test_ray_bundle_carries_a_weight_but_refuses_to_call_it_an_amplitude",
            "tests/test_coupler_contracts.py::test_weight_to_amplitude_conversion_must_be_declared_by_the_caller",
        ),
        reason=(
            "the contract-gap resolution is asserted directly: the bundle carries a "
            "real weight, refuses to be read as an amplitude, and requires the caller "
            "to declare the weight-to-amplitude map"
        ),
    ),
)


_MODEL_CHROMATIX: tuple[Coverage, ...] = (
    Coverage(
        component="M_WAVE_CHROMATIX",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Sampling and propagation method satisfy the relevant band-limit conditions",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=("B1-WAVE-ASM-VALIDITY", "benchmarks/instances/b1_wave.py"),
        reason=(
            "this is the declaration B1-WAVE-ASM-VALIDITY exists to make executable: "
            "it sweeps across the transfer-function sampling boundary with a signed "
            "normalized margin and measures what the invalid side returns, which is a "
            "plausible field rather than an exception"
        ),
    ),
    Coverage(
        component="M_WAVE_CHROMATIX",
        kind=DeclarationKind.WARNING,
        anchor="Scalar propagation is not valid for every high-NA or strongly vectorial task",
        coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
        reason=(
            "a scope boundary, not a behaviour. This repository has no vectorial "
            "model to compare against, so there is nothing an assertion could measure "
            "the scalar result's inadequacy against -- the archived L1-WAVE-01 high-NA "
            "case was blocked on a defective upstream lens rather than resolved. The "
            "executable half of the statement, the sampling adequacy boundary, is "
            "carried by B1-WAVE-ASM-VALIDITY; the vectorial half stays declared."
        ),
        evidence=("archive/benchmarks/gen1/README.md",),
    ),
    Coverage(
        component="M_WAVE_CHROMATIX",
        kind=DeclarationKind.WARNING,
        anchor="M1 validates scalar angular-spectrum cases and a Gaussian scaling",
        coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
        reason=(
            "a provenance statement about which pinned distribution was verified and "
            "what that verification covered. The pinned version and commit are "
            "asserted by the dependency test; the sentence itself is a scope claim "
            "about past work, and there is no runtime behaviour for a check to "
            "observe."
        ),
        evidence=("tests/test_package_dependencies.py",),
    ),
    Coverage(
        component="M_WAVE_CHROMATIX",
        kind=DeclarationKind.WARNING,
        anchor="A complex128 input array is silently downcast to complex64",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B0-DTYPE-01",
            "benchmarks/instances/b0_contract.py",
            "tests/test_precision_contract.py",
        ),
        reason=(
            "B0-DTYPE-01 refuses complex128 under SAFE and records the truncation as a "
            "measured loss under ALLOW_DOWNCAST, so the downcast cannot happen "
            "unreported inside ScalarField; the declared negative control is the "
            "silent truncation and it must fail"
        ),
    ),
)


_C_RAY_TO_WAVE: tuple[Coverage, ...] = (
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Rays adequately sample the pupil and represent a locally single-valued wavefront",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=("B2-R2W-EXACT", "benchmarks/instances/b2_transitions.py"),
        reason=(
            "the enumeration-limit instance has zero sampling error by construction, "
            "so it is the point at which the sampling assumption is exactly satisfied "
            "and any residual is a transform defect rather than a sampling one"
        ),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Amplitude reconstruction and reference sphere are explicitly defined",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_coupler_contracts.py::test_weight_to_amplitude_conversion_must_be_declared_by_the_caller",
            "tests/test_ray_to_wave.py",
        ),
        reason=(
            "the amplitude map is refused unless declared, and the reference surface "
            "is a plane read from the system rather than an implied sphere -- the "
            "reference-sphere pupil is not implemented and is not declared as "
            "available"
        ),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.ASSUMPTION,
        anchor="The optical path length carries a declared reference",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B0-HANDOFF-01",
            "tests/test_coupler_contracts.py::test_unverified_opl_reference_is_refused_not_defaulted",
        ),
        reason=(
            "a bare opd_native is refused with OPL_REFERENCE_UNVERIFIED through the "
            "shipping path, and the refusal carries the remedy naming the two edge "
            "declarations that would fix it"
        ),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.ASSUMPTION,
        anchor="the declared OPL reference is versioned",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=("B1-RAY-OFFAXIS-OPL", "benchmarks/instances/b2_transitions.py"),
        reason=(
            "the off-axis reconstruction is gated, and a record with no object-space "
            "reference term is refused with OBJECT_SPACE_REFERENCE_MISSING rather "
            "than reconstructed approximately"
        ),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.ASSUMPTION,
        anchor="two declarations must be supplied ON THE EDGE",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_ray_to_wave_node.py",
            "tests/test_coupler_contracts.py::test_ray_bundle_names_the_missing_declaration",
        ),
        reason=(
            "the edge configuration is required rather than defaulted, and the "
            "refusal names which declaration is missing; the reference-plane mismatch "
            "case is separately refused rather than reconciled"
        ),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.WARNING,
        anchor="Caustics, multi-valued phase, hard vignetting, and sparse rays",
        coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
        reason=(
            "the coupler does not interpolate a wavefront -- it sums plane wavelets "
            "per ray -- so 'can invalidate interpolation' describes a step this "
            "implementation does not perform, and there is no interpolation stage for "
            "a check to break. The sparse-ray half is covered quantitatively by the "
            "convergence ladder in B2-R2W-ROUTE instead. The declaration is left in "
            "place because it is true of the physical situation and an agent choosing "
            "the coupler needs it; what it is not is a behaviour of this code."
        ),
        evidence=("src/couplers/ray_to_wave.py", "B2-R2W-ROUTE"),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.WARNING,
        anchor="Omitting the <n, d> projection factor",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-R2W-EXACT",
            "benchmarks/instances/b2_transitions.py",
            "tests/test_wave_to_ray.py::test_flipping_the_normal_component_sign_is_detected_off_plane",
        ),
        reason=(
            "the projection factor is one of the four terms the exact-route negative "
            "controls remove individually, and the blind-spot test states in code why "
            "an on-axis-only control cannot establish it: the factor is exactly 1 at "
            "normal incidence"
        ),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.INVARIANT,
        anchor="phase_reference_consistency",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-ROUNDTRIP",
            "tests/test_coupler_round_trip.py::test_a_mismatched_phase_sign_pairing_breaks_the_round_trip",
        ),
        reason=(
            "asserted as the round-trip residual with its deliberately mismatched "
            "phase-sign twin, which is what makes the round trip mean anything"
        ),
        tolerance_basis=(
            "numerical_precision_floor: float64 round-off over the accumulated phase "
            "of the enumerated round trip, 1e-12 against a measured 1.32e-15, with "
            "the broken twin at 1.40"
        ),
    ),
    Coverage(
        component="C_RAY_TO_WAVE",
        kind=DeclarationKind.INVARIANT,
        anchor="pupil_power_consistency",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=("B2-R2W-EXACT", "benchmarks/instances/b2_transitions.py"),
        reason=(
            "the exact-route power ratio is asserted as a declared invariant of the "
            "family, measured on the enumerated instance where there is no sampling "
            "budget to hide inside"
        ),
        tolerance_basis=(
            "numerical_precision_floor: the enumerated route is a relabelling of the "
            "same modes, so the only admissible discrepancy is float64 summation "
            "round-off; 1e-12"
        ),
    ),
)


_C_WAVE_TO_RAY: tuple[Coverage, ...] = (
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Scalar, monochromatic, fully coherent field on a planar tangent patch",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_coupler_contracts.py::test_complex_field_builds_from_an_unmodified_chromatix_artifact",
            "tests/test_wave_to_ray.py::test_emitted_rays_declare_their_own_contract",
        ),
        reason=(
            "the artifact contract carries scalar polarization, one wavelength and a "
            "declared coherence model, and a field that does not is refused at "
            "construction rather than decomposed"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Sampling density is strictly positive wherever the spectrum is nonzero",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=("tests/test_wave_to_ray.py::test_a_density_with_a_hole_is_refused_as_inconsistent",),
        reason=(
            "a density with a zero where the spectrum is nonzero is refused as "
            "inconsistent, because the 1/p weight would be unbounded there and the "
            "estimator would silently stop being unbiased"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Tangent-plane approximation holds within eps_curv",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B0-VALIDITY-01",
            "tests/test_curvature_bound.py::test_exceeding_the_threshold_raises_with_a_usable_remedy",
        ),
        reason=(
            "the bound is the closed form of SI eq S9 and is asserted against it; "
            "crossing it is reported as out_of_validity rather than unsupported, "
            "because the tangent-plane picture still computes -- it computes the wrong "
            "thing"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.WARNING,
        anchor="Evanescent modes with ku^2 + kv^2 > k^2 are discarded",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-W2R-STOCH",
            "tests/test_wave_to_ray.py::test_evanescent_power_is_reported_as_a_named_loss",
        ),
        reason=(
            "the family gates the power ledger -- propagated plus reported discarded "
            "equals input -- so a path that absorbs the discarded power rather than "
            "reporting it fails"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.WARNING,
        anchor="Omitting the 1/p importance weight biases the estimator",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-W2R-STOCH",
            "tests/test_wave_to_ray.py::test_omitting_the_importance_weight_is_detected_as_a_bias",
            "tests/test_wave_to_ray.py::test_under_uniform_sampling_the_missing_weight_is_only_a_scale_factor",
        ),
        reason=(
            "run as a negative control through the shipping estimator with the weight "
            "removed, reported as a bias in measured standard errors; the second test "
            "is the blind-spot statement that a uniform-sampling control could not "
            "have established it"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.WARNING,
        anchor="Bitwise reproducibility at a fixed seed is not evidence of accuracy",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-W2R-STOCH",
            "tests/test_wave_to_ray.py::test_same_seed_gives_bitwise_identical_rays",
        ),
        reason=(
            "the schema enforces the distinction rather than leaving it to prose: a "
            "stochastic family must require more than one seed, reproducibility is "
            "reported under its own name, and the family's accuracy claims rest on "
            "ensemble statistics"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.HARD_LIMIT,
        anchor="grazing_bin",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=("tests/test_wave_to_ray.py::test_selecting_an_evanescent_bin_is_refused",),
        reason=(
            "the boundary bin at kn = 0 is excluded by the propagating-mode selection "
            "and asking for it is refused, rather than yielding a ray with an "
            "undefined direction"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.INVARIANT,
        anchor="evanescent_power_accounted",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-W2R-STOCH",
            "tests/test_wave_to_ray.py::test_evanescent_power_is_reported_as_a_named_loss",
        ),
        reason="the power ledger is asserted to close, not merely to be reported",
        tolerance_basis=(
            "conservation_law: propagated + reported discarded = input, exactly, up "
            "to float64 summation round-off over the mode grid; 1e-9 covers the "
            "accumulation over the largest declared grid"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.INVARIANT,
        anchor="importance_weight_applied",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-W2R-STOCH",
            "tests/test_wave_to_ray.py::test_omitting_the_importance_weight_is_detected_as_a_bias",
        ),
        reason=(
            "asserted through its negative control rather than by reading a flag: the "
            "weight is removed from the shipping estimator and the ensemble mean must "
            "move past its measured standard error"
        ),
        tolerance_basis=(
            "analytic_derivation: the bias is gated in units of the MEASURED ensemble "
            "standard error at 3 sigma, so the tolerance is a property of the run "
            "rather than a chosen field-space constant"
        ),
    ),
    Coverage(
        component="C_WAVE_TO_RAY",
        kind=DeclarationKind.INVARIANT,
        anchor="unit_direction_norm",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_artifacts.py",
            "tests/test_wave_to_ray.py::test_emitted_rays_declare_their_own_contract",
        ),
        reason=(
            "enforced at the artifact boundary on every bundle the coupler emits, so "
            "it cannot be satisfied by a test that happens to look"
        ),
        tolerance_basis=(
            "numerical_precision_floor: the boundary uses a dtype-dependent tolerance "
            "on ||d|| - 1, float32 and float64 each against their own epsilon, "
            "because a fixed absolute bound would be either vacuous at float32 or "
            "unsatisfiable at float64"
        ),
    ),
)


_C_PLANAR_DOE_STEP: tuple[Coverage, ...] = (
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.ASSUMPTION,
        anchor="The surface is PLANAR",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_patch_wft.py::test_a_conformal_substrate_is_refused_rather_than_approximated",
            "tests/test_curvature_bound.py",
        ),
        reason=(
            "a conformal substrate is refused rather than approximated, so the planar "
            "case's confidence cannot be inherited by a curved one; the curvature "
            "module supplies the patch-sizing bound the refusal points at"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.ASSUMPTION,
        anchor="The declared plane is where the transmission lives",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_coupler_round_trip.py::test_the_cascade_applies_the_doe_to_the_accumulated_field",
            "tests/test_planar_doe_step.py::test_the_node_refuses_an_incident_record_with_no_declared_opl",
        ),
        reason=(
            "the transmission is applied to the field accumulated ON the declared "
            "plane, asserted directly, and the step does not propagate the incident "
            "bundle to it -- a record whose reference plane is not declared is refused"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.WARNING,
        anchor=(
            "The step RESETS optical path length to zero on the outgoing rays. "
            "The incident path"
        ),
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_planar_doe_step.py::test_the_outgoing_optical_path_length_is_actually_zero",
        ),
        reason=(
            "the reset is asserted as a value rather than documented, and the rebasing "
            "consequence is the entry the knowledge pack keys by symptom for a "
            "downstream OPL that reads short"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.WARNING,
        anchor="Power is not conserved by default and should not be",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_planar_doe_step.py::test_preserve_energy_is_off_by_default_and_reported_when_on",
            "tests/test_coupler_round_trip.py::test_an_absorbing_doe_loses_power_and_says_so",
        ),
        reason=(
            "the default is asserted to be off, the renormalization factor is asserted "
            "to be reported when on, and an absorbing DOE is asserted to lose power "
            "and say so rather than being silently renormalized"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.WARNING,
        anchor="The outgoing amplitude is a spectral amplitude",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_planar_doe_step.py::test_the_outgoing_count_does_not_depend_on_the_incident_count",
            "knowledge/couplers/README.md",
        ),
        reason=(
            "no per-ray correspondence can survive when the outgoing count is "
            "independent of the incident count, which is asserted; the prose "
            "companion states it as one of the quantities that provably does not "
            "survive the transition"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.WARNING,
        anchor="Primary launch positions are sampled UNIFORMLY",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_planar_doe_step.py::test_primary_positions_can_be_sampled_instead_of_supplied",
            "tests/test_planar_doe_step.py::test_supplying_both_position_sources_is_a_conflict_not_a_precedence",
        ),
        reason=(
            "the asymmetry is offered as an option rather than adopted: sampling is "
            "opt-in, supplying positions is the alternative, and supplying both is a "
            "conflict rather than a precedence question"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.INVARIANT,
        anchor="evanescent_power_accounted",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_coupler_round_trip.py::test_power_terms_are_reported_separately_rather_than_netted",
            "tests/test_planar_doe_step.py::test_padding_widens_the_grid_without_moving_the_power",
        ),
        reason=(
            "inherited from C_WAVE_TO_RAY, which the step composes, and asserted at "
            "the step's own boundary: the power terms are reported separately rather "
            "than netted into one number that could hide a loss"
        ),
        tolerance_basis=(
            "conservation_law: the discrete power sum is |U|^2 times pixel area and "
            "the ledger closes to float64 round-off over the grid; asserted at 1e-9, "
            "the same basis the underlying coupler uses"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.INVARIANT,
        anchor="importance_weight_applied",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_planar_doe_step.py::test_full_enumeration_still_reproduces_the_transmitted_field",
        ),
        reason=(
            "under full enumeration the weighted sum must reproduce the transmitted "
            "field exactly, which it cannot do if the weight is missing or wrong -- "
            "that is a stronger statement than checking a flag"
        ),
        tolerance_basis=(
            "numerical_precision_floor: full enumeration has zero sampling error, so "
            "the residual is float64 round-off on the mode sum and nothing else"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.INVARIANT,
        anchor="unit_direction_norm",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_artifacts.py",
            "tests/test_planar_doe_step.py::test_the_node_runs_and_reports_the_conventions",
        ),
        reason=(
            "enforced at the artifact boundary on the outgoing bundle, so every run of "
            "the step checks it rather than one test doing so"
        ),
        tolerance_basis=(
            "numerical_precision_floor: the dtype-dependent boundary tolerance on "
            "||d|| - 1, as for C_WAVE_TO_RAY"
        ),
    ),
    Coverage(
        component="C_PLANAR_DOE_STEP",
        kind=DeclarationKind.INVARIANT,
        anchor="outgoing_count_is_the_budget",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_coupler_round_trip.py::test_ray_count_after_a_planar_doe_is_the_budget_not_the_input_size",
            "tests/test_planar_doe_step.py::test_two_stacked_does_keep_the_outgoing_count_at_the_budget",
        ),
        reason=(
            "the composability invariant: two DOEs in series give 256 then 256 rays "
            "rather than 256*64, asserted on a real cascade, which is what makes a "
            "multi-element diffractive system runnable at all"
        ),
        tolerance_basis=(
            "analytic_derivation: an exact integer identity, so the tolerance is zero "
            "by construction and 1e-12 is the schema's way of writing that for a "
            "float-valued metric"
        ),
    ),
)


_C_PATCH_WFT: tuple[Coverage, ...] = (
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.ASSUMPTION,
        anchor="The substrate is PLANAR, declared rather than inferred",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_patch_wft.py::test_a_conformal_substrate_is_refused_rather_than_approximated",
            "tests/test_patch_wft.py::test_a_patch_wider_than_the_curvature_budget_is_refused",
        ),
        reason=(
            "conformal is refused rather than approximated, and a patch wider than the "
            "eq S9 budget is refused on a curved substrate, so the planar convergence "
            "argument cannot be applied where it does not hold"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.ASSUMPTION,
        anchor="patch_px is ODD",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_patch_wft.py::test_an_even_patch_is_refused_rather_than_silently_rounded",
        ),
        reason=(
            "an even patch is refused rather than rounded, which matters because the "
            "paper's own sizes are all even and rounding would silently change the "
            "operator the caller asked for"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.ASSUMPTION,
        anchor="pad_px is DERIVED, not taken",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_patch_wft.py::test_the_derived_pad_satisfies_all_three_conditions",
            "tests/test_patch_wft.py::test_a_pad_that_violates_clearance_produces_a_plausible_wrong_field",
            "tests/test_patch_wft.py::test_the_full_aperture_patch_is_the_one_exemption_from_clearance",
        ),
        reason=(
            "the derivation is asserted to satisfy clearance, centring and oddness "
            "together; the consequence of violating clearance is measured as a "
            "plausible field wrong by order 100%, and the one legitimate exemption is "
            "asserted as an exemption rather than left implicit"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Patch centres are drawn on the sample grid over the aperture DILATED",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=(
            "B2-EQUIV",
            "tests/test_patch_wft.py::test_a_dilated_plan_recovers_the_familiar_grid_plus_patch_clearance",
        ),
        reason=(
            "the dilation is asserted structurally, and continuous rather than "
            "grid-snapped centres is a declared negative control on B2-EQUIV that must "
            "break convergence -- the plateau is the symptom, so the control has to "
            "measure a sweep rather than one point"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.ASSUMPTION,
        anchor="The centre DENSITY is uniform by default and that is a choice",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=("tests/test_patch_positions.py",),
        reason=(
            "unbiasedness of each offered density is gated against the enumerated "
            "exact field rather than inferred from the noise level, and the default "
            "stays uniform so every committed record remains the estimator that "
            "produced it"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.WARNING,
        anchor=(
            "The step RESETS optical path length to zero on the outgoing rays, "
            "for the same reason"
        ),
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_patch_wft.py::test_the_emitted_bundle_declares_its_normalization_and_reference",
        ),
        reason=(
            "the emitted bundle's OPL reference is asserted to be the patch plane, so "
            "the rebasing is a declared property of the artifact rather than a "
            "footnote a downstream consumer has to remember"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.WARNING,
        anchor="NO apodization",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=("B2-EQUIV", "benchmarks/instances/b2_equiv.py"),
        reason=(
            "applying an apodization taper is a declared negative control on the "
            "sub-aperture convergence sweep and must break it, which is the executable "
            "form of the partition-of-unity argument"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.WARNING,
        anchor="Reconstruction is O(secondary_rays * pixels)",
        coverage_kind=CoverageKind.EXPLICIT_NON_EXECUTABLE,
        reason=(
            "a cost asymptote and a statement that batching is part of the method. "
            "The calibrated half -- the emitter term -- has an executable estimator "
            "with a domain and a refusal outside it, and that is tested. The "
            "reconstruction term is explicitly NOT calibrated, and asserting an "
            "asymptote would mean fitting an exponent on this machine and calling it a "
            "property of the algorithm, which is the thing B4 exists to keep out of a "
            "gate."
        ),
        evidence=(
            "tests/test_patch_wft.py::test_the_cost_estimate_names_the_downstream_term_not_its_own",
            "B4-COST",
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.INVARIANT,
        anchor="evanescent_power_accounted",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_patch_wft.py::test_a_grid_with_evanescent_modes_still_takes_the_boolean_path",
            "tests/test_wave_to_ray.py::test_evanescent_power_is_reported_as_a_named_loss",
        ),
        reason=(
            "inherited from the per-patch spectral decomposition, which is "
            "C_WAVE_TO_RAY's, and exercised on a grid that actually has evanescent "
            "content -- a coarse grid has none, so a test on one would prove nothing"
        ),
        tolerance_basis=(
            "conservation_law: per-patch propagated + discarded = patch power, to "
            "float64 round-off; 1e-9, the same basis as the underlying coupler"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.INVARIANT,
        anchor="importance_weight_applied",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_patch_wft.py::test_enumerating_every_patch_position_is_exact_not_merely_convergent",
        ),
        reason=(
            "enumerating every patch position is exact rather than merely convergent, "
            "and it cannot be if the per-mode weight is missing; the exactness is the "
            "assertion"
        ),
        tolerance_basis=(
            "numerical_precision_floor: complete enumeration has zero sampling error, "
            "measured at 5.9e-15 against a 1.4e-12 gate derived from the accumulated "
            "float64 round-off of the patch sum"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.INVARIANT,
        anchor="unit_direction_norm",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_artifacts.py",
            "tests/test_patch_wft.py::test_the_emitted_bundle_declares_its_normalization_and_reference",
        ),
        reason="enforced at the artifact boundary on the emitted bundle, as for the other couplers",
        tolerance_basis=(
            "numerical_precision_floor: the dtype-dependent boundary tolerance on "
            "||d|| - 1"
        ),
    ),
    Coverage(
        component="C_PATCH_WFT",
        kind=DeclarationKind.INVARIANT,
        anchor="patch_coverage_corrected",
        coverage_kind=CoverageKind.BENCHMARK_CASE,
        evidence=("B2-EQUIV", "benchmarks/instances/b2_equiv.py"),
        reason=(
            "gated on a SUB-APERTURE case, because the correction is exactly 1 on the "
            "full-aperture anchor -- which is why an inverted correction once survived. "
            "Omitting it is a declared negative control that must fail."
        ),
        tolerance_basis=(
            "conservation_law: the coverage-corrected power ratio is 1 exactly when "
            "the correction is A_draw/A_patch rather than its inverse, so the "
            "tolerance is float64 round-off on the ratio; 1e-9"
        ),
    ),
)


_C_GENERALIZED_SNELL: tuple[Coverage, ...] = (
    Coverage(
        component="C_GENERALIZED_SNELL",
        kind=DeclarationKind.ASSUMPTION,
        anchor="The substrate is PLANAR, declared rather than inferred",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_diffractive_interaction.py::test_generalized_snell_on_a_conformal_substrate_is_refused",
        ),
        reason=(
            "a conformal substrate is refused with MISSING_DECLARATION rather than "
            "approximated with the flat-plane frame, on every call -- there is no "
            "conditional path that would let one through"
        ),
    ),
    Coverage(
        component="C_GENERALIZED_SNELL",
        kind=DeclarationKind.ASSUMPTION,
        anchor="n_incident and n_transmitted are declared and used directly",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_diffractive_interaction.py::test_the_zero_phase_limit_is_ordinary_snells_law",
        ),
        reason=(
            "the zero-phase limit is tested at n_incident=1.0, n_transmitted=1.5 and "
            "reproduces ordinary Snell's law, which exercises the declared indices "
            "directly rather than only at the n=1 default"
        ),
    ),
    Coverage(
        component="C_GENERALIZED_SNELL",
        kind=DeclarationKind.ASSUMPTION,
        anchor="The diffraction order m is declared, defaulting to +1",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_diffractive_interaction.py::test_control_an_order_sign_flip_conjugates_the_deflection",
        ),
        reason=(
            "requesting order=-1 instead of the +1 default measurably changes the "
            "result (conjugates the deflection), so the declared default is a real "
            "value the equation reads rather than a hidden constant"
        ),
    ),
    Coverage(
        component="C_GENERALIZED_SNELL",
        kind=DeclarationKind.ASSUMPTION,
        anchor="Three validity predicates, each a signed normalized margin",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_diffractive_interaction.py::test_propagating_order_margin_sign_convention",
            "tests/test_diffractive_interaction.py::test_local_gradient_smoothness_margin_is_perfect_for_a_pure_ramp",
            "tests/test_diffractive_interaction.py::test_local_gradient_smoothness_margin_degrades_with_curvature",
            "tests/test_diffractive_interaction.py::test_single_order_dominance_is_high_for_a_well_resolved_grating",
            "tests/test_diffractive_interaction.py::test_single_order_dominance_is_low_for_a_two_tone_surface",
            "tests/test_diffractive_interaction.py::test_an_evanescent_requested_order_is_refused_not_returned_as_nonsense",
            "tests/test_diffractive_interaction.py::test_a_local_phase_discontinuity_is_refused_and_the_refusal_is_local",
        ),
        reason=(
            "each margin function's sign convention is tested directly, and the two "
            "that are runtime hard limits are separately shown to actually refuse the "
            "call rather than only compute a number"
        ),
    ),
    Coverage(
        component="C_GENERALIZED_SNELL",
        kind=DeclarationKind.WARNING,
        anchor="The gradient estimator reads the complex transmission directly",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_diffractive_interaction.py::test_a_local_phase_discontinuity_is_refused_and_the_refusal_is_local",
        ),
        reason=(
            "the known blind spot -- a discontinuity that breaks the estimator -- is "
            "demonstrated to be caught when local to a ray's own stencil; the warning "
            "is honest that a uniformly aliased signal is not caught, which is a "
            "negative claim the test suite cannot demonstrate by definition and is "
            "instead reasoned about in knowledge/couplers/generalized_snell/failure_guide.md"
        ),
    ),
    Coverage(
        component="C_GENERALIZED_SNELL",
        kind=DeclarationKind.WARNING,
        anchor="The OPL convention is additive, not reset-to-zero",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_diffractive_interaction.py::test_a_linear_phase_ramp_deflects_to_the_exact_grating_angle",
        ),
        reason=(
            "the exact grating-angle test constructs its incident bundle with a "
            "nonzero OPL reference and the outgoing OPL is read as incident-plus-"
            "surface-phase in couplers/generalized_snell.py, exercised on every "
            "closed-form test rather than asserted separately"
        ),
    ),
    Coverage(
        component="C_GENERALIZED_SNELL",
        kind=DeclarationKind.INVARIANT,
        anchor="unit_direction_norm",
        coverage_kind=CoverageKind.EXECUTABLE_TEST,
        evidence=(
            "tests/test_artifacts.py",
            "tests/test_diffractive_interaction.py::test_a_linear_phase_ramp_deflects_to_the_exact_grating_angle",
        ),
        reason="enforced at the artifact boundary on the outgoing bundle, as for the other couplers",
        tolerance_basis=(
            "numerical_precision_floor: the dtype-dependent boundary tolerance on "
            "||d|| - 1, as for the other three couplers"
        ),
    ),
)


#: The ledger. One entry per registry declaration, in both directions.
COVERAGE_LEDGER: tuple[Coverage, ...] = (
    *_MODEL_OPTILAND,
    *_MODEL_CHROMATIX,
    *_C_RAY_TO_WAVE,
    *_C_WAVE_TO_RAY,
    *_C_PLANAR_DOE_STEP,
    *_C_PATCH_WFT,
    *_C_GENERALIZED_SNELL,
)
