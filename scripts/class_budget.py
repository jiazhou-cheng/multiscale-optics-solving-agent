"""Production classes per package, against a declared budget.

CHE-171 (R01.1). The reference implementation reached **280 production classes**
under stated principles comparable to the new ones. That is the entire argument
for this script existing in R01 rather than in R15: a principle nothing counts is
a principle that gets restated in every ticket and enforced in none.

The rule this counts against is the project's, quoted so it is checkable rather
than remembered. A class is justified only if:

1. several fields share an invariant enforced together;
2. it is a public serialized / versioned data model;
3. it owns a genuine mutable resource lifecycle;
4. at least two *current* implementations need runtime polymorphism;
5. it is a real plugin boundary used by the runtime or registry.

Otherwise the answer is a function, a module, a frozen dataclass, a TypedDict, a
tuple, a Literal or an Enum.

**What this script can and cannot do, stated plainly.** It counts. It cannot tell
whether a class satisfies one of the five rules -- that is a judgement, and
pretending otherwise would make the gate authoritative about something it does
not know. So the budget is the reviewed artifact: raising a number requires a
ticket to say which rule the new class satisfies, and this script makes an
unjustified raise visible instead of silent. `docs/architecture_principles.md`
labels this a judgement call for exactly that reason.

`PROJECT_CEILING` is the second half. Without it, a package could raise its own
budget indefinitely and every individual raise would look local; the ceiling makes
the sum of the budgets fail against the project's declared target of 22.

**Why this is a script *and* a test.** Same two-layer split as
`scripts/check_dependencies.py`, for the same reason:

* **This file is the CLI and report layer.** No pytest dependency. `make
  check-arch` prints every package, its count, its budget and the fully-qualified
  name of each class counted -- which is the output you want when deciding
  whether a raise is justified, and which an assertion failure does not give you.
* **`tests/unit/test_class_budget.py` is the gate and the meta-test.** It runs
  `verify()` in the default suite, and it drives `_classes_in` against synthetic
  modules to prove the counter behaves as claimed: that it finds nested classes
  (the cheapest way to hide growth under one budgeted name), that it does not
  charge for the five rules' sanctioned alternatives, and that it *does* charge
  for `TypedDict` and `Enum` -- a known limit pinned as a test rather than left as
  a surprise.

Run directly for a report, or through `tests/unit/test_class_budget.py`, which is
what puts it in the default suite.
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

# The repo root, so `scripts` resolves as a package whether this file is run
# directly (`python scripts/class_budget.py`) or imported by the test that gates
# it. `LANDED` is imported rather than restated: the migration state is one fact
# and a second copy would drift.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_dependencies import LANDED

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

#: Production classes allowed per package of the new tree.
#:
#: One entry per package in `LANDED`, and the numbers are the reviewed artifact:
#: a raise belongs to the ticket that adds the class and must name the rule that
#: class satisfies. `tests/unit/test_class_budget.py` requires each number to
#: equal what is actually on disk, so a budget cannot be raised in advance of the
#: code -- headroom is not a pre-authorization.
#:
#: `representations` is 2, raised from 0 by CHE-174 (R02.2). Both are rule 1 --
#: a shared invariant across several fields -- and the invariant is named, because
#: "these fields go together" is the justification every unjustified class also
#: claims:
#:
#:   `Frame`             rule 1 -- axis order, handedness, origin rule and
#:                       propagation axis are one mapping from array indices to
#:                       physical directions. Each fixes part of the same mapping,
#:                       and getting any one wrong silently mirrors, transposes or
#:                       shifts a wavefront instead of raising. All four are
#:                       validated in `__post_init__`.
#:   `ReferenceSurface`  rule 1 -- axial coordinate, unit normal and medium index
#:                       are only meaningful together: an optical path is `n * s`
#:                       projected onto the normal, so a valid `z_m` beside an
#:                       unnormalized normal or an index nobody set yields an OPL
#:                       wrong by a factor no later check can attribute back.
#:
#: Raised again to 5 by CHE-175 (R02.3, +2) and CHE-176 (R02.4, +1):
#:
#:   `ContractError`     exception -- the one catchable failure type in the
#:                       package. None of the five rules claims an exception and
#:                       R00.2 counted 22 of them in the old tree the same way;
#:                       it is a class because `except ContractError` is what
#:                       lets a coupler return a diagnostic instead of an
#:                       invented field, and `except ValueError` would also
#:                       swallow every unrelated arithmetic error.
#:   `RayBundle`         rules 1 + 2 -- geometry, coherent state and sampling
#:                       measure are three groups with joint invariants (per-ray
#:                       length, one device and one namespace, an optical path
#:                       without its reference or a measure without its kind is
#:                       unusable), and it is the public model a solver produces
#:                       and a coupler consumes.
#:   `ScalarField`       rules 1 + 2 -- array, pitch, wavelength, surface and pad
#:                       state are one physical object; a pitch that does not
#:                       belong to this array gives a plausible extent that is
#:                       wrong by a factor.
#:
#: Seven names did *not* land against that +3: `CoherentRayBatch`,
#: `WavefrontSamples`, `GeometricRayBundle`, `CoherentRayBundle`, `RayBundleBase`,
#: `TrackedRayBundle` and `RayBatch`, plus `PSF` as a representation.
#: `tests/representations/test_rays.py` and `test_scalar.py` assert their absence,
#: because a budget records what exists and cannot record what was avoided.
#:
#: `numerics` is 7, raised from 0 by CHE-173 (R02.1). Three of the seven are
#: production classes and the ticket names the rule each satisfies:
#:
#:   `DevicePlacement`        rule 1 -- kind and index are one invariant; an index
#:                            without a kind is meaningless and a host index is a
#:                            contradiction, refused in `__post_init__`.
#:   `ArrayState`             rule 1 -- namespace, device and dtype are one
#:                            observation of one buffer. Split apart they invite a
#:                            namespace read from data beside a device read from a
#:                            config value, describing no array that exists.
#:   `ComponentCapabilities`  rule 2 -- the public, probe-backed capability model
#:                            every descriptor and solver reasons against, and the
#:                            thing a probe re-run confirms or falsifies.
#:
#: The other four -- `Precision`, `DType`, `DeviceKind`, `ArrayNamespace` -- are
#: `StrEnum`s, which the five rules list as a *sanctioned alternative* to a class.
#: They are counted anyway because `StrEnum` is written with `class` syntax and
#: this gate is an AST count, not a judgement; `tests/unit/test_class_budget.py`
#: pins that behaviour deliberately. Four enums is the honest arithmetic, not four
#: classes the architecture wanted avoided.
#: `operations` is 2, raised from 0 by CHE-177 (R03.1). CHE-178 (R03.2) adds none:
#: the registry is module-level state plus four functions, and a `Registry` class
#: would need two consumers wanting independent registries to justify it.
#:
#:   `OperationDescriptor`  rules 2 + 5 -- the public model planning and the
#:                          runtime read, and the plugin boundary between an
#:                          operation and the layer that selects one. The four
#:                          operation kinds are a *field* on it; four subclasses
#:                          would share every field and override nothing, and the
#:                          distinction they would encode (a coupler changes
#:                          representation, a physical operator changes physical
#:                          state) is not one an `isinstance` check can enforce.
#:   `OperationKind`        a `StrEnum`, counted for the same reason the four in
#:                          `numerics` are: this gate is an AST count and
#:                          `StrEnum` is written with `class` syntax. CHE-177 calls
#:                          the class delta +1 on the grounds that an enum is a
#:                          sanctioned alternative to a class; 2 is what is on
#:                          disk, and the budget records disk.
#:
#: Not landed against that +2, and asserted absent by
#: `tests/operations/test_descriptors.py`: `SolverDescriptor`, `CouplerDescriptor`,
#: `PhysicalOperatorDescriptor`, `MeasurementDescriptor`, `Registry`, and the
#: 13 spec classes of the old `core/specs.py` that became fields on one record.
#: `problems` is 3, raised from 0 by CHE-156 (R04). The ticket budgets "2 public
#: classes, with material / aperture / source / wavelength as TypedDict, Literal
#: or frozen tuples"; three is what an AST count sees, and the third is the
#: TypedDict:
#:
#:   `RayTraceProblem`  rules 1 + 2 -- `stop_index` has to index `surfaces` and
#:                      `primary_wavelength_index` has to index `wavelengths_um`,
#:                      so the fields are jointly constrained and none of them
#:                      means anything alone; and it is the public model a solver
#:                      adapter consumes.
#:   `SurfaceSpec`      rule 1 -- curvature, conic, following medium and spacing
#:                      describe one interface. A radius given twice in two forms
#:                      makes the surface silently a *different* one rather than
#:                      an invalid one, which is the failure mode the joint
#:                      validation exists for.
#:   `Material`         a `TypedDict`, which the five rules list as a sanctioned
#:                      alternative to a class, counted for the same reason the
#:                      enums in `numerics` and `operations` are: this gate is an
#:                      AST count. Its runtime validation is a *function*
#:                      (`_check_material`), because a TypedDict is an annotation
#:                      that disappears at run time.
#:
#: Twenty class names did not land against that +3 -- the whole of
#: `core/optical_system.py`'s geometry, interaction, material, aperture, field and
#: wavelength hierarchies, its four kind enums and its error type, plus
#: `core/optical_assembly.py`'s three and every builder.
#: `tests/problems/test_ray_trace.py` asserts their absence.
#: `solvers` is 2, raised from 0 by CHE-181 (R05.3). **Both are `TypedDict`s and
#: neither is a public class**: R05 budgets "0 public classes, at most 1 private",
#: and the private one (`_OptilandTraceBatch`, for grouped native ray state) did
#: not land -- the native columns are a local dict inside one function, and no
#: second function exchanges them. Two is what an AST count sees, for the same
#: reason `problems.Material` is counted: this gate counts `class` syntax and does
#: not judge, and `TypedDict` is written with it.
#:
#:   `Sampling`   a `TypedDict`, the sanctioned alternative -- the fields share no
#:                invariant enforced across them, nothing subclasses it, and its
#:                runtime validation is a *function* (`_require_keys`) because a
#:                TypedDict is an annotation that disappears at run time. It is
#:                the public argument schema of `trace`, and the alternative was
#:                `Mapping[str, Any]`, which is the untyped config dict this
#:                rewrite is removing.
#:   `Execution`  the same, for device and precision. Both keys are required
#:                rather than defaulted, because this is the process-global solver
#:                state whose inheritance was a measured source of
#:                nondeterminism.
#:
#: Six class names did **not** land against that +2, and the tests name them:
#: `OptilandAdapter` (a one-instance facade behind `get_adapter()`),
#: `OptilandExecutionState`, `TracePlans`, `OptilandRayRequest` /
#: `OptilandRayFailure` / `OptilandRayResult`, `PatchEmitterCostModel`, and
#: `HandoffPlaneError` as a separate exception type -- an unresolvable plane is a
#: `ContractError` with a code, because a coupler branches on the code and not on
#: the class. `solvers/optiland/baseline.py` did not land either, in its entirety.
#: `operators` is 0, declared by CHE-211 (R06.6), and `sources` is 0, declared by
#: CHE-210 (R06.5). Both are genuinely zero rather than budgeted-at-zero-for-now: a
#: mask is an array plus the grid it was built on and the grid already lives on the
#: `ScalarField`, and an illumination is three keyword arguments and a return
#: value. Nine class names did **not** land against those two packages, and the
#: tests name them: `ThinElement`, `PhaseMask`, `AmplitudeMask`, `Pupil` and
#: `Grating` as separate operators or result types, and `Illumination`,
#: `PlaneWaveSource`, `IlluminationAngle` and `SourceResult` on the source side.
#: The `Illumination` frozen dataclass is the one that has a real argument -- lambda,
#: `k_t` and the medium index are coupled by `|k_t| <= n k0`, which is rule 1 -- and
#: it did not land because the coupling is checked *at the point the field is
#: built*, where the grid's Nyquist limit is also known and is the second refusal;
#: a declaration object could not check that half, so it would validate less than
#: the function does while adding a type.
BUDGETS: dict[str, int] = {
    "numerics": 7,
    "operations": 2,
    "operators": 0,
    "problems": 3,
    "representations": 5,
    "solvers": 2,
    "sources": 0,
}

#: The project's declared target for the whole production tree. The reference
#: implementation had 280; R00's inventory found 66 of them would satisfy a rule
#: at all, which is still three times this number, so the collapse R02-R11 owes is
#: real work and not rounding.
#:
#: **Open inconsistency, recorded rather than resolved here.** `AGENTS.md` and
#: `docs/architecture_principles.md` were both rewritten for the clean slate and
#: now state that the rewrite does *not* inherit a project-wide class ceiling from
#: the reference implementation, on the grounds that 22 was derived from a tree
#: that no longer exists. This script still enforces it. Nothing fails today (7 of
#: 22) and the per-package budgets above are the part doing the real work either
#: way, but the two documents and this constant disagree and the owner has not
#: settled it. Do not quietly delete the ceiling to make a raise fit -- that is the
#: exact move it exists to make visible.
PROJECT_CEILING = 22


@dataclass(frozen=True)
class PackageCount:
    package: str
    counted: int
    budget: int
    classes: tuple[str, ...]

    @property
    def over(self) -> int:
        return max(0, self.counted - self.budget)

    @property
    def headroom(self) -> int:
        return max(0, self.budget - self.counted)


def _classes_in(path: Path) -> list[str]:
    """Top-level and nested class names defined in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def _modules_of(package: str) -> list[Path]:
    base = SRC / package
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in str(p))


def count() -> list[PackageCount]:
    counts: list[PackageCount] = []
    for package in sorted(LANDED):
        found: list[str] = []
        for module in _modules_of(package):
            where = module.relative_to(SRC).as_posix()
            found.extend(f"{where}::{name}" for name in _classes_in(module))
        counts.append(
            PackageCount(
                package=package,
                counted=len(found),
                budget=BUDGETS.get(package, 0),
                classes=tuple(sorted(found)),
            )
        )
    return counts


def verify() -> tuple[list[str], list[str], list[PackageCount]]:
    """Return (budget failures, structural problems, per-package counts)."""
    counts = count()
    failures: list[str] = []
    structural: list[str] = []

    if not LANDED:
        structural.append(
            "no package of the new tree has been landed, so this gate would pass "
            "without counting anything"
        )

    for package in sorted(LANDED):
        if package not in BUDGETS:
            structural.append(
                f"src/{package}/ is a landed package with no entry in BUDGETS, so its "
                "class count is unbudgeted. Declare a number, even if it is 0."
            )
    for package in sorted(BUDGETS.keys() - LANDED):
        structural.append(
            f"BUDGETS declares {package!r}, which is not landed. A budget for a package "
            "that does not exist reads as headroom nobody is using."
        )

    declared = sum(BUDGETS.values())
    if declared > PROJECT_CEILING:
        failures.append(
            f"the budgets sum to {declared}, over the project ceiling of {PROJECT_CEILING}. "
            "Raising one package's budget cannot be a local decision: the target is for the "
            "whole production tree."
        )

    for entry in counts:
        if entry.over:
            failures.append(
                f"src/{entry.package}/ has {entry.counted} production class(es) against a "
                f"budget of {entry.budget} -- {entry.over} over.\n"
                + "\n".join(f"      {name}" for name in entry.classes)
                + "\n    Either collapse them (function, module, frozen dataclass, TypedDict, "
                "tuple, Literal, Enum) or raise the budget in a ticket that names which of the "
                "five minimality rules each new class satisfies."
            )

    return failures, structural, counts


def _report() -> int:
    failures, structural, counts = verify()
    total = sum(entry.counted for entry in counts)
    declared = sum(BUDGETS.values())
    print(f"production classes: {total} / {declared} budgeted (ceiling {PROJECT_CEILING})")
    for entry in counts:
        print(f"  src/{entry.package}/: {entry.counted} / {entry.budget}")
        for name in entry.classes:
            print(f"      {name}")
    if structural:
        print("\nSTRUCTURAL PROBLEM -- this gate cannot be trusted as it stands:")
        for problem in structural:
            print(f"  {problem}")
    if failures:
        print("\nBUDGET EXCEEDED:")
        for failure in failures:
            print(f"  {failure}")
    if not structural and not failures:
        print("\nOK: within budget.")
    return 1 if (structural or failures) else 0


if __name__ == "__main__":
    sys.exit(_report())
