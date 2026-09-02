"""The package surface: what `sources` exports, what it excludes, and what it may not be.

CHE-215 (R06.10), acceptance criterion 1 -- "the docstring and `__all__` never
disagree". That criterion exists because items 2 and 3 of that ticket **reversed
part of a landed declaration**: `src/sources/__init__.py` used to exclude "point
sources, Gaussian beams as a source primitive" outright, and landing them while
the sentence stood would have left the package's own canonical prose contradicting
its `__all__` -- with nothing anywhere to catch it.

CHE-219 (R05.8) adds the second half, and it is a *semantic* rule rather than a
dependency-direction one: **no system-launch ray initialization lives in
`sources/`**. The dependency check already forbids `sources -> solvers`, and that
was never the hazard. The hazard is a function in this package that returns a
`RayBundle` while importing nothing at all -- built from caller-supplied points and
a shared direction, with no optical system in scope, and therefore unable to say
whether those points are the entrance pupil, the stop, the first traced surface, a
valid finite-conjugate aim, or anything in the constructed system. That is what
`collimated_bundle` was, and its removal is what section 2 below pins: two ways to
initialize rays -- one system-independent here, one system-aware in
`solvers/optiland/launch.py` -- would preserve exactly the ambiguity R05.8
removed.

So this file asserts the halves against each other and against that rule. It is
the only test that reads the package docstring as a contract, and it is
deliberately small: what is being pinned is the *agreement*, not the wording.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sources

PACKAGE = Path(sources.__file__).resolve().parent

#: The three sources and the one pure converter. Every one initializes a
#: `ScalarField`; CHE-219 (R05.8) removed the two ray-launch names, and the
#: exact-set assertion below is what makes a re-addition a deliberate act.
EXPECTED_EXPORTS = {
    "gaussian_beam",
    "plane_wave",
    "spherical_wave",
    "transverse_wavevector_from_angle",
}

#: Names that must not come back. `collimated_bundle` built a launch `RayBundle`
#: from explicit points; `direction_from_angle` turned a source field into a
#: launched ray direction. Both are now test helpers in
#: `tests/fixtures/ray_bundles.py`, and both are checked by name because "it is
#: gone" and "it may not return" are different claims.
REMOVED_RAY_LAUNCH_EXPORTS = ("collimated_bundle", "direction_from_angle")

#: Vocabulary that only a system-dependent launch needs. Matched as substrings of
#: the package's source, in code positions only, because each one names a quantity
#: that requires a constructed optical system to resolve -- and resolving one here
#: would mean this package had started guessing at geometry it cannot see.
LAUNCH_VOCABULARY = (
    "entrance_pupil_diameter",
    "stop_index",
    "aiming",
    "ray_aiming",
    "launch_surface",
    "paraxial",
    "max_field",
    "num_rings",
    "hexapolar",
)


def test_every_export_is_public_and_importable() -> None:
    """`__all__` is the surface, and every name on it resolves."""
    assert set(sources.__all__) == EXPECTED_EXPORTS
    assert len(sources.__all__) == len(set(sources.__all__))
    assert sources.__all__ == sorted(sources.__all__)
    for name in EXPECTED_EXPORTS:
        assert callable(getattr(sources, name))


def test_the_return_representation_is_unambiguous_per_operation() -> None:
    """`AGENTS.md`: representation-independent as a package, explicit per operation.

    Three `ScalarField` sources in one flat package -- no subpackage per
    representation, and no constructor whose return representation depends on its
    arguments. Asserted on the annotations rather than by calling, because the
    requirement is that it is unambiguous *in the signature*.

    The *rule* is what R06.10 landed and it is unchanged by CHE-219 removing the
    ray source: a package spanning one representation and a package spanning two
    are held to the same requirement, which is why this reads the annotation rather
    than counting representations.
    """
    from representations import ScalarField

    for name in ("plane_wave", "gaussian_beam", "spherical_wave"):
        annotation = getattr(sources, name).__annotations__["return"]
        assert annotation == ScalarField.__name__

    # The converter returns a plain tuple, so it is not a source at all and cannot
    # be mistaken for one.
    assert "tuple" in sources.transverse_wavevector_from_angle.__annotations__["return"]

    assert not any(path.is_dir() for path in PACKAGE.iterdir() if path.name != "__pycache__")


# ---------------------------------------------------------------------------
# 2. No system-launch ray initialization lives here. CHE-219 (R05.8).
# ---------------------------------------------------------------------------


def test_no_public_operation_returns_a_ray_bundle() -> None:
    """The invariant, on the signature: `sources/` does not produce a `RayBundle`.

    Read off the annotations of everything the package exports, and separately off
    the *source* of every module including the private one -- because a function
    that is not on `__all__` but is importable is reachable, and a return
    annotation is the one place this requirement can be checked without calling
    anything.
    """
    for name in sources.__all__:
        annotation = getattr(sources, name).__annotations__.get("return", "")
        assert "RayBundle" not in str(annotation), f"{name} returns a RayBundle"

    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            returns = "" if node.returns is None else ast.unparse(node.returns)
            assert "RayBundle" not in returns, f"{path.name}:{node.name} returns {returns}"


def test_the_removed_ray_launch_names_are_gone_and_stay_gone() -> None:
    """Neither name is importable from the package, under any spelling.

    `hasattr` as well as `__all__`, because a module-level import that is merely
    absent from `__all__` is still `sources.collimated_bundle` to a caller. The
    submodule is checked too: `src/sources/collimated_bundle.py` moved to
    `tests/fixtures/ray_bundles.py`, and a file left behind would be importable
    whatever `__init__` exported.
    """
    for name in REMOVED_RAY_LAUNCH_EXPORTS:
        assert name not in sources.__all__
        assert not hasattr(sources, name)
    assert not (PACKAGE / "collimated_bundle.py").exists()

    with pytest.raises(ImportError):
        __import__("sources.collimated_bundle")


def test_no_pupil_stop_or_aiming_resolution_happens_here() -> None:
    """The stronger claim: not merely no `RayBundle`, but none of the vocabulary.

    Every term in `LAUNCH_VOCABULARY` names a quantity that needs a *constructed*
    optical system to resolve -- the entrance pupil's diameter, the stop, the
    aimer and its mode, the launch surface, the backend's paraxial
    characterization, the pupil sampling density. A source that resolved any of
    them would be guessing at geometry it cannot see, and it would be doing so
    without importing anything the dependency gate could catch.

    Docstrings are excluded, as `tests/solvers/test_optiland_boundary.py` excludes
    them for the same reason: this package's `__init__` explains at length *why*
    the stop and the aim are the solver's, and flagging that sentence would make
    the only correct response "stop explaining the rule". Every other string
    constant is checked, because a dict key is how a concept actually arrives.
    """
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        tokens: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                tokens.append(node.id)
            elif isinstance(node, ast.Attribute):
                tokens.append(node.attr)
            elif isinstance(node, ast.arg) or (isinstance(node, ast.keyword) and node.arg):
                tokens.append(node.arg)
            elif isinstance(node, ast.alias):
                tokens.append(node.name)
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                tokens.append(node.value)
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                tokens.append(node.name)
        joined = " ".join(tokens)
        for term in LAUNCH_VOCABULARY:
            assert term not in joined, f"{path.name} resolves {term!r}, which needs a system"


def test_the_lifted_exclusion_is_gone_from_the_docstring() -> None:
    """The sentence that contradicted `__all__` is not there any more.

    The exact phrase, because a docstring that merely *mentions* Gaussian beams
    somewhere while still excluding them is the failure this guards.
    """
    docstring = sources.__doc__ or ""
    assert "Gaussian beams as a source primitive" not in docstring
    assert "Not here: point sources" not in docstring
    # ...and the reversal is recorded as a decision rather than made silently.
    assert "exclusion was lifted for CHE-215 on the owner's" in docstring


@pytest.mark.parametrize(
    "still_excluded",
    [
        "spectra and chromatic fields",
        "polarization",
        "partially coherent illumination",
        "any physical model of an illumination unit",
        "delta-function emitters",
        "pupil-aware or finite-conjugate launches",
        "automatic aperture or NA inference",
        "arbitrary-`z` Gaussian beams",
    ],
)
def test_what_is_still_out_of_scope_is_named(still_excluded: str) -> None:
    """Lifting part of an exclusion does not lift the rest, so the rest is listed.

    Each of these is a thing a reader could reasonably expect from a package called
    `sources`, and each one absent-but-unnamed would read as an oversight rather
    than a decision.
    """
    assert still_excluded in (sources.__doc__ or "")


def test_the_removal_is_recorded_as_a_decision_rather_than_an_absence() -> None:
    """A capability that left has to say so, the way a lifted exclusion did.

    R06.10 recorded lifting an exclusion in the docstring rather than silently
    contradicting it; R05.8 narrowing the package back to one representation is the
    same obligation in the other direction. A reader who finds no ray source has to
    find out whether that is a decision or an oversight, and where the operation
    went.
    """
    docstring = sources.__doc__ or ""
    assert "CHE-219 (R05.8) removed ray initialization from this package" in docstring
    assert "A source may be described without a system. A ray launch may not" in docstring
    assert "solvers.optiland.launch" in docstring


def test_the_docstring_records_the_layer_decomposition_and_the_examples() -> None:
    """Criterion: the package docstring, not a new `docs/` file, is the documentation.

    Per the clean-slate rule -- `docs/` holds `architecture_principles.md` and
    `docs/rewrite/reference_inventory.md`, and adding a third file for one package
    is the scaffolding that rule exists to prevent.
    """
    docstring = sources.__doc__ or ""
    for layer in ("a **source**", "an **operator**", "**propagation**", "a **solver/problem**"):
        assert layer in docstring
    assert "Prefer composition over a new constructor" in docstring
    # One minimal example per source.
    for name in sorted(EXPECTED_EXPORTS):
        assert f"{name}(" in docstring


def test_the_package_defines_no_class() -> None:
    """`BUDGETS["sources"] == 0`, across every module including the private one.

    Per-module assertions live with each module's own tests; this is the one that
    would catch a class added in a *new* module nobody wrote a test file for.
    """
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        assert classes == [], f"{path.name} defines {classes}"


def test_the_package_imports_no_backend_and_no_downstream_package() -> None:
    """The allowlist row is `sources -> problems, representations, numerics`.

    `scripts/check_dependencies.py` is the gate; this is the same claim stated
    where a reader of this package will see it, and it also covers the *forbidden*
    direction, which is the half that matters: a source is upstream of everything
    that consumes state, so reaching for `couplers`, `operators`, `measurements` or
    a solver backend would invert the graph.
    """
    forbidden = {"solvers", "couplers", "operators", "measurements", "optiland", "chromatix"}
    for path in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = {node.module.split(".")[0]}
            else:
                continue
            assert not (roots & forbidden), f"{path.name} imports {roots & forbidden}"
