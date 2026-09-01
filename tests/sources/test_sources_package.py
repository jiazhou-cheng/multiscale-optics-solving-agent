"""The package surface: what `sources` exports, and what its docstring still excludes.

CHE-215 (R06.10), acceptance criterion 1 -- "the docstring and `__all__` never
disagree". That criterion exists because items 2 and 3 of that ticket **reversed
part of a landed declaration**: `src/sources/__init__.py` used to exclude "point
sources, Gaussian beams as a source primitive" outright, and landing them while
the sentence stood would have left the package's own canonical prose contradicting
its `__all__` -- with nothing anywhere to catch it.

So this file asserts the two halves against each other. It is the only test that
reads the package docstring as a contract, and it is deliberately small: what is
being pinned is the *agreement*, not the wording.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sources

PACKAGE = Path(sources.__file__).resolve().parent

#: The four sources and the two pure converters. Both landed representations are
#: initialized here, which is the asymmetry R06.5 left on purpose and R06.10 closed.
EXPECTED_EXPORTS = {
    "collimated_bundle",
    "direction_from_angle",
    "gaussian_beam",
    "plane_wave",
    "spherical_wave",
    "transverse_wavevector_from_angle",
}


def test_every_export_is_public_and_importable() -> None:
    """`__all__` is the surface, and every name on it resolves."""
    assert set(sources.__all__) == EXPECTED_EXPORTS
    assert len(sources.__all__) == len(set(sources.__all__))
    assert sources.__all__ == sorted(sources.__all__)
    for name in EXPECTED_EXPORTS:
        assert callable(getattr(sources, name))


def test_the_return_representation_is_unambiguous_per_operation() -> None:
    """`AGENTS.md`: representation-independent as a package, explicit per operation.

    One `RayBundle` source and three `ScalarField` sources in one flat package --
    no subpackage per representation, and no constructor whose return
    representation depends on its arguments. Asserted on the annotations rather
    than by calling, because the requirement is that it is unambiguous *in the
    signature*.
    """
    from representations import RayBundle, ScalarField

    expected = {
        "collimated_bundle": RayBundle,
        "plane_wave": ScalarField,
        "gaussian_beam": ScalarField,
        "spherical_wave": ScalarField,
    }
    for name, representation in expected.items():
        annotation = getattr(sources, name).__annotations__["return"]
        assert annotation == representation.__name__

    # The two converters return plain tuples, so they are not sources at all and
    # cannot be mistaken for one.
    for name in ("direction_from_angle", "transverse_wavevector_from_angle"):
        assert "tuple" in getattr(sources, name).__annotations__["return"]

    assert not any(path.is_dir() for path in PACKAGE.iterdir() if path.name != "__pycache__")


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
