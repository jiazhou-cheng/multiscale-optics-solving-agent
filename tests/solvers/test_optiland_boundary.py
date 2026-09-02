"""The anti-corruption boundary, asserted structurally and at run time.

CHE-157 (R05) acceptance criterion 2: no `RealRays`, `.i`, `.opd`, `opd_native`,
`optiland_intensity` or millimetre concept is observable outside
`solvers/optiland/` -- by an AST walk **and** a runtime check, the way the old
tree asserted the coupler core's solver-freedom.

Why both halves are needed
--------------------------
The AST walk is the one that scales: it reads every production module and every
test outside the package and fails on a native name used as *code* wherever it
appears -- an identifier, an attribute access, an import, a keyword argument, or a
runtime string such as a dict key. What it cannot see is *transitive* loading -- a
module that looks backend-free while pulling optiland three levels down -- so the
runtime half runs a fresh interpreter and asks `sys.modules`.

Code, not prose
---------------
The walk deliberately ignores docstrings. `representations/rays.py` explains at
length why `opd_native` may not be promoted to an optical path; that sentence is
the reason the rule exists and flagging it would make the only correct response
"stop explaining the rule". Every other string constant *is* checked, because a
dict key or a metadata field is how native state actually escapes.

`test_the_walk_would_actually_catch_a_violation` is the meta-test: it drives the
detector against synthetic modules, including the docstring case, because a gate
that cannot fail is not a gate and one that fails on its own documentation is
worse than none.

And criterion 3: the public boundary receives neutral project types and emits a
neutral `RayBundle`. That is checked as an absence claim on the emitted artifact,
because a bundle whose arrays were torch tensors would satisfy the type
annotation and not the boundary.

CHE-217 (R05.6) extends the last section to the second entry point and changes
nothing else: the AST walk and the `sys.modules` check are unaltered, and the
supplied-bundle path and its own test file are held to them as written -- which
is why `tests/solvers/test_optiland_bundle_trace.py` is in neither exemption set
and states its lengths in SI.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fixtures.systems import singlet_ref

from representations import RayBundle, ReferenceSurface
from solvers.optiland import trace, trace_rays

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
PACKAGE = SRC / "solvers" / "optiland"

#: Native ray attributes. `.i`, `.opd` and `.w` are the ticket's; `.L`, `.M` and
#: `.N` are added because they are how the native ray object spells a direction,
#: so reading one outside the package means something got hold of a `RealRays`
#: however it was annotated. Measured against the tree: no false positive.
NATIVE_ATTRIBUTES = frozenset({"i", "opd", "w", "L", "M", "N"})

#: Native names in any code position: an identifier, an import, a keyword
#: argument, or a runtime string.
NATIVE_NAMES = frozenset({"RealRays", "real_rays", "opd_native", "optiland_intensity"})

#: The millimetre concept, matched as a substring of any code token or runtime
#: string. A project module outside the package has no business naming one: SI is
#: the boundary unit and `rays.NATIVE_LENGTH_M` is the only place the conversion
#: lives.
MILLIMETRE_FRAGMENTS = ("_mm", "mm_", "millimet")

#: Modules exempt from the millimetre rule, each with its reason.
#:
#: `problems/ray_trace.py` and the tests that hold it to its schema are
#: *prescriptions*. R04 fixed that schema in millimetres deliberately: it is the
#: unit optical prescriptions are written in, and a schema whose numbers no longer
#: match the literature they were transcribed from is a worse schema. That is a
#: declared and tested unit on a problem statement, not native solver state
#: leaking out -- and `solvers/optiland/system.py` checks the declaration before
#: passing a single number through.
MILLIMETRE_EXEMPT = frozenset(
    {
        SRC / "problems" / "ray_trace.py",
        TESTS / "fixtures" / "systems.py",
        TESTS / "problems" / "test_ray_trace.py",
        TESTS / "problems" / "test_fixtures.py",
    }
    | {
        # The tests that verify this boundary, and the parity tests that compare
        # against records frozen in native units. A test that may not name what it
        # forbids cannot check that it is forbidden.
        TESTS / "solvers" / "test_optiland_boundary.py",
        TESTS / "solvers" / "test_optiland_system.py",
        TESTS / "solvers" / "test_optiland_solver.py",
        TESTS / "physics" / "test_optiland_opl_convention.py",
        TESTS / "physics" / "test_optiland_rays.py",
        # CHE-207. It measures the finite-object *launch state*, which is native
        # solver state by definition -- the whole point is that no project type
        # describes where a ray started -- and it checks the closed-form conjugate
        # in the prescription's own millimetres.
        TESTS / "physics" / "test_optiland_finite_conjugate.py",
    }
)

#: Modules exempt from the native-name and native-attribute rules. The boundary
#: and physics tests, for the same reason, and nothing in `src/`.
#:
#: `test_optiland_finite_conjugate.py` is here because CHE-207's central evidence
#: *is* a native reading: it regenerates the launch state and asserts the origin
#: spread is exactly zero, which is a claim about `RealRays` columns that no
#: neutral type carries. Reading them in a test that says so is the opposite of the
#: leak this gate exists to catch -- and the gate caught this file on its first run,
#: which is the check working rather than the exemption weakening it.
NATIVE_EXEMPT = frozenset(
    {
        TESTS / "solvers" / "test_optiland_boundary.py",
        TESTS / "solvers" / "test_optiland_system.py",
        TESTS / "physics" / "test_optiland_opl_convention.py",
        TESTS / "physics" / "test_optiland_rays.py",
        TESTS / "physics" / "test_optiland_finite_conjugate.py",
    }
)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """The `id()` of every string constant that is a docstring."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            continue
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            found.add(id(body[0].value))
    return found


def code_tokens(source: str, *, filename: str = "<probe>") -> set[str]:
    """Every name this module uses as code, plus every non-docstring string.

    Comments are not in the AST at all, so they are outside the rule for the same
    reason docstrings are exempted from it: a comment is an explanation, not a
    reference to native state.
    """
    tree = ast.parse(source, filename=filename)
    docstrings = _docstring_nodes(tree)
    tokens: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            tokens.add(node.id)
        elif isinstance(node, ast.Attribute):
            tokens.add(node.attr)
        elif isinstance(node, ast.alias):
            tokens.update(node.name.split("."))
            if node.asname:
                tokens.add(node.asname)
        elif isinstance(node, ast.ImportFrom) and node.module:
            tokens.update(node.module.split("."))
        elif isinstance(node, ast.arg) or (isinstance(node, ast.keyword) and node.arg):
            tokens.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            tokens.add(node.name)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            tokens.add(node.value)
    return tokens


def native_attributes(source: str, *, filename: str = "<probe>") -> set[str]:
    """Native ray attributes read off something in this module, `x.opd` style."""
    tree = ast.parse(source, filename=filename)
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in NATIVE_ATTRIBUTES
    }


def _modules_outside_the_package() -> list[Path]:
    """Every production module and every test that is not inside the package."""
    outside = sorted(
        path
        for tree in (SRC, TESTS)
        for path in tree.rglob("*.py")
        if "__pycache__" not in str(path) and PACKAGE not in path.parents
    )
    assert len(outside) > 10, "the walk found almost nothing, so it cannot fail"
    return outside


# ---------------------------------------------------------------------------
# 1. The AST walk
# ---------------------------------------------------------------------------


def test_no_native_ray_name_outside_the_package() -> None:
    """`RealRays`, `opd_native`, `optiland_intensity`: nowhere else, in any code."""
    offenders = [
        f"{path.relative_to(ROOT)}: {name!r}"
        for path in _modules_outside_the_package()
        if path not in NATIVE_EXEMPT
        for name in sorted(NATIVE_NAMES & code_tokens(path.read_text(encoding="utf-8")))
    ]
    assert offenders == [], (
        "native Optiland ray state is named outside solvers/optiland/:\n  "
        + "\n  ".join(offenders)
    )


def test_no_native_ray_attribute_read_outside_the_package() -> None:
    """Nothing outside the package reads `.i`, `.opd`, `.w`, `.L`, `.M` or `.N`."""
    offenders = [
        f"{path.relative_to(ROOT)}: .{attribute}"
        for path in _modules_outside_the_package()
        if path not in NATIVE_EXEMPT
        for attribute in sorted(native_attributes(path.read_text(encoding="utf-8")))
    ]
    assert offenders == [], (
        "a native ray attribute is read outside solvers/optiland/:\n  " + "\n  ".join(offenders)
    )


def test_no_millimetre_concept_outside_the_package_or_the_prescription() -> None:
    """SI at every boundary. The prescription schema is the one declared exception."""
    offenders = [
        f"{path.relative_to(ROOT)}: {token!r}"
        for path in _modules_outside_the_package()
        if path not in MILLIMETRE_EXEMPT
        for token in sorted(code_tokens(path.read_text(encoding="utf-8")))
        if any(fragment in token for fragment in MILLIMETRE_FRAGMENTS)
    ]
    assert offenders == [], (
        "a millimetre concept appears in code outside solvers/optiland/ and the "
        "prescription schema:\n  " + "\n  ".join(offenders)
    )


def test_the_walk_would_actually_catch_a_violation() -> None:
    """The meta-test: every detector fires, and none fires on its own explanation."""
    assert native_attributes("def f(rays):\n    return rays.opd * 1e-3\n") == {"opd"}
    assert "RealRays" in code_tokens("from optiland.rays import RealRays\n")
    assert "opd_native" in code_tokens("ARRAYS = {'opd_native': None}\n")
    assert "opd_native" in code_tokens("opd_native = 1\n")
    assert any(
        "_mm" in token for token in code_tokens("def f(thickness_mm: float) -> None: ...\n")
    )
    # ...and the exemption that makes the rule statable at all.
    assert code_tokens('"""opd_native must not become an OPL; lengths are millimetres."""\n') == (
        set()
    )


# ---------------------------------------------------------------------------
# 2. The runtime check
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "import numerics",
        "import representations",
        "import problems",
        "import operations",
        "import solvers.optiland",
        "from solvers.optiland import trace",
        "from solvers.optiland import trace_rays",
        "import solvers.optiland.rays",
        "import solvers.optiland.system",
    ],
)
def test_importing_pulls_no_solver(statement: str) -> None:
    """A fresh interpreter, `sys.modules`, and no optiland or torch in it.

    Reading the package -- including the module that translates native rays -- must
    not import the backend. The failure this guards against is transitive, which
    is why it needs a subprocess: a module that looks backend-free in isolation can
    pull one three levels down, and an in-process check would be satisfied by
    whatever the test session had already loaded.
    """
    probe = (
        f"{statement}\n"
        "import sys, json\n"
        "print(json.dumps(sorted(\n"
        "    name for name in sys.modules\n"
        "    if name.split('.')[0] in ('optiland', 'torch', 'jax', 'chromatix')\n"
        ")))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
        cwd=ROOT,
    )
    assert result.stdout.strip().endswith("[]"), (
        f"{statement!r} loaded a backend: {result.stdout.strip()}"
    )


# ---------------------------------------------------------------------------
# 3. What crosses the boundary
# ---------------------------------------------------------------------------


def test_trace_emits_a_neutral_bundle_and_nothing_native() -> None:
    """Criterion 3, as an absence claim on the artifact rather than on its type."""
    bundle = trace(
        singlet_ref(),
        sampling={"num_rings": 8, "reference_surface": "exit_pupil", "wavelength_um": 0.55},
        execution={"device": "cpu", "precision": "fp64"},
    )
    assert isinstance(bundle, RayBundle)
    # Every array is a plain NumPy buffer in a compute namespace. `RayBundle`
    # refuses a torch tensor outright, so this restates the contract at the
    # boundary that produced it rather than trusting it.
    for name in ("positions_m", "directions", "amplitude", "optical_path_m", "measure_weight"):
        array = getattr(bundle, name)
        assert isinstance(array, np.ndarray), f"{name} is {type(array).__name__}"
    # The three quantities the boundary owes, all present and all declared.
    assert bundle.optical_path_m is not None
    assert bundle.amplitude is not None
    assert bundle.measure_weight is not None
    assert bundle.measure_kind == "quadrature_area_m2"
    assert bundle.reference_surface.name in ("exit_pupil", "image_surface")
    # And no text field on it names the solver's types or its units. The optical
    # path reference is the one to watch: it is a long declaration written inside
    # the package, and it quotes the plane coordinate and the removed piston --
    # both of which have to be in metres by the time they are written down.
    for value in (
        bundle.optical_path_reference,
        bundle.reference_surface.name,
        bundle.frame.axis_order,
    ):
        text = str(value)
        for name in NATIVE_NAMES:
            assert name not in text
        for fragment in MILLIMETRE_FRAGMENTS:
            assert fragment not in text


def test_trace_rays_emits_a_neutral_bundle_and_nothing_native() -> None:
    """Criterion 3 on CHE-217's second entry point, on the same absence claim.

    The extension matters because this path is the one that *receives* a project
    representation as well as emitting one, so there are two directions for
    native state to escape in. The AST walk above already covers every module
    outside the package, including the tests for this path; what is checked here
    is the artifact.

    The composed optical-path reference is the field to watch: it is written
    inside the package, and it quotes the incoming bundle's own declaration, both
    surface coordinates and the object-space index. Every one of those has to be
    in metres and free of the solver's type names by the time it is written down.
    """
    count = 5
    radii = np.linspace(0.0, 2.0e-4, count)
    supplied = RayBundle(
        positions_m=np.column_stack([radii, np.zeros(count), np.zeros(count)]),
        directions=np.tile(np.array([0.0, 0.0, 1.0]), (count, 1)),
        wavelength_m=0.55e-6,
        reference_surface=ReferenceSurface(
            name="emitting surface", z_m=0.0, medium_index=1.0
        ),
        amplitude=np.linspace(0.3, 2.1, count) * np.exp(1j * np.linspace(0.2, 2.2, count)),
        optical_path_m=np.zeros(count),
        optical_path_reference="zero at the emitting surface",
        measure_weight=np.linspace(1.0, 5.0, count),
        measure_kind="importance_weight",
    )
    bundle = trace_rays(
        singlet_ref(), supplied, execution={"device": "cpu", "precision": "fp64"}
    )

    assert isinstance(bundle, RayBundle)
    for name in ("positions_m", "directions", "amplitude", "optical_path_m", "measure_weight"):
        array = getattr(bundle, name)
        assert isinstance(array, np.ndarray), f"{name} is {type(array).__name__}"
    assert bundle.measure_kind == "importance_weight"
    assert bundle.reference_surface.name == "image_surface"
    for value in (
        bundle.optical_path_reference,
        bundle.reference_surface.name,
        bundle.frame.axis_order,
    ):
        text = str(value)
        for name in NATIVE_NAMES:
            assert name not in text
        for fragment in MILLIMETRE_FRAGMENTS:
            assert fragment not in text
