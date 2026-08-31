"""The anti-corruption boundary around the wave backend, asserted structurally.

CHE-183 (R06.1) acceptance criterion 3 and CHE-158 (R06) acceptance criterion 2:
no chromatix or JAX object is observable outside `solvers/chromatix/` -- by an AST
walk **and** a runtime check, the same two halves
`tests/solvers/test_optiland_boundary.py` applies to the ray backend, and
`code_tokens` is imported from there rather than written again so its meta-test
covers both walks.

Four rules
----------
1. **Nobody else imports the backend.** No module outside the package may
   `import chromatix` or `import jax` -- checked as a top-level import rather than
   as a token, because `chromatix` is also the name of *this* subpackage and
   `from solvers.chromatix import propagate` is the sanctioned route in.
   `numerics/` is the one production exception for JAX: it is the project's
   array-namespace bridge, `ArrayNamespace.JAX` is a declared compute namespace,
   and `numerics.arrays._to_jax` is the only sanctioned conversion into it, so
   forbidding the import there would forbid the mechanism that makes a GPU field
   expressible at all. Tests are outside this rule because several exist to assert
   the backend's absence and have to name it to do so.
2. **No chromatix symbol anywhere else**, in any code position -- an identifier,
   an attribute, a keyword argument or a runtime string. JAX gets no symbol rule:
   `jnp` is a legitimate alias inside the bridge, and every route to it starts
   with an import rule 1 already catches.
3. **Nothing loads the backend by being imported.** A fresh interpreter and
   `sys.modules`, because the failure is transitive: a module that looks
   backend-free can pull one three levels down.
4. **The artifact holds no backend buffer.** Rules 1-3 are about names; the last
   section is about the object a caller actually receives.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
from chromatix_support import a_scalar_field
from test_optiland_boundary import code_tokens

from representations import ScalarField
from solvers.chromatix import propagate

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TESTS = ROOT / "tests"
PACKAGE = SRC / "solvers" / "chromatix"

#: Backend symbols in any code position: an identifier, an attribute, a keyword
#: argument, or a runtime string. `Field` is the backend's own field type and is
#: checked as an exact token, so `ScalarField` -- the neutral type this boundary
#: exists to emit -- does not match it.
#:
#: The bare name `chromatix` is **not** here: it is also this subpackage's name,
#: so `from solvers.chromatix import propagate` would trip it. The import rule
#: below is what covers the distribution.
CHROMATIX_NAMES = frozenset(
    {
        "Field",
        "asm_propagate",
        "compute_asm_propagator",
        "kernel_propagate",
        "compute_padding_transfer",
        "l2_sq_norm",
        "broadcasted_wavelength",
        "f_grid",
        "spatial_dims",
    }
)

#: Backend distributions, matched against a module's top-level imports.
BACKEND_IMPORTS = frozenset({"chromatix", "jax", "jaxlib"})

#: JAX has no symbol rule of its own. `jnp` is a legitimate local alias inside
#: `numerics/arrays.py`, which is the sanctioned bridge, and every route to it
#: starts with an `import jax...` that the import rule already catches. Adding a
#: token rule here would only re-report the exception.

#: Files whose job is to name a backend symbol: this walk itself, and the ray
#: boundary walk, which enumerates the same names to forbid them.
SYMBOL_EXEMPT = frozenset(
    {
        TESTS / "solvers" / "test_chromatix_boundary.py",
        TESTS / "solvers" / "test_optiland_boundary.py",
    }
)

#: `numerics/` is the array-namespace bridge and is the one production package
#: allowed to import JAX. Enumerated rather than expressed as a path prefix
#: nobody notices.
JAX_EXEMPT_PACKAGES = frozenset({SRC / "numerics"})


def _top_level_imports(source: str) -> set[str]:
    """The first segment of every absolute import in one module.

    The same rule `scripts/check_dependencies.py` applies to production packages,
    restated here because that gate does not walk `tests/` and this criterion
    covers the whole tree.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def _production_modules_outside_the_package() -> list[Path]:
    outside = sorted(
        path
        for path in SRC.rglob("*.py")
        if "__pycache__" not in str(path) and PACKAGE not in path.parents
    )
    assert len(outside) > 10, "the walk found almost nothing, so it cannot fail"
    return outside


def _modules_outside_the_package() -> list[Path]:
    outside = sorted(
        path
        for tree in (SRC, TESTS)
        for path in tree.rglob("*.py")
        if "__pycache__" not in str(path) and PACKAGE not in path.parents
    )
    assert len(outside) > 20, "the walk found almost nothing, so it cannot fail"
    return outside


# ---------------------------------------------------------------------------
# 1. The AST walk
# ---------------------------------------------------------------------------


def test_no_chromatix_symbol_outside_the_package() -> None:
    """`Field`, `asm_propagate`, `f_grid`: nowhere else, in any code position."""
    offenders = [
        f"{path.relative_to(ROOT)}: {name!r}"
        for path in _modules_outside_the_package()
        if path not in SYMBOL_EXEMPT
        for name in sorted(CHROMATIX_NAMES & code_tokens(path.read_text(encoding="utf-8")))
    ]
    assert offenders == [], (
        "a chromatix symbol appears outside solvers/chromatix/:\n  " + "\n  ".join(offenders)
    )


def test_nothing_in_production_imports_the_backend_except_numerics() -> None:
    """Rule 1. `numerics/` may import JAX; nothing anywhere may import chromatix."""
    offenders = []
    for path in _production_modules_outside_the_package():
        imported = _top_level_imports(path.read_text(encoding="utf-8"))
        forbidden = BACKEND_IMPORTS & imported
        if any(package in path.parents for package in JAX_EXEMPT_PACKAGES):
            forbidden -= {"jax", "jaxlib"}
        offenders.extend(f"{path.relative_to(ROOT)}: {name!r}" for name in sorted(forbidden))
    assert offenders == [], (
        "a backend is imported in production outside solvers/chromatix/:\n  "
        + "\n  ".join(offenders)
    )


def test_the_walk_would_actually_catch_a_violation() -> None:
    """The detectors fire, and neither one fires on the sanctioned route in."""
    assert "Field" in code_tokens("from chromatix.core.field import Field\n")
    assert "asm_propagate" in code_tokens("METHODS = ('asm_propagate',)\n")
    assert CHROMATIX_NAMES & code_tokens("from representations import ScalarField\n") == set()

    assert BACKEND_IMPORTS & _top_level_imports("import chromatix.functional as cf\n")
    assert BACKEND_IMPORTS & _top_level_imports("from jax import numpy\n")
    # ...and the import every caller of this package makes is not a violation.
    assert not (
        BACKEND_IMPORTS & _top_level_imports("from solvers.chromatix import propagate\n")
    )


# ---------------------------------------------------------------------------
# 2. The runtime check
# ---------------------------------------------------------------------------


def test_importing_pulls_no_backend() -> None:
    """A fresh interpreter, `sys.modules`, and no chromatix or jax in it.

    Reading the package -- including the translation module and the solver -- must
    not import the backend. Parametrized inside one subprocess batch rather than
    one process per statement, because each interpreter start is the expensive part
    and the statements are independent.
    """
    statements = (
        "import solvers",
        "import solvers.chromatix",
        "from solvers.chromatix import propagate",
        "import solvers.chromatix.fields",
        "import solvers.chromatix.solver",
    )
    for statement in statements:
        probe = (
            f"{statement}\n"
            "import sys, json\n"
            "print(json.dumps(sorted(\n"
            "    name for name in sys.modules\n"
            "    if name.split('.')[0] in ('chromatix', 'jax', 'jaxlib', 'optiland', 'torch')\n"
            ")))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
        )
        assert result.stdout.strip().endswith("[]"), (
            f"{statement!r} loaded a backend: {result.stdout.strip()}"
        )


# ---------------------------------------------------------------------------
# 3. What crosses the boundary
# ---------------------------------------------------------------------------


def test_propagate_emits_a_neutral_field_holding_no_backend_buffer() -> None:
    """An absence claim on the artifact, not on its type annotation.

    A JAX array would satisfy `ScalarField` perfectly well -- `ArrayNamespace.JAX`
    is a declared compute namespace -- so the claim that has to be checked is the
    one the boundary actually makes: the field comes back in the namespace it went
    in as.
    """
    source = a_scalar_field()
    out = propagate(
        source,
        distance_m=10e-6,
        model={"method": "asm", "pad_width": 4, "target_surface": "target"},
    )
    assert isinstance(out, ScalarField)
    assert isinstance(out.u, np.ndarray)

    for name in ("u", "sample_pitch_m", "wavelength_m", "reference_surface", "frame"):
        value = getattr(out, name)
        module = type(value).__module__
        assert not module.startswith(("jax", "chromatix")), f"{name} is a {module} object"
