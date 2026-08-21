"""Shared harness for the repo-owned Chromatix example reproductions (CHE-57 / PB6).

Each `cNN_<slug>.py` module in this directory is a *repo-owned* reimplementation
of one page from the official Chromatix documentation
(https://chromatix.readthedocs.io/en/latest/) -- Chromatix 101 plus the 15
examples -- rewritten against the pinned `chromatix==0.6.0` install (commit
`d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee`) rather than executed as an upstream
notebook.

A module exposes:

``TUTORIAL``
    A :class:`TutorialMeta` describing which upstream tutorial it reproduces.
``run()``
    Executes the reproduction and returns a :class:`TutorialResult` holding
    (a) ``metrics``: JSON-serializable quantities recorded as durable evidence,
    and (b) ``checks``: the validation applied to those quantities.

Validation strength is declared per check, mirroring the CHE-57 priority order:

``reference``
    Compared against a quantitative value published by the upstream tutorial.
``analytic``
    Compared against a closed-form/independently computable expectation that
    does not go through the code path under test.
``invariant``
    A structural or physical invariant (shape, finiteness, monotonicity,
    symmetry, conservation, expected direction of change).
``qualitative``
    The upstream example is inherently visual and offers no machine-checkable
    oracle beyond "it produced a plausible artifact". Used sparingly and never
    as the only check for a tutorial that has a computable quantity.

Run one reproduction on its own::

    ./run.sh python knowledge/solvers/chromatix/tutorials/c00_chromatix_101.py

Run every reproduction and refresh the recorded evidence::

    ./run.sh python knowledge/solvers/chromatix/tutorials/run_all.py --write-expected

``jax_enable_x64`` is pinned to ``False`` at import, matching
``tests/test_chromatix_adapter.py``'s autouse fixture: another module may set
it to ``True`` as an import side effect and the adapter registry eagerly imports
every adapter module, so collection order can otherwise flip it process-wide and
change these numbers (see ``conventions.md`` "Numerical dtype").
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# Every reproduction must be headless: the examples call `plt.show()` freely and
# the container has no display. Set before any pyplot import.
os.environ.setdefault("MPLBACKEND", "Agg")


def pin_jax_precision() -> None:
    """Pin `jax_enable_x64=False`, the pinned environment's documented default.

    Another module may flip this to True as an import side effect, and
    `multiscale_optics_agent.adapters.registry._discover()` imports every adapter
    module eagerly, so a chromatix reproduction can otherwise observe complex128
    output purely because of test collection order.
    """
    import jax

    jax.config.update("jax_enable_x64", False)

CheckKind = Literal["reference", "analytic", "invariant", "qualitative"]

EXPECTED_DIR = Path(__file__).resolve().parent / "expected"


@dataclass(frozen=True)
class TutorialMeta:
    """Identifies the upstream tutorial a module reproduces."""

    slug: str
    title: str
    level: Literal["beginner", "intermediate", "advanced"]
    url: str
    #: What solver capability / physical concept the tutorial establishes.
    demonstrates: str
    #: True when the reproduction is individually expensive (>~2 s), which
    #: routes its regression test to the `slow` marker instead of Tier A.
    slow: bool = False
    #: True when the reproduction needs the torch backend (optional install).
    #: Unused for Chromatix (which is JAX-based) but kept so the two harnesses
    #: share one shape.
    needs_torch: bool = False
    #: Relative tolerance for replaying this reproduction's recorded metrics.
    #: Leave None for a deterministic reproduction (the regression test then uses
    #: a float64-reassociation budget). Set it when the reproduction is genuinely
    #: stochastic -- e.g. Optiland's numba-backed BSDF scattering, whose RNG is
    #: not reachable from numpy's seed -- and say why in the module docstring.
    metric_rtol: float | None = None


@dataclass
class Check:
    name: str
    kind: CheckKind
    passed: bool
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class TutorialResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    #: Free-text notes worth keeping with the evidence (API drift, caveats).
    notes: list[str] = field(default_factory=list)

    # -- recording ---------------------------------------------------------
    def record(self, **metrics: Any) -> None:
        for key, value in metrics.items():
            self.metrics[key] = _jsonable(value)

    def note(self, text: str) -> None:
        self.notes.append(text)

    # -- checks ------------------------------------------------------------
    def check(self, name: str, kind: CheckKind, passed: bool, detail: str) -> bool:
        self.checks.append(Check(name=name, kind=kind, passed=bool(passed), detail=detail))
        return bool(passed)

    def check_true(self, name: str, kind: CheckKind, condition: bool, detail: str) -> bool:
        return self.check(name, kind, condition, detail)

    def check_close(
        self,
        name: str,
        kind: CheckKind,
        observed: float,
        expected: float,
        *,
        rel: float | None = None,
        abs_: float | None = None,
    ) -> bool:
        observed = float(observed)
        expected = float(expected)
        if rel is None and abs_ is None:
            rel = 1e-9
        ok = math.isclose(
            observed,
            expected,
            rel_tol=0.0 if rel is None else rel,
            abs_tol=0.0 if abs_ is None else abs_,
        )
        tol = f"rel={rel}" if rel is not None else ""
        tol += (", " if tol and abs_ is not None else "") + (f"abs={abs_}" if abs_ is not None else "")
        detail = f"observed={observed!r} expected={expected!r} ({tol})"
        return self.check(name, kind, ok, detail)

    def check_finite(self, name: str, values: Any, detail: str = "") -> bool:
        import numpy as np

        arr = np.asarray(values)
        ok = bool(np.all(np.isfinite(arr)))
        return self.check(
            name,
            "invariant",
            ok,
            detail or f"all finite over shape {tuple(arr.shape)}",
        )

    def check_shape(self, name: str, values: Any, expected: tuple[int, ...]) -> bool:
        import numpy as np

        got = tuple(np.asarray(values).shape)
        return self.check(
            name, "invariant", got == tuple(expected), f"shape {got} == {tuple(expected)}"
        )

    # -- reporting ---------------------------------------------------------
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]

    def to_json(self, meta: TutorialMeta) -> dict[str, Any]:
        return {
            "slug": meta.slug,
            "title": meta.title,
            "level": meta.level,
            "url": meta.url,
            "demonstrates": meta.demonstrates,
            "metrics": self.metrics,
            "checks": [c.to_json() for c in self.checks],
            "notes": self.notes,
            "all_checks_passed": self.passed,
        }


def _jsonable(value: Any) -> Any:
    """Convert numpy/torch scalars and arrays into JSON-serializable values."""
    import numpy as np

    if isinstance(value, (str, bool, int, float, type(None))):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "detach"):  # torch.Tensor
        value = value.detach().cpu().numpy()
    if type(value).__module__.startswith("jax"):  # jax.Array / tracers
        value = np.asarray(value)
    arr = np.asarray(value)
    if arr.dtype == object:
        return str(value)
    if arr.ndim == 0:
        item = arr.item()
        return item if isinstance(item, (bool, int, str)) else float(item)
    return [_jsonable(v) for v in arr.tolist()]


def expected_path(slug: str) -> Path:
    return EXPECTED_DIR / f"{slug}.json"


def emit(meta: TutorialMeta, result: TutorialResult, *, write_expected: bool = False) -> int:
    """Print the JSON evidence for one reproduction; optionally record it.

    Returns a process exit code so a module's ``main()`` can be used directly as
    a standalone probe.
    """
    payload = result.to_json(meta)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if write_expected:
        EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
        expected_path(meta.slug).write_text(text + "\n")
    if not result.passed:
        names = ", ".join(c.name for c in result.failures)
        print(f"FAILED checks: {names}")
        return 1
    return 0


def standalone_main(meta: TutorialMeta, run) -> int:
    """Boilerplate ``main()`` for a tutorial module."""
    import argparse

    parser = argparse.ArgumentParser(description=meta.title)
    parser.add_argument(
        "--write-expected",
        action="store_true",
        help="record this run into knowledge/solvers/optiland/tutorials/expected/",
    )
    args = parser.parse_args()
    return emit(meta, run(), write_expected=args.write_expected)


def tutorial_module_paths() -> list[Path]:
    """Every ``cNN_*.py`` reproduction in this directory, in slug order."""
    here = Path(__file__).resolve().parent
    return sorted(p for p in here.glob("c[0-9][0-9]_*.py"))


def load_tutorial_module(path: Path):
    """Import one reproduction by file path without requiring a package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def iter_tutorial_modules():
    for path in tutorial_module_paths():
        yield load_tutorial_module(path)
