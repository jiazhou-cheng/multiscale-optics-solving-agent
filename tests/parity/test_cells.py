"""Meta-tests: the fixture's own behaviour, falsified rather than described.

A parity fixture that cannot fail is worse than no fixture, because the suite
then reports agreement it never checked. Everything here is about
`tests/parity/` itself: where the cell set comes from (AC-1), that a misplaced
buffer fails (AC-2), that the four unavailability outcomes stay distinguishable
(AC-3), that tolerances have exactly one home (AC-4), and that the torch cells
are the strict xfails CHE-248 (T4) will trip over (AC-7).
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from numerics.arrays import COMPUTE_NAMESPACES, xp_for
from numerics.precision import ArrayNamespace, DeviceKind, DevicePlacement, DType
from parity.cells import Cell, cells_for, tolerance_for
from parity.conftest import unavailable_reason, verify_placement

_PACKAGE = Path(__file__).resolve().parent

_SYNTHETIC = {
    "schema_version": 1,
    "component": "SYNTHETIC_PARITY_SUBJECT",
    "probe": "benchmarks/probes/precision/synthetic_parity_subject.py",
    "probe_tag": "test-only",
    "evidence": "a synthetic record written by tests/parity/test_cells.py; measures nothing",
    "notes": "not a component. Exists so AC-1's derivation can be falsified against a "
    "declaration nobody has to edit src/ to change.",
    "devices": ["cpu", "cuda"],
    "precisions": ["fp32"],
    "minimum_compute_precision": "fp32",
    "accepted_input_dtypes": ["float32", "complex64"],
    "native_compute_dtypes": ["float32"],
    "output_dtypes": ["float32"],
    "lossy_input_dtypes": [],
    "device_namespaces": {"cpu": ["numpy", "jax"], "cuda": ["jax"]},
}


def _write(directory: Path, record: dict[str, Any]) -> str:
    component = str(record["component"])
    (directory / f"{component}.json").write_text(json.dumps(record), encoding="utf-8")
    return component


# ---------------------------------------------------------------------------
# AC-1: the cell set is derived from a declaration, and changing the
# declaration changes it.
# ---------------------------------------------------------------------------


def test_cells_are_derived_from_the_capability_record(tmp_path: Path) -> None:
    """The falsifier AC-1 asks for: a synthetic record, a hand-written expectation.

    Hand-written *here* on purpose. AC-1 forbids a hand-written namespace,
    device or dtype list in the fixture, because that would be a second
    declaration that drifts. A meta-test is the one place the expectation
    should be spelled out, since a derivation nobody checked by hand is a
    derivation of something nobody intended.
    """
    component = _write(tmp_path, _SYNTHETIC)

    assert {str(cell) for cell in cells_for(component, complex_data=False, directory=tmp_path)} == {
        "jax-cpu-float32",
        "jax-cuda-float32",
        "numpy-cpu-float32",
    }
    # `numpy-cuda-float32` is absent, and that is the fourth outcome of AC-3:
    # the record does not name numpy as a driver for cuda, so the cell never
    # enters the parameter list. It is not a skip, because there is no run to
    # skip and nothing a reader would need to be told about.
    assert {str(cell) for cell in cells_for(component, complex_data=True, directory=tmp_path)} == {
        "jax-cpu-complex64",
        "jax-cuda-complex64",
        "numpy-cpu-complex64",
    }


def test_changing_one_field_of_the_record_changes_the_cell_set(tmp_path: Path) -> None:
    """The second half of AC-1's falsifier: the derivation is live, not decorative."""
    before = cells_for(
        _write(tmp_path, _SYNTHETIC), complex_data=False, directory=tmp_path
    )

    host_only = dict(_SYNTHETIC)
    host_only["component"] = "SYNTHETIC_PARITY_SUBJECT_HOST_ONLY"
    host_only["devices"] = ["cpu"]
    host_only["device_namespaces"] = {"cpu": ["numpy", "jax"]}
    after = cells_for(_write(tmp_path, host_only), complex_data=False, directory=tmp_path)

    assert {str(cell) for cell in after} == {"jax-cpu-float32", "numpy-cpu-float32"}
    assert {str(cell) for cell in after} < {str(cell) for cell in before}


def test_a_pack_less_component_derives_from_the_project_declarations() -> None:
    """AC-1's documented asymmetry, asserted so it cannot be mistaken for an omission.

    Ten of the seventeen catalog records carry `capabilities=None`, including
    both subjects of this package. Their cells come from `COMPUTE_NAMESPACES`
    and `ArrayNamespace.can_leave_host` because those are the only declarations
    that exist for them -- nobody measured device or dtype rows for a repo-owned
    operation with no external backend. The expectation below is the whole cell
    set that follows, and `torch` is absent from it for the same reason it is
    absent from `COMPUTE_NAMESPACES`.
    """
    expected = {"numpy-cpu-complex64", "jax-cpu-complex64", "jax-cuda-complex64"}
    assert {str(cell) for cell in cells_for("M_PSF", complex_data=True)} == expected
    assert {str(cell) for cell in cells_for("C_RAY_TO_SCALAR", complex_data=True)} == expected
    assert ArrayNamespace.TORCH not in COMPUTE_NAMESPACES


# ---------------------------------------------------------------------------
# AC-2: a buffer that did not land where it was requested fails the fixture.
# ---------------------------------------------------------------------------


def test_a_buffer_that_landed_on_the_host_fails_a_cuda_cell() -> None:
    """AC-2's falsifier, and it needs no device -- which is better than AC-2 asked for.

    AC-2 specified "in a CUDA-capable session". It does not need one: the
    condition being forced is "requested cuda, observed host", and a host buffer
    beside a cuda-requesting cell is constructible anywhere. So this runs in the
    default gate, on any host, which is where a guard against
    `M_WAVE_CHROMATIX.json`'s recorded failure -- "a process-global JAX platform
    pin produces a successful complex64 run on the host while the caller asked
    for CUDA, with no error raised" -- is actually useful.
    """
    pytest.importorskip("jax")
    host_cell = Cell(
        namespace=ArrayNamespace.JAX,
        device=DevicePlacement(DeviceKind.CPU),
        dtype=DType.COMPLEX64,
    )
    device_cell = Cell(
        namespace=ArrayNamespace.JAX,
        device=DevicePlacement(DeviceKind.CUDA),
        dtype=DType.COMPLEX64,
    )

    import jax.numpy as jnp

    on_the_host = jnp.asarray(np.zeros(4, dtype=np.complex64))

    # The positive control first: without it, a `verify_placement` that failed
    # unconditionally would pass the test below.
    assert verify_placement(host_cell, on_the_host) is on_the_host

    with pytest.raises(pytest.fail.Exception) as refused:
        verify_placement(device_cell, on_the_host)
    message = str(refused.value)
    assert "requested=jax:complex64@cuda" in message, message
    assert "observed=jax:complex64@cpu" in message, message


def test_the_read_back_also_catches_a_namespace_or_dtype_that_drifted() -> None:
    """Placement is three fields, so all three are observed rather than one.

    A namespace that silently became NumPy is exactly the T1 failure -- an
    Optiland exit that copies to the host -- and a dtype that silently narrowed
    is the `jax_enable_x64` failure `verify_dtype` exists for. Neither is a
    device question, and a fixture that only checked the device would pass both.
    """
    host = np.zeros(4, dtype=np.complex64)
    numpy_cell = Cell(
        namespace=ArrayNamespace.NUMPY,
        device=DevicePlacement(DeviceKind.CPU),
        dtype=DType.COMPLEX64,
    )
    assert verify_placement(numpy_cell, host) is host

    wrong_dtype = Cell(
        namespace=ArrayNamespace.NUMPY,
        device=DevicePlacement(DeviceKind.CPU),
        dtype=DType.COMPLEX128,
    )
    with pytest.raises(pytest.fail.Exception, match="dtype complex64 != requested complex128"):
        verify_placement(wrong_dtype, host)

    wrong_namespace = Cell(
        namespace=ArrayNamespace.JAX,
        device=DevicePlacement(DeviceKind.CPU),
        dtype=DType.COMPLEX64,
    )
    with pytest.raises(pytest.fail.Exception, match="namespace numpy != requested jax"):
        verify_placement(wrong_namespace, host)


# ---------------------------------------------------------------------------
# AC-3: four outcomes, and they stay distinguishable.
# ---------------------------------------------------------------------------


def test_a_missing_dependency_is_reported_as_a_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Outcome 1, forced by pointing the namespace at a module that is not there.

    Simulated rather than provoked: both frameworks are pinned in both images,
    so the only honest way to exercise the branch is to make the import fail.
    """
    from parity import conftest as parity_conftest

    monkeypatch.setitem(
        parity_conftest._NAMESPACE_MODULES,
        ArrayNamespace.JAX,
        "a_namespace_module_that_does_not_exist",
    )
    cell = Cell(
        namespace=ArrayNamespace.JAX,
        device=DevicePlacement(DeviceKind.CUDA),
        dtype=DType.COMPLEX64,
    )
    reason = unavailable_reason(cell)
    assert reason is not None and reason.startswith("dependency missing: jax"), reason


def test_a_detached_device_is_reported_as_a_detached_device() -> None:
    """Outcome 2, delegating the judgement to the landed helper rather than repeating it.

    Asserted in both directions because this test has to be true in both
    containers: in `agent_solver` there is no device and the reason is reported;
    in `agent_solver_gpu` with a device attached there is no reason at all.
    """
    from conftest import cuda_unavailable_reason

    cell = Cell(
        namespace=ArrayNamespace.JAX,
        device=DevicePlacement(DeviceKind.CUDA),
        dtype=DType.COMPLEX64,
    )
    landed = cuda_unavailable_reason()
    reason = unavailable_reason(cell)
    if landed is None:
        assert reason is None
    else:
        assert reason == f"device not attached: {landed}"


def test_the_dedicated_session_rule_is_delegated_and_not_reimplemented() -> None:
    """Outcome 3: every CUDA cell carries `gpu`, and that is the whole mechanism.

    `tests/conftest.py::pytest_collection_modifyitems` owns the dedicated-session
    skip. Reimplementing it here would let this package call a session usable
    that `make test-gpu` calls unusable, so the only thing done here is to mark
    the cells -- and a torch-cpu cell must **not** be marked, since that is what
    keeps CHE-248's acceptance gate running in the default suite.
    """
    optiland = cells_for("M_RAY_OPTILAND", complex_data=False)
    for cell in optiland:
        marks = {mark.name for mark in cell.param.marks}
        assert ("gpu" in marks) is (cell.device.kind is DeviceKind.CUDA), (str(cell), marks)


def test_an_inadmissible_cell_never_enters_the_parameter_list() -> None:
    """Outcome 4. Optiland reaches CUDA only through torch, so `numpy-cuda` is absent.

    Not a skip: `ArrayState` refuses `numpy` on a CUDA device outright
    (`NUMPY_CANNOT_LEAVE_HOST`), so a cell like that could never produce a
    result to compare, and reporting it as skipped would imply it was a run this
    environment happened not to support.
    """
    cells = cells_for("M_RAY_OPTILAND", complex_data=False)
    assert cells, "the Optiland pack should derive cells"
    for cell in cells:
        if cell.device.kind is DeviceKind.CUDA:
            assert cell.namespace is ArrayNamespace.TORCH, str(cell)
    assert not [
        cell
        for cell in cells
        if cell.namespace is ArrayNamespace.NUMPY and cell.device.kind is DeviceKind.CUDA
    ]


def test_the_four_outcomes_do_not_collide(monkeypatch: pytest.MonkeyPatch) -> None:
    """The property AC-3 is actually about: a reader can tell them apart.

    All four are produced here, in one place, and compared. If two collapsed
    into the same message the suite could report "skipped" for a reason nobody
    could act on -- which is the failure the criterion exists to prevent, and it
    is a failure of *distinctness*, so it cannot be caught by the tests above
    that each check one outcome alone.
    """
    from parity import conftest as parity_conftest

    cell = Cell(
        namespace=ArrayNamespace.JAX,
        device=DevicePlacement(DeviceKind.CUDA),
        dtype=DType.COMPLEX64,
    )

    # 1. dependency missing, forced.
    monkeypatch.setitem(
        parity_conftest._NAMESPACE_MODULES, ArrayNamespace.JAX, "not_an_importable_module"
    )
    dependency = unavailable_reason(cell)
    monkeypatch.undo()

    # 2. device not attached -- or `None` in a container that has one, which is
    #    itself the distinction being asserted.
    device = unavailable_reason(cell)

    # 3. session not dedicated: delegated, and visible only as the marker.
    optiland = cells_for("M_RAY_OPTILAND", complex_data=False)
    marks = {mark.name for one in optiland for mark in one.param.marks}

    # 4. not admissible: an absence from the parameter list.
    inadmissible = [
        one
        for one in optiland
        if one.namespace is ArrayNamespace.NUMPY and one.device.kind is DeviceKind.CUDA
    ]

    assert dependency is not None and dependency.startswith("dependency missing"), dependency
    assert device is None or device.startswith("device not attached"), device
    assert device != dependency, "outcomes 1 and 2 report the same thing"
    assert "gpu" in marks
    assert inadmissible == [], "an inadmissible cell reached the parameter list"


# ---------------------------------------------------------------------------
# AC-4: one derivation point for tolerances, and nowhere else.
# ---------------------------------------------------------------------------


def test_the_tolerance_scales_with_what_its_docstring_says_it_scales_with() -> None:
    """AC-4: the derivation is checked, not just documented.

    Each assertion is one clause of `tolerance_for`'s docstring. A future edit
    that swaps `sqrt(n)` for `n`, or drops the complex factor, changes a
    scientific claim about error growth and has to come past this test.
    """
    complex_cell = Cell(
        namespace=ArrayNamespace.NUMPY,
        device=DevicePlacement(DeviceKind.CPU),
        dtype=DType.COMPLEX64,
    )
    real_cell = Cell(
        namespace=ArrayNamespace.NUMPY,
        device=DevicePlacement(DeviceKind.CPU),
        dtype=DType.FLOAT32,
    )
    wide_cell = Cell(
        namespace=ArrayNamespace.NUMPY,
        device=DevicePlacement(DeviceKind.CPU),
        dtype=DType.COMPLEX128,
    )

    one = tolerance_for(complex_cell, accumulation_length=1, matmul=False)
    four = tolerance_for(complex_cell, accumulation_length=4, matmul=False)
    assert four == pytest.approx(2.0 * one)  # sqrt(4), not 4

    assert tolerance_for(real_cell, accumulation_length=1, matmul=False) == pytest.approx(one / 2.0)
    assert tolerance_for(complex_cell, accumulation_length=1, matmul=True) == pytest.approx(
        2.0 * one
    )
    assert tolerance_for(wide_cell, accumulation_length=1, matmul=False) < one

    with pytest.raises(ValueError, match="accumulation_length"):
        tolerance_for(complex_cell, accumulation_length=0, matmul=False)


def test_no_tolerance_is_spelled_at_a_comparison() -> None:
    """AC-4's mechanical half: `tolerance_for` is the only home for one.

    Scans every module in this package for a float literal used as a comparison
    bound or handed to an approximate-equality helper. `0.0` is allowed, because
    an exact-zero degeneracy guard is not a tolerance -- there is no value of it
    that could be widened to make a failing test pass.

    `AGENTS.md`: "Do not widen a tolerance merely to make a benchmark pass."
    A literal at a comparison site is how that happens without a reviewer
    noticing, which is why this is a test rather than a convention.
    """
    approximate = {"approx", "allclose", "isclose", "assert_allclose"}
    offenders: list[str] = []

    for path in sorted(_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        allowed: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "tolerance_for":
                allowed = {id(inner) for inner in ast.walk(node)}

        for node in ast.walk(tree):
            if id(node) in allowed:
                continue
            literals: list[ast.Constant] = []
            if isinstance(node, ast.Compare):
                literals = [
                    operand
                    for operand in node.comparators
                    if isinstance(operand, ast.Constant) and isinstance(operand.value, float)
                ]
            elif isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(
                    node.func, "id", ""
                )
                if name in approximate:
                    literals = [
                        argument
                        for argument in node.args
                        if isinstance(argument, ast.Constant) and isinstance(argument.value, float)
                    ]
            offenders += [
                f"{path.name}:{literal.lineno} compares against {literal.value!r}"
                for literal in literals
                if literal.value != 0.0
            ]

    assert not offenders, (
        "a floating-point bound is spelled at a comparison site; tolerance_for is the only "
        f"place one may be decided: {offenders}"
    )


# ---------------------------------------------------------------------------
# AC-7: the torch cells, and why they are CHE-248's acceptance gate.
# ---------------------------------------------------------------------------


def test_torch_cells_are_strict_xfails_that_run_without_a_device() -> None:
    """AC-7, including the part the plan had wrong.

    `M_RAY_OPTILAND` declares `device_namespaces = {"cpu": ["numpy", "torch"],
    "cuda": ["torch"]}`, so **torch is admitted on the host as well as on
    CUDA**. The strict-xfail torch cells are therefore not GPU-only: two of them
    are host cells, they run in the default gate with no device attached, and
    that is what makes CHE-248 (T4) checkable by `make test` on any machine.
    """
    cells = cells_for("M_RAY_OPTILAND", complex_data=False)
    torch_cells = [cell for cell in cells if cell.namespace is ArrayNamespace.TORCH]
    host_torch = [cell for cell in torch_cells if cell.device.kind is DeviceKind.CPU]

    assert host_torch, "the Optiland pack admits torch on cpu; those cells must exist"
    for cell in torch_cells:
        marks = {mark.name: mark for mark in cell.param.marks}
        assert "xfail" in marks, str(cell)
        assert marks["xfail"].kwargs["strict"] is True, str(cell)
        assert "COMPUTE_NAMESPACES" in marks["xfail"].kwargs["reason"]
    for cell in host_torch:
        assert "gpu" not in {mark.name for mark in cell.param.marks}, str(cell)

    for cell in cells:
        if cell.namespace is not ArrayNamespace.TORCH:
            assert "xfail" not in {mark.name for mark in cell.param.marks}, str(cell)


@pytest.mark.parametrize(
    "cell", [cell.param for cell in cells_for("M_RAY_OPTILAND", complex_data=False)]
)
def test_a_cell_can_compute_in_its_own_namespace(cell: Any, place: Any) -> None:
    """The gate itself: placement succeeds for every cell, computation does not.

    A torch buffer can be *placed* today -- `to_namespace` bridges into torch on
    purpose -- so the refusal these cells trip is the compute one: `xp_for`
    raises `NAMESPACE_NOT_A_COMPUTE_NAMESPACE` rather than hand back a namespace
    that would produce a second physics implementation by accident. When
    CHE-248 opens it, this test passes for torch, the strict xfail becomes an
    unexpected pass, and the suite fails until the marks come off. That is the
    intended alarm, not a regression.
    """
    array = place(cell, np.arange(8, dtype=np.float64))
    xp = xp_for(cell.namespace)
    doubled = array * 2
    verify_placement(cell, doubled)
    assert float(xp.sum(doubled)) == float(np.arange(8).sum() * 2)
