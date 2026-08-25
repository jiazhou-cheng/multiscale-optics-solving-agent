"""What the two solver adapters do today, pinned before CHE-91 splits them.

These tests are not written to be pretty, and several assert on things nobody
designed on purpose. That is the point. A characterization suite records
*current observable behaviour* -- including anything accidental -- so that a
mechanical refactor can be judged by "did anything change" rather than by "does
it still look right".

The rule for CHE-91: these must stay green **without edits** after the split. A
test that needs editing to pass is a behaviour change, and has to be justified
in the issue or reverted.

Four things are pinned, chosen because each is a way a split can silently move
behaviour:

1. **The failure-code inventory**, exhaustively and from the AST. 56 codes exist
   across the two adapters and 27 of them had no assertion anywhere before this
   file. A behavioural test per code is not achievable -- several need a missing
   dependency, a CUDA fault, or a solver that raises mid-trace -- but "the set of
   codes is exactly this" is achievable, and it is what AC 3 actually asks for.
   Moving a code string to a new module keeps it in the inventory; renaming or
   dropping one does not.
2. **Recorded array hashes.** The strongest available signal that no physics
   moved, because it is computed from the arrays themselves.
3. **Artifact metadata and diagnostic key sets.** A split that drops a metadata
   key produces a result that still validates and still looks like a field.
4. **Lazy imports.** Neither adapter may import its solver at module scope. This
   is easy to break by hoisting an import while tidying, and the damage --
   `import registry` pulling in jax -- shows up far from the cause.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPTILAND = ROOT / "src" / "solvers" / "optiland"
CHROMATIX = ROOT / "src" / "solvers" / "chromatix"

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# 1. The failure-code inventory
# ---------------------------------------------------------------------------

def _code_literals(package: Path) -> set[str]:
    """Every SCREAMING_SNAKE string literal in a package's modules.

    Collected from the AST rather than by regex so a code inside a comment or a
    docstring is not counted -- a comment mentioning a code is not the code
    existing. The `MODEL_ID` values match the shape too and are kept: they are
    contract strings for the same reason the failure codes are.
    """
    found: set[str] = set()
    for module in sorted(package.rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                text = node.value
                if len(text) >= 5 and text.replace("_", "").isalnum() and text.isupper():
                    found.add(text)
    return found


#: Frozen at bd0cf57, before the split, and generated from the AST rather than
#: hand-listed. Sorted so a diff reads as one line added or removed rather than
#: as a reflow.
#:
#: Two entries are not failure codes and are kept anyway. `"OPTILAND_"` is the
#: prefix the codes are built from -- if it moved or changed, every code built
#: through it would change with it, and nothing else would notice.
#: `CARRIER_REMOVED_ASM_ID` and `GLOBAL_PHASE_POLICY` are contract identifiers
#: for the same reason a code is: something downstream matches on them.
#: Two more that are not failure codes, enrolled by CHE-118 (M5.1) because the
#: guard asked for them by name. `MONOCHROMATIC_WAVELENGTH_RULE` is the declared
#: rule for handing Optiland one wavelength rather than one per ray, and it is
#: reported in `trace_ray_batch`'s diagnostics and provenance, so it is a contract
#: string in the same sense the others are. `CUDA_FP32_TRACE_COST` is the trace
#: cost calibration; both appear here only as `__all__` entries, which is the
#: shape this scan matches.
OPTILAND_CODES = frozenset({
    "CUDA_FP32_TRACE_COST",
    "MONOCHROMATIC_WAVELENGTH_RULE",
    "M_RAY_OPTILAND",
    "OPTILAND_",
    "OPTILAND_BASELINE_FAILED",
    "OPTILAND_CUDA_REQUIRES_TORCH",
    "OPTILAND_CUDA_UNAVAILABLE",
    "OPTILAND_CUSTOM_SYSTEM_NOT_IMPLEMENTED",
    "OPTILAND_DEPENDENCY_UNAVAILABLE",
    "OPTILAND_EXIT_PUPIL_UNRESOLVED",
    "OPTILAND_GRADIENTS_REQUIRE_DESIGN_PARAMETERS",
    "OPTILAND_GRADIENTS_REQUIRE_TORCH_BACKEND",
    "OPTILAND_HANDOFF_PLANE_UNREACHABLE",
    "OPTILAND_INVALID_BASELINE_REQUEST",
    "OPTILAND_INVALID_NUM_RAYS",
    "OPTILAND_INVALID_OR_EMPTY_OUTPUT",
    "OPTILAND_INVALID_PRESCRIPTION",
    "OPTILAND_INVALID_REQUEST",
    "OPTILAND_INVALID_WAVELENGTH",
    "OPTILAND_REQUEST_OK",
    "OPTILAND_TRACE_FAILED",
    "OPTILAND_UNSUPPORTED_BACKEND",
    "OPTILAND_UNSUPPORTED_BASELINE_REQUEST",
    "OPTILAND_UNSUPPORTED_DESIGN_PARAMETER",
    "OPTILAND_UNSUPPORTED_HANDOFF_PLANE",
    "OPTILAND_UNSUPPORTED_SAMPLE",
    "PRESCRIPTION_CATALOG_FILE_MISMATCH",
    "PRESCRIPTION_CATALOG_MATERIAL_INEXACT",
    "PRESCRIPTION_CATALOG_MATERIAL_UNRESOLVED",
    "PRESCRIPTION_CONFLICTING_SOURCES",
    "PRESCRIPTION_GEOMETRY_UNSUPPORTED",
    "PRESCRIPTION_GRATING_GEOMETRY_UNSUPPORTED",
    "PRESCRIPTION_MATERIAL_UNSUPPORTED",
    "PRESCRIPTION_NAME_UNKNOWN",
    "PRESCRIPTION_NOT_A_SPEC",
})

CHROMATIX_CODES = frozenset({
    "CARRIER_REMOVED_ASM_ID",
    "CHROMATIX_CUDA_UNAVAILABLE",
    "CHROMATIX_DEPENDENCY_UNAVAILABLE",
    "CHROMATIX_DEVICE_ORDINAL_UNAVAILABLE",
    "CHROMATIX_DEVICE_UNAVAILABLE",
    "CHROMATIX_GRADIENTS_NOT_SUPPORTED",
    "CHROMATIX_INPUT_FIELD_NOT_2D",
    "CHROMATIX_INPUT_FIELD_NOT_COMPLEX",
    "CHROMATIX_INPUT_FIELD_NOT_FINITE",
    "CHROMATIX_INPUT_FIELD_UNREADABLE",
    "CHROMATIX_INVALID_BASELINE_REQUEST",
    "CHROMATIX_INVALID_METADATA",
    "CHROMATIX_INVALID_PADDING",
    "CHROMATIX_INVALID_SAMPLING",
    "CHROMATIX_MISSING_CONFIG",
    "CHROMATIX_MISSING_INPUT",
    "CHROMATIX_MISSING_METADATA",
    "CHROMATIX_MISSING_PROPAGATION_DISTANCE",
    "CHROMATIX_NONFINITE_OUTPUT",
    "CHROMATIX_PHASOR_MISMATCH",
    "CHROMATIX_PROPAGATION_DISTANCE_MISMATCH",
    "CHROMATIX_REQUEST_VALID",
    "CHROMATIX_RESOURCE_ESTIMATE_EXCEEDED",
    "CHROMATIX_SOLVER_EXECUTION_FAILED",
    "CHROMATIX_SOURCE_PLANE_UNDECLARED",
    "CHROMATIX_UNSUPPORTED_CAPABILITY",
    "CHROMATIX_UNSUPPORTED_DEVICE",
    "CHROMATIX_UNSUPPORTED_DTYPE",
    "CHROMATIX_UNSUPPORTED_FIELD_KIND",
    "CHROMATIX_UNSUPPORTED_PROPAGATION",
    "CHROMATIX_UNSUPPORTED_PROPAGATION_METHOD",
    "GLOBAL_PHASE_POLICY",
    "M_WAVE_CHROMATIX",
})


@pytest.mark.parametrize(
    ("package", "expected"),
    [(OPTILAND, OPTILAND_CODES), (CHROMATIX, CHROMATIX_CODES)],
    ids=["optiland", "chromatix"],
)
def test_the_failure_code_inventory_is_unchanged(package: Path, expected: frozenset[str]) -> None:
    """Every structured code that existed before the split still exists.

    Deliberately an equality check, not a superset check. A code that vanished
    is a contract break for anyone matching on it; a code that appeared without
    being added here is a new contract nobody reviewed. Both are worth failing
    on, and they need different fixes.

    Renaming a code is out of scope for a refactor. If a rename is genuinely
    wanted, it is a breaking change and belongs in its own issue with its
    consumers identified.
    """
    actual = _code_literals(package)
    missing = expected - actual
    added = actual - expected
    assert not missing, (
        f"{package.name}: these structured codes no longer appear anywhere in the "
        f"package: {sorted(missing)}. Moving a code to a new module keeps it here; "
        "renaming or deleting one is a contract change and is out of scope for a "
        "mechanical split."
    )
    assert not added, (
        f"{package.name}: new structured codes appeared: {sorted(added)}. Add them "
        "to the inventory in this file together with the issue that introduced "
        "them, so the contract surface stays reviewed rather than discovered."
    )


# ---------------------------------------------------------------------------
# 2-3. A real run: array hash, metadata, diagnostics
# ---------------------------------------------------------------------------

#: The canonical Optiland request used below. Small enough to run in seconds,
#: real enough to exercise the exit-pupil handoff path the M3 slice uses.
RAY_CONFIG: dict[str, Any] = {
    "sample": "M3SingletRef",
    "num_rays": 8,
    "wavelength": 0.55,
    "handoff_plane": "exit_pupil",
}

#: Measured at bd0cf57 on the CPU image, numpy backend, float64. This is the
#: strongest signal available that no physics moved during the split: it is
#: computed from the traced arrays themselves, not from a summary of them.
RAY_ARRAY_SHA256 = "a84fe53f6184c097072bce9ef4c245470f865cf4f3099d492bc3a7afe6f3434a"

#: Not a count. A split that drops one of these produces a record that still
#: validates and still looks like a ray bundle.
RAY_METADATA_KEYS = frozenset({
    "backend", "conventions", "coordinate_fields", "direction_fields", "execution",
    "intensity_field", "intensity_is_not_amplitude", "length_unit", "native_length_unit",
    "native_to_si_scale", "native_wavelength_to_si_scale", "native_wavelength_unit",
    "pupil_boundary", "requested_Hx", "requested_Hy", "requested_num_rays", "sample",
    "scientific_array_sha256", "serialization", "summary_metrics", "survival_field",
    "survival_semantics", "traced_num_rays", "wavelength_m", "wavelength_unit",
})

WAVEFRONT_METADATA_KEYS = frozenset({
    "backend", "coordinate_fields", "length_unit", "missing_declared_metadata",
    "optical_path_length_source", "sample", "wavelength", "wavelength_unit",
})

RAY_DIAGNOSTIC_KEYS = frozenset({
    "Hx", "Hy", "actual_surviving_ray_count", "backend_used", "cpu_device", "device",
    "dtype", "execution", "package_version", "prescription_fingerprint",
    "prescription_source", "prescription_spec_version", "requested_num_rays",
    "runtime_seconds", "sample", "scientific_array_sha256", "seed", "seed_semantics",
    "summary_metrics", "wavelength_native_units",
})


@pytest.fixture(scope="module")
def traced_rays(tmp_path_factory: pytest.TempPathFactory) -> Any:
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    out = tmp_path_factory.mktemp("char_rays")
    return get_adapter().run(
        ModelRunRequest(
            run_id="characterization",
            node_id="lens",
            config={**RAY_CONFIG, "output_directory": str(out)},
        )
    )


@pytest.mark.optiland
def test_the_ray_array_hash_is_byte_identical(traced_rays: Any) -> None:
    from core.execution import RunStatus

    assert traced_rays.status is RunStatus.SUCCEEDED, traced_rays.error_message
    assert traced_rays.diagnostics["scientific_array_sha256"] == RAY_ARRAY_SHA256, (
        "the traced arrays changed. If this fires during a mechanical refactor, "
        "revert rather than investigate forward -- CHE-91 says so explicitly, and "
        "the reason is that a hash change with no intended cause is far more "
        "likely to be a reordered dtype resolution than a discovered improvement."
    )
    # The same hash is carried on the artifact, and the two must agree: the
    # record is what a downstream node reads, the diagnostic is what a human
    # reads, and a split that updated one of them would be undetectable.
    assert traced_rays.outputs["rays"].metadata["scientific_array_sha256"] == RAY_ARRAY_SHA256


@pytest.mark.optiland
def test_the_traced_ray_count_and_provenance_are_unchanged(traced_rays: Any) -> None:
    diagnostics = traced_rays.diagnostics
    assert diagnostics["actual_surviving_ray_count"] == 217
    assert diagnostics["requested_num_rays"] == 8
    assert diagnostics["backend_used"] == "numpy"
    assert diagnostics["device"] == "cpu"
    assert diagnostics["dtype"] == "float64"
    assert diagnostics["package_version"] == "0.6.0"
    assert diagnostics["seed"] == 20260811
    assert diagnostics["wavelength_native_units"] == 0.55
    assert diagnostics["prescription_source"] == "config['sample']"
    assert diagnostics["prescription_spec_version"] == "optical-system-spec/1"
    assert diagnostics["prescription_fingerprint"] == (
        "e0d03eae536e8cce784c6cfad2027aebc0aaf73f302831c32200e80a74c85f02"
    )


@pytest.mark.optiland
def test_the_ray_artifact_carries_exactly_these_declarations(traced_rays: Any) -> None:
    assert set(traced_rays.outputs["rays"].metadata) == RAY_METADATA_KEYS
    assert set(traced_rays.outputs["wavefront"].metadata) == WAVEFRONT_METADATA_KEYS
    assert set(traced_rays.diagnostics) == RAY_DIAGNOSTIC_KEYS
    assert sorted(traced_rays.outputs) == ["rays", "wavefront"]


@pytest.mark.optiland
def test_the_wavefront_port_still_declares_what_it_does_not_write(traced_rays: Any) -> None:
    """An accidental-looking behaviour, pinned because it is load-bearing.

    CHE-34 narrowed the `rays` port to what the adapter actually writes but left
    the `wavefront` port over-claiming amplitude, polarization and pupil_mask,
    on the grounds that narrowing a port nothing in M3 executes would be an
    unverified change. The adapter compensates by *reporting* the gap in
    `missing_declared_metadata`.

    That reporting is the only thing standing between a consumer and three
    declarations that are not backed by data, so the split must not lose it.
    """
    missing = traced_rays.outputs["wavefront"].metadata["missing_declared_metadata"]
    assert sorted(missing) == ["amplitude", "polarization", "pupil_mask"]


# ---------------------------------------------------------------------------
# 4. Lazy imports
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("module", "forbidden"),
    [
        ("solvers.optiland.adapter", ("optiland", "torch")),
        ("solvers.chromatix.adapter", ("chromatix", "jax")),
        ("solvers.registry", ("optiland", "torch", "chromatix", "jax")),
        ("registry.loader", ("optiland", "torch", "chromatix", "jax")),
        ("core.boundary", ("optiland", "torch", "chromatix", "jax")),
    ],
)
def test_importing_the_module_does_not_import_a_physics_solver(
    module: str, forbidden: tuple[str, ...]
) -> None:
    """Importing an adapter must not import its solver.

    Checked in a subprocess against `sys.modules`, not by reading the source: an
    import hoisted into a helper that the module body calls is just as much a
    module-scope import as one at the top, and only the runtime can tell.

    The cost of losing this is not the seconds. `registry.loader` is imported by
    the CLI and by every graph validation, and `core/capabilities.py` is imported
    by the precision policy; if either started pulling in jax, an unrelated
    command would begin initialising a GPU backend, which is exactly the class of
    process-global surprise AGENTS.md records two traps for.
    """
    probe = (
        f"import sys; import {module}; "
        f"leaked = sorted(m for m in {forbidden!r} if m in sys.modules); "
        "print(','.join(leaked))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tempfile.gettempdir(),
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": tempfile.gettempdir()},
    )
    assert result.returncode == 0, result.stderr
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, (
        f"importing {module} pulled in {leaked} at module scope. Keep the solver "
        "import inside the private _import_<solver>() helper called from "
        "run()/estimate(). Hoisting it while tidying is the easy mistake here, and "
        "the damage shows up far from the cause."
    )


# ---------------------------------------------------------------------------
# 5. Failure paths, behaviourally
# ---------------------------------------------------------------------------
#
# The inventory above proves a code still *exists*. These prove the paths that
# reach it still reach it, and -- more easily broken by a split -- that each one
# still fails in the same *manner*. Three manners are in play here and they are
# not interchangeable:
#
#   * a structured result with `status = FAILED` and an `error_type`;
#   * a raised `UnsupportedCapabilityError`, which AGENTS.md requires to be
#     raised *eagerly*, before any solver call;
#   * a `ValidationReport` issue from `validate_request`, which runs no solver.
#
# A refactor that converted a raise into a failed result would keep the code
# string, keep the message, and silently move a pre-flight refusal to after the
# solver ran. Only asserting the manner catches that.
#
# Not every code is reachable this way. Several need a missing dependency, a
# CUDA fault, or a solver that raises mid-trace, and manufacturing those would
# test the mock rather than the adapter. The inventory covers them.

@pytest.mark.optiland
@pytest.mark.parametrize(
    ("config", "fragment"),
    [
        ({"num_rays": 0}, "config['num_rays'] must be a positive number, got 0"),
        ({"wavelength": -1.0}, "config['wavelength'] must be a positive number, got -1.0"),
    ],
    ids=["num_rays", "wavelength"],
)
def test_optiland_invalid_scalars_fail_as_a_structured_result(
    tmp_path: Path, config: dict[str, Any], fragment: str
) -> None:
    from core.execution import RunStatus
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    result = get_adapter().run(
        ModelRunRequest(
            run_id="char", node_id="lens", config={**config, "output_directory": str(tmp_path)}
        )
    )
    assert result.status is RunStatus.FAILED
    assert result.error_type == "ValueError"
    assert fragment in (result.error_message or "")


@pytest.mark.optiland
@pytest.mark.parametrize(
    ("request_kwargs", "fragment"),
    [
        ({"config": {"backend": "bogus"}}, "is not supported; use 'numpy' or 'torch'"),
        (
            {"require_gradients": True, "config": {"backend": "torch"}},
            "needs at least one entry in design_parameters",
        ),
        (
            {
                "require_gradients": True,
                "design_parameters": {"nope": 1.0},
                "config": {"backend": "torch"},
            },
            "is not one of the parameter paths this adapter has validated",
        ),
        ({"config": {"sample": "NotASample"}}, "PRESCRIPTION_NAME_UNKNOWN"),
        ({"config": {"handoff_plane": "nope"}}, "('image_surface', 'exit_pupil')"),
    ],
    ids=["backend", "grad-no-params", "bad-design-param", "unknown-sample", "bad-handoff"],
)
def test_optiland_capability_refusals_raise_before_any_solver_call(
    tmp_path: Path, request_kwargs: dict[str, Any], fragment: str
) -> None:
    """These raise; they do not return a failed result.

    The distinction is the point. AGENTS.md requires an unsupported request to
    be refused *eagerly*, before any solver call, so that "the solver ran and
    failed" and "we declined to ask" stay different outcomes.
    """
    from core.errors import UnsupportedCapabilityError
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    config = {**request_kwargs.pop("config", {}), "output_directory": str(tmp_path)}
    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        get_adapter().run(
            ModelRunRequest(run_id="char", node_id="lens", config=config, **request_kwargs)
        )
    assert fragment in str(excinfo.value)


@pytest.mark.optiland
def test_optiland_validate_request_reports_without_running_anything(tmp_path: Path) -> None:
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    adapter = get_adapter()
    ok = adapter.validate_request(
        ModelRunRequest(run_id="c", node_id="l", config={"output_directory": str(tmp_path)})
    )
    assert [issue.code for issue in ok.issues] == ["OPTILAND_REQUEST_OK"]

    bad = adapter.validate_request(
        ModelRunRequest(
            run_id="c", node_id="l", config={"num_rays": 0, "output_directory": str(tmp_path)}
        )
    )
    assert [issue.code for issue in bad.issues] == ["OPTILAND_INVALID_NUM_RAYS"]


@pytest.mark.chromatix
def test_chromatix_missing_input_is_a_structured_failure(tmp_path: Path) -> None:
    from core.execution import RunStatus
    from solvers.base import ModelRunRequest
    from solvers.chromatix.adapter import get_adapter

    result = get_adapter().run(
        ModelRunRequest(
            run_id="char",
            node_id="wave",
            config={"propagation": "angular_spectrum", "output_dir": str(tmp_path)},
        )
    )
    assert result.status is RunStatus.FAILED
    assert result.error_type == "SolverExecutionError"
    assert "input_field" in (result.error_message or "")


@pytest.mark.chromatix
def test_chromatix_validate_request_accumulates_every_problem(tmp_path: Path) -> None:
    """Order matters and is pinned: it is what a reader is told to fix first.

    A validator that stopped at the first problem would send someone round the
    loop three times, so `validate_request` reports all of them -- and the
    sequence below is the one the adapter emits today.
    """
    from solvers.base import ModelRunRequest
    from solvers.chromatix.adapter import get_adapter

    report = get_adapter().validate_request(
        ModelRunRequest(
            run_id="c",
            node_id="w",
            config={"propagation": "nope", "output_dir": str(tmp_path)},
        )
    )
    assert [issue.code for issue in report.issues] == [
        "CHROMATIX_UNSUPPORTED_CAPABILITY",
        "CHROMATIX_MISSING_INPUT",
        "CHROMATIX_MISSING_CONFIG",
    ]


@pytest.mark.chromatix
def test_chromatix_refuses_a_gradient_request_before_running(tmp_path: Path) -> None:
    from core.errors import UnsupportedCapabilityError
    from solvers.base import ModelRunRequest
    from solvers.chromatix.adapter import get_adapter

    with pytest.raises(UnsupportedCapabilityError) as excinfo:
        get_adapter().run(
            ModelRunRequest(
                run_id="char",
                node_id="wave",
                require_gradients=True,
                config={"propagation": "angular_spectrum", "output_dir": str(tmp_path)},
            )
        )
    assert "does not claim a verified derivative" in str(excinfo.value)
