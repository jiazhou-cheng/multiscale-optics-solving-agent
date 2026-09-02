"""One entry point, an explicit capability gate, and determinism under repetition.

CHE-181 (R05.3), acceptance criteria:

1. `trace(setup, source, sampling=..., execution=...)` returns a neutral `RayBundle`,
   and it is the only entry point -- **superseded by CHE-217 (R05.6)**, which
   added `trace_rays` for a supplied bundle. What survives of the criterion is the
   exact list: the public surface is these two functions plus the execution
   configuration, and a third name has to be justified. `trace` itself is
   numerically untouched by that addition;
2. backend / device / precision configuration is explicit and idempotent, and two
   consecutive calls with the same request produce bit-identical results;
3. a requested precision or device the measured capability table does not support
   is refused **before** any solver call, not discovered inside one;
4. no gradient claim is made across the framework boundary -- the descriptor says
   `forward_only`;
5. GPU verification on one device -- `test_device_gpu` below, `-m gpu`;
6. class delta 0 -- the two names on disk are `TypedDict`s, and
   `test_class_budget.py` holds the count.

On criterion 3, and what "before any solver call" is worth
----------------------------------------------------------
Two things: the refusal names the capability row and its probe evidence rather
than surfacing whatever the solver happened to raise, and -- the part that matters
on a shared GPU host -- an inadmissible request costs no import, no CUDA context
and no allocation.

On criterion 4, and where the descriptor is
-------------------------------------------
`operations/` may not import `solvers/` and `solvers/` may not import
`operations/`, so the descriptor for this trace cannot live in either package
today and no package that could hold a registration site has landed. What is
checked here is the whole of the claim that can be executed: the record is
constructible with `derivative="forward_only"`, its `implementation` string
resolves to this function through `operations.resolve`, and it cites a capability
row that exists. Where the registration site lives is R12/R13's, and is reported
as follow-up on CHE-181.
"""

from __future__ import annotations

import math
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fixtures.systems import (
    REVERSE_TELEPHOTO,
    reverse_telephoto_source,
    singlet_ref,
    singlet_source,
)

from numerics import OPTILAND_CAPABILITIES, ArrayNamespace, DevicePlacement, Precision
from operations import OperationDescriptor, OperationKind, registry, resolve
from representations import RayBundle
from solvers import optiland
from solvers.optiland import CAPABILITIES, DERIVATIVE, configure_execution, trace

ROOT = Path(__file__).resolve().parents[2]
CPU64 = {"device": "cpu", "precision": "fp64"}
ON_AXIS = {"num_rings": 8, "reference_surface": "exit_pupil"}

#: CHE-218 (R05.7): the wavelength and the field angle are the *source's*, not the
#: sampling's. `singlet_source()` and `reverse_telephoto_source()` both default to
#: on axis at 550 nm, which is what every assertion in this file was written at.
LIGHT = singlet_source()

#: Names the reference implementation used for this job, none of which landed.
AVOIDED_NAMES = (
    "OptilandAdapter",
    "get_adapter",
    "OptilandExecutionState",
    "TracePlans",
    "OptilandRayRequest",
    "OptilandRayResult",
    "OptilandRayFailure",
    "PatchEmitterCostModel",
    "HandoffPlaneError",
    "run_standalone",
)


# ---------------------------------------------------------------------------
# 1. One entry point
# ---------------------------------------------------------------------------


def test_the_public_entry_points_are_the_two_kinds_of_input() -> None:
    """`trace` and `trace_rays` are the API; `configure_execution` is the state they own.

    Was `test_one_public_entry_point` through CHE-181. CHE-217 (R05.6) added
    `trace_rays`, and the two are not a facade over one implementation: they differ
    in what the rays *are*. `trace` generates them inside the solver from a field
    coordinate and a ring count; `trace_rays` consumes a `RayBundle` the project
    already holds, with its own amplitude and its own quadrature. This assertion
    stays an exact list for the reason it always was -- a third name has to be
    justified rather than accumulated.
    """
    callables = sorted(
        name
        for name in optiland.__all__
        if callable(getattr(optiland, name)) and not isinstance(getattr(optiland, name), type)
    )
    assert callables == ["configure_execution", "trace", "trace_rays"]


def test_no_adapter_facade_anywhere_in_the_package() -> None:
    package = ROOT / "src" / "solvers" / "optiland"
    for module in sorted(package.rglob("*.py")):
        source = module.read_text(encoding="utf-8")
        for name in AVOIDED_NAMES:
            assert f"class {name}" not in source, f"{module.name} defines {name}"
            assert f"def {name}" not in source, f"{module.name} defines {name}"


def test_trace_returns_a_neutral_bundle() -> None:
    bundle = trace(singlet_ref(), LIGHT, sampling=ON_AXIS, execution=CPU64)
    assert isinstance(bundle, RayBundle)
    assert bundle.count == 1 + 3 * 8 * 9
    assert bundle.wavelength_m == pytest.approx(0.55e-6, abs=0.0)


# ---------------------------------------------------------------------------
# 2. Explicit, idempotent, deterministic
# ---------------------------------------------------------------------------


def test_two_consecutive_calls_are_bit_identical() -> None:
    """Criterion 2. Not `approx`: the same request must give the same bits.

    Process-global solver state was a *measured* source of nondeterminism, and the
    fix -- setting backend, precision and device on every call rather than
    inheriting them -- is part of the behaviour being reproduced, not an
    incidental detail.
    """
    first = trace(REVERSE_TELEPHOTO, reverse_telephoto_source(), sampling=ON_AXIS, execution=CPU64)
    second = trace(REVERSE_TELEPHOTO, reverse_telephoto_source(), sampling=ON_AXIS, execution=CPU64)
    for name in ("positions_m", "directions", "amplitude", "optical_path_m", "measure_weight"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    assert first.optical_path_reference == second.optical_path_reference


def test_a_foreign_backend_selection_does_not_leak_in() -> None:
    """The state is set on every call, so what another component left cannot win.

    This is the CHE-102 failure in miniature: something else in the process
    selects the torch backend, and a trace that every artifact describes as NumPy
    executes in torch, converting every array on entry.
    """
    baseline = trace(singlet_ref(), LIGHT, sampling=ON_AXIS, execution=CPU64)

    import optiland.backend as be

    be.set_backend("torch")
    be.set_precision("float32")
    after = trace(singlet_ref(), LIGHT, sampling=ON_AXIS, execution=CPU64)
    try:
        np.testing.assert_array_equal(baseline.positions_m, after.positions_m)
        assert str(be.get_backend()) == "numpy"
    finally:
        be.set_backend("numpy")
        be.set_precision("float64")


def test_configure_execution_reports_observed_state_not_the_request() -> None:
    report = configure_execution(
        device=DevicePlacement.parse("cpu"),
        precision=Precision.FP64,
        namespace=ArrayNamespace.NUMPY,
    )
    assert report["requested"] == {
        "backend": "numpy",
        "device": "cpu",
        "precision": "float64",
    }
    assert report["observed"]["backend"] == "numpy"
    assert report["observed"]["precision"] == "float64"
    assert report["capabilities"] == CAPABILITIES
    assert report["derivative"] == DERIVATIVE == "forward_only"
    # Idempotent: the second call leaves the same state and says the same thing.
    assert (
        configure_execution(
            device=DevicePlacement.parse("cpu"),
            precision=Precision.FP64,
            namespace=ArrayNamespace.NUMPY,
        )
        == report
    )


def test_float32_traces_in_float32_and_says_so() -> None:
    """The requested precision is executed and *observed*, not echoed.

    The dtype the solver produced is preserved out to the bundle. Forcing float64
    at the export was the single line that made a float32 trace indistinguishable
    from a float64 one downstream.

    FP32 selects the torch backend even on the host, and that is measured rather
    than preferred -- see `_resolve_namespace`. The numpy backend honours
    `set_precision('float32')` for its own array constructor while `Optic.trace`
    still returns float64, so a numpy FP32 trace would carry float32-scale error
    inside float64 buffers, which is worse than either honest precision because
    nothing downstream can see it.
    """
    bundle = trace(
        singlet_ref(), LIGHT, sampling=ON_AXIS, execution={"device": "cpu", "precision": "fp32"}
    )
    assert str(bundle.state.dtype) == "float32"
    assert bundle.amplitude.dtype == np.complex64, (
        "sqrt of a float32 weight is a complex64 amplitude, not a complex128 one: "
        "widening would fabricate precision the producer never had"
    )
    # Restore the process default for anything that follows in this worker.
    configure_execution(
        device=DevicePlacement.parse("cpu"),
        precision=Precision.FP64,
        namespace=ArrayNamespace.NUMPY,
    )


# ---------------------------------------------------------------------------
# 3. The capability gate, before the solver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("execution", "code"),
    [
        ({"device": "cpu", "precision": "fp16"}, "UNSUPPORTED_DTYPE"),
        ({"device": "tpu", "precision": "fp64"}, "UNSUPPORTED_DEVICE_SPELLING"),
        ({"device": "cpu:0", "precision": "fp64"}, "UNSUPPORTED_DEVICE_SPELLING"),
        ({"device": "cpu", "precision": "float8"}, "UNKNOWN_PRECISION"),
    ],
)
def test_an_inadmissible_request_is_refused_with_a_code(
    execution: dict[str, str], code: str
) -> None:
    with pytest.raises(ValueError) as excinfo:
        trace(singlet_ref(), LIGHT, sampling=ON_AXIS, execution=execution)
    assert getattr(excinfo.value, "code", None) == code


def test_fp16_refusal_cites_the_measured_table() -> None:
    """A refusal that names the probe is one a reader can check or widen."""
    with pytest.raises(ValueError) as excinfo:
        trace(
            singlet_ref(),
            LIGHT,
            sampling=ON_AXIS,
            execution={"device": "cpu", "precision": "fp16"},
        )
    message = str(excinfo.value)
    assert CAPABILITIES in message
    assert "fp32" in message and "fp64" in message
    assert "set_precision" in message, "the evidence sentence from the capability row"
    assert OPTILAND_CAPABILITIES.precisions == frozenset({Precision.FP32, Precision.FP64})


def test_the_refusal_happens_before_the_solver_is_imported() -> None:
    """Criterion 3, and the reason it is worth stating: it costs no import.

    A fresh interpreter, an inadmissible request, and `sys.modules` afterwards.
    In-process this would be meaningless -- the session has already loaded
    optiland for every other test in this file.
    """
    probe = (
        "import sys, json\n"
        "sys.path.insert(0, 'tests')\n"
        "from fixtures.systems import singlet_ref, singlet_source\n"
        "from solvers.optiland import trace\n"
        "try:\n"
        "    trace(singlet_ref(), singlet_source(),\n"
        "          sampling={'num_rings': 8, 'reference_surface': 'exit_pupil'},\n"
        "          execution={'device': 'cpu', 'precision': 'fp16'})\n"
        "except ValueError as exc:\n"
        "    code = getattr(exc, 'code', None)\n"
        "else:\n"
        "    code = 'NOT REFUSED'\n"
        "print(json.dumps([code, sorted(\n"
        "    n for n in sys.modules if n.split('.')[0] in ('optiland', 'torch'))]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip().endswith('["UNSUPPORTED_DTYPE", []]'), result.stdout


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("sampling", {"num_rings": 8}),
        ("sampling", {"num_rings": 8, "reference_surfce": "exit_pupil"}),
        ("sampling", {"num_rings": 8, "reference_surface": "exit_pupil", "num_rays": 8}),
        ("execution", {"device": "cpu"}),
        ("execution", {"device": "cpu", "precision": "fp64", "backend": "torch"}),
    ],
)
def test_a_misspelled_or_incomplete_argument_is_refused(
    argument: str, value: dict[str, object]
) -> None:
    """An unrecognized key is discarded silently by the pinned solver, so it is
    refused here -- the same hazard `problems._check_material` closes one layer up."""
    kwargs: dict[str, object] = {"sampling": dict(ON_AXIS), "execution": dict(CPU64)}
    kwargs[argument] = value
    with pytest.raises(ValueError, match=f"{argument}= "):
        trace(singlet_ref(), LIGHT, **kwargs)  # type: ignore[arg-type]


def test_a_field_no_record_enumerated_is_traceable() -> None:
    """CHE-218 (R05.7) acceptance criterion 1, and it replaces two refusals.

    Both of these used to raise. `singlet_ref()` declared only the axis, so 6 deg
    was "not expressible on this problem"; `REVERSE_TELEPHOTO` declared at most
    30 deg, so 45 deg was "outside the largest field this problem declares". Both
    refusals were correct given the old schema and both were artifacts of it: the
    field set existed only to give the backend a `max_field`, so asking for a new
    field angle meant *editing the optical system*.

    The setups here are the same two objects, unedited, and neither carries a field
    at all. This is the capability the split bought, so it is asserted as a
    capability rather than left as the absence of a test.
    """
    for setup, source in (
        (singlet_ref(), singlet_source(field_angle_deg=(0.0, 6.0))),
        (REVERSE_TELEPHOTO, reverse_telephoto_source(field_angle_deg=(0.0, 45.0))),
    ):
        bundle = trace(setup, source, sampling=ON_AXIS, execution=CPU64)
        assert isinstance(bundle, RayBundle)
        assert bundle.count == 1 + 3 * 8 * 9


def test_the_same_setup_traces_at_two_fields_without_being_reconstructed() -> None:
    """Acceptance criterion 3: one setup object, two field angles, two traces.

    The setup is constructed **once** and reused, so nothing about it can have
    been edited between the calls. The two bundles have to differ -- otherwise the
    field angle reached nothing -- and the axial one has to stay centred on the
    axis, which is what says the off-axis call did not perturb the on-axis one.
    """
    setup = singlet_ref()
    on_axis = trace(setup, singlet_source(), sampling=ON_AXIS, execution=CPU64)
    off_axis = trace(
        setup,
        singlet_source(field_angle_deg=(0.0, 3.0)),
        sampling=ON_AXIS,
        execution=CPU64,
    )
    assert on_axis.count == off_axis.count
    axial_y = np.asarray(on_axis.positions_m)[:, 1]
    tilted_y = np.asarray(off_axis.positions_m)[:, 1]
    assert abs(float(axial_y.mean())) < 1.0e-18, "the on-axis fan is centred on the axis"
    assert float(tilted_y.mean()) != 0.0, "the off-axis field has to have moved the fan"
    # ...and re-tracing on axis reproduces the first call bitwise, so the setup
    # carries no state either call could have left behind.
    again = trace(setup, singlet_source(), sampling=ON_AXIS, execution=CPU64)
    np.testing.assert_array_equal(
        np.asarray(again.positions_m), np.asarray(on_axis.positions_m)
    )


def test_field_degrees_convert_to_the_solvers_normalized_coordinate() -> None:
    """The declared field IS the maximum field, so the coordinate is the unit one.

    Before R05.7 this read `normalized_field(lens, (0.0, 6.0)) == (0.0, 0.2)`,
    because `max_field` was 30 -- the largest of a list the caller had to declare.
    `build_lens` now declares exactly the field being traced, so the same 6 deg
    normalizes against 6 and the round trip `max_field * H` recovers it exactly.
    `max_field` is still read off the constructed lens rather than assumed.

    CHE-219 (R05.8) moved the function from `solver` to `launch`, unchanged: it is
    the first step of aiming a declarative source into a particular system, which
    is what that module owns.
    """
    from solvers.optiland.launch import normalized_field as _normalized_field
    from solvers.optiland.system import build_lens

    for field_deg, expected in (
        ((0.0, 6.0), (0.0, 1.0)),
        ((0.0, 30.0), (0.0, 1.0)),
        ((0.0, 0.0), (0.0, 0.0)),
    ):
        lens = build_lens(
            REVERSE_TELEPHOTO, reverse_telephoto_source(field_angle_deg=field_deg)
        )
        assert _normalized_field(lens, field_deg) == expected
        # The round trip the backend performs, exactly: `field = max_field * H`.
        max_field = float(lens.fields.max_field)
        normalized = _normalized_field(lens, field_deg)
        assert (max_field * normalized[0], max_field * normalized[1]) == field_deg

    # A two-component field, where `max_field` is `hypot(x, y)` rather than either
    # component. The round trip still has to be exact, because
    # `test_optiland_finite_conjugate.py` asserts the launch position at `abs=0.0`.
    lens = build_lens(REVERSE_TELEPHOTO, reverse_telephoto_source(field_angle_deg=(2.0, 3.0)))
    max_field = float(lens.fields.max_field)
    assert max_field == math.hypot(2.0, 3.0)
    normalized = _normalized_field(lens, (2.0, 3.0))
    assert (max_field * normalized[0], max_field * normalized[1]) == (2.0, 3.0)


# ---------------------------------------------------------------------------
# 4. No gradient claim
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_registry() -> Iterator[None]:
    """The registry is module-level state, so the isolation belongs in the test."""
    saved = dict(registry._REGISTERED)
    registry._REGISTERED.clear()
    yield
    registry._REGISTERED.clear()
    registry._REGISTERED.update(saved)


def test_the_descriptor_says_forward_only(isolated_registry: None) -> None:
    """Criterion 4, executed end to end: the record resolves to this function.

    A descriptor with `derivative='differentiable'` and no evidence is refused at
    construction (R03.1), so `forward_only` here is the project's rule holding
    rather than a string someone typed.
    """
    descriptor = registry.register(
        OperationDescriptor(
            operation_id="S_RAY_OPTILAND",
            kind=OperationKind.SOLVER,
            input="ray_bundle",
            output="ray_bundle",
            implementation="solvers.optiland.solver:trace",
            approximation=(
                "sequential geometric ray tracing: rays are plane wavelets, diffraction "
                "is not modelled, and a surface interaction is refraction at a real "
                "interface"
            ),
            evidence=("tests/physics/test_optiland_rays.py",),
            capabilities=CAPABILITIES,
            derivative=DERIVATIVE,
        )
    )
    assert descriptor.derivative == "forward_only"
    assert descriptor.derivative_evidence is None
    assert resolve("S_RAY_OPTILAND") is trace


def test_there_is_no_gradient_knob() -> None:
    """No argument asks for one, so a caller cannot inherit a live graph.

    The reference implementation exposed exactly one characterized
    design-parameter path -- a surface radius under the torch backend -- and
    re-exposing it would be a gradient claim across a framework boundary this
    rewrite has not validated.
    """
    import inspect

    signature = inspect.signature(trace)
    # `aiming` joined the four settled arguments at CHE-219 (R05.8). It is a
    # declaration of how the backend resolves a launch into this system, which
    # before that ticket was an unstated inheritance of the solver's constructor
    # default -- and its own default is that same value, measured bit-identical.
    assert set(signature.parameters) == {
        "setup",
        "source",
        "sampling",
        "execution",
        "aiming",
    }
    package = ROOT / "src" / "solvers" / "optiland"
    for module in sorted(package.rglob("*.py")):
        source = module.read_text(encoding="utf-8")
        assert "requires_grad" not in source
        assert "grad_mode.enable" not in source
    assert "grad_mode.disable" in (package / "solver.py").read_text(encoding="utf-8"), (
        "autograd is switched off explicitly on the torch path rather than assumed off"
    )


# ---------------------------------------------------------------------------
# 5. GPU
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_trace_on_cuda_matches_the_host_within_float32() -> None:
    """Criterion 5. One device, `MOA_GPUS=device=6 make test-gpu`.

    CUDA is reachable only through the torch backend -- `set_device` raises on the
    numpy backend -- so this exercises the one path where the array namespace
    changes. The comparison is against the *float32 host* trace at the same
    precision, not against float64: a device change must not be allowed to hide
    behind a precision change.
    """
    fp32 = {"precision": "fp32"}
    host = trace(singlet_ref(), LIGHT, sampling=ON_AXIS, execution={"device": "cpu", **fp32})
    device = trace(singlet_ref(), LIGHT, sampling=ON_AXIS, execution={"device": "cuda", **fp32})
    assert device.count == host.count
    assert str(device.state.dtype) == "float32"
    np.testing.assert_allclose(
        np.asarray(device.positions_m), np.asarray(host.positions_m), rtol=0.0, atol=1e-9
    )
    np.testing.assert_allclose(
        np.asarray(device.optical_path_m), np.asarray(host.optical_path_m), rtol=0.0, atol=1e-9
    )
    configure_execution(
        device=DevicePlacement.parse("cpu"),
        precision=Precision.FP64,
        namespace=ArrayNamespace.NUMPY,
    )


def test_cuda_is_refused_on_a_container_with_no_device() -> None:
    """No silent host fallback: a CUDA request that cannot be served is refused.

    Skipped rather than inverted when a device *is* attached, because the claim
    here is about the refusal and not about the device.
    """
    import torch

    if torch.version.cuda is not None and torch.cuda.is_available():
        pytest.skip("a CUDA device is attached; the refusal path is not reachable here")
    with pytest.raises(ValueError) as excinfo:
        trace(
            singlet_ref(),
            LIGHT,
            sampling=ON_AXIS,
            execution={"device": "cuda", "precision": "fp32"},
        )
    assert getattr(excinfo.value, "code", None) == "DEVICE_NOT_AVAILABLE"
    message = str(excinfo.value)
    assert "CPU-only build" in message or "no CUDA device is attached" in message
    assert "--gpu" in message, "the refusal names the way to actually get a device"
