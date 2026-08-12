"""Tests for the M_CIRCUIT_SAX adapter (src/multiscale_optics_agent/adapters/sax_adapter.py).

All tests require the optional `circuit` extra (sax + jax), so they are
marked `jax` and `integration` under the repository test-marker policy and
pyproject.toml's marker declarations, and skip cleanly (via
`pytest.importorskip`) when sax/jax are not installed.
"""

from __future__ import annotations

import math

import pytest

sax = pytest.importorskip("sax")
jax = pytest.importorskip("jax")

from conftest import load_probe_expected  # noqa: E402

import multiscale_optics_agent.adapters.sax_adapter as sax_adapter  # noqa: E402
from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus  # noqa: E402
from multiscale_optics_agent.adapters.sax_adapter import SaxAdapter  # noqa: E402
from multiscale_optics_agent.core.errors import (  # noqa: E402
    AdapterDependencyError,
    UnsupportedCapabilityError,
)
from multiscale_optics_agent.core.specs import ArtifactKind  # noqa: E402

pytestmark = [pytest.mark.jax, pytest.mark.integration]


def _component_request(**config_overrides) -> ModelRunRequest:
    config = {
        "mode": "component",
        "port_naming_strategy": "optical",
        "wavelength_m": 1.55e-6,
        "component": {"model": "coupler_ideal", "params": {"coupling": 0.5}},
    }
    config.update(config_overrides)
    return ModelRunRequest(run_id="run-1", node_id="coupler", config=config)


def _mzi_request(**config_overrides) -> ModelRunRequest:
    config = {
        "mode": "mzi_circuit",
        "port_naming_strategy": "optical",
        "wavelength_m": 1.55e-6,
        "coupling1": 0.5,
        "coupling2": 0.5,
        "neff": 2.34,
        "ng": 3.4,
        "length_short_m": 10.0e-6,
        "length_long_m": 15.0e-6,
        "loss_dB_cm": 0.0,
    }
    config.update(config_overrides)
    return ModelRunRequest(run_id="run-2", node_id="mzi", config=config)


# ---------------------------------------------------------------------
# Registry contract
# ---------------------------------------------------------------------


def test_spec_matches_registry(registry) -> None:
    adapter = SaxAdapter()
    spec = adapter.spec
    assert spec.id == "M_CIRCUIT_SAX"
    assert spec == registry.models["M_CIRCUIT_SAX"]
    assert spec.derivative.verified is False


# ---------------------------------------------------------------------
# 1. Smoke test: single coupler_ideal component
# ---------------------------------------------------------------------


def test_component_smoke_single_coupler() -> None:
    adapter = SaxAdapter()
    request = _component_request()

    result = adapter.run(request)

    assert result.status is RunStatus.SUCCEEDED, result.error_message
    assert "response" in result.outputs
    artifact = result.outputs["response"]
    assert artifact.kind is ArtifactKind.CIRCUIT_RESPONSE
    assert artifact.dtype == "complex128"
    # coupler_ideal has 4 optical ports (o1..o4) -> declared dense shape (4, 4)
    assert artifact.shape == (4, 4)
    assert artifact.metadata["port_naming_strategy"] == "optical"
    assert set(artifact.metadata["port_names"]) == {"o1", "o2", "o3", "o4"}
    # thru/cross power should match the analytic 50/50 split from
    # knowledge/solvers/sax/expected/component_model_probe.json
    thru_re, thru_im = artifact.metadata["s_parameters"]["o1->o4"]
    cross_re, cross_im = artifact.metadata["s_parameters"]["o1->o3"]
    thru_power = thru_re**2 + thru_im**2
    cross_power = cross_re**2 + cross_im**2
    assert thru_power == pytest.approx(0.5, abs=1e-12)
    assert cross_power == pytest.approx(0.5, abs=1e-12)
    assert any("derivative boundary" in w for w in result.warnings)


def test_component_smoke_matches_recorded_probe_evidence() -> None:
    probe = load_probe_expected("sax", "component_model_probe")

    adapter = SaxAdapter()
    result = adapter.run(_component_request())
    artifact = result.outputs["response"]

    thru_re, thru_im = artifact.metadata["s_parameters"]["o1->o4"]
    cross_re, cross_im = artifact.metadata["s_parameters"]["o1->o3"]
    assert [thru_re, thru_im] == pytest.approx(probe["coupler_ideal"]["thru_o1_o4"], abs=1e-12)
    assert [cross_re, cross_im] == pytest.approx(probe["coupler_ideal"]["cross_o1_o3"], abs=1e-12)


# ---------------------------------------------------------------------
# 2. Independent-comparison test: assembled MZI vs. analytic oracle
# ---------------------------------------------------------------------


def test_mzi_circuit_matches_analytic_oracle_and_probe_evidence() -> None:
    probe = load_probe_expected("sax", "circuit_probe")

    adapter = SaxAdapter()
    request = _mzi_request()
    result = adapter.run(request)

    assert result.status is RunStatus.SUCCEEDED, result.error_message
    artifact = result.outputs["response"]
    assert artifact.shape == (4, 4)
    assert set(artifact.metadata["port_names"]) == {"in0", "in1", "out0", "out1"}

    t00_re, t00_im = artifact.metadata["s_parameters"]["in0->out0"]
    t01_re, t01_im = artifact.metadata["s_parameters"]["in0->out1"]
    power_00 = t00_re**2 + t00_im**2
    power_01 = t01_re**2 + t01_im**2

    # Regression-pin against the recorded probe evidence (the adapter must
    # reproduce the same numbers the knowledge pack captured directly from sax).
    assert power_00 == pytest.approx(probe["in0_to_out0"]["power"], rel=1e-9)
    assert power_01 == pytest.approx(probe["in0_to_out1"]["power"], rel=1e-9)

    # Independent analytic oracle: T = sin^2(pi * n_eff * dl / wavelength).
    neff = request.config["neff"]
    dl = request.config["length_long_m"] - request.config["length_short_m"]
    wavelength_m = request.config["wavelength_m"]
    dphi = 2 * math.pi * neff * dl / wavelength_m
    analytic_power_00 = math.sin(dphi / 2) ** 2
    relative_error = abs(power_00 - analytic_power_00) / analytic_power_00
    assert relative_error < 1e-9, relative_error
    # The recorded probe's own independently verified relative error was
    # 1.36e-15; assert the adapter is at least consistent with that order
    # of magnitude having been achievable.
    assert probe["relative_error_vs_analytic"] < 1e-9

    # Exact energy conservation and reciprocity, as captured by the probe.
    assert power_00 + power_01 == pytest.approx(1.0, abs=1e-9)
    out0_in0_re, out0_in0_im = artifact.metadata["s_parameters"]["out0->in0"]
    assert (out0_in0_re, out0_in0_im) == (t00_re, t00_im)
    assert probe["energy_conservation_from_in0"] == pytest.approx(1.0, abs=1e-9)
    assert probe["reciprocity_out0_in0_equals_in0_out0"] is True


# ---------------------------------------------------------------------
# 3. Gradient regression test (component-level closed-form path only)
# ---------------------------------------------------------------------


def test_gradient_regression_against_probe_evidence() -> None:
    """Regression-pin the one gradient path this repository has evidence for.

    Reuses knowledge/solvers/sax/expected/gradient_probe.json's
    AD-vs-finite-difference comparison for `coupler_ideal`'s `coupling`
    parameter, evaluated directly (NOT through sax.circuit). This is NOT
    the full repository gradient-verification bundle (that would
    need multiple step sizes, a convergence table, and an ill-conditioned
    case) -- it is a regression check that the installed sax/jax still
    reproduce the recorded probe evidence.
    """

    probe = load_probe_expected("sax", "gradient_probe")

    import jax.numpy as jnp
    import sax.models as sm

    sax.set_port_naming_strategy("optical")

    def thru_power(coupling: float) -> jnp.ndarray:
        s = sm.coupler_ideal(wl=1.55, coupling=coupling)
        return jnp.abs(s[("o1", "o4")]) ** 2

    c0 = probe["c0"]
    eps = probe["fd_step"]
    grad_ad = float(jax.grad(thru_power)(c0))
    grad_fd = float((thru_power(c0 + eps) - thru_power(c0 - eps)) / (2 * eps))
    relative_error = abs(grad_ad - grad_fd) / abs(grad_fd)

    assert grad_ad == pytest.approx(probe["grad_native_autodiff"], rel=1e-9)
    # Regression bound loosely anchored to the recorded probe evidence
    # (4.4e-13); this only demonstrates the installed sax/jax reproduce
    # that closed-form-path result, not a new independent verification.
    assert relative_error < 1e-6, (
        f"relative_error={relative_error} regressed past probe evidence ({probe['relative_error']})"
    )


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Gradient through the assembled sax.circuit matrix-solve path "
        "(mode='mzi_circuit') had never been probed in this repository before "
        "this test was written -- see gradient_probe.py's docstring and "
        "knowledge/solvers/sax/capability_notes.md 'not yet exercised' list. "
        "A single point-check performed while writing this adapter did agree "
        "with finite differences away from a degenerate critical point (at "
        "coupling1=coupling2=0.5 the objective is at a local extremum and the "
        "relative-error denominator vanishes, which is why c0=0.3 is used "
        "here instead). strict=False: an XPASS here is real, useful evidence "
        "for a human to review, but it is only one parameter/point, not the "
        "full repository gradient-verification bundle (multiple step sizes, a "
        "convergence table, an ill-conditioned case) -- it must not be used "
        "to set derivative.verified=true on its own, and a failure (xfail) "
        "must not be papered over either."
    ),
)
def test_gradient_through_assembled_circuit_not_yet_verified() -> None:
    import jax.numpy as jnp

    sax.set_port_naming_strategy("optical")
    models = {
        "coupler_ideal": sax.models.coupler_ideal,
        "straight": sax.models.straight,
    }
    mzi, _info = sax.circuit(sax_adapter._MZI_NETLIST, models)

    def power_00(coupling1: float) -> jnp.ndarray:
        result = mzi(
            wl=1.55,
            c1={"coupling": coupling1},
            c2={"coupling": 0.5},
            wg_short={"length": 10.0, "neff": 2.34, "wl0": 1.55, "ng": 3.4, "loss_dB_cm": 0.0},
            wg_long={"length": 15.0, "neff": 2.34, "wl0": 1.55, "ng": 3.4, "loss_dB_cm": 0.0},
        )
        return jnp.abs(result[("in0", "out0")]) ** 2

    c0 = 0.3  # away from the coupling1=coupling2=0.5 critical point (see reason above)
    eps = 1e-4
    grad_ad = float(jax.grad(power_00)(c0))
    grad_fd = float((power_00(c0 + eps) - power_00(c0 - eps)) / (2 * eps))
    relative_error = abs(grad_ad - grad_fd) / max(abs(grad_fd), 1e-12)

    assert relative_error < 1e-6, relative_error


# ---------------------------------------------------------------------
# 4. Failure / resource tests
# ---------------------------------------------------------------------


def test_missing_dependency_raises_adapter_dependency_error(monkeypatch) -> None:
    def _raise_dependency_error():
        raise AdapterDependencyError("sax/jax not importable in this environment (forced for test)")

    monkeypatch.setattr(sax_adapter, "_import_sax", _raise_dependency_error)

    adapter = SaxAdapter()
    request = _component_request()

    with pytest.raises(AdapterDependencyError):
        adapter.run(request)


def test_invalid_port_naming_strategy_returns_failed_result() -> None:
    adapter = SaxAdapter()
    request = _mzi_request(port_naming_strategy="not_a_real_strategy")

    result = adapter.run(request)

    assert result.status is RunStatus.FAILED
    assert result.error_type == "ValueError"
    assert "port naming strategy" in result.error_message.lower()
    assert result.outputs == {}


def test_unsupported_component_raises_unsupported_capability_error() -> None:
    adapter = SaxAdapter()
    request = _component_request(component={"model": "mmi1x2", "params": {}})

    with pytest.raises(UnsupportedCapabilityError):
        adapter.run(request)


def test_require_gradients_raises_unsupported_capability_error() -> None:
    adapter = SaxAdapter()
    request = ModelRunRequest(
        run_id="run-3",
        node_id="mzi",
        config=_mzi_request().config,
        require_gradients=True,
    )

    with pytest.raises(UnsupportedCapabilityError):
        adapter.run(request)

    report = adapter.validate_request(request)
    assert not report.valid
    assert any(issue.code == "SAX_GRADIENT_NOT_IMPLEMENTED" for issue in report.errors)


def test_validate_request_flags_missing_mzi_params() -> None:
    adapter = SaxAdapter()
    config = _mzi_request().config
    del config["neff"]
    request = ModelRunRequest(run_id="run-4", node_id="mzi", config=config)

    report = adapter.validate_request(request)

    assert not report.valid
    assert any(issue.code == "SAX_MISSING_MZI_PARAMS" for issue in report.errors)


def test_estimate_returns_cost_estimate_for_each_mode() -> None:
    adapter = SaxAdapter()
    component_estimate = adapter.estimate(_component_request())
    mzi_estimate = adapter.estimate(_mzi_request())

    assert component_estimate.solver_calls == 1
    assert mzi_estimate.solver_calls == 1
    assert component_estimate.wall_time_s is not None
    assert mzi_estimate.wall_time_s is not None
