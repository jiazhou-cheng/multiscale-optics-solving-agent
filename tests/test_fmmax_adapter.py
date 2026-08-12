"""Tests for the ``M_RCWA_FMMAX`` (FMMAX RCWA) adapter.

Grounded in ``knowledge/solvers/fmmax/`` (solver_card.yaml, conventions.md,
capability_notes.md, api_minimal_examples.md, failure_guide.md) and the
recorded probe outputs under ``knowledge/solvers/fmmax/expected/``. All new
tests here require the real ``fmmax``/``jax`` packages (pinned
``fmmax==1.7.1`` in the ``agent_solver`` Docker image) and are marked
``@pytest.mark.jax`` and ``@pytest.mark.integration`` accordingly; run with:

    docker run --rm -v "$(pwd)":/workspace -w /workspace agent_solver \\
        python -m pytest -q tests/test_fmmax_adapter.py
"""

from __future__ import annotations

import builtins

import pytest
from conftest import load_probe_expected

import multiscale_optics_agent.adapters.fmmax_adapter as fmmax_adapter
from multiscale_optics_agent.adapters.base import ModelRunRequest, RunStatus
from multiscale_optics_agent.core.artifacts import ArtifactRecord
from multiscale_optics_agent.core.errors import AdapterDependencyError, UnsupportedCapabilityError
from multiscale_optics_agent.core.specs import ArtifactKind

MODEL_ID = fmmax_adapter.MODEL_ID


def _bare_interface_structure(
    *,
    n_ambient: float = 1.0,
    n_substrate: float = 1.5,
    wavelength_m: float = 0.55e-6,
) -> ArtifactRecord:
    """A homogeneous ambient/substrate interface -- the Fresnel-oracle case.

    Matches knowledge/solvers/fmmax/probes/fresnel_oracle_probe.py exactly
    (same n_ambient/n_substrate/wavelength, approximate_num_terms=1 is set
    via request.config by the caller), except lengths are expressed in
    meters per this project's SI convention (repository scientific conventions) rather
    than fmmax's bare, scale-agnostic numbers.
    """

    return ArtifactRecord(
        id="structure-bare-interface",
        kind=ArtifactKind.PERIODIC_STRUCTURE,
        uri="memory://bare-interface",
        metadata={
            "period": {"u_m": [1.0e-6, 0.0], "v_m": [0.0, 1.0e-6]},
            "wavelength_m": wavelength_m,
            "incidence": {"polar_angle_rad": 0.0, "azimuthal_angle_rad": 0.0},
            "polarization": "te",
            "material_convention": "permittivity",
            "layers": [
                {"permittivity": n_ambient**2},
                {"permittivity": n_substrate**2},
            ],
        },
    )


def _lamellar_grating_structure() -> ArtifactRecord:
    """A tiny binary lamellar grating: ambient / patterned layer / substrate.

    Numerically identical in construction to the ad hoc probe run while
    building this adapter (not previously in the knowledge pack): a 1 um
    period, half n=1.0 / half n=2.0 patterned layer on a 32x32 grid,
    0.3 um thick, sandwiched between n=1.0 ambient and n=1.5 substrate.
    """

    grid_size = 32
    row = [4.0] * (grid_size // 2) + [1.0] * (grid_size // 2)  # eps = n^2, n=2.0 / n=1.0
    grating_permittivity = [row for _ in range(grid_size)]
    return ArtifactRecord(
        id="structure-lamellar-grating",
        kind=ArtifactKind.PERIODIC_STRUCTURE,
        uri="memory://lamellar-grating",
        metadata={
            "period": {"u_m": [1.0e-6, 0.0], "v_m": [0.0, 1.0e-6]},
            "wavelength_m": 0.55e-6,
            "incidence": {"polar_angle_rad": 0.0, "azimuthal_angle_rad": 0.0},
            "polarization": "te",
            "material_convention": "permittivity",
            "layers": [
                {"permittivity": 1.0},
                {"permittivity": grating_permittivity, "thickness_m": 0.3e-6},
                {"permittivity": 1.5**2},
            ],
        },
    )


def _request(structure: ArtifactRecord, *, config: dict | None = None, **kwargs) -> ModelRunRequest:
    return ModelRunRequest(
        run_id="test-run",
        node_id="fmmax-node",
        inputs={"structure": structure},
        config=config or {},
        **kwargs,
    )


# --------------------------------------------------------------------------------------
# 1. Registry / spec wiring
# --------------------------------------------------------------------------------------


@pytest.mark.jax
@pytest.mark.integration
def test_spec_matches_registry(registry) -> None:
    adapter = fmmax_adapter.get_adapter()
    assert adapter.spec.id == MODEL_ID
    assert adapter.spec.id == registry.models[MODEL_ID].id
    assert adapter.spec.derivative.verified is False  # must never be flipped by this adapter


# --------------------------------------------------------------------------------------
# 2. Smoke test: tiny unit cell, few Fourier orders
# --------------------------------------------------------------------------------------


@pytest.mark.jax
@pytest.mark.integration
def test_smoke_grating_run_succeeds(registry) -> None:
    adapter = fmmax_adapter.get_adapter()
    request = _request(_lamellar_grating_structure(), config={"approximate_num_terms": 9})

    result = adapter.run(request)

    assert result.status == RunStatus.SUCCEEDED, result.error_message
    assert "diffraction" in result.outputs
    record = result.outputs["diffraction"]
    assert record.kind == ArtifactKind.DIFFRACTION_CHANNELS

    declared_metadata = registry.models[MODEL_ID].outputs[0].provides_metadata
    for key in declared_metadata:
        assert key in record.metadata, f"missing declared output metadata key {key!r}"

    # A real grating (9 Fourier orders, 18 modes with TE/TM) still closes energy conservation.
    assert result.diagnostics["n_modes"] > 2
    assert result.diagnostics["approximate_num_terms_actual"] == 9
    energy_residual = result.diagnostics["energy_residual"]
    assert energy_residual < 1e-4, (
        f"R+T did not close for the grating case: residual={energy_residual}"
    )
    assert 0.0 <= record.metadata["reflectance_total"] <= 1.0
    assert 0.0 <= record.metadata["transmittance_total"] <= 1.0


# --------------------------------------------------------------------------------------
# 3. Independent comparison: analytic Fresnel oracle (magnitude only) + Poynting-flux
#    energy conservation (R + T ~= 1). Phase/sign convention is explicitly NOT validated
#    here -- see knowledge/solvers/fmmax/conventions.md and the module docstring.
# --------------------------------------------------------------------------------------


@pytest.mark.jax
@pytest.mark.integration
def test_bare_interface_matches_fresnel_oracle_and_closes_energy() -> None:
    expected = load_probe_expected("fmmax", "fresnel_oracle_probe")
    adapter = fmmax_adapter.get_adapter()

    structure = _bare_interface_structure(
        n_ambient=expected["n_ambient"],
        n_substrate=expected["n_substrate"],
        wavelength_m=expected["wavelength"] * 1e-6,  # probe's raw number treated as microns here
    )
    request = _request(structure, config={"approximate_num_terms": 1})

    result = adapter.run(request)
    assert result.status == RunStatus.SUCCEEDED, result.error_message
    record = result.outputs["diffraction"]

    # Magnitude-only independent verification against the analytic Fresnel formula
    # Under the repository scientific contract, reflectance |r|^2 matched to ~1e-7 in the
    # recorded probe. We recompute R via a different formula (Poynting flux, not |s21|^2) and still
    # expect close agreement, since both are physically the reflectance of the same
    # lossless bare interface.
    reflectance = record.metadata["reflectance_total"]
    relative_error = abs(reflectance - expected["fmmax_R_te"]) / abs(expected["fmmax_R_te"])
    assert relative_error < 1e-5, (
        f"Poynting-flux reflectance {reflectance} disagrees with the recorded amplitude^2 "
        f"oracle value {expected['fmmax_R_te']} (relative error {relative_error})."
    )
    also_vs_analytic = abs(reflectance - expected["analytic_fresnel_R"]) / abs(
        expected["analytic_fresnel_R"]
    )
    assert also_vs_analytic < 1e-5

    # New physics work for this adapter: R + T ~= 1 via directional_poynting_flux, which the
    # knowledge pack explicitly flagged as NOT YET DONE (naive |s11|^2 attempt gave >1 and did
    # not close; see failure_guide.md). This closes to ~2e-7 in practice.
    reflectance_plus_transmittance = reflectance + record.metadata["transmittance_total"]
    assert abs(reflectance_plus_transmittance - 1.0) < 1e-5, (
        "R + T did not close to 1 via the Poynting-flux path: "
        f"R={reflectance}, T={record.metadata['transmittance_total']}"
    )

    # Explicitly NOT validated: the sign/phase of the complex amplitude. We only ever expose
    # (and only ever verify) the magnitude/power quantities R and T in this adapter.
    assert "reflectance_total" in record.metadata
    assert "transmittance_total" in record.metadata


# --------------------------------------------------------------------------------------
# 4. Gradient regression test against the recorded directional-derivative probe.
# --------------------------------------------------------------------------------------


@pytest.mark.jax
@pytest.mark.integration
def test_gradient_through_run_matches_recorded_probe() -> None:
    """jax.grad wrapped around FmmaxAdapter.run, regression-tested against gradient_probe.json.

    This is NOT a repository five-part gradient-verification bundle (single step size, single
    parameter, non-periodic homogeneous-limit structure) -- it is a regression check that this
    adapter's require_gradients=True output-boundary policy (module docstring) actually preserves
    a traceable JAX value all the way through ModelRunResult/ArtifactRecord construction, and that
    the resulting gradient matches the value already independently checked against a centered
    finite difference in knowledge/solvers/fmmax/probes/gradient_probe.py.
    """

    import jax
    import jax.numpy as jnp

    expected = load_probe_expected("fmmax", "gradient_probe")
    adapter = fmmax_adapter.get_adapter()

    def reflectance_of(n_substrate_index: jnp.ndarray) -> jnp.ndarray:
        structure = ArtifactRecord(
            id="structure-bare-interface-grad",
            kind=ArtifactKind.PERIODIC_STRUCTURE,
            uri="memory://bare-interface-grad",
            metadata={
                "period": {"u_m": [1.0e-6, 0.0], "v_m": [0.0, 1.0e-6]},
                "wavelength_m": 0.55e-6,
                "incidence": {"polar_angle_rad": 0.0, "azimuthal_angle_rad": 0.0},
                "polarization": "te",
                "material_convention": "permittivity",
                "layers": [{"permittivity": 1.0}, {"permittivity": 1.0}],
            },
        )
        request = _request(
            structure,
            config={"approximate_num_terms": 1},
            design_parameters={"layers[1].permittivity": n_substrate_index**2},
            require_gradients=True,
        )
        result = adapter.run(request)
        assert result.status == RunStatus.SUCCEEDED, result.error_message
        return result.outputs["diffraction"].metadata["reflectance_total"]

    n0 = jnp.asarray(expected["n_substrate_0"])
    value = float(reflectance_of(n0))
    grad_ad = float(jax.grad(reflectance_of)(n0))

    value_relative_error = abs(value - expected["objective_value"]) / abs(
        expected["objective_value"]
    )
    assert value_relative_error < 1e-5

    grad_relative_error = abs(grad_ad - expected["grad_native_autodiff"]) / abs(
        expected["grad_native_autodiff"]
    )
    assert grad_relative_error < 1e-4, (
        f"jax.grad through FmmaxAdapter.run ({grad_ad}) diverged from the recorded "
        f"gradient_probe.json autodiff value ({expected['grad_native_autodiff']})."
    )

    # Also sanity-check against the probe's own finite-difference cross-check, since that is the
    # actual independent verification recorded for this narrow path.
    fd_relative_error = abs(grad_ad - expected["grad_finite_difference"]) / abs(
        expected["grad_finite_difference"]
    )
    assert fd_relative_error < 5e-4


# --------------------------------------------------------------------------------------
# 5. Failure / resource tests.
# --------------------------------------------------------------------------------------


@pytest.mark.jax
@pytest.mark.integration
def test_import_error_maps_to_adapter_dependency_error(monkeypatch) -> None:
    """(a) Force the lazy-import helper's underlying import to fail -> AdapterDependencyError."""

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "fmmax":
            raise ImportError("simulated: fmmax not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(AdapterDependencyError):
        fmmax_adapter._import_fmmax()

    # And confirm the same failure propagates as a raised exception (not swallowed into a
    # FAILED ModelRunResult) when it occurs inside FmmaxAdapter.run, per the module docstring's
    # stated convention that AdapterDependencyError/UnsupportedCapabilityError propagate.
    adapter = fmmax_adapter.get_adapter()
    request = _request(_bare_interface_structure(), config={"approximate_num_terms": 1})
    with pytest.raises(AdapterDependencyError):
        adapter.run(request)


@pytest.mark.jax
@pytest.mark.integration
def test_non_convergent_truncation_reports_failed_status() -> None:
    """(b) A degenerate/non-convergent truncation order (0 Fourier terms) -> FAILED, not raised.

    fmmax itself does not reject approximate_num_terms=0 (it silently returns a zero-mode
    expansion, see knowledge/solvers/fmmax/failure_guide.md-style investigation performed for
    this adapter); this adapter's own incident-excitation-vector construction then fails with an
    IndexError deep in JAX, which is caught at the run() boundary and reported as a structured
    failure rather than raised, per this adapter's documented solver-execution-error convention.
    """

    adapter = fmmax_adapter.get_adapter()
    request = _request(_bare_interface_structure(), config={"approximate_num_terms": 0})

    result = adapter.run(request)

    assert result.status == RunStatus.FAILED
    assert result.error_type is not None
    assert result.error_message
    assert result.outputs == {}


@pytest.mark.jax
@pytest.mark.integration
def test_unsupported_oblique_incidence_raises_eagerly_before_solver_call() -> None:
    """Requesting physics outside this adapter's verified scope must fail before any solver call."""

    adapter = fmmax_adapter.get_adapter()
    structure = _bare_interface_structure()
    structure.metadata["incidence"] = {"polar_angle_rad": 0.3, "azimuthal_angle_rad": 0.0}
    request = _request(structure, config={"approximate_num_terms": 1})

    with pytest.raises(UnsupportedCapabilityError):
        adapter.run(request)
