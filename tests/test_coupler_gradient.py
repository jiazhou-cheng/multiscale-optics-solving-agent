"""CHE-28 — measure what the gradient estimator computes; certify nothing.

The expected outcome of this issue was "the estimator is biased, here is how
much". What the measurement found is narrower and different, so that is what is
recorded. Either way ``derivative.verified`` stays false: one parameter, one
grid, one wavelength and two objectives is a characterization, not a
certification.
"""

from __future__ import annotations

import numpy as np
import pytest

from couplers.contracts import ComplexField, ReferencePlane
from couplers.gradient import (
    GradientProblem,
    characterize,
    finite_difference_table,
    surrogate_derivative,
    true_derivative,
)
from couplers.wave_to_ray import SamplingDensity

pytestmark = [pytest.mark.coupler, pytest.mark.slow]

WAVELENGTH_M = 500e-9
PITCH_M = 1e-6
N_GRID = 16
PLANE = ReferencePlane(name="doe plane", z_m=0.0)
OBSERVATION_DISTANCE_M = 20e-6
K_SIGMA = 3.0


def _problem(kind: str, distance: float = OBSERVATION_DISTANCE_M) -> GradientProblem:
    rng = np.random.default_rng(1)
    incident = ComplexField(
        u=(rng.normal(size=(N_GRID, N_GRID)) + 1j * rng.normal(size=(N_GRID, N_GRID))).astype(
            np.complex128
        ),
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        reference_plane=PLANE,
    )
    mask = rng.uniform(-1.0, 1.0, size=(N_GRID, N_GRID))

    if kind == "linear":
        weight = rng.normal(size=(N_GRID, N_GRID)) + 1j * rng.normal(size=(N_GRID, N_GRID))
        objective = lambda field: float(np.real(np.vdot(weight, field)))  # noqa: E731
    else:
        window = np.zeros((N_GRID, N_GRID))
        window[4:12, 4:12] = 1.0
        objective = lambda field: float(np.sum(window * np.abs(field) ** 2))  # noqa: E731

    return GradientProblem(incident, mask, objective, kind, distance)


# --- The finite-difference reference must be trustworthy first ------------------


def test_the_true_derivative_is_stable_across_step_sizes() -> None:
    """Required by the protocol so a step-size artifact cannot be mistaken for a
    bias. A central difference carries O(h^2) truncation and O(eps/h) round-off,
    so the estimate is only usable where both are small."""
    table = finite_difference_table(_problem("quadratic"), 1.0, (1e-3, 1e-4, 1e-5))

    values = [row["derivative"] for row in table]
    scale = max(abs(v) for v in values)
    spread = max(values) - min(values)
    assert spread / scale < 1e-4, f"derivative varies with step size: {values}"


def test_an_intensity_objective_at_the_doe_plane_has_no_derivative_at_all() -> None:
    """Recorded because it was the first thing measured, and the zero was the
    objective's fault rather than the estimator's.

    A pure-phase mask leaves |U| pointwise unchanged, so at the DOE plane every
    intensity functional is exactly theta-independent. Any 'gradient bias' study
    run there would be comparing two estimates of zero.
    """
    at_plane = _problem("quadratic", distance=0.0)
    assert abs(true_derivative(at_plane, 1.0, 1e-4)) < 1e-8

    propagated = _problem("quadratic")
    assert abs(true_derivative(propagated, 1.0, 1e-4)) > 1.0


# --- The main measurement ------------------------------------------------------------


@pytest.mark.parametrize("kind", ["linear", "quadratic"])
def test_the_fixed_direction_estimator_is_unbiased_on_a_fixed_spectral_grid(kind: str) -> None:
    """CHE-28's finding.

    The paper's caveat is that the estimator "neglects gradients associated with
    changes in the sampled secondary-ray directions". On a fixed spectral grid
    there is no such gradient to neglect: the direction belongs to the BIN, not
    to the DOE parameter. So the gap is structurally absent here, and the
    caveat describes the continuous-wavevector formulation that SI S7.3's
    Gumbel-Softmax relaxation addresses.

    This is scoped, not a promotion. See the claim test at the bottom.
    """
    report = characterize(_problem(kind), count=2048, realizations=32)

    assert report.claim == "characterized_unbiased_in_regime"
    assert report.bias_in_standard_errors <= K_SIGMA, (
        f"{kind}: bias {report.bias:.4e} is "
        f"{report.bias_in_standard_errors:.2f} SE from the true derivative"
    )
    assert report.realizations >= 32


def test_detaching_the_density_is_what_makes_the_estimator_unbiased() -> None:
    """The measurable half of SI S7.2, and the opposite of the natural reading.

    ``p.detach()`` is not a convenience. Letting the density track the parameter
    leaves in a term that the omitted score-function term of the discrete
    sampling distribution would have cancelled, and the result is badly biased.
    """
    problem = _problem("quadratic")

    detached = characterize(problem, count=2048, realizations=32, detach_density=True)
    live = characterize(problem, count=2048, realizations=32, detach_density=False)

    assert detached.claim == "characterized_unbiased_in_regime"
    assert live.claim == "characterized_biased"
    assert live.bias_in_standard_errors > 10 * detached.bias_in_standard_errors
    assert abs(live.bias) > abs(detached.bias)


def test_no_bias_appears_as_the_ray_count_grows() -> None:
    """A bias hidden under Monte Carlo noise at small N would emerge as the
    standard error shrinks. Sweeping N is how that is ruled out rather than
    assumed: the standard error falls by ~8x across this sweep while the bias
    stays inside 3 SE throughout."""
    problem = _problem("quadratic")
    standard_errors = []
    for count in (512, 2048, 8192):
        report = characterize(problem, count=count, realizations=32)
        assert report.bias_in_standard_errors <= K_SIGMA, (
            f"N={count}: {report.bias_in_standard_errors:.2f} SE"
        )
        standard_errors.append(report.surrogate_standard_error)

    assert standard_errors[-1] < standard_errors[0] / 3.0


def test_uniform_and_magnitude_sampling_are_both_unbiased_for_the_gradient() -> None:
    problem = _problem("linear")
    for density_kind in (SamplingDensity.UNIFORM, SamplingDensity.MAGNITUDE):
        report = characterize(
            problem, count=2048, realizations=32, density_kind=density_kind
        )
        assert report.bias_in_standard_errors <= K_SIGMA, density_kind


def test_the_surrogate_holds_its_sampled_indices_fixed_across_the_difference() -> None:
    """What 'fixed-direction' means operationally. Same seed, same frozen
    indices, so the derivative is reproducible bitwise -- and the difference is
    of the surrogate, not of two independent realizations."""
    problem = _problem("linear")
    first = surrogate_derivative(
        problem, 1.0, 1e-4, count=512, rng=np.random.default_rng(3)
    )
    second = surrogate_derivative(
        problem, 1.0, 1e-4, count=512, rng=np.random.default_rng(3)
    )
    assert first == second


# --- What is reported, and what is not claimed ------------------------------------------


def test_the_report_names_the_omitted_terms_and_carries_the_step_table() -> None:
    report = characterize(_problem("linear"), count=1024, realizations=32)
    record = report.as_dict()

    assert record["omitted_terms"], "the estimator's omissions must be named, not implied"
    assert any("score-function" in term for term in record["omitted_terms"])
    assert any("directions" in term for term in record["omitted_terms"])

    table = record["finite_difference_comparison"]["table"]
    assert len(table) >= 3
    assert all("step" in row and "derivative" in row for row in table)

    # The regime is part of the result: a bias figure without one is not usable.
    assert record["sampled_ray_count"] == 1024
    assert record["realizations"] == 32
    assert record["objective_kind"] == "linear"


def test_nothing_here_promotes_a_gradient_claim() -> None:
    """AGENTS.md: never claim a gradient across an untested boundary. An
    unbiased measurement in one narrow regime is not the evidence
    coupler_protocol.yaml requires for promotion, and the registry must still
    say so."""
    import yaml

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    couplers = yaml.safe_load(
        (root / "src/registry/couplers.yaml").read_text()
    )["couplers"]
    by_id = {entry["id"]: entry for entry in couplers}

    for coupler_id in ("C_RAY_TO_WAVE", "C_WAVE_TO_RAY"):
        assert by_id[coupler_id]["derivative"]["verified"] is False, coupler_id

    # And the report's own vocabulary cannot express a promotion by accident.
    report = characterize(_problem("linear"), count=512, realizations=32)
    assert report.claim != "verified"
