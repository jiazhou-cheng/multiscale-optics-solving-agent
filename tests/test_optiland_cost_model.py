"""The trace cost model says what the committed record says, or this fails (CHE-118).

A calibrated cost model is a number copied out of a measurement into source, and
the copy is where it rots. These tests are the tie: every constant in
``solvers.optiland.cost_model`` is checked against
``benchmarks/perf/records/optiland_trace_chunk_sweep.json``, so re-running the
sweep and forgetting to update the constant fails here rather than quietly
mispricing an executor's plan for the next several months.

The other half is the refusals. A cost model that answers on any host is worse
than one that answers on none, because a portable-looking number gets planned
against; ``core.performance.compare`` already refuses to divide records across
environment fingerprints, and these tests hold the estimator to the same rule.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.execution import CostEstimate
from solvers.optiland.cost_model import (
    CUDA_FP32_TRACE_COST,
    estimate_trace_seconds,
    hexapolar_ray_count,
    traced_surface_count,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "benchmarks" / "perf" / "records" / "optiland_trace_chunk_sweep.json"
SCALING = ROOT / "benchmarks" / "perf" / "records" / "scaling_ray_axis.json"


@pytest.fixture(scope="module")
def sweep() -> dict:
    return json.loads(SWEEP.read_text())


def test_the_constants_are_the_committed_fit(sweep) -> None:
    """The one test that keeps the model honest as the record is re-measured."""
    fit = sweep["affine_cost_model"]
    model = CUDA_FP32_TRACE_COST
    assert model.fixed_per_call_s == fit["fixed_per_call_s"]
    assert model.seconds_per_ray_per_surface == fit["seconds_per_ray_per_surface"]
    assert list(model.domain_rays) == fit["domain_rays"]
    assert model.fitted_on_surfaces == fit["fitted_on_surfaces"]
    assert model.max_relative_error_in_domain == fit["max_relative_error_in_domain"]
    assert model.environment_sha256 == sweep["environment"]["sha256"]
    assert model.gpu_name == sweep["environment"]["gpu_name"]


def test_the_model_reproduces_every_measured_point_in_its_domain(sweep) -> None:
    """Not a re-fit: the constants in source, against the measurements on disk.

    `max_relative_error_in_domain` is the record's own claim about the fit; this
    checks that the *committed constants* deliver it, which is a different
    statement and the one a caller depends on.
    """
    # `+ 1e-4` for the record's own rounding: `relative_error` is stored to four
    # decimal places, so the worst point's true error is up to 5e-5 above the
    # figure the model carries as its bound.
    tolerance = CUDA_FP32_TRACE_COST.max_relative_error_in_domain + 1e-4
    surfaces = CUDA_FP32_TRACE_COST.fitted_on_surfaces
    checked = 0
    for residual in sweep["affine_cost_model"]["residuals"]:
        if not residual["in_domain"]:
            continue
        predicted = CUDA_FP32_TRACE_COST.predict_s(
            rays=residual["rays"], surfaces=surfaces
        )
        assert predicted == pytest.approx(
            residual["measured_s"], rel=tolerance
        ), f"{residual['rays']} rays"
        checked += 1
    assert checked >= 4, "the domain has to contain enough points to be a fit"


def test_the_excluded_point_is_excluded_because_the_model_fails_there(sweep) -> None:
    """The domain is a measurement too, so its boundary needs evidence.

    Above ~1 M rays a chunk the cost per ray turns back up and the affine form
    stops holding. If that ever stops being true the domain should be widened --
    and this test failing is how anyone would find out.
    """
    outside = [
        r for r in sweep["affine_cost_model"]["residuals"] if not r["in_domain"]
    ]
    assert outside, "an unbounded domain is not a domain"
    floor = 5 * CUDA_FP32_TRACE_COST.max_relative_error_in_domain
    for residual in outside:
        assert abs(residual["relative_error"]) > floor, (
            f"{residual['rays']} rays is excluded from the model's domain but the "
            "model predicts it about as well as the points inside. Either widen the "
            "domain or say why it is still excluded."
        )


def test_the_hexapolar_ray_count_matches_every_committed_ring_count() -> None:
    """`1 + 3 N (N + 1)` is verified, not assumed.

    This is what lets `estimate()` know a ray count the old estimator said was
    unknowable until after the call, so the formula needs evidence rather than
    plausibility. `scaling_ray_axis.json` traced six ring counts and recorded the
    ray count each produced.
    """
    rows = json.loads(SCALING.read_text())["rows"]
    assert len(rows) >= 4
    for row in rows:
        assert hexapolar_ray_count(row["rings"]) == row["rays"], row["rings"]


def test_rings_and_rays_are_not_confused() -> None:
    """The distinction the whole estimator rests on, held at one obvious point."""
    assert hexapolar_ray_count(0) == 1
    assert hexapolar_ray_count(32) == 3169
    with pytest.raises(ValueError, match="rings must be >= 0"):
        hexapolar_ray_count(-1)


def test_the_traced_surface_count_matches_a_built_system() -> None:
    """The convention, checked against Optiland rather than against arithmetic.

    A built `Optic` carries an object surface that `skip=1` never traces and an
    image plane the prescription does not list. `traced_surface_count` encodes
    that; this is the check that it encodes it the way the builder builds it.
    """
    pytest.importorskip("optiland")
    from registry.prescriptions import resolve_prescription
    from solvers.optiland.builder import build_optiland_system
    from solvers.optiland.coherent_trace import surface_positions_m

    for name in ("M3SingletRef", "ReverseTelephoto"):
        spec = resolve_prescription(name)
        built = build_optiland_system(spec)
        # `skip=1` traces everything after the object surface.
        traced = len(surface_positions_m(built)) - 1
        assert traced_surface_count(len(spec.surfaces)) == traced, name


def test_a_prediction_is_given_on_the_calibrated_environment() -> None:
    """The point of the exercise: a planner gets a number, with its confidence."""
    estimate = estimate_trace_seconds(
        rays=200_000,
        surfaces=4,
        environment_sha256=CUDA_FP32_TRACE_COST.environment_sha256,
    )
    assert isinstance(estimate, CostEstimate)
    assert estimate.wall_time_s == pytest.approx(
        CUDA_FP32_TRACE_COST.predict_s(rays=200_000, surfaces=4)
    )
    assert estimate.confidence == "medium"
    # Never guessed. A planner would size a batch with it.
    assert estimate.peak_memory_bytes is None


def test_a_different_environment_gets_no_prediction_and_is_told_why() -> None:
    estimate = estimate_trace_seconds(
        rays=200_000, surfaces=4, environment_sha256="0" * 64
    )
    assert estimate.wall_time_s is None
    assert estimate.confidence == "unknown"
    joined = " ".join(estimate.notes)
    assert "NO PREDICTION" in joined
    assert "fingerprint" in joined
    # The reference value may be quoted in the notes, but it must not be reachable
    # as a prediction -- that is the whole distinction being drawn.
    assert "not a prediction on this host" in joined


@pytest.mark.parametrize(
    ("rays", "expected_word"),
    [(1.0, "below"), (1.0e8, "above")],
)
def test_outside_the_fitted_domain_gets_no_prediction(rays, expected_word) -> None:
    estimate = estimate_trace_seconds(
        rays=rays,
        surfaces=4,
        environment_sha256=CUDA_FP32_TRACE_COST.environment_sha256,
    )
    assert estimate.wall_time_s is None
    joined = " ".join(estimate.notes)
    assert "NO PREDICTION" in joined
    assert expected_word in joined


def test_the_estimator_refuses_nonsense_rather_than_extrapolating() -> None:
    for kwargs in ({"rays": 0, "surfaces": 4}, {"rays": 100, "surfaces": 0}):
        with pytest.raises(ValueError):
            estimate_trace_seconds(**kwargs)


def test_the_adapter_estimate_uses_the_model_and_names_the_ring_conversion(
    tmp_path,
) -> None:
    """`estimate()` no longer returns a bare `None` with a note saying it cannot.

    It either predicts or refuses for a stated reason, and either way it now
    reports the surface count and the traced ray count -- the two things the
    previous estimator said it did not know.
    """
    pytest.importorskip("optiland")
    from solvers.base import ModelRunRequest
    from solvers.optiland.adapter import get_adapter

    estimate = get_adapter().estimate(
        ModelRunRequest(
            run_id="estimate",
            node_id="lens",
            config={
                "sample": "M3SingletRef",
                "num_rays": 32,
                "wavelength": 0.55,
                "Hx": 0.0,
                "Hy": 0.0,
                "handoff_plane": "exit_pupil",
                "output_directory": str(tmp_path),
            },
        )
    )
    joined = " ".join(estimate.notes)
    assert "3169 rays" in joined, "the ring-to-ray conversion has to be visible"
    assert "RING count" in joined
    assert "3 are traced" in joined
    assert CUDA_FP32_TRACE_COST.record in joined
    # On the calibrated host this is a number; anywhere else it is a stated
    # refusal. Both are correct outcomes and the test accepts either, because
    # pinning one would make this test pass or fail on where it is run.
    if estimate.wall_time_s is None:
        assert "NO PREDICTION" in joined
    else:
        assert estimate.confidence == "medium"
