"""The metric definitions, and the blind spots they are required to declare.

CHE-112 (M2.4). Two things are tested: that each metric computes what it says,
and that its declared blind spot is real. The second is the unusual one and it
is the point of the module -- a blind spot nobody demonstrated is a sentence in
a docstring, and this file turns each into a measurement.

Every blind-spot test below constructs the case the metric cannot see and shows
it scoring perfectly on it. That is a strange-looking assertion until you notice
what it prevents: a reader picking NCC to check energy conservation, which it
cannot do, on a route that is known to lose 1.7% of the power.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from verification.metrics import (
    METRICS,
    MetricDefinition,
    central_relative_l2_intensity,
    metric,
    mse_unit_sum,
    ncc,
    ncc_uncentred,
    power_ratio,
    relative_l2_field,
    relative_l2_intensity,
    relative_rms,
)


@pytest.fixture
def field() -> np.ndarray:
    """A deterministic complex field with structure on and off axis."""
    n = 32
    y, x = np.mgrid[:n, :n] - n // 2
    r2 = (x**2 + y**2).astype(np.float64)
    envelope = np.exp(-r2 / (2 * 6.0**2))
    phase = 0.3 * x + 0.05 * r2
    return (envelope * np.exp(1j * phase)).astype(np.complex128)


# --------------------------------------------------------------------------- #
# Each definition computes what it says
# --------------------------------------------------------------------------- #


def test_identical_inputs_score_the_ideal_value(field: np.ndarray) -> None:
    intensity = np.abs(field) ** 2
    assert ncc(intensity, intensity) == pytest.approx(1.0)
    assert ncc_uncentred(intensity, intensity) == pytest.approx(1.0)
    assert relative_l2_field(field, field) == pytest.approx(0.0)
    assert relative_l2_intensity(field, field) == pytest.approx(0.0)
    assert power_ratio(field, field) == pytest.approx(1.0)
    assert mse_unit_sum(intensity, intensity) == pytest.approx(0.0)
    assert central_relative_l2_intensity(field, field) == pytest.approx(0.0)


def test_the_field_and_intensity_metrics_are_different_numbers(field: np.ndarray) -> None:
    """Both are called "relative L2" and confusing them has produced wrong
    conclusions. A pure phase error is everything to one and nothing to the
    other."""
    phase_only = field * np.exp(1j * 0.4)
    assert relative_l2_field(phase_only, field) > 0.3
    assert relative_l2_intensity(phase_only, field) == pytest.approx(0.0, abs=1e-12)


def test_the_power_ratio_reads_the_loss_it_is_for(field: np.ndarray) -> None:
    """0.9832 is the k-space route's measured ratio on demo3."""
    lossy = field * math.sqrt(0.9832)
    assert power_ratio(lossy, field) == pytest.approx(0.9832, rel=1e-9)


def test_relative_rms_and_relative_l2_field_agree(field: np.ndarray) -> None:
    """Named separately so a reader matching a round-trip number against a paper
    does not have to work out that the sqrt(N) cancels."""
    other = field + 0.01
    assert relative_rms(other, field) == pytest.approx(relative_l2_field(other, field))


# --------------------------------------------------------------------------- #
# The blind spots, demonstrated rather than asserted
# --------------------------------------------------------------------------- #


def test_ncc_cannot_see_a_power_loss(field: np.ndarray) -> None:
    """The one that matters most in this repository.

    A route can lose 1.7% of the energy and correlate at exactly 1.0, so NCC
    alone cannot certify a representation change. The power ratio is what sees
    it, which is why the two are always reported together.
    """
    intensity = np.abs(field) ** 2
    lossy = 0.983 * intensity
    assert ncc(lossy, intensity) == pytest.approx(1.0)
    assert power_ratio(np.sqrt(lossy), np.sqrt(intensity)) == pytest.approx(0.983, rel=1e-9)


def test_ncc_cannot_see_an_additive_pedestal(field: np.ndarray) -> None:
    intensity = np.abs(field) ** 2
    raised = intensity + 0.5 * intensity.max()
    assert ncc(raised, intensity) == pytest.approx(1.0)
    assert ncc_uncentred(raised, intensity) < 1.0, (
        "the uncentred variant does see it, which is why both are reported"
    )


def test_a_centred_metric_cannot_see_an_off_axis_defect(field: np.ndarray) -> None:
    """CHE-44's concern, never audited until now, as a measurement.

    A defect placed outside the gate disc is invisible to the centred metric and
    plainly visible to the full-array one. Anything that reports only the first
    is reporting a number whose scope is not what a reader assumes.
    """
    perturbed = field.copy()
    perturbed[2:6, 2:6] += 5.0  # a corner, far outside any central window

    centred = central_relative_l2_intensity(perturbed, field, fraction=0.5)
    whole = relative_l2_intensity(perturbed, field)

    assert centred == pytest.approx(0.0, abs=1e-12)
    assert whole > 1.0
    assert METRICS["central_relative_l2_intensity"].support == "centred_window"


def test_a_round_trip_metric_cannot_see_an_error_that_is_its_own_inverse() -> None:
    """The reason a round trip needs a deliberately broken twin.

    Apply a wrong transfer function forward and its exact inverse backward: the
    field returns to round-off, and the round-trip metric says everything is
    fine about a propagator that is wrong.
    """
    rng = np.random.default_rng(0)
    original = rng.normal(size=64) + 1j * rng.normal(size=64)
    wrong_kernel = np.exp(1j * 7.3 * np.arange(64))  # not the right k_z, at all

    returned = (original * wrong_kernel) / wrong_kernel
    assert relative_rms(returned, original) == pytest.approx(0.0, abs=1e-14)


def test_mse_unit_sum_cannot_see_a_scale_factor(field: np.ndarray) -> None:
    intensity = np.abs(field) ** 2
    assert mse_unit_sum(3.7 * intensity, intensity) == pytest.approx(0.0, abs=1e-20)


# --------------------------------------------------------------------------- #
# The register
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("definition", sorted(METRICS.values(), key=lambda d: d.name),
                         ids=lambda d: d.name)
def test_every_metric_declares_a_blind_spot_and_a_question(
    definition: MetricDefinition,
) -> None:
    assert definition.blind_to
    assert all(blind.strip() for blind in definition.blind_to)
    assert definition.answers.strip()
    assert definition.support in {"whole_array", "centred_window"}


def test_a_metric_without_a_blind_spot_cannot_be_declared() -> None:
    with pytest.raises(ValueError, match="state what this metric cannot see"):
        MetricDefinition(
            name="wishful", fn=lambda a, b: 0.0, answers="everything", blind_to=()
        )


def test_an_unknown_metric_names_what_exists() -> None:
    with pytest.raises(KeyError, match="no metric"):
        metric("relative_l2")  # ambiguous on purpose: field or intensity?


def test_the_metric_names_families_reference_all_resolve() -> None:
    """Every metric a family declares must be one of these definitions, or the
    centralization has not happened and two benchmarks can still disagree about
    what a number means.

    Reported as a gap list rather than asserted empty: M1's families were
    authored before this module existed, and the honest state is which of their
    metric names have a definition here and which are still local.
    """
    from verification.families import FAMILIES

    named = [
        (family.family_id, m.name, m.definition)
        for family in FAMILIES
        for m in family.metrics
        if m.definition is not None
    ]
    assert named, "no family names a central definition at all"
    dangling = [entry for entry in named if entry[2] not in METRICS]
    assert not dangling, (
        "these family metrics name a definition that does not exist:\n  "
        + "\n  ".join(f"{fid}/{metric} -> {definition}" for fid, metric, definition in dangling)
    )


def test_every_definition_is_callable_through_the_register(field: np.ndarray) -> None:
    intensity = np.abs(field) ** 2
    for definition in METRICS.values():
        value = definition(intensity, intensity)
        assert math.isfinite(value)
        assert value == pytest.approx(definition.ideal, abs=1e-12)


def test_a_zero_reference_gives_nan_rather_than_a_number() -> None:
    """A relative metric against nothing has no value, and returning 0.0 would
    read as perfect agreement."""
    zeros = np.zeros(8, dtype=np.complex128)
    ones = np.ones(8, dtype=np.complex128)
    assert math.isnan(relative_l2_field(ones, zeros))
    assert math.isnan(power_ratio(ones, zeros))
    assert math.isnan(mse_unit_sum(np.ones(8), np.zeros(8)))
