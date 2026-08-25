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
from pathlib import Path

import numpy as np
import pytest

from verification.metrics import (
    METRICS,
    SUPPORTS,
    MetricDefinition,
    central_relative_l2_intensity,
    disc_mask,
    disc_relative_l2_intensity,
    metric,
    mse_unit_sum,
    ncc,
    ncc_uncentred,
    peak_normalized_masked_relative_l2,
    power_ratio,
    radial_profile_relative_l2,
    radial_profile_residuals,
    relative_l2_field,
    relative_l2_intensity,
    relative_rms,
)

ROOT = Path(__file__).resolve().parents[1]


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
    assert definition.support in SUPPORTS


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


#: Geometry for the metrics that cannot be called without it. Keyed by metric
#: name, and asserted below to cover exactly the metrics declaring
#: ``required_kwargs`` -- so a new geometry-taking metric cannot be added and
#: silently skipped by the smoke test that would have exercised it.
_SMOKE_KWARGS: dict[str, dict[str, object]] = {
    "radial_profile_relative_l2": {
        "measured_pitch": (1e-6, 1e-6),
        "reference_pitch": (1e-6, 1e-6),
        "max_radius_m": 8e-6,
    },
    "disc_relative_l2_intensity": {
        "sample_pitch_m": 1e-6,
        "max_radius_m": 8e-6,
    },
}


def test_the_smoke_geometry_covers_every_metric_that_needs_geometry() -> None:
    needs = {name for name, d in METRICS.items() if d.required_kwargs}
    assert set(_SMOKE_KWARGS) == needs, (
        "a metric declaring required_kwargs with no entry here would be skipped by "
        "the register smoke test rather than exercised by it"
    )
    for name, supplied in _SMOKE_KWARGS.items():
        assert set(METRICS[name].required_kwargs) <= set(supplied)


def test_every_definition_is_callable_through_the_register(field: np.ndarray) -> None:
    intensity = np.abs(field) ** 2
    for definition in METRICS.values():
        value = definition(intensity, intensity, **_SMOKE_KWARGS.get(definition.name, {}))
        assert math.isfinite(value)
        assert value == pytest.approx(definition.ideal, abs=1e-12)


# --------------------------------------------------------------------------- #
# The promoted radial-profile residual (CHE-115)
# --------------------------------------------------------------------------- #


def _airy_like(n: int, pitch_m: float, radius_m: float, *, center_px: float = 0.0) -> np.ndarray:
    """A rotationally symmetric pattern with a known centre, for profile tests."""
    axis = (np.arange(n) - n // 2) * pitch_m
    y = axis[:, None] - center_px * pitch_m
    x = axis[None, :]
    r = np.hypot(y, x) / radius_m
    return np.exp(-(r**2))


def test_the_profile_residual_is_zero_against_itself() -> None:
    pattern = _airy_like(128, 1e-6, 4e-6)
    assert radial_profile_relative_l2(
        pattern,
        pattern,
        measured_pitch=(1e-6, 1e-6),
        reference_pitch=(1e-6, 1e-6),
        max_radius_m=16e-6,
    ) == pytest.approx(0.0, abs=1e-12)


def test_the_profile_residual_is_blind_to_absolute_scale() -> None:
    """Both patterns are peak-normalized first, which is the frozen M3 oracle
    normalization -- so a power loss is invisible here by construction and must
    be measured beside it."""
    pattern = _airy_like(128, 1e-6, 4e-6)
    assert radial_profile_relative_l2(
        7.3 * pattern,
        pattern,
        measured_pitch=(1e-6, 1e-6),
        reference_pitch=(1e-6, 1e-6),
        max_radius_m=16e-6,
    ) == pytest.approx(0.0, abs=1e-12)


def test_the_profile_residual_compares_across_different_samplings_and_converges() -> None:
    """The whole reason it exists: the oracle and the measurement are on
    different pitches, and each is profiled on its own grid before the reference
    is interpolated onto the measured radii.

    What is asserted is convergence rather than a threshold, because the metric
    is NOT free of the measured pattern's own sampling: against a finely sampled
    reference, the same analytic pattern sampled at 1 um reads 1.5e-2 -- the
    polar interpolation of a coarse grid, not a difference in the patterns. A
    fixed tolerance here would either bless that or be tuned to it. The
    meaningful statement is that refining the measured grid drives it away.
    """
    reference = _airy_like(1024, 0.125e-6, 4e-6)
    values = [
        radial_profile_relative_l2(
            _airy_like(int(128 * 1e-6 / pitch), pitch, 4e-6),
            reference,
            measured_pitch=(pitch, pitch),
            reference_pitch=(0.125e-6, 0.125e-6),
            max_radius_m=16e-6,
        )
        for pitch in (1.0e-6, 0.5e-6, 0.25e-6)
    ]
    assert values == sorted(values, reverse=True), values
    assert values[0] > 10 * values[-1], values


def test_the_profile_residual_sees_a_width_error() -> None:
    wide = _airy_like(128, 1e-6, 4.4e-6)
    reference = _airy_like(128, 1e-6, 4e-6)
    value = radial_profile_relative_l2(
        wide,
        reference,
        measured_pitch=(1e-6, 1e-6),
        reference_pitch=(1e-6, 1e-6),
        max_radius_m=16e-6,
    )
    assert value > 0.05, value


def test_the_profile_residual_is_blind_to_an_off_centre_pattern_averaged_on_axis() -> None:
    """CHE-41's finding, kept as a live demonstration rather than a warning.

    Averaging an off-axis pattern about the ORIGIN smears it into an annulus.
    The metric declares this blindness; this asserts the blindness is real, so
    that the declaration cannot quietly become false.
    """
    reference = _airy_like(256, 1e-6, 4e-6)
    shifted = _airy_like(256, 1e-6, 4e-6, center_px=30.0)

    about_the_origin = radial_profile_relative_l2(
        shifted, reference,
        measured_pitch=(1e-6, 1e-6), reference_pitch=(1e-6, 1e-6), max_radius_m=16e-6,
    )
    about_the_true_centre = radial_profile_relative_l2(
        shifted, reference,
        measured_pitch=(1e-6, 1e-6), reference_pitch=(1e-6, 1e-6), max_radius_m=16e-6,
        measured_center_m=(30.0e-6, 0.0),
    )
    # Told where the pattern is, the metric sees agreement; left at the origin it
    # reports a large residual for a pattern that is identically shaped.
    assert about_the_true_centre < 1e-9
    assert about_the_origin > 0.1


def test_a_zero_reference_profile_raises_rather_than_reading_as_agreement() -> None:
    """A peak-normalized comparison against nothing has no value. It is refused
    at the normalization, before the norm, so the message names the actual
    problem rather than reporting an undefined ratio."""
    pattern = _airy_like(64, 1e-6, 4e-6)
    with pytest.raises(ValueError, match="needs a positive peak in both patterns"):
        radial_profile_relative_l2(
            pattern,
            np.zeros_like(pattern),
            measured_pitch=(1e-6, 1e-6),
            reference_pitch=(1e-6, 1e-6),
            max_radius_m=16e-6,
        )


def test_the_probe_and_the_promoted_metric_are_the_same_computation() -> None:
    """CHE-115 copied this out of ``benchmarks/probes/psf_oracle_verification.py``.

    Copied, not moved, and that is deliberate. Editing the probe changes the
    source hash its stamped record is enrolled against
    (``tests/test_provenance_fingerprint.py``), so switching it to the promoted
    definition requires regenerating ``m3_psf_verification.json`` -- a real
    multi-minute run of the full slice, and a record must be produced by the code
    it claims. That regeneration is the remaining step.

    Until then there are two implementations, and this is what stops them
    drifting: they are asserted equal here on a non-trivial input rather than
    assumed equal because one was pasted from the other.
    """
    import importlib.util
    import sys

    path = ROOT / "benchmarks/probes/psf_oracle_verification.py"
    spec = importlib.util.spec_from_file_location("psf_oracle_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["psf_oracle_probe"] = module
    spec.loader.exec_module(module)

    measured = _airy_like(128, 1e-6, 4.2e-6)
    reference = _airy_like(256, 0.5e-6, 4e-6)
    geometry = {
        "measured_pitch": (1e-6, 1e-6),
        "reference_pitch": (0.5e-6, 0.5e-6),
        "max_radius_m": 16e-6,
    }
    from_probe = module._profile_residual(measured, reference, **geometry)
    from_metric = radial_profile_residuals(measured, reference, **geometry)
    assert from_probe == from_metric
    assert from_probe["relative_l2_profile_residual"] == pytest.approx(
        radial_profile_relative_l2(measured, reference, **geometry)
    )


def test_a_zero_reference_gives_nan_rather_than_a_number() -> None:
    """A relative metric against nothing has no value, and returning 0.0 would
    read as perfect agreement."""
    zeros = np.zeros(8, dtype=np.complex128)
    ones = np.ones(8, dtype=np.complex128)
    assert math.isnan(relative_l2_field(ones, zeros))
    assert math.isnan(power_ratio(ones, zeros))
    assert math.isnan(mse_unit_sum(np.ones(8), np.zeros(8)))


# --------------------------------------------------------------------------- #
# The promoted frozen gate metric (CHE-115)
# --------------------------------------------------------------------------- #


def test_the_disc_metric_is_the_frozen_gate_computation() -> None:
    """It must reproduce ``sensor_handoff_convergence._relative_l2`` exactly.

    That probe helper is what produced ``2.2072391812867093e-3`` -- the number
    the B3-PSF-SINGLET gate is NOT_MET against -- so a promotion that changed it
    by a rounding would silently move a frozen result.
    """
    rng = np.random.default_rng(11)
    measured = np.abs(rng.normal(size=(64, 64))) + 0.1
    reference = np.abs(rng.normal(size=(64, 64))) + 0.1
    mask = disc_mask((64, 64), 1e-6, 12e-6)

    def frozen(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> float:
        a64 = np.asarray(a, dtype=np.float64)
        b64 = np.asarray(b, dtype=np.float64)
        pa, pb = float(np.max(a64)), float(np.max(b64))
        difference = (a64 / pa - b64 / pb)[m]
        denominator = float(np.linalg.norm((b64 / pb)[m]))
        return float(np.linalg.norm(difference) / denominator)

    assert peak_normalized_masked_relative_l2(measured, reference, mask=mask) == frozen(
        measured, reference, mask
    )
    assert disc_relative_l2_intensity(
        measured, reference, sample_pitch_m=1e-6, max_radius_m=12e-6
    ) == frozen(measured, reference, mask)


def test_the_disc_metric_is_blind_to_absolute_scale() -> None:
    """Each pattern is divided by its own peak, so a power loss cannot be seen
    here and power_ratio has to be reported beside it."""
    rng = np.random.default_rng(3)
    pattern = np.abs(rng.normal(size=(32, 32))) + 0.1
    assert disc_relative_l2_intensity(
        41.7 * pattern, pattern, sample_pitch_m=1e-6, max_radius_m=8e-6
    ) == pytest.approx(0.0, abs=1e-14)


def test_a_defect_outside_the_disc_still_moves_the_number_through_the_peak() -> None:
    """The disc bounds the DIFFERENCE, not the normalization -- and that is the
    trap.

    The obvious reading of a windowed metric is that nothing outside the window
    matters. Here each pattern is divided by its own **global** maximum before
    the difference is taken, so a single bright pixel anywhere in the array
    rescales everything inside the disc. A stray 5x pixel in the corner, well
    outside a 6 um disc on a 1 um grid, takes an otherwise perfect agreement to
    0.8.

    Asserted rather than warned about, because it is the difference between "the
    gate is blind to a halo" and "a halo silently corrupts the gate", and the
    two lead to opposite conclusions when a number moves.
    """
    pattern = np.ones((64, 64))
    corrupted = pattern.copy()
    corrupted[0, 0] = 5.0
    assert disc_relative_l2_intensity(
        corrupted, pattern, sample_pitch_m=1e-6, max_radius_m=6e-6
    ) == pytest.approx(0.8, rel=1e-9)

    # The difference itself IS windowed: the same magnitude of defect placed
    # outside the disc, but small enough not to become the peak, does not appear.
    faint = pattern.copy()
    faint[0, 0] = 0.4
    assert disc_relative_l2_intensity(
        faint, pattern, sample_pitch_m=1e-6, max_radius_m=6e-6
    ) == pytest.approx(0.0, abs=1e-14)

    inside = pattern.copy()
    inside[32, 32] = 0.4
    assert disc_relative_l2_intensity(
        inside, pattern, sample_pitch_m=1e-6, max_radius_m=6e-6
    ) > 0.01


def test_the_disc_metric_returns_nan_against_an_empty_reference() -> None:
    pattern = np.ones((16, 16))
    assert math.isnan(
        disc_relative_l2_intensity(
            pattern, np.zeros((16, 16)), sample_pitch_m=1e-6, max_radius_m=6e-6
        )
    )


def test_the_disc_window_matches_the_probes_own_mask() -> None:
    """The window is half the gate: a caller building a square instead would
    report "the same" metric over a different region.

    Same situation as the profile residual above -- the probe keeps its copy
    until its stamped record is regenerated, so the two are pinned equal here
    rather than merged.
    """
    import importlib.util
    import sys

    path = ROOT / "benchmarks/probes/sensor_handoff_convergence.py"
    spec = importlib.util.spec_from_file_location("sensor_probe_for_mask", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["sensor_probe_for_mask"] = module
    spec.loader.exec_module(module)

    assert np.array_equal(
        module._disc_mask((37, 41), 0.5e-6, 7.3e-6), disc_mask((37, 41), 0.5e-6, 7.3e-6)
    )


def test_the_singlet_gate_metric_now_names_a_central_definition() -> None:
    """CHE-115's blocker: the family's gate metric pointed at
    ``central_relative_l2_intensity`` -- a centred SQUARE window with no peak
    normalization -- which is not the computation the frozen 2.2072e-3 came
    from. It now names the promoted one."""
    from verification.families.b3_composed import B3_PSF_SINGLET

    gate = {m.name: m for m in B3_PSF_SINGLET.metrics}["fft_oracle_intensity_relative_l2"]
    assert gate.definition == "disc_relative_l2_intensity"
    assert METRICS[gate.definition].support == "centred_disc"
