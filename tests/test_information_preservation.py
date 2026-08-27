"""M2.4's two enforcement layers: the invariant battery and the blindness audit.

CHE-112. The round trips, the metric definitions, the negative-control battery
and the "what does not survive" statement are all built elsewhere -- in
``verification/metrics.py``, in ``B2-ROUNDTRIP`` and its instances, and in
``knowledge/couplers/README.md``. What this file adds is the part that has to be
enforced rather than written: that no coupler can declare an invariant nothing
asserts, and that no metric can be blind to an off-axis defect without saying so.

Both are mechanical for the same reason. A registry declaration and a metric
docstring are both prose that keeps reading correctly after the thing it
describes has changed, and the failure is silent in both directions -- an
invariant with no assertion looks like coverage, and a centred metric looks like
a measurement of the whole field.

CHE-44 was opened to audit the control metrics for centre-dependent blindness
and was cancelled without the audit being done. This is the audit, executed:
each metric is handed the *same* localized defect on axis and off axis, and any
metric whose off-axis sensitivity is materially lower has to declare it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.paths import repository_root
from registry.loader import Registry
from verification.declaration_ledger import DeclarationKind, resolve_ledger
from verification.families import FAMILIES
from verification.metrics import METRICS

pytestmark = [pytest.mark.coupler]

ROOT = repository_root()

#: Metrics that need a physical geometry (a pitch, a radius) to be defined at
#: all, so they cannot be called on a bare array pair. They are audited by the
#: declaration check below rather than by the numerical sweep.
_NEEDS_GEOMETRY = frozenset({"radial_profile_relative_l2", "disc_relative_l2_intensity"})


# --------------------------------------------------------------------------- #
# The invariant battery
# --------------------------------------------------------------------------- #


def _registry_invariants() -> dict[str, tuple[str, ...]]:
    registry = Registry.from_package()
    return {cid: tuple(coupler.invariants) for cid, coupler in registry.couplers.items()}


def test_every_coupler_declares_at_least_one_invariant() -> None:
    """A coupler declared ``lossy: true`` with nothing preserved is either not a
    coupler or an undocumented one."""
    for coupler_id, invariants in _registry_invariants().items():
        assert invariants, f"{coupler_id} declares no invariant"


@pytest.mark.parametrize(
    ("coupler_id", "invariant"),
    [
        (coupler_id, invariant)
        for coupler_id, invariants in sorted(_registry_invariants().items())
        for invariant in invariants
    ],
)
def test_every_declared_invariant_has_an_executable_assertion(
    coupler_id: str, invariant: str
) -> None:
    """M2.4's third acceptance criterion, enforced rather than reviewed.

    "Every invariant declared in ``registry/couplers.yaml`` has an executable
    assertion with a stated tolerance basis, or is removed from the registry as
    unsupported." The parameterization is over the *registry*, not over a list
    here, so adding an invariant to a coupler adds a test that fails until it has
    evidence -- which is the direction that matters. A hand-maintained list would
    let the new declaration in silently.

    The declaration ledger is the shared mechanism. It also refuses to classify
    an invariant as ``EXPLICIT_NON_EXECUTABLE`` at construction, so "nothing can
    assert this" is not an available answer for an invariant: the honest move is
    to remove the declaration.
    """
    report = resolve_ledger()
    entries = [
        entry
        for entry in report.covered.values()
        if entry.kind is DeclarationKind.INVARIANT
        and entry.component == coupler_id
        and entry.anchor == invariant
    ]
    assert entries, (
        f"{coupler_id} declares invariant {invariant!r} and the ledger covers no such "
        "entry. Either assert it or remove the declaration."
    )
    assert len(entries) == 1, f"{coupler_id}/{invariant} is claimed twice"
    entry = entries[0]

    assert entry.coverage_kind.is_executable, (
        f"{coupler_id}/{invariant} is covered by something non-executable"
    )
    assert entry.evidence, f"{coupler_id}/{invariant} names no evidence"
    assert entry.tolerance_basis, (
        f"{coupler_id}/{invariant}: an invariant asserted at a tolerance nobody "
        "derived is a number somebody liked"
    )


def test_every_invariant_tolerance_basis_names_a_recognized_kind() -> None:
    """A basis that is prose but not a *kind* cannot be checked for consistency.

    So each one has to open with the class of argument it is: a conservation law,
    an analytic derivation, or a floating-point floor. Those are the only three
    that can carry an invariant -- a fitted or measured tolerance is a
    characterization, and an invariant is not.
    """
    permitted = ("conservation_law", "analytic_derivation", "numerical_precision_floor")
    for key, entry in resolve_ledger().covered.items():
        if entry.kind is not DeclarationKind.INVARIANT:
            continue
        basis = entry.tolerance_basis or ""
        assert basis.startswith(permitted), f"{key}: basis {basis[:60]!r}"
        assert len(basis) > 60, f"{key}: a basis this short is a label, not a derivation"


def test_the_invariant_battery_covers_all_five_couplers() -> None:
    """Stated as a count so a silently dropped coupler is visible.

    Fourteen invariant declarations across five couplers, because the two
    composed couplers inherit their halves' invariants and add one each --
    ``outgoing_count_is_the_budget`` for the cascade and
    ``patch_coverage_corrected`` for the patch route -- and CHE-143 (M2.7)
    added ``unit_direction_norm`` for ``C_GENERALIZED_SNELL``, its one
    invariant.
    """
    entries = [
        entry
        for entry in resolve_ledger().covered.values()
        if entry.kind is DeclarationKind.INVARIANT
    ]
    assert len(entries) == 14, len(entries)
    assert {entry.component for entry in entries} == set(_registry_invariants())


def test_the_composability_invariant_is_asserted_on_a_cascade_not_one_step() -> None:
    """``outgoing_count_is_the_budget`` is why a multi-element diffractive system
    is runnable at all: two planar DOEs in series give 256 then 256 rays, not
    256 x 64. A single step cannot demonstrate that -- the product and the budget
    coincide at N = 1 -- so the evidence has to be a cascade.
    """
    entry = next(
        entry
        for entry in resolve_ledger().covered.values()
        if entry.kind is DeclarationKind.INVARIANT
        and entry.anchor == "outgoing_count_is_the_budget"
    )
    cascade = [
        reference
        for reference in entry.evidence
        if "stacked" in reference or "cascade" in reference or "two_" in reference
    ]
    assert cascade, (
        f"the composability invariant cites only {entry.evidence}, none of which is a "
        "cascade. At one step the budget and the product are the same number."
    )
    for reference in entry.evidence:
        path = reference.split("::")[0]
        assert (ROOT / path).exists(), reference


# --------------------------------------------------------------------------- #
# CHE-44 — the centre-dependent blindness audit, executed
# --------------------------------------------------------------------------- #


#: Grid and carrier period. 72 = 3 x 24, so the carrier is exactly periodic and
#: the two probe positions below have IDENTICAL local neighbourhoods.
_N = 72
_PERIOD = 24
#: Near the array centre (35.5, 35.5), and one full period away from it.
_NEAR_CENTRE = (36, 36)
_OFF_CENTRE = (12, 12)


def _probe_pair(*, position: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """A reference field and the same field with one localized defect.

    The carrier is **periodic** with period 24 on a 72-sample grid, and the two
    probe positions are exactly one period apart -- so their local neighbourhoods
    are bit-for-bit identical and the only thing that differs between the two
    calls is the defect's distance from the array centre.

    That construction is the whole point, and the first version of this helper
    got it wrong in a way that produced a plausible result: a Gaussian carrier
    with a fixed-amplitude defect flagged ``power_ratio`` and
    ``relative_l2_intensity`` as centre-dependent, because at radius 34 the
    carrier has fallen to 0.17 and an intensity metric's response to an additive
    complex defect scales with the local carrier level. It was measuring the
    carrier's falloff, not the metric's window.
    """
    grid = np.arange(_N, dtype=np.float64)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    phase = 2.0 * np.pi / _PERIOD
    # Amplitude structure so the mean-subtracted metrics have variance to work
    # with, and a phase ramp so the ones that see phase have phase to see. Both
    # are periodic on _PERIOD.
    # BOTH ramp coefficients are integers, so the phase is periodic on _PERIOD in
    # each axis and a shift of one period reproduces the carrier exactly. An
    # earlier version used 0.5 on the y ramp, giving a y period of 48: the two
    # positions then differed by exactly pi, which leaves complex-field metrics
    # untouched but flips the sign of the 2 Re(ref* d) cross term an intensity
    # metric sees. That was ~10% -- far inside the factor-3 threshold, so no
    # verdict changed -- but the docstring claimed an identity the probe did not
    # have, and a probe that does not do what it says is how an audit stops being
    # one.
    reference = (1.0 + 0.4 * np.cos(phase * xx) * np.cos(phase * yy)) * np.exp(
        1j * (phase * xx + 2.0 * phase * yy)
    )

    row, col = position
    blob = np.exp(-((xx - col) ** 2 + (yy - row) ** 2) / (2.0 * 2.5**2))
    perturbed = reference + 0.35 * blob * np.exp(1j * 0.7)
    return reference, perturbed


def test_the_two_probe_positions_see_an_identical_carrier() -> None:
    """The audit's whole validity rests on this, so it is checked, not asserted.

    If the carrier differs between the two positions then a difference in a
    metric's response is the carrier's, not the metric's window -- which is
    exactly the confound that made the first version of this audit report
    ``power_ratio`` and ``relative_l2_intensity`` as centre-dependent. With both
    ramp coefficients integral the carrier is periodic on 24 samples in each axis,
    so a one-period shift reproduces it exactly.
    """
    near, _ = _probe_pair(position=_NEAR_CENTRE)
    off, _ = _probe_pair(position=_OFF_CENTRE)
    assert np.array_equal(near, off), "the reference field must not depend on the defect"

    half = 6
    dy = _NEAR_CENTRE[0] - _OFF_CENTRE[0]
    dx = _NEAR_CENTRE[1] - _OFF_CENTRE[1]
    assert (dy, dx) == (_PERIOD, _PERIOD), (dy, dx)
    a = near[
        _NEAR_CENTRE[0] - half : _NEAR_CENTRE[0] + half,
        _NEAR_CENTRE[1] - half : _NEAR_CENTRE[1] + half,
    ]
    b = near[
        _OFF_CENTRE[0] - half : _OFF_CENTRE[0] + half,
        _OFF_CENTRE[1] - half : _OFF_CENTRE[1] + half,
    ]
    assert np.allclose(a, b, rtol=0, atol=1e-12), (
        "the two neighbourhoods differ, so any response difference below is the "
        f"carrier's and not the metric's: max |delta| = {np.abs(a - b).max():.3e}"
    )


_AUDITABLE = tuple(sorted(set(METRICS) - _NEEDS_GEOMETRY))


# An intensity metric handed a complex field discards the imaginary part, which
# is exactly what its definition says it does -- the audit deliberately hands the
# same array to every metric so the spatial comparison is like for like.
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
@pytest.mark.parametrize("name", _AUDITABLE)
def test_every_metric_is_audited_for_centre_dependent_blindness(name: str) -> None:
    """CHE-44's concern, measured rather than reasoned about.

    The same defect is placed at the centre and near the corner, and the metric's
    two responses are compared. A metric that responds much less to the off-axis
    copy is *spatially windowed*, and the rule is not that it must not be -- a
    windowed metric is often the right instrument -- but that it must SAY SO, so
    a family cannot gate on it while believing it sees the whole field.

    The threshold is deliberately generous (a factor of 3). What is being caught
    is a metric that is essentially blind outside a window, not one that weights
    the centre slightly more.
    """
    definition = METRICS[name]
    on_axis_ref, on_axis = _probe_pair(position=_NEAR_CENTRE)
    off_axis_ref, off_axis = _probe_pair(position=_OFF_CENTRE)

    centre = definition(on_axis_ref, on_axis)
    edge = definition(off_axis_ref, off_axis)
    baseline = definition(on_axis_ref, on_axis_ref)

    centre_response = abs(centre - baseline)
    edge_response = abs(edge - definition(off_axis_ref, off_axis_ref))

    if centre_response == 0.0 and edge_response == 0.0:
        pytest.skip(f"{name} is blind to this defect in both positions")
    if not np.isfinite(centre_response) or not np.isfinite(edge_response):
        pytest.skip(f"{name} is undefined on this probe pair")

    windowed = edge_response < centre_response / 3.0
    declares = any(
        "OUTSIDE" in blind or "outside" in blind or "off-axis" in blind
        for blind in definition.blind_to
    )
    if windowed:
        assert declares, (
            f"{name} responds {centre_response:.3e} to a centred defect and "
            f"{edge_response:.3e} to the same defect off axis -- a factor of "
            f"{centre_response / max(edge_response, 1e-300):.1f} -- and declares no "
            "outside-the-window blindness. That is exactly CHE-44's concern."
        )


@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_the_centred_metric_is_the_one_the_audit_catches() -> None:
    """The audit has to be able to fail, so this asserts it does on the metric
    designed to be windowed.

    ``central_relative_l2_intensity`` evaluates a fraction of the array about the
    centre. It is a legitimate instrument -- a PSF core comparison should not be
    diluted by a quiet background -- and it is blind to everything outside its
    window by construction. An audit that flagged nothing would not be an audit.
    """
    definition = METRICS["central_relative_l2_intensity"]
    reference, centred = _probe_pair(position=_NEAR_CENTRE)
    _, off_axis = _probe_pair(position=_OFF_CENTRE)

    centre_response = definition(reference, centred)
    edge_response = definition(reference, off_axis)
    assert centre_response > 3.0 * edge_response, (
        f"the windowed metric should be far less sensitive off axis: "
        f"{centre_response:.3e} vs {edge_response:.3e}"
    )
    assert any("OUTSIDE THE WINDOW" in blind for blind in definition.blind_to)
    assert any("CHE-44" in blind for blind in definition.blind_to), (
        "the metric that motivated the audit should name it"
    )


@pytest.mark.parametrize("name", sorted(_NEEDS_GEOMETRY))
def test_a_geometric_metric_declares_its_radius_blindness(name: str) -> None:
    """These two cannot be called without a pitch and a radius, so the numerical
    sweep does not reach them -- and a metric with a hard radius cutoff is the
    most windowed kind there is. The declaration is what carries it."""
    blind = METRICS[name].blind_to
    assert any("OUTSIDE" in item or "outside" in item for item in blind), blind
    assert any("radius" in item for item in blind), blind


def test_at_least_one_negative_control_is_off_axis() -> None:
    """CHE-44's second half: a control battery evaluated only on axis cannot see
    an off-axis defect, so at least one control has to be off axis.

    Two are, and they are different kinds. ``B1-RAY-OFFAXIS-OPL`` is an off-axis
    family outright -- its whole subject is the object-space term that vanishes on
    axis -- and ``B2-R2W-ROUTE`` measures the splat kernel's residual GROWTH off
    axis (3.8x, whole grid against a centred window), which is the case where a
    centred metric would have understated the error rather than missed it.
    """
    families = {family.family_id: family for family in FAMILIES}

    off_axis_family = families["B1-RAY-OFFAXIS-OPL"]
    controls = {control.control_id for control in off_axis_family.negative_controls}
    assert controls, "the off-axis family declares no control"

    route = families["B2-R2W-ROUTE"]
    note = route.gate_disposition.note
    assert "off axis" in note, (
        "the route family must record the off-axis growth, not only the centred "
        "residual -- a centred metric would understate the splat kernel's error "
        f"where it is worst. Note: {note[:200]!r}"
    )
    assert "CHE-44" in note, "and it must say which concern that answers"
    # The residual metric owes the matching blindness, since it is the one a
    # caller would otherwise read as a whole-field number.
    residual = route.metric("route_field_relative_l2")
    assert any("where the error is" in blind for blind in residual.blind_to), (
        residual.blind_to
    )


def test_the_off_axis_blindness_is_declared_where_it_was_moved_to() -> None:
    """It used to be a negative control on the off-axis family, which was wrong.

    "On axis this cannot be detected" is not a broken twin -- nothing was
    mutated. It is a statement about what a metric can see, so it belongs in the
    metric's ``blind_to`` where a consumer reading the metric finds it, not in a
    control list where it would have to be reported as NOT_RUN forever.
    """
    family = next(f for f in FAMILIES if f.family_id == "B1-RAY-OFFAXIS-OPL")
    controls = {control.control_id for control in family.negative_controls}
    assert "on-axis-cannot-detect-it" not in controls

    blind = [
        item
        for metric in family.metrics
        for item in metric.blind_to
        if "on axis" in item or "on-axis" in item
    ]
    assert blind, (
        "the on-axis blindness has to be somewhere, and a metric's blind_to is the "
        "place a consumer will actually read it"
    )


# --------------------------------------------------------------------------- #
# The written statement
# --------------------------------------------------------------------------- #


def test_the_what_does_not_survive_statement_names_every_transition() -> None:
    """The prose and the executable table must agree, and the prose is the one an
    agent reads. So this checks the README against the registry -- not against
    the table -- because both could go stale together.
    """
    readme = (ROOT / "knowledge/couplers/README.md").read_text()
    assert "## What does not survive a representation change" in readme
    for coupler_id in Registry.from_package().couplers:
        assert f"`{coupler_id}`" in readme, f"{coupler_id} has no discard statement"


def test_the_statement_names_the_four_discarded_quantities_che112_lists() -> None:
    readme = (ROOT / "knowledge/couplers/README.md").read_text()
    for quantity in (
        "Per-ray identity",
        "Evanescent power",
        "The incident OPL reference",
        "Everything the Monte Carlo sample did not draw",
    ):
        assert quantity in readme, quantity
    # And the composability invariant, which is why this belongs in M2.4.
    assert "outgoing_count_is_the_budget" in readme
    assert "256" in readme, "the concrete cascade number is what makes the point"


def test_the_statement_points_at_the_executable_table_as_the_source_of_truth() -> None:
    """Prose is not evidence. The README has to say where the checked version is,
    or a reader has no way to tell which one moved."""
    readme = " ".join((ROOT / "knowledge/couplers/README.md").read_text().split())
    assert "WHAT_DOES_NOT_SURVIVE" in readme
    assert "source of truth" in readme
    assert "b2_transitions.py" in readme


def test_the_broken_twin_rule_is_stated_with_its_measured_pair() -> None:
    """A rule with no number beside it reads as a policy. The pair is the
    argument: 5.31e-16 against 1.414, and the 1.414 only appeared after the probe
    was changed to one the twins could actually damage."""
    readme = (ROOT / "knowledge/couplers/README.md").read_text()
    assert "deliberately broken twin" in readme
    assert "1.414" in readme
    assert "no-ops" in readme, (
        "the fact that the first probe made both twins inert is the part worth "
        "recording -- a round trip that cannot fail proves nothing"
    )
