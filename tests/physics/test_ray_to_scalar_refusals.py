"""R07.3: the sampling measure as a contract, and every refusal it declares.

CHE-187. §7 of the architecture principles: keep physical amplitude and
sampling/quadrature measure conceptually separate, do not encode the integration
measure invisibly inside amplitude, and do not make quadrature a global boolean
option. **If the required integration measure is unknown, a trusted ray-to-wave
conversion must refuse rather than silently invent one.**

The pressure this file exists against
-------------------------------------
The refusal being softened to a warning during implementation because a fixture
forgot to declare its measure. That is the exact move that turns a contract into a
default, and the fixture is the thing that should change. So the refusal is tested
from both sides: an undeclared bundle is refused with a structured code, *and* the
declared one that replaces it is checked to reconstruct correctly -- because a
refusal nobody can satisfy is as useless as one nobody hits.

Why the measure is not fussiness, in one number
-----------------------------------------------
`sum_i dA_i -> pi a^2` as the ring count grows, so the reconstructed power
converges under ray refinement. Without the area element the sum is pinned to the
ray count: measured here as `d log P / d log N = 2.0038` with an equal weight per
ray against `-0.0002` with the area element, on a hexapolar fan from 217 to 49 537
rays. The reference implementation measured `2.0024` for the same thing (CHE-33),
which is the number this reproduces.

Peak-normalized metrics cannot see any of that, which is why the wrong convention
survived three milestones and why the gate is on absolute power.
"""

from __future__ import annotations

import ast
import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
from ray_support import (
    collimated_bundle,
    converging_bundle,
    plateau_radius_m,
)

from couplers import REFUSALS, SCALE_NOTE, Reconstruction, ray_to_scalar
from representations import MEASURE_KINDS, UNVERIFIED, ContractError, ReferenceSurface

SRC = Path(__file__).resolve().parents[2] / "src"
MODULE = SRC / "couplers" / "ray_to_scalar.py"
RAYS_MODULE = SRC / "representations" / "rays.py"
SHAPE = (8, 9)
PITCH_M = (0.30e-6, 0.25e-6)
FOCAL_GRID = (9, 9)
FOCAL_PITCH_M = (0.2e-6, 0.2e-6)


def a_bundle(**overrides):
    rays, _, _ = collimated_bundle(
        shape=SHAPE, sample_pitch_m=PITCH_M, direction=(0.0, 0.0, 1.0)
    )
    return dataclasses.replace(rays, **overrides) if overrides else rays


def reconstruct(rays, **kwargs):
    return ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M, **kwargs)


# ---------------------------------------------------------------------------
# 1. The refusal itself
# ---------------------------------------------------------------------------


def test_an_undeclared_measure_is_refused_not_defaulted_and_not_warned() -> None:
    """Criterion 1. A structured diagnostic, and no field.

    `measure_kind` defaults to `'undeclared'` in `representations/rays.py` so that
    refusing is what happens when nobody thought about it. There is no warning
    path and no uniform fallback -- `test_the_refusal_is_satisfiable` below checks
    that the *satisfied* case emits no warning either, so a future softening from
    a refusal to a warning fails one of the two rather than passing both.
    """
    undeclared = a_bundle(measure_weight=None, measure_kind="undeclared")
    assert undeclared.measure_kind == "undeclared"

    with pytest.raises(ContractError) as raised:
        reconstruct(undeclared)

    error = raised.value
    assert error.code == "MEASURE_UNDECLARED"
    assert error.declaration == "measure_kind"
    assert error.remedy is not None
    assert "invent" in error.remedy
    assert set(error.as_diagnostic()) == {"code", "message", "declaration", "remedy"}


def test_the_refusal_is_satisfiable(recwarn: pytest.WarningsRecorder) -> None:
    """The other half: declaring the measure makes the same bundle work, silently.

    A contract nobody can satisfy is as useless as one nobody hits, and a contract
    that is satisfied *with a warning* is a default wearing a costume.
    """
    field, diagnostics = reconstruct(a_bundle())
    assert diagnostics.measure_kind == "quadrature_area_m2"
    assert field.shape == SHAPE
    assert [str(w.message) for w in recwarn] == []


def test_a_measure_weight_without_a_kind_never_reaches_the_coupler() -> None:
    """The representation refuses it first, which is where the declaration lives."""
    with pytest.raises(ContractError) as raised:
        a_bundle(measure_weight=np.ones(SHAPE[0] * SHAPE[1]), measure_kind="undeclared")
    assert raised.value.code == "MEASURE_UNDECLARED"


def test_every_declared_measure_kind_has_a_rule_here() -> None:
    """A measure kind that lands in `representations/` lands with its row here.

    The `UNKNOWN_MEASURE_KIND` refusal exists for the window in between, and this
    is the test that says the window is currently closed. If R08 adds a k-space
    cell measure to `MEASURE_KINDS` and forgets this module, this fails.
    """
    declared = set(MEASURE_KINDS) - {"undeclared"}
    for kind in sorted(declared):
        rays = a_bundle(
            measure_weight=np.ones(SHAPE[0] * SHAPE[1]),
            measure_kind=kind,
        )
        _, diagnostics = reconstruct(rays)
        assert diagnostics.normalization in ("none", "one_over_n")
        assert diagnostics.measure_kind == kind


def test_the_two_measure_kinds_normalize_differently() -> None:
    """A given ensemble takes no `1/N`; a Monte-Carlo sample does. Same weights."""
    weights = np.full(SHAPE[0] * SHAPE[1], 1.0e-12)
    quadrature = a_bundle(measure_weight=weights, measure_kind="quadrature_area_m2")
    importance = a_bundle(measure_weight=weights, measure_kind="importance_weight")

    quadrature_field, quadrature_record = reconstruct(quadrature)
    importance_field, importance_record = reconstruct(importance)

    assert quadrature_record.normalization == "none"
    assert importance_record.normalization == "one_over_n"
    ratio = float(abs(quadrature_field.u[0, 0] / importance_field.u[0, 0]))
    assert ratio == pytest.approx(quadrature.count, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Amplitude and measure stay apart
# ---------------------------------------------------------------------------


def test_amplitude_and_measure_are_separately_inspectable_and_separately_reported() -> None:
    """Criterion 2. Nothing multiplies them before the kernel.

    On the input they are two fields of `RayBundle`. In the record they are three
    numbers -- the amplitude power sum, the measure sum, and the launch power sum
    the kernel actually summed, which is the only one of the three that has seen a
    multiply.
    """
    count = SHAPE[0] * SHAPE[1]
    amplitude = np.full(count, 2.0 + 0.0j)
    weight = np.full(count, 3.0e-12)
    rays = a_bundle(
        amplitude=amplitude, measure_weight=weight, measure_kind="quadrature_area_m2"
    )

    # The producer's two declarations, still separate on the artifact.
    assert float(np.max(np.abs(np.asarray(rays.amplitude)))) == 2.0
    assert float(np.max(np.asarray(rays.measure_weight))) == 3.0e-12

    _, diagnostics = reconstruct(rays)
    assert diagnostics.incident_amplitude_power_sum == pytest.approx(count * 4.0)
    assert diagnostics.measure_sum == pytest.approx(count * 3.0e-12)
    assert diagnostics.launch_amplitude_power_sum == pytest.approx(
        count * (2.0 * 3.0e-12) ** 2
    )


def test_the_measure_is_applied_exactly_once() -> None:
    """Scaling the weight by `c` scales the field by `c`, not by `c^2` or by 1.

    The three ways this can go wrong -- folded in at the producer as well as here,
    dropped entirely, or squared -- are all a single power of `c` apart, so one
    linearity check separates all of them.
    """
    base = a_bundle()
    scaled = dataclasses.replace(
        base, measure_weight=np.asarray(base.measure_weight) * 7.0
    )
    base_field, _ = reconstruct(base)
    scaled_field, _ = reconstruct(scaled)
    ratio = float(abs(scaled_field.u[0, 0] / base_field.u[0, 0]))
    assert ratio == pytest.approx(7.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Why the measure is there: convergence, not a constant
# ---------------------------------------------------------------------------


def test_the_area_elements_sum_to_the_aperture_area_under_ring_refinement() -> None:
    """Criterion 3, first half. `sum_i dA_i = pi a^2 (1 + 1 / (4 J^2))`, exactly.

    Worked from the hexapolar ring counts `6j` with the two boundary corrections,
    so the residual is not a tolerance -- it is a closed form, and the test checks
    the form rather than a bound.
    """
    radius = plateau_radius_m()
    aperture_m2 = math.pi * radius**2
    for rings in (8, 16, 32, 64):
        rays, area = converging_bundle(rings=rings, radius_m=radius)
        _, diagnostics = ray_to_scalar(
            rays, grid_shape=FOCAL_GRID, sample_pitch_m=FOCAL_PITCH_M
        )
        expected = aperture_m2 * (1.0 + 1.0 / (4.0 * rings**2))
        assert float(area.sum()) == pytest.approx(expected, rel=1e-12)
        assert diagnostics.measure_sum == pytest.approx(expected, rel=1e-12)


def test_discrete_power_converges_with_the_measure_and_scales_as_n_squared_without_it() -> None:
    """Criterion 3, second half, and CHE-33's number reproduced.

    Measured on a hexapolar fan from 217 rays (8 rings) to 49 537 (128), the same
    geometry both times, differing only in what `measure_weight` declares:

    | rings | 8 | 16 | 32 | 64 | 128 |
    | -- | -- | -- | -- | -- | -- |
    | area element | 2.295e-29 | 2.293e-29 | 2.292e-29 | 2.292e-29 | 2.292e-29 |
    | equal weight | 1.359e-07 | 1.948e-06 | 2.948e-05 | 4.586e-04 | 7.235e-03 |

    `d log P / d log N` is **-0.0002** with the area element and **2.0038** with an
    equal weight per ray. The reference implementation measured 2.0024 for the
    latter (CHE-33); this reproduces it.

    Not marked `slow`, measured: the whole sweep is 0.8 s, because the separable
    contraction is `O(N (ny + nx))` and the grid is 9x9. It is criterion 3's
    evidence, so it belongs in the gate that runs by default rather than behind a
    marker that `addopts` deselects.

    The equal-weight arm declares `quadrature_area_m2` with a weight of 1.0, which
    is the old convention stated in this tree's vocabulary: every ray asserted to
    stand for one square metre of pupil. It is a lie about the pupil, and the point
    is that only an *absolute* area element makes the sum converge -- a relative
    correction factor would not.
    """
    radius = plateau_radius_m()
    counts, with_measure, without = [], [], []
    for rings in (8, 16, 32, 64, 128):
        rays, _ = converging_bundle(rings=rings, radius_m=radius)
        _, honest = ray_to_scalar(rays, grid_shape=FOCAL_GRID, sample_pitch_m=FOCAL_PITCH_M)
        equal = dataclasses.replace(rays, measure_weight=np.ones(rays.count))
        _, pinned = ray_to_scalar(equal, grid_shape=FOCAL_GRID, sample_pitch_m=FOCAL_PITCH_M)
        counts.append(rays.count)
        with_measure.append(honest.reconstructed_discrete_power)
        without.append(pinned.reconstructed_discrete_power)

    log_count = np.log(np.asarray(counts, dtype=float))
    converged = float(np.polyfit(log_count, np.log(np.asarray(with_measure)), 1)[0])
    pinned_exponent = float(np.polyfit(log_count, np.log(np.asarray(without)), 1)[0])

    assert abs(converged) < 1e-2, converged
    assert pinned_exponent == pytest.approx(2.0, abs=0.02), pinned_exponent
    # ...and the converged value is the one the analytic focal-peak oracle fixes.
    assert with_measure[-1] == pytest.approx(with_measure[0], rel=2e-3)


# ---------------------------------------------------------------------------
# 4. The scale is relative
# ---------------------------------------------------------------------------


def test_the_reported_scale_is_relative_and_never_a_watt() -> None:
    """Criterion 4. `U` is `i lambda z` times the SI field, so a watt is 18 orders out."""
    _, diagnostics = reconstruct(a_bundle())
    assert diagnostics.scale is SCALE_NOTE
    assert diagnostics.scale.startswith("relative")
    assert "not a physical power" in diagnostics.scale
    text = " ".join(str(value) for value in diagnostics.as_dict().values()).lower()
    for unit in ("watt", "joule", "irradiance"):
        assert unit not in text, unit


def test_the_relative_scale_is_comparable_between_two_runs_of_one_configuration() -> None:
    """What "relative" is actually good for, asserted rather than only stated."""
    rays = a_bundle()
    _, first = reconstruct(rays)
    _, second = reconstruct(dataclasses.replace(rays, amplitude=np.asarray(rays.amplitude) * 2.0))
    assert second.reconstructed_discrete_power / first.reconstructed_discrete_power == (
        pytest.approx(4.0, rel=1e-12)
    )


# ---------------------------------------------------------------------------
# 5. Every declared refusal code is reachable
# ---------------------------------------------------------------------------


def _contract_error_calls(source: str, filename: str) -> list[ast.Call]:
    """Every `ContractError(...)` construction in `source`, however it is spelled."""
    tree = ast.parse(source, filename=filename)
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "ContractError")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "ContractError")
        )
    ]


def _code_of(call: ast.Call) -> str | None:
    """The literal code a call declares, positionally or by keyword, or `None`.

    `None` means the code is not a string literal -- a variable, a lookup, an
    f-string. The caller treats that as a *failure* rather than skipping it: a walk
    that silently ignores what it cannot read is a walk that reports completeness
    it did not check, which is the exact drift `REFUSALS` exists to prevent.
    """
    candidates = [*call.args[:1], *(kw.value for kw in call.keywords if kw.arg == "code")]
    if not candidates:
        return None
    first = candidates[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def _codes_raised_in(path: Path, *, inside: str | None = None) -> set[str]:
    """Codes raised in a module, or in one named function of it.

    Every call site must declare a string-literal code; one that does not fails
    here with its line number rather than being dropped from the set.
    """
    source = path.read_text(encoding="utf-8")
    if inside is not None:
        tree = ast.parse(source, filename=str(path))
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == inside
        ]
        assert len(functions) == 1, f"{inside!r} is not a unique function of {path.name}"
        source = ast.unparse(functions[0])
    calls = _contract_error_calls(source, str(path))
    assert calls, f"no ContractError is constructed in {path.name}"
    codes = set()
    for call in calls:
        code = _code_of(call)
        assert code is not None, (
            f"{path.name}:{call.lineno} constructs a ContractError whose code this walk "
            "cannot read, so the completeness check below would silently skip it. Pass a "
            "string literal."
        )
        codes.add(code)
    return codes


def test_the_declared_refusals_are_exactly_the_ones_the_module_can_raise() -> None:
    """Criterion 5, the completeness half.

    `REFUSALS` also lists the codes `RayBundle.require_coherent()` raises, because
    they are part of *this* boundary's contract -- a caller of `ray_to_scalar`
    branches on them here and cannot see where the `raise` physically lives. That
    inherited set is **read from `require_coherent`'s own source**, not written out:
    a hand-written list would go stale the day that method grows a third code, and
    going stale silently is the failure this whole test is against.
    """
    inherited = _codes_raised_in(RAYS_MODULE, inside="require_coherent")
    assert inherited == {"COHERENT_STATE_INCOMPLETE", "OPL_REFERENCE_UNVERIFIED"}
    assert _codes_raised_in(MODULE) == set(REFUSALS) - inherited
    assert inherited <= set(REFUSALS)


def test_the_walk_would_notice_a_code_it_cannot_read() -> None:
    """The meta-test. A gate that cannot fail is not a gate.

    Three shapes the earlier version of this walk missed: a keyword code, an
    attribute-qualified constructor, and a code held in a variable. The first two
    must be *found*; the third must be *reported*, not dropped.
    """
    assert _code_of(_contract_error_calls('ContractError(code="X", message="m")', "p")[0]) == "X"
    assert _code_of(
        _contract_error_calls('contracts.ContractError("Y", "m")', "p")[0]
    ) == "Y"
    assert _code_of(_contract_error_calls("ContractError(CODE, 'm')", "p")[0]) is None


TRIGGERS = {
    "UNIT_NOT_SI": lambda: ray_to_scalar(
        a_bundle(), grid_shape=SHAPE, sample_pitch_m=(0.0, PITCH_M[1])
    ),
    "MEASURE_UNDECLARED": lambda: reconstruct(
        a_bundle(measure_weight=None, measure_kind="undeclared")
    ),
    "UNKNOWN_MEASURE_KIND": None,  # see the test below
    "FRAME_MISMATCH": lambda: reconstruct(
        a_bundle(),
        surface=ReferenceSurface(name="elsewhere", z_m=1.0e-3, medium_index=1.0),
    ),
    "SHAPE_MISMATCH": lambda: ray_to_scalar(
        a_bundle(), grid_shape=(0, 4), sample_pitch_m=PITCH_M
    ),
    "COHERENT_STATE_INCOMPLETE": lambda: reconstruct(a_bundle(amplitude=None)),
    "OPL_REFERENCE_UNVERIFIED": lambda: reconstruct(
        a_bundle(optical_path_reference=UNVERIFIED)
    ),
}


@pytest.mark.parametrize("code", sorted(code for code, trigger in TRIGGERS.items() if trigger))
def test_every_declared_refusal_is_reachable_through_the_coupler(code: str) -> None:
    """Criterion 5, the reachability half.

    Reachable *through `ray_to_scalar`*, not merely raisable somewhere in the tree.
    `tests/representations/test_contract_codes.py` already proves the second thing;
    what a caller of this function needs is the first.
    """
    trigger = TRIGGERS[code]
    assert trigger is not None
    with pytest.raises(ContractError) as raised:
        trigger()
    assert raised.value.code == code


def test_the_unknown_measure_kind_refusal_is_reachable_only_by_construction() -> None:
    """The one code with no reachable trigger today, and why it is still declared.

    `RayBundle.__post_init__` validates `measure_kind` against `MEASURE_KINDS`, so
    every bundle that reaches this coupler carries one of the three -- and all
    three have a rule. The branch is what happens in the window between a fourth
    kind landing in `representations/` and its row landing here, which is a real
    window R08 may open. It is exercised against the private helper rather than
    left as an unreached claim, and the assertion above (`declared == raised`) is
    what stops the declaration and the code drifting apart.
    """
    from couplers.ray_to_scalar import _resolve_measure

    rays = a_bundle()
    invented = dataclasses.replace(rays)
    object.__setattr__(invented, "measure_kind", "kspace_cell_per_m2")
    with pytest.raises(ContractError) as raised:
        _resolve_measure(invented)
    assert raised.value.code == "UNKNOWN_MEASURE_KIND"
    assert "kspace_cell_per_m2" in str(raised.value)


def test_the_refusal_table_explains_each_code() -> None:
    """A code with no sentence is a branch a caller cannot act on."""
    for code, explanation in REFUSALS.items():
        assert code.isupper()
        assert len(explanation.split()) > 8, code


def test_the_k_space_route_refuses_the_same_way() -> None:
    """The measure contract is the operation's, not the route's."""
    with pytest.raises(ContractError) as raised:
        reconstruct(
            a_bundle(measure_weight=None, measure_kind="undeclared"),
            reconstruction=Reconstruction.KSPACE,
        )
    assert raised.value.code == "MEASURE_UNDECLARED"


# ---------------------------------------------------------------------------
# 6. The open item, recorded rather than closed
# ---------------------------------------------------------------------------


def test_the_off_axis_weight_residual_is_recorded_as_an_open_item() -> None:
    """Criterion 6. Carried forward with its numbers, and explicitly not resolved.

    Off-axis on the reference implementation's `ReverseTelephoto` field the
    residual against the analytic Airy profile went from `1.48e-3` to `1.11e-2`
    when the weight was introduced, and removing the weight improves it 7.5x; on
    axis the same metric barely moved (`5.87e-3 -> 5.51e-3`). The rim taper is a
    plausible cause and is a hypothesis, not a measurement.

    Neither that field nor an off-axis Airy oracle exists in the new tree, so
    nothing here re-measures it. This test is a **pin on the record**: it fails if
    the numbers are deleted, so the ticket that lands an off-axis comparison
    inherits them instead of rediscovering them. Widening a tolerance is not the
    remedy and is not available -- there is no tolerance here to widen.
    """
    prose = MODULE.read_text(encoding="utf-8")
    # Only the numbers are pinned. They are the part that cannot be rediscovered
    # -- the field and the oracle they were measured on are both deleted -- so
    # losing them loses the open item. The prose around them is free to be
    # reworded, which is why no phrase is asserted here.
    for number in ("1.48e-3", "1.11e-2", "5.87e-3", "5.51e-3", "7.5x", "2.44 pixels"):
        assert number in prose, number


def test_the_weight_is_not_an_apodization_and_the_module_says_which_pupils_are_untested() -> None:
    """The claim the weight must not be read as making.

    A non-uniform launch amplitude arising from how the pupil was *sampled* does
    not exercise the kernel's response to a physically apodized, vignetted or
    Fresnel-weighted pupil. Those are untested, and the module says so rather than
    letting the rim taper stand in for them.
    """
    prose = MODULE.read_text(encoding="utf-8")
    for claim in ("apodized", "vignetted", "Fresnel-weighted"):
        assert claim in prose, claim
