"""`RayBundle`: the coherent contract, the declared measure, and one ray type.

CHE-175 (R02.3). The construction-time refusals live in
`test_contract_codes.py`, which enumerates them against `CONTRACT_CODES`. This
file is the behaviour the acceptance criteria name: what `require_coherent()`
returns and what it says when it cannot, what the three measure kinds mean, and
the two properties that are easy to break silently -- that nothing multiplies the
measure into the amplitude, and that the direction tolerance follows the dtype.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import representations
from numerics import DType
from representations import (
    MEASURE_KINDS,
    UNVERIFIED,
    ContractError,
    Frame,
    RayBundle,
    ReferenceSurface,
    direction_norm_tolerance,
)

SURFACE = ReferenceSurface(name="exit_pupil", z_m=-3.2e-3, medium_index=1.0)
WAVELENGTH_M = 550e-9
COUNT = 5


def _bundle(**overrides: Any) -> RayBundle:
    fields: dict[str, Any] = {
        "positions_m": np.linspace(-1e-3, 1e-3, COUNT * 3).reshape(COUNT, 3),
        "directions": np.tile(np.array([0.0, 0.0, 1.0]), (COUNT, 1)),
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": SURFACE,
    }
    fields.update(overrides)
    return RayBundle(**fields)


def _coherent(**overrides: Any) -> RayBundle:
    fields: dict[str, Any] = {
        "amplitude": np.full(COUNT, 0.5 + 0.25j),
        "optical_path_m": np.linspace(0.0, 1e-6, COUNT),
        "optical_path_reference": "chief ray at the exit pupil",
    }
    fields.update(overrides)
    return _bundle(**fields)


# --- geometry and the declared boundary ---


def test_a_geometric_bundle_is_complete_without_coherent_state() -> None:
    """The absence of the optional fields is what makes the type useful.

    A ray trace that produced no amplitude and no verified path still produces a
    valid bundle; it just does not pass the coherent gate.
    """
    bundle = _bundle()
    assert bundle.count == COUNT
    assert bundle.amplitude is None
    assert bundle.optical_path_m is None
    assert bundle.measure_kind == "undeclared"
    assert bundle.frame == Frame()
    assert bundle.reference_surface is SURFACE


def test_the_wavenumber_is_derived_not_declared() -> None:
    assert _bundle().wavenumber == pytest.approx(2.0 * np.pi / WAVELENGTH_M)


def test_the_state_is_read_off_the_geometry() -> None:
    """Observed, never caller-declared, so it cannot contradict the data."""
    bundle = _bundle(
        positions_m=np.zeros((COUNT, 3), dtype=np.float32),
        directions=np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (COUNT, 1)),
    )
    assert bundle.state.dtype is DType.FLOAT32
    assert bundle.xp is np


def test_a_bundle_is_frozen() -> None:
    bundle = _bundle()
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.wavelength_m = 1e-6  # type: ignore[misc]


def test_a_list_input_takes_the_historical_host_default() -> None:
    """An input with no representation of its own becomes host float64, as before."""
    bundle = _bundle(
        positions_m=[[0.0, 0.0, 0.0], [1e-3, 0.0, 0.0]],
        directions=[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
    )
    assert bundle.state.dtype is DType.FLOAT64
    assert bundle.count == 2


def test_a_float32_bundle_stays_float32() -> None:
    """The CHE-61 finding: intake preserves the representation it was handed.

    A single `np.asarray(value, dtype=np.float64)` at the boundary would move the
    data to the host, change its dtype and break any autograd graph, so a float32
    GPU artifact could not exist whatever the producing solver supported.
    """
    directions = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (COUNT, 1))
    bundle = _bundle(
        positions_m=np.zeros((COUNT, 3), dtype=np.float32),
        directions=directions,
        amplitude=np.ones(COUNT, dtype=np.complex64),
    )
    assert bundle.state.dtype is DType.FLOAT32
    assert bundle.amplitude is not None
    assert bundle.amplitude.dtype == np.complex64


# --- the direction-norm tolerance, as a function of dtype ---


def test_the_direction_tolerance_reduces_to_the_historical_constant_at_float64() -> None:
    assert direction_norm_tolerance(DType.FLOAT64) == 1e-9


def test_the_direction_tolerance_widens_for_float32() -> None:
    """`64 * eps`, derived rather than picked."""
    assert direction_norm_tolerance(DType.FLOAT32) == pytest.approx(
        64.0 * float(np.finfo(np.float32).eps)
    )
    assert direction_norm_tolerance(DType.FLOAT32) > direction_norm_tolerance(DType.FLOAT64)


def test_the_same_deviation_is_refused_at_float64_and_accepted_at_float32() -> None:
    """The whole point of making the bound a function of the dtype.

    1e-7 of norm error is a modelling mistake in float64 and is below the noise
    floor in float32 -- casting an exactly normalized float64 direction to float32
    already perturbs `|d|` by about one float32 epsilon before the norm is
    computed. A single absolute bound is either vacuous at float32 or
    unsatisfiable at it.
    """
    off_unit = np.tile(np.array([0.0, 0.0, 1.0 + 1e-7]), (COUNT, 1))
    with pytest.raises(ContractError) as caught:
        _bundle(directions=off_unit)
    assert caught.value.code == "NON_UNIT_DIRECTION"

    bundle = _bundle(
        positions_m=np.zeros((COUNT, 3), dtype=np.float32),
        directions=off_unit.astype(np.float32),
    )
    assert bundle.state.dtype is DType.FLOAT32


def test_a_genuinely_unnormalized_float32_bundle_is_still_refused() -> None:
    """Widening the bound for float32 is not the same as removing it."""
    with pytest.raises(ContractError, match="NON_UNIT_DIRECTION"):
        _bundle(
            positions_m=np.zeros((COUNT, 3), dtype=np.float32),
            directions=np.tile(np.array([0.0, 0.0, 1.001], dtype=np.float32), (COUNT, 1)),
        )


# --- acceptance criterion 1: require_coherent ---


def test_require_coherent_returns_the_two_coherent_fields() -> None:
    bundle = _coherent()
    amplitude, optical_path_m = bundle.require_coherent()
    assert amplitude is bundle.amplitude
    assert optical_path_m is bundle.optical_path_m


def test_require_coherent_names_everything_that_is_missing() -> None:
    """Not the first missing declaration -- all of them.

    A caller that fixes one, re-runs, and discovers the next is a caller who
    starts guessing at the third round trip.
    """
    with pytest.raises(ContractError) as caught:
        _bundle().require_coherent()
    message = str(caught.value)
    assert caught.value.code == "COHERENT_STATE_INCOMPLETE"
    assert "amplitude" in message
    assert "optical_path_m" in message
    assert caught.value.declaration == "amplitude, optical_path_m"


def test_require_coherent_names_only_what_is_actually_missing() -> None:
    with pytest.raises(ContractError) as caught:
        _bundle(
            optical_path_m=np.zeros(COUNT), optical_path_reference="chief ray"
        ).require_coherent()
    assert caught.value.declaration == "amplitude"


def test_a_measure_weight_is_named_as_not_being_an_amplitude() -> None:
    """The distinction the reference implementation spent a whole code on.

    Optiland supplies a real `intensity` weight. Whether it is a power (so
    `a = sqrt(w)`), a photon count, or already an amplitude is a modelling
    decision, and this type does not make it for the caller.
    """
    with pytest.raises(ContractError) as caught:
        _bundle(measure_weight=np.ones(COUNT), measure_kind="quadrature_area_m2").require_coherent()
    assert "not an amplitude" in str(caught.value)
    assert "quadrature_area_m2" in str(caught.value)


def test_an_unverified_optical_path_is_carried_but_not_readable_as_physics() -> None:
    """Carrying an unverified quantity is fine; reading it as physics is not.

    Optiland's `opd_native` has an unverified sign, and a wrong OPL *sign*
    conjugates the wavefront -- a converging beam reconstructs as a diverging one,
    which no intensity check distinguishes.
    """
    bundle = _coherent(optical_path_reference=UNVERIFIED)
    assert bundle.optical_path_m is not None  # carried
    with pytest.raises(ContractError) as caught:
        bundle.require_coherent()
    assert caught.value.code == "OPL_REFERENCE_UNVERIFIED"


def test_a_real_amplitude_is_widened_at_the_same_precision() -> None:
    """`sqrt(w)` of a declared power weight is a phase-free amplitude, not an error.

    float32 in gives complex64, never complex128: widening the precision would
    fabricate accuracy the producer never had.
    """
    bundle = _coherent(amplitude=np.ones(COUNT, dtype=np.float32))
    assert bundle.amplitude is not None
    assert bundle.amplitude.dtype == np.complex64
    bundle64 = _coherent(amplitude=np.ones(COUNT, dtype=np.float64))
    assert bundle64.amplitude is not None
    assert bundle64.amplitude.dtype == np.complex128


# --- acceptance criteria 2 and 3: the measure ---


def test_the_three_measure_kinds_are_the_declared_vocabulary() -> None:
    assert MEASURE_KINDS == ("quadrature_area_m2", "importance_weight", "undeclared")


def test_undeclared_is_the_default_so_refusing_is_what_happens_by_default() -> None:
    """AC 2: the third value is what makes R07's refusal possible.

    `undeclared` is not a synonym for uniform. A coupler that treats it as uniform
    has invented a quadrature -- which is exactly the 3.84e-3 residual CHE-38
    measured.
    """
    assert _bundle().measure_kind == "undeclared"
    assert _bundle().measure_weight is None


@pytest.mark.parametrize("kind", ["quadrature_area_m2", "importance_weight"])
def test_a_declared_measure_is_carried_and_inspectable(kind: str) -> None:
    weights = np.linspace(1e-9, 2e-9, COUNT)
    bundle = _bundle(measure_weight=weights, measure_kind=kind)
    assert bundle.measure_kind == kind
    np.testing.assert_array_equal(bundle.measure_weight, weights)


def test_a_consumer_can_branch_on_the_measure_kind() -> None:
    """What R07 does: refuse the undeclared case, accept the declared ones."""

    def would_reconstruct(bundle: RayBundle) -> bool:
        return bundle.measure_kind != "undeclared"

    assert not would_reconstruct(_bundle())
    assert would_reconstruct(
        _bundle(measure_weight=np.ones(COUNT), measure_kind="importance_weight")
    )


def test_a_weight_without_a_kind_is_refused() -> None:
    """An area element and an importance weight differ by the aperture area."""
    with pytest.raises(ContractError) as caught:
        _bundle(measure_weight=np.ones(COUNT))
    assert caught.value.code == "MEASURE_UNDECLARED"


def test_a_kind_without_a_weight_is_refused() -> None:
    with pytest.raises(ContractError) as caught:
        _bundle(measure_kind="quadrature_area_m2")
    assert caught.value.declaration == "measure_weight"


def test_a_negative_measure_weight_is_refused() -> None:
    with pytest.raises(ContractError, match="UNIT_NOT_SI"):
        _bundle(
            measure_weight=np.array([-1.0, 1.0, 1.0, 1.0, 1.0]),
            measure_kind="quadrature_area_m2",
        )


def test_construction_does_not_fold_the_measure_into_the_amplitude() -> None:
    """AC 3, and the reason the two fields exist separately.

    `a_i` is the light and `w_i` is how the pupil was sampled. Once they are
    multiplied, no consumer can recover either, and a second application of the
    measure -- by a coupler that assumes it has not been applied -- is invisible.
    """
    amplitude = np.full(COUNT, 0.5 + 0.25j)
    weights = np.full(COUNT, 3.0e-9)
    bundle = _bundle(
        amplitude=amplitude.copy(),
        optical_path_m=np.zeros(COUNT),
        optical_path_reference="chief ray",
        measure_weight=weights.copy(),
        measure_kind="quadrature_area_m2",
    )
    np.testing.assert_array_equal(bundle.amplitude, amplitude)
    np.testing.assert_array_equal(bundle.measure_weight, weights)
    returned_amplitude, _ = bundle.require_coherent()
    np.testing.assert_array_equal(returned_amplitude, amplitude)


# --- acceptance criterion 5: exactly one ray representation ---


def test_the_module_defines_exactly_one_class() -> None:
    import ast
    from pathlib import Path

    source = Path(representations.rays.__file__).read_text(encoding="utf-8")
    classes = [n.name for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)]
    assert classes == ["RayBundle"]


@pytest.mark.parametrize(
    "banned",
    [
        "CoherentRayBatch",
        "WavefrontSamples",
        "GeometricRayBundle",
        "CoherentRayBundle",
        "RayBundleBase",
        "AbstractRayBundle",
        "TrackedRayBundle",
        "RayBatch",
    ],
)
def test_the_collapsed_ray_types_did_not_come_back(banned: str) -> None:
    """The seven names R00.2 dispositioned, asserted rather than trusted.

    `WavefrontSamples` had zero production consumers
    (`docs/rewrite/reference_inventory.md` §1.1); the rest are named in
    `docs/architecture_principles.md`'s audit list. A bundle capable of ray-to-wave
    conversion is not another class -- it is a bundle that passes
    `require_coherent()`.
    """
    assert not hasattr(representations, banned)
