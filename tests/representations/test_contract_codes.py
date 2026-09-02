"""Every declared contract code is reachable, and every raise declares its code.

CHE-175 (R02.3). The same enumeration `tests/numerics/test_refusals.py` runs over
`REFUSAL_CODES`, for the same reason: a code in `CONTRACT_CODES` that nothing can
raise is a claim about a failure path that does not exist. In R02.1 this test
*removed* a code -- `NO_ADMISSIBLE_NAMESPACE` turned out unreachable once an
earlier check refused the state that would have produced it.

It matters more here than in `numerics/`, because these are the strings a coupler
branches on. R07 is required to return an explicit diagnostic rather than an
invented field; a declared code with no trigger is a branch it would write and
never take.

Two stub arrays appear below. They are doubles for `numerics.arrays`'
*classifier*, not for a backend: `namespace_of` reads
`type(value).__module__`, so a class whose module is spelled `torch` is
classified as torch without torch being installed or imported. That keeps the
default suite free of a multi-second framework import for two refusal paths, and
`tests/numerics/` is where the classifier itself is tested against real arrays.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from couplers import ray_to_scalar
from measurements import NORMALIZATION_DECLARATIONS, PsfResult
from representations import (
    CONTRACT_CODES,
    ContractError,
    Frame,
    RayBundle,
    ReferenceSurface,
    ScalarField,
)

SURFACE = ReferenceSurface(name="exit_pupil", z_m=0.0, medium_index=1.0)
WAVELENGTH_M = 550e-9


def _positions(count: int = 4) -> np.ndarray[Any, Any]:
    return np.zeros((count, 3), dtype=np.float64)


def _directions(count: int = 4) -> np.ndarray[Any, Any]:
    return np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))


def _bundle(**overrides: Any) -> RayBundle:
    fields: dict[str, Any] = {
        "positions_m": _positions(),
        "directions": _directions(),
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": SURFACE,
    }
    fields.update(overrides)
    return RayBundle(**fields)


def _field(**overrides: Any) -> ScalarField:
    fields: dict[str, Any] = {
        "u": np.ones((4, 6), dtype=np.complex128),
        "sample_pitch_m": (1e-6, 1e-6),
        "wavelength_m": WAVELENGTH_M,
        "reference_surface": SURFACE,
    }
    fields.update(overrides)
    return ScalarField(**fields)


class _TorchLike:
    """A buffer the namespace classifier reads as torch. See the module docstring."""

    dtype = np.dtype("float64")
    shape = (4,)
    ndim = 1


_TorchLike.__module__ = "torch"


class _JaxDevice:
    platform = "gpu"
    id = 0


class _JaxLikeOnGpu:
    """A buffer the classifier reads as JAX resident on `cuda:0`."""

    dtype = np.dtype("complex128")
    shape = (4,)
    ndim = 1

    def devices(self) -> list[_JaxDevice]:
        return [_JaxDevice()]


_JaxLikeOnGpu.__module__ = "jax"


#: One trigger per declared code. Each is the smallest construction that reaches
#: exactly that branch.
TRIGGERS: dict[str, Callable[[], object]] = {
    "MISSING_DECLARATION": lambda: _bundle(optical_path_m=np.zeros(4), optical_path_reference=None),
    "UNIT_NOT_SI": lambda: _bundle(wavelength_m=0.0),
    "NON_FINITE": lambda: _bundle(positions_m=np.full((4, 3), np.nan)),
    "FRAME_MISMATCH": lambda: Frame(handedness="left-handed"),
    "SHAPE_MISMATCH": lambda: _bundle(positions_m=np.zeros((4, 2))),
    "NON_UNIT_DIRECTION": lambda: _bundle(directions=np.tile([0.0, 0.0, 2.0], (4, 1))),
    "EMPTY_ENSEMBLE": lambda: _bundle(positions_m=_positions(0), directions=_directions(0)),
    "DTYPE_KIND_MISMATCH": lambda: _field(u=np.ones((4, 6), dtype=np.float64)),
    "REPRESENTATION_INCONSISTENT": lambda: _bundle(amplitude=_JaxLikeOnGpu()),
    "PHASOR_MISMATCH": lambda: _bundle(phasor="exp(+i omega t)"),
    "COHERENT_STATE_INCOMPLETE": lambda: _bundle().require_coherent(),
    "OPL_REFERENCE_UNVERIFIED": lambda: _bundle(
        amplitude=np.ones(4, dtype=np.complex128),
        optical_path_m=np.zeros(4),
        optical_path_reference="unverified",
    ).require_coherent(),
    "MEASURE_UNDECLARED": lambda: _bundle(measure_weight=np.ones(4)),
    "UNKNOWN_MEASURE_KIND": lambda: _bundle(measure_weight=np.ones(4), measure_kind="equal_weight"),
    "UNKNOWN_VALIDITY_FLAG": lambda: _field(validity=frozenset({"probably_fine"})),
    "PAD_STATE_UNKNOWN": lambda: _field(padded=True, pad_width=0),
    # The one code whose only raiser is a *consumer* rather than a representation's
    # own `__post_init__`: whether a mode's constant phase can be represented
    # depends on the compute precision the consumer chose, so nothing here can
    # decide it at construction (CHE-188). Its trigger is a one-ray bundle whose
    # 0.5 m optical path over a 1e-4 axial cosine costs 1.5 rad of float32 phase
    # against a 0.01 rad budget. It lives in this file because this file *is* the
    # enumeration; the physics is in `tests/physics/test_grazing_phase_floor.py`.
    "GRAZING_PHASE_UNREPRESENTABLE": lambda: ray_to_scalar(
        _bundle(
            positions_m=np.array([[0.5, 0.0, 0.0]], dtype=np.float32),
            directions=np.array([[1.0 - 5e-9, 0.0, 1e-4]], dtype=np.float32),
            amplitude=np.ones(1, dtype=np.complex64),
            optical_path_m=np.array([0.5], dtype=np.float32),
            optical_path_reference="the plane z = 0",
            measure_weight=np.ones(1, dtype=np.float32),
            measure_kind="quadrature_area_m2",
        ),
        grid_shape=(4, 4),
        sample_pitch_m=(0.25e-6, 0.25e-6),
    ),
    # The second consumer-owned code (CHE-197). Intensity is not a representation
    # -- `ScalarField` holds an *amplitude*, and refuses a real array precisely so
    # that `|U|` cannot be read as `U` -- so no `__post_init__` here can enforce
    # non-negativity. The measurement that derives the observable does, and this is
    # the smaller half of what the retired `C_FIELD_TO_PSF` entry declared.
    "NEGATIVE_INTENSITY": lambda: PsfResult(
        intensity=np.array([[1.0, -1e-30], [1.0, 1.0]]),
        sample_pitch_m=(1e-6, 1e-6),
        wavelength_m=0.55e-6,
        normalization="raw",
        normalization_declaration=NORMALIZATION_DECLARATIONS["raw"],
        scale_factor=1.0,
        raw_peak_intensity=1.0,
        raw_window_energy=1.0,
        peak_index=(0, 0),
        peak_position_m=(0.0, 0.0),
        border_energy_fraction=1.0,
    ),
}


@pytest.mark.parametrize("code", sorted(TRIGGERS))
def test_the_contract_error_carries_its_code(code: str) -> None:
    with pytest.raises(ContractError) as caught:
        TRIGGERS[code]()
    assert caught.value.code == code, (
        f"the trigger for {code} raised {caught.value.code} instead: {caught.value}"
    )
    assert str(caught.value).startswith(f"[{code}] ")


def test_every_declared_code_has_a_trigger() -> None:
    unreachable = set(CONTRACT_CODES) - set(TRIGGERS)
    assert not unreachable, (
        f"{sorted(unreachable)} are declared contract codes with no way to reach them. "
        "Either delete the code or delete the branch that claims to raise it."
    )


def test_no_trigger_names_a_code_that_is_not_declared() -> None:
    undeclared = set(TRIGGERS) - set(CONTRACT_CODES)
    assert not undeclared, f"{sorted(undeclared)} are triggered but not declared"


def test_an_undeclared_code_cannot_be_raised() -> None:
    """The detection half: `ContractError` refuses to invent a code."""
    with pytest.raises(ValueError, match="not a declared contract code"):
        ContractError("MADE_UP_CODE", "nope")


def test_a_torch_buffer_is_refused_at_intake() -> None:
    """A representation holds data in a compute namespace; torch crosses a bridge.

    Shares `REPRESENTATION_INCONSISTENT` with the mixed-device case above, so it
    needs its own test rather than a second entry in the table.
    """
    with pytest.raises(ContractError) as caught:
        _bundle(measure_weight=_TorchLike())
    assert caught.value.code == "REPRESENTATION_INCONSISTENT"
    assert "torch" in str(caught.value)


def test_the_diagnostic_is_what_a_coupler_returns() -> None:
    """`as_diagnostic()` is the structured form, not the prose."""
    try:
        _bundle().require_coherent()
    except ContractError as error:
        diagnostic = error.as_diagnostic()
    assert diagnostic["code"] == "COHERENT_STATE_INCOMPLETE"
    assert diagnostic["declaration"] == "amplitude, optical_path_m"
    assert diagnostic["remedy"]
    assert set(diagnostic) == {"code", "message", "declaration", "remedy"}


def test_a_contract_error_is_still_a_value_error() -> None:
    """R02.2's constructions moved onto this class and must not have changed meaning."""
    assert issubclass(ContractError, ValueError)
    with pytest.raises(ValueError):
        ReferenceSurface(name="", z_m=0.0, medium_index=1.0)
