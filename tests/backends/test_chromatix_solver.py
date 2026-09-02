"""The propagation entry point: its contract, its phase claim, and its descriptors.

CHE-184 (R06.2) acceptance criteria 2, 4 and 5, and CHE-158 (R06) criteria 3 and
4. Criterion 1 -- the analytic-oracle cases -- is
`tests/physics/test_scalar_wave_propagation.py`, and criterion 3 -- the phasor
sign on an axis-asymmetric case -- is there too, because both are physics rather
than contract.

The claim this file exists for is criterion 2: **a returned field says which
phase it carries, in its typed `validity` and not in prose.** Absolute and
carrier-removed differ by the constant `k n z`, `|U|^2` is identical under it, and
a composition that mixes the two inherits a silent piston. So the flag is asserted
both ways round, and the constant is measured against `carrier_phase_rad` so that
what the flag announces is the difference that is actually there.
"""

from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
from chromatix_support import WAVELENGTH_M, a_scalar_field

from backends import chromatix
from backends.chromatix import CAPABILITIES, DERIVATIVE, MODELS, carrier_phase_rad, propagate
from operations import CATALOG, OperationKind, registry, resolve
from representations import ContractError, ScalarField

DISTANCE_M = 30e-6


def a_model(**overrides: object) -> dict[str, object]:
    model: dict[str, object] = {"method": "asm", "pad_width": 8, "target_surface": "focus"}
    model.update(overrides)
    return model


# ---------------------------------------------------------------------------
# 1. One entry point, and what it refuses
# ---------------------------------------------------------------------------


def test_the_public_entry_points_are_the_two_operations() -> None:
    """Two operations, and everything else exported is a declaration or a diagnostic.

    `propagate` (CHE-184) is free-space evolution by a distance; CHE-209 added
    `focal_plane_transform`, the ideal lens between its two focal planes. They are
    separate entry points because they are separate physics -- one preserves the
    sampling and the other must change it -- and neither is a mode of the other.
    """
    callables = {
        name
        for name in chromatix.__all__
        if callable(getattr(chromatix, name)) and not isinstance(getattr(chromatix, name), str)
    }
    assert callables == {
        "propagate",
        "focal_plane_transform",
        "carrier_phase_rad",
        "edge_energy_fraction",
        "fourier_plane_pitch_m",
        "padded_field_bytes",
        "padded_shape",
    }
    assert inspect.signature(propagate).parameters.keys() == {"field", "distance_m", "model"}


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ({"method": "asm", "pad_width": 8}, "target_surface"),
        ({"method": "asm", "target_surface": "f"}, "pad_width"),
        ({"pad_width": 8, "target_surface": "f"}, "method"),
        (
            {"method": "asm", "pad_width": 8, "target_surface": "f", "pad_witdh": 4},
            "pad_witdh",
        ),
    ],
)
def test_a_missing_or_misspelled_model_key_is_refused(
    model: dict[str, object], expected: str
) -> None:
    """An unrecognized key would be discarded silently: a different run, no error."""
    with pytest.raises(ValueError, match=expected):
        propagate(a_scalar_field(), distance_m=DISTANCE_M, model=model)


def test_an_unknown_method_and_a_negative_pad_width_are_refused() -> None:
    with pytest.raises(ValueError, match="asm_carrier_removed"):
        propagate(a_scalar_field(), distance_m=DISTANCE_M, model=a_model(method="fresnel"))
    with pytest.raises(ValueError, match="pad_width"):
        propagate(a_scalar_field(), distance_m=DISTANCE_M, model=a_model(pad_width=-1))
    with pytest.raises(ValueError, match="finite"):
        propagate(a_scalar_field(), distance_m=math.inf, model=a_model())


def test_a_surface_only_field_is_refused_rather_than_propagated() -> None:
    """`surface_only` is a claim about where the field is valid, and this is that place.

    Propagating it is not a loss of accuracy, it is a different physical claim, and
    nothing in the result would record that the claim had been made.
    """
    source = a_scalar_field()
    pinned = ScalarField(
        u=source.u,
        sample_pitch_m=source.sample_pitch_m,
        wavelength_m=source.wavelength_m,
        reference_surface=source.reference_surface,
        validity=frozenset({"surface_only"}),
    )
    with pytest.raises(ContractError) as caught:
        propagate(pinned, distance_m=DISTANCE_M, model=a_model())
    assert caught.value.code == "REPRESENTATION_INCONSISTENT"


# ---------------------------------------------------------------------------
# 2. Where the result is declared to be
# ---------------------------------------------------------------------------


def test_the_result_is_declared_on_the_named_target_plane() -> None:
    """The surface is advanced by the distance, and the medium is carried, not re-declared."""
    source = a_scalar_field(z_m=1e-3, medium_index=1.5)
    out = propagate(source, distance_m=DISTANCE_M, model=a_model(target_surface="sensor"))

    assert out.reference_surface.name == "sensor"
    assert out.reference_surface.z_m == pytest.approx(1e-3 + DISTANCE_M, rel=1e-15)
    assert out.reference_surface.medium_index == 1.5
    assert out.reference_surface.normal == source.reference_surface.normal


def test_the_medium_comes_from_the_field_and_there_is_no_second_place_to_say_it() -> None:
    """Index 1.5 must actually change the propagation; the argument list must not offer it."""
    vacuum = a_scalar_field()
    glass = a_scalar_field(medium_index=1.5)
    model = a_model(method="asm_carrier_removed")

    in_vacuum = np.asarray(propagate(vacuum, distance_m=DISTANCE_M, model=model).u)
    in_glass = np.asarray(propagate(glass, distance_m=DISTANCE_M, model=model).u)
    assert not np.allclose(in_vacuum, in_glass)
    assert "refractive_index" not in inspect.signature(propagate).parameters


# ---------------------------------------------------------------------------
# 3. Criterion 2: which phase the field carries, in the type
# ---------------------------------------------------------------------------


def test_the_absolute_path_declares_no_carrier_removal() -> None:
    out = propagate(a_scalar_field(), distance_m=DISTANCE_M, model=a_model(method="asm"))
    assert "carrier_removed_phase" not in out.validity
    assert out.validity == frozenset()


def test_the_carrier_removed_path_says_so_in_its_validity() -> None:
    out = propagate(
        a_scalar_field(), distance_m=DISTANCE_M, model=a_model(method="asm_carrier_removed")
    )
    assert "carrier_removed_phase" in out.validity


def test_an_inherited_validity_flag_survives_the_propagation() -> None:
    """A field that already lacked a curvature term still lacks it afterwards."""
    source = a_scalar_field()
    limited = ScalarField(
        u=source.u,
        sample_pitch_m=source.sample_pitch_m,
        wavelength_m=source.wavelength_m,
        reference_surface=source.reference_surface,
        validity=frozenset({"no_wavefront_curvature_term"}),
    )
    out = propagate(
        limited, distance_m=DISTANCE_M, model=a_model(method="asm_carrier_removed")
    )
    assert out.validity == frozenset({"no_wavefront_curvature_term", "carrier_removed_phase"})


def test_the_two_methods_differ_by_exactly_the_declared_carrier() -> None:
    """The falsifiable twin: what the flag announces is the difference that is there.

    `|U|^2` is identical under the piston, which is the whole reason the flag has
    to exist -- so the intensities are asserted equal and the *complex* fields are
    asserted to differ by `exp(i k n z)` and by nothing else. The residual is
    complex64 round-off of a phase of `k z` = 354 rad, which is where one float32
    epsilon per radian puts it.
    """
    source = a_scalar_field()
    absolute = propagate(source, distance_m=DISTANCE_M, model=a_model(method="asm"))
    removed = propagate(
        source, distance_m=DISTANCE_M, model=a_model(method="asm_carrier_removed")
    )

    carrier = carrier_phase_rad(
        wavelength_m=WAVELENGTH_M, distance_m=DISTANCE_M, refractive_index=1.0
    )
    assert carrier == pytest.approx(2 * math.pi * DISTANCE_M / WAVELENGTH_M)

    a, r = np.asarray(absolute.u), np.asarray(removed.u)
    reconstructed = r * np.exp(1j * carrier)
    residual = float(np.linalg.norm(reconstructed - a) / np.linalg.norm(a))
    float32_floor = float(np.finfo(np.float32).eps) * carrier
    assert residual < float32_floor, f"{residual} is above the complex64 floor {float32_floor}"

    # And the thing intensity cannot see. Compared in L2 over the grid rather than
    # pixelwise: complex64 round-off of a 354 rad phase is an *absolute* error, so
    # in the dim wings -- five orders below the peak -- it is percents relative,
    # and a pixelwise relative test would be measuring the floor rather than the
    # piston.
    intensity_residual = float(
        np.linalg.norm(np.abs(a) ** 2 - np.abs(r) ** 2) / np.linalg.norm(np.abs(a) ** 2)
    )
    assert intensity_residual < float32_floor

    # ...while the complex fields differ by exactly the piston the flag announces:
    # |exp(i k n z) - 1|, which is 1.88 here, seven orders above the residual above.
    piston = abs(complex(np.exp(1j * carrier)) - 1.0)
    field_residual = float(np.linalg.norm(a - r) / np.linalg.norm(a))
    assert field_residual == pytest.approx(piston, rel=1e-3)
    assert piston > 1.0


def test_the_carrier_is_not_folded_back_into_the_field() -> None:
    """`carrier_phase_rad` is arithmetic in float64 and touches no array.

    Reapplying it in complex64 would reintroduce exactly the rounding the method
    removes, so the reconstruction route is a number the caller adds to a float64
    phase, not a field this package hands back.
    """
    assert inspect.signature(carrier_phase_rad).parameters.keys() == {
        "wavelength_m",
        "distance_m",
        "refractive_index",
    }
    one_metre = carrier_phase_rad(wavelength_m=5e-7, distance_m=1.0, refractive_index=1.0)
    assert isinstance(one_metre, float)
    # Signed and unwrapped: a backward leg removes the opposite carrier, and the
    # value is the accumulated phase before any modular reduction.
    forward = carrier_phase_rad(wavelength_m=5e-7, distance_m=1e-3, refractive_index=1.0)
    assert forward > 1e4
    assert carrier_phase_rad(
        wavelength_m=5e-7, distance_m=-1e-3, refractive_index=1.0
    ) == pytest.approx(-forward)


# ---------------------------------------------------------------------------
# 4. Criterion 4: the two descriptors, and neither is a coupler
# ---------------------------------------------------------------------------


def test_the_solver_and_the_propagation_register_as_themselves() -> None:
    """Criterion 4, executed end to end: both PRODUCTION records resolve to this
    function.

    Two descriptors over one implementation, because they answer different
    questions. `S_WAVE_CHROMATIX` is the *backend* -- what this project can drive,
    and the capability row it executes within. `O_ASM_PROPAGATE` is the *physical
    operation* -- what happens to the state, which is evolution through a declared
    medium from one reference surface to another.

    Neither is a coupler, and that is the substantive half of the criterion: a
    coupler changes representation while preserving physical state. This changes
    physical state and preserves the representation, which is the exact opposite,
    and heavy numerics is not what decides the question.

    The descriptor used to be constructed here, inside a fixture that emptied the
    registry, because `backends/` may not import `operations/` and there was no
    production registration site anywhere. CHE-221 (R03.4) put one *inside*
    `operations/`: the catalog names the implementation as a
    `"module.path:attribute"` string, so it needs no dependency edge in either
    direction, and the allowlist is unchanged. What is read below is the shipped
    record rather than a copy this file kept in step by hand.
    """
    catalogued = {d.operation_id: d for d in CATALOG}
    solver = catalogued["S_WAVE_CHROMATIX"]
    operator = catalogued["O_ASM_PROPAGATE"]

    assert solver.kind is OperationKind.SOLVER
    assert operator.kind is OperationKind.PHYSICAL_OPERATOR
    assert OperationKind.COUPLER not in {solver.kind, operator.kind}
    assert solver.implementation == operator.implementation
    assert solver.capabilities == operator.capabilities == CAPABILITIES
    assert solver.derivative == operator.derivative == DERIVATIVE
    # The catalog now HAS couplers -- two of them -- so the old assertion that
    # `find(kind=COUPLER)` is empty is no longer a statement about these records.
    # What still holds, and is what the criterion meant, is that neither of these
    # two is among them.
    assert solver not in registry.find(kind=OperationKind.COUPLER)
    assert operator not in registry.find(kind=OperationKind.COUPLER)
    assert resolve("S_WAVE_CHROMATIX") is propagate
    assert resolve("O_ASM_PROPAGATE") is propagate


def test_no_gradient_is_claimed() -> None:
    """`forward_only`, with no argument that changes it."""
    assert DERIVATIVE == "forward_only"
    assert MODELS == ("asm", "asm_carrier_removed")
    parameters = inspect.signature(propagate).parameters
    assert not any("grad" in name or "differen" in name for name in parameters)
