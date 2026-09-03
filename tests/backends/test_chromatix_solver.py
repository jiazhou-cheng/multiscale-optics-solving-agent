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
from backends.chromatix import (
    CAPABILITIES,
    DERIVATIVE,
    MODELS,
    carrier_phase_rad,
    fresnel_propagate,
    propagate,
)
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


def test_the_public_entry_points_are_the_three_operations() -> None:
    """Three operations, and everything else exported is a declaration or a diagnostic.

    `propagate` (CHE-184) is free-space evolution by a distance; CHE-209 added
    `focal_plane_transform`, the ideal lens between its two focal planes. They are
    separate entry points because they are separate physics -- one preserves the
    sampling and the other must change it -- and neither is a mode of the other.

    CHE-228 (R06.11) added `fresnel_propagate`, and it is the case where that rule
    was hardest to apply: it *does* preserve the sampling, so it is not separate
    from `propagate` the way the lens is. What makes it a third entry point rather
    than a third `method` is the claim, not the plumbing -- `O_ASM_PROPAGATE`'s
    `approximation` says no term is dropped, the Fresnel kernel drops one, and one
    record per implementation means a second claim needs a second callable.
    """
    callables = {
        name
        for name in chromatix.__all__
        if callable(getattr(chromatix, name)) and not isinstance(getattr(chromatix, name), str)
    }
    assert callables == {
        "propagate",
        "fresnel_propagate",
        "focal_plane_transform",
        "carrier_phase_rad",
        "edge_energy_fraction",
        "fourier_plane_pitch_m",
        "padded_field_bytes",
        "padded_shape",
    }
    assert inspect.signature(propagate).parameters.keys() == {"field", "distance_m", "model"}
    # Deliberately the same signature: the two differ in their `model=` vocabulary
    # and in what they approximate, not in what a caller has to supply.
    assert (
        inspect.signature(fresnel_propagate).parameters.keys()
        == {"field", "distance_m", "model"}
    )


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


def test_the_propagation_registers_as_itself_and_declares_its_backend() -> None:
    """Criterion 4, executed end to end: the PRODUCTION record resolves to this
    function.

    **One descriptor over this implementation, not two.** There used to be two,
    because they answered different questions: `S_WAVE_CHROMATIX` was the *backend*
    -- what this project can drive, and the capability row it executes within -- and
    `O_ASM_PROPAGATE` was the *physical operation*, what happens to the state.
    CHE-224 (R15.1) made the first question a field: `backend="chromatix"` on the
    surviving record says what the deleted one was for, and `kind` is left saying
    only what happens to the physical state.

    It is not a coupler, and that is the substantive half of the criterion: a
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
    assert "S_WAVE_CHROMATIX" not in catalogued, (
        "the merged record is back. The backend question is a descriptor field; a "
        "second record over one callable answers it twice."
    )
    operator = catalogued["O_ASM_PROPAGATE"]

    assert operator.kind is OperationKind.PHYSICAL_OPERATOR
    assert operator.backend == "chromatix"
    assert operator.capabilities == CAPABILITIES
    assert operator.derivative == DERIVATIVE
    # The catalog HAS couplers -- two of them -- so the old assertion that
    # `find(kind=COUPLER)` is empty is no longer a statement about this record.
    # What still holds, and is what the criterion meant, is that it is not one.
    assert operator not in registry.find(kind=OperationKind.COUPLER)
    assert resolve("O_ASM_PROPAGATE") is propagate
    # And this is the only record over `propagate`, which is what the merge means.
    assert [d.operation_id for d in CATALOG if d.implementation == operator.implementation] == [
        "O_ASM_PROPAGATE"
    ]


def test_no_gradient_is_claimed() -> None:
    """`forward_only`, with no argument that changes it."""
    assert DERIVATIVE == "forward_only"
    assert MODELS == ("asm", "asm_carrier_removed")
    parameters = inspect.signature(propagate).parameters
    assert not any("grad" in name or "differen" in name for name in parameters)


# ---------------------------------------------------------------------------
# 5. `fresnel_propagate` -- CHE-228 (R06.11)
# ---------------------------------------------------------------------------
#
# The physics is `tests/physics/test_fresnel_propagation.py`. What is here is the
# contract: which `model=` keys it takes, what it refuses, what pitch it declares,
# and what its returned field says about itself.


def a_fresnel_model(**overrides: object) -> dict[str, object]:
    model: dict[str, object] = {"pad_width": 8, "target_surface": "focus"}
    model.update(overrides)
    return model


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ({"pad_width": 8}, "target_surface"),
        ({"target_surface": "f"}, "pad_width"),
        ({"pad_width": 8, "target_surface": "f", "pad_witdh": 4}, "pad_witdh"),
    ],
)
def test_the_fresnel_model_refuses_a_missing_or_misspelled_key(
    model: dict[str, object], expected: str
) -> None:
    with pytest.raises(ValueError, match=expected):
        fresnel_propagate(a_scalar_field(), distance_m=DISTANCE_M, model=model)


def test_the_fresnel_model_refuses_method_with_its_own_reason() -> None:
    """Not as an unrecognized spelling: as the wrong belief it actually is.

    A caller who writes `method=` here thinks this function offers the
    absolute/carrier-removed choice `propagate` does. It does not, and it cannot:
    the backend's Fresnel kernel carries no `exp(i k n z)` factor at all, so there
    is no absolute variant to select. The generic "does not take ['method']"
    message would leave that belief in place.
    """
    with pytest.raises(ValueError, match="no absolute-phase variant"):
        fresnel_propagate(
            a_scalar_field(),
            distance_m=DISTANCE_M,
            model=a_fresnel_model(method="fresnel"),
        )


def test_the_fresnel_path_refuses_a_negative_pad_width_and_a_non_finite_distance() -> None:
    with pytest.raises(ValueError, match="pad_width"):
        fresnel_propagate(
            a_scalar_field(), distance_m=DISTANCE_M, model=a_fresnel_model(pad_width=-1)
        )
    with pytest.raises(ValueError, match="finite"):
        fresnel_propagate(a_scalar_field(), distance_m=math.inf, model=a_fresnel_model())


def test_a_surface_only_field_is_refused_by_the_fresnel_path_too() -> None:
    """The same claim `propagate` refuses, and it is not weaker for being paraxial."""
    source = a_scalar_field()
    pinned = ScalarField(
        u=source.u,
        sample_pitch_m=source.sample_pitch_m,
        wavelength_m=source.wavelength_m,
        reference_surface=source.reference_surface,
        validity=frozenset({"surface_only"}),
    )
    with pytest.raises(ContractError) as caught:
        fresnel_propagate(pinned, distance_m=DISTANCE_M, model=a_fresnel_model())
    assert caught.value.code == "REPRESENTATION_INCONSISTENT"


def test_a_complex128_field_is_refused_before_jax_is_asked() -> None:
    """The capability row, reached on this path as well as on `propagate`'s.

    The refusal is `native_state`'s and is shared, so this is a check that the new
    entry point actually goes through the boundary rather than around it.
    """
    with pytest.raises(ValueError) as caught:
        fresnel_propagate(
            a_scalar_field(dtype="complex128"),
            distance_m=DISTANCE_M,
            model=a_fresnel_model(),
        )
    assert getattr(caught.value, "code", None) == "LOSSY_DOWNCAST_REQUIRED"


@pytest.mark.parametrize("pad_width", [0, 64])
def test_the_fresnel_path_preserves_the_sampling_it_declares(pad_width: int) -> None:
    """Criterion 4. The transfer method is a convolution, so the pitch is the input's.

    `from_native` checks the declaration against what the backend returned and
    refuses a regrid the caller did not predict, so this passing at two pad widths
    is the statement that padding changes the wraparound and nothing else -- unlike
    the single-FFT Fresnel method, whose output pitch is
    `lambda |z| / (n (N + 2 pad_width) dx)` and which is deliberately not here.
    """
    source = a_scalar_field()
    out = fresnel_propagate(
        source, distance_m=DISTANCE_M, model=a_fresnel_model(pad_width=pad_width)
    )
    assert out.sample_pitch_m == source.sample_pitch_m
    assert out.shape == source.shape


def test_the_fresnel_result_declares_both_paraxial_and_carrier_removed() -> None:
    """Criterion 5. Both, always, and neither is conditional on an argument.

    `paraxial` because the kernel drops a term whose error `|U|^2` cannot show, and
    `carrier_removed_phase` because the kernel has no `exp(i k n z)` in it -- so the
    phase is relative to the same piston `asm_carrier_removed` removes. An
    unconditional flag is a stronger statement than a conditional one and is the
    reason this function takes no `method`.
    """
    out = fresnel_propagate(a_scalar_field(), distance_m=DISTANCE_M, model=a_fresnel_model())
    assert out.validity == frozenset({"paraxial", "carrier_removed_phase"})

    # And an inherited flag survives, exactly as it does through `propagate`.
    source = a_scalar_field()
    limited = ScalarField(
        u=source.u,
        sample_pitch_m=source.sample_pitch_m,
        wavelength_m=source.wavelength_m,
        reference_surface=source.reference_surface,
        validity=frozenset({"no_wavefront_curvature_term"}),
    )
    inherited = fresnel_propagate(limited, distance_m=DISTANCE_M, model=a_fresnel_model())
    assert inherited.validity == frozenset(
        {"no_wavefront_curvature_term", "paraxial", "carrier_removed_phase"}
    )


def test_the_fresnel_result_lands_on_the_named_target_plane() -> None:
    out = fresnel_propagate(
        a_scalar_field(),
        distance_m=DISTANCE_M,
        model=a_fresnel_model(target_surface="image"),
    )
    assert out.reference_surface.name == "image"
    assert out.reference_surface.z_m == pytest.approx(DISTANCE_M)


def test_the_fresnel_propagation_registers_as_its_own_record() -> None:
    """Criterion 8, and the reason there are two records rather than one with a mode.

    `O_ASM_PROPAGATE` says "no Fresnel approximation and no term dropped". A
    paraxial method under that record would make the record's own prose false, and
    the id would read `O_ASM_` for a run that is not one. Since CHE-224 (R15.1) the
    catalog also keys uniqueness on `implementation`, so a second claim needs a
    second callable either way -- and this asserts both records exist, resolve to
    different functions, and disagree in their `approximation` rather than only in
    their ids.
    """
    catalogued = {d.operation_id: d for d in CATALOG}
    record = catalogued["O_FRESNEL_PROPAGATE"]

    assert record.kind is OperationKind.PHYSICAL_OPERATOR
    assert record.composes == ()
    assert record.backend == "chromatix"
    assert record.capabilities == CAPABILITIES
    assert record.derivative == DERIVATIVE
    assert record.inputs == ("scalar_field",)
    assert record.returns == ("scalar_field",)
    assert resolve("O_FRESNEL_PROPAGATE") is fresnel_propagate
    assert resolve("O_ASM_PROPAGATE") is propagate

    # The two claims, side by side, so the pair cannot drift into saying one thing.
    exact = catalogued["O_ASM_PROPAGATE"]
    assert "no Fresnel approximation and no term dropped" in exact.approximation
    assert "One term IS dropped" in record.approximation
    assert "paraxial" in record.validity[1]

    # One record per implementation, still.
    for descriptor in (record, exact):
        assert [
            d.operation_id for d in CATALOG if d.implementation == descriptor.implementation
        ] == [descriptor.operation_id]


def test_no_gradient_is_claimed_by_the_fresnel_path_either() -> None:
    assert not any(
        "grad" in name or "differen" in name
        for name in inspect.signature(fresnel_propagate).parameters
    )
    assert MODELS == ("asm", "asm_carrier_removed"), (
        "the Fresnel kernel is a separate callable, so it must not appear in "
        "`propagate`'s method vocabulary"
    )
