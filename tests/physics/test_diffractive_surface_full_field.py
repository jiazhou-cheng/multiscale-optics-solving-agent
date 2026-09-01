"""R10.2: the full-field diffractive surface, against the grating equation.

CHE-194. The composition is `ray_to_scalar -> complex_transmission ->
scalar_to_ray`, and its identity is a **physical operator**: an optical surface
changes the physical state, and that the implementation converts representation
twice on the way is its implementation rather than what it is.

The oracle is outside this repository. `sin theta_m = m lambda / Lambda` -- the
grating equation -- predicts where a periodic surface sends light, and a binary
pi-phase grating additionally suppresses every even order. Those are two
independent statements, one about the period and one about the phase depth, and a
composition that got the phase sign or the pitch convention wrong would fail the
second while passing the first.

What is *not* tested here, and where it is
------------------------------------------
The two couplers' own conventions -- the projection factor, the measure, the
grazing floor, the `1/p` -- are R07's and R08's, tested there. This file tests
what the *composition* adds: that it acts at the declared surface, that the
transmission is applied once with the right sign, that the identity case reduces
to the couplers' round trip, and that the interior field's limitations are
declared on the way out. Duplicating the couplers' gates here would make two
places to update and would not add evidence.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from ray_support import WAVELENGTH_M, a_surface, collimated_bundle, propagating_only

from couplers import ray_to_scalar, scalar_to_ray
from operations import OperationDescriptor, OperationKind, registry, resolve
from operators import DiffractiveSurface, complex_transmission, diffractive_surface
from representations import ContractError, ReferenceSurface

SRC = Path(__file__).resolve().parents[2] / "src"
MODULE = SRC / "operators" / "diffractive_surface.py"

SHAPE = (64, 64)
PITCH_M = (0.25e-6, 0.25e-6)


#: The one surface everything in this file is declared on. Named once, because the
#: precondition R10.1 settled is that the bundle and the surface agree, and a test
#: that had to keep two names in step would be testing its own bookkeeping.
DOE_SURFACE = a_surface("doe")


def an_incident_bundle(*, direction=(0.0, 0.0, 1.0), shape=SHAPE, pitch=PITCH_M, surface=None):
    """A collimated bundle already expressed on the surface -- the precondition."""
    rays, unit, area = collimated_bundle(
        shape=shape, sample_pitch_m=pitch, direction=direction, wavelength_m=WAVELENGTH_M
    )
    return dataclasses.replace(rays, reference_surface=surface or DOE_SURFACE), unit, area


def a_binary_phase_grating(*, period_px: int, shape=SHAPE, pitch=PITCH_M):
    """A `pi`-phase binary grating along `x`, and its period in metres.

    Binary rather than sinusoidal because the even-order suppression is a second,
    independent oracle: a sinusoidal grating puts power in `m = 0, +-1` and says
    nothing about the phase depth, while a `pi`-deep binary one must put **zero**
    in every even order including `m = 0`.
    """
    column = np.arange(shape[1])
    sign = np.where(((column // (period_px // 2)) % 2) == 0, 1.0, -1.0)
    transmission = np.tile(sign, (shape[0], 1)).astype(complex)
    return (
        DiffractiveSurface(
            transmission=transmission,
            sample_pitch_m=pitch,
            reference_surface=DOE_SURFACE,
        ),
        period_px * pitch[1],
    )


def emitted_power_by_direction(rays, *, bins) -> dict[float, float]:
    """Launch power `|a w|^2` accumulated onto the nearest of `bins` in `d_u`."""
    direction_u = np.asarray(rays.directions)[:, 0]
    power = np.abs(np.asarray(rays.amplitude) * np.asarray(rays.measure_weight)) ** 2
    totals = dict.fromkeys(bins, 0.0)
    for value, weight in zip(direction_u, power, strict=True):
        nearest = min(bins, key=lambda b: abs(b - value))
        if abs(nearest - value) < 1e-3:
            totals[nearest] += float(weight)
    return totals


# ---------------------------------------------------------------------------
# 1. The grating equation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("period_px", [8, 16])
def test_the_orders_land_where_the_grating_equation_says(period_px: int) -> None:
    """Criterion 1, against an oracle outside this repository.

    `sin theta_m = m lambda / Lambda`, so a transmitted ray of the `m`-th order has
    transverse direction cosine `d_u = m lambda / Lambda`. At a 2.0 um period and
    0.55 um light that is `+-0.275` for `m = +-1` and `+-0.825` for `m = +-3`, and
    all four are among the strongest directions the composition emits.

    Nothing about this check involves a number this repository produced: the period
    is the fixture's, the wavelength is the bundle's, and the prediction is
    trigonometry.
    """
    surface, period_m = a_binary_phase_grating(period_px=period_px)
    rays, _, _ = an_incident_bundle()

    outgoing, record = diffractive_surface(rays, surface=surface)

    orders = {m: m * WAVELENGTH_M / period_m for m in (-3, -1, 1, 3)}
    assert all(abs(value) < 1.0 for value in orders.values())

    direction_u = np.asarray(outgoing.directions)[:, 0]
    power = np.abs(np.asarray(outgoing.amplitude) * np.asarray(outgoing.measure_weight)) ** 2
    strongest = {
        float(np.round(direction_u[i], 4)) for i in np.argsort(power)[-8:]
    }
    for order, expected in orders.items():
        assert any(abs(found - expected) < 1e-3 for found in strongest), (order, expected)
    assert record["model"] == "full_field"


def test_a_binary_pi_grating_suppresses_its_even_orders() -> None:
    """The second oracle, about the phase depth rather than the period.

    A binary grating of phase depth `pi` has Fourier coefficients that vanish for
    every even `m`, including `m = 0`: `t = +-1` averages to zero over a period. So
    the zeroth order must be *empty*, not merely small.

    What this catches, measured rather than asserted -- odd / even launch power:

    | transmission | odd | even | ratio |
    | -- | -- | -- | -- |
    | `t` (correct) | 4.64e-13 | 1.74e-45 | **2.7e32** |
    | `t^2` (applied twice) | 0 | 4.64e-13 | **0** |
    | `|t|` (modulus taken) | 0 | 4.64e-13 | **0** |

    A complete inversion in both wrong cases, not a small shift. What it does
    **not** catch, and the reason is worth stating rather than leaving as an
    unstated limit: a `+-1` binary transmission is *real*, so it is its own
    conjugate and `exp(-i phi)` gives an identical result. The phase sign is
    covered by `test_from_phase_applies_the_repository_phasor_sign`, which uses a
    linear ramp for exactly that reason.
    """
    surface, period_m = a_binary_phase_grating(period_px=8)
    rays, _, _ = an_incident_bundle()
    outgoing, _ = diffractive_surface(rays, surface=surface)

    odd = [m * WAVELENGTH_M / period_m for m in (-3, -1, 1, 3)]
    even = [m * WAVELENGTH_M / period_m for m in (-2, 0, 2)]
    totals = emitted_power_by_direction(outgoing, bins=[*odd, *even])

    odd_power = sum(totals[b] for b in odd)
    even_power = sum(totals[b] for b in even)
    assert odd_power > 0.0
    assert odd_power / max(even_power, 1e-300) > 1e6, (odd_power, even_power)

    # ...and the two perturbations it does catch, run through the shipping
    # composition on a perturbed *surface* rather than a parallel implementation.
    transmission = np.asarray(surface.transmission)
    for wrong in (transmission**2, np.abs(transmission).astype(complex)):
        broken = DiffractiveSurface(
            transmission=wrong,
            sample_pitch_m=PITCH_M,
            reference_surface=DOE_SURFACE,
        )
        emitted, _ = diffractive_surface(rays, surface=broken)
        wrong_totals = emitted_power_by_direction(emitted, bins=[*odd, *even])
        assert sum(wrong_totals[b] for b in odd) == 0.0
        assert sum(wrong_totals[b] for b in even) > 0.0


def test_a_tilted_incident_bundle_shifts_every_order_by_its_own_direction() -> None:
    """The grating equation's general form: `d_u_out = d_u_in + m lambda / Lambda`.

    Momentum conservation along the surface, and it is the check that the incident
    bundle is really an input to the transformation rather than something the
    composition happens to carry. R10.1 found the reference implementation's patch
    branch reading only the wavelength; this is the property that would have caught
    it.
    """
    surface, period_m = a_binary_phase_grating(period_px=8)
    incident_u = 4 * WAVELENGTH_M / (SHAPE[1] * PITCH_M[1])  # an exact spectral bin
    rays, direction, _ = an_incident_bundle(
        direction=(incident_u, 0.0, math.sqrt(1.0 - incident_u**2))
    )
    assert direction[0] == pytest.approx(incident_u, rel=1e-12)

    outgoing, _ = diffractive_surface(rays, surface=surface)
    direction_u = np.asarray(outgoing.directions)[:, 0]
    power = np.abs(np.asarray(outgoing.amplitude) * np.asarray(outgoing.measure_weight)) ** 2
    strongest = {float(np.round(direction_u[i], 4)) for i in np.argsort(power)[-8:]}

    for order in (-1, 1):
        expected = incident_u + order * WAVELENGTH_M / period_m
        assert any(abs(found - expected) < 1e-3 for found in strongest), (order, expected)


# ---------------------------------------------------------------------------
# 2. The composition, and the identity case
# ---------------------------------------------------------------------------


def test_the_identity_transmission_is_the_couplers_own_round_trip() -> None:
    """Criterion 3, and R10.1's shared-boundary measurement.

    With `t = 1` there is no physics between the two couplers, so the composition
    must be exactly their round trip -- and it is, **bit-identically**, not to a
    tolerance. That is the degenerate case of every model of this operation and the
    sharpest statement that they act at one boundary.

    It is also the reason a ray-wave-ray operation with no transformation between
    the couplers exists only here: this test *is* that operation, and shipping it
    would advertise a physical capability that is a consistency check.
    """
    rays, _, _ = an_incident_bundle()
    identity = DiffractiveSurface(
        transmission=np.ones(SHAPE, dtype=complex),
        sample_pitch_m=PITCH_M,
        reference_surface=DOE_SURFACE,
    )
    outgoing, _ = diffractive_surface(rays, surface=identity)
    through, _ = ray_to_scalar(outgoing, grid_shape=SHAPE, sample_pitch_m=PITCH_M)

    # ...and the round trip the two couplers do on their own, with nothing between.
    field, _ = ray_to_scalar(rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)
    direct_rays, _ = scalar_to_ray(field, surface=field.reference_surface)
    direct, _ = ray_to_scalar(direct_rays, grid_shape=SHAPE, sample_pitch_m=PITCH_M)

    assert np.array_equal(np.asarray(through.u), np.asarray(direct.u))
    residual = float(
        np.max(np.abs(np.asarray(through.u) - propagating_only(field)))
        / np.max(np.abs(np.asarray(field.u)))
    )
    assert residual == 0.0


def test_the_two_couplers_inside_are_the_functions_r07_and_r08_built() -> None:
    """Criterion 2. No private copy and no variant kernel, asserted by identity.

    A composition that had its own reconstruction would be a second kernel whose
    conventions could drift from the tested one, and the drift would be invisible:
    both would produce plausible fields. Checked as an object identity through the
    module's own globals, not as an import statement, so an aliased copy fails too.
    """
    import operators.diffractive_surface  # noqa: F401  (ensure it is loaded)

    module = sys.modules["operators.diffractive_surface"]
    assert module.ray_to_scalar is ray_to_scalar
    assert module.scalar_to_ray is scalar_to_ray
    assert module.complex_transmission is complex_transmission

    # ...and the module defines no function that looks like a second kernel.
    defined = {
        node.name
        for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    assert defined == {"__post_init__", "from_phase", "grid_shape", "diffractive_surface"}


def test_the_package_attribute_is_the_function_and_not_the_module() -> None:
    """The name collision the ticket's own API specifies, pinned rather than left.

    `src/operators/diffractive_surface.py` holds `def diffractive_surface(...)`, so
    `operators.diffractive_surface` is ambiguous on its face. It resolves to the
    **function**, and it keeps resolving to the function after the submodule is
    imported by name -- CPython sets the parent attribute when the submodule is
    first initialized, which happens inside `operators/__init__.py` *before* the
    function is bound.

    That is stable but not obvious, and a reader arriving at
    `sys.modules["operators.diffractive_surface"]` in the test above deserves to
    find out why here rather than by experiment.
    """
    import operators

    assert callable(operators.diffractive_surface)
    assert operators.diffractive_surface is diffractive_surface
    import operators.diffractive_surface

    assert operators.diffractive_surface is diffractive_surface


def test_the_transmission_is_applied_exactly_once() -> None:
    """A linearity check that separates "applied twice" from "not applied".

    Scaling the transmission by `c` scales every emitted amplitude by `c`; applying
    it twice would scale by `c^2`, and dropping it would not scale at all. One
    check separates all three, and it needs no oracle.
    """
    base, _ = a_binary_phase_grating(period_px=8)
    scaled = DiffractiveSurface(
        transmission=0.4 * np.asarray(base.transmission),
        sample_pitch_m=base.sample_pitch_m,
        reference_surface=base.reference_surface,
    )
    rays, _, _ = an_incident_bundle()

    plain, _ = diffractive_surface(rays, surface=base)
    dimmed, _ = diffractive_surface(rays, surface=scaled)
    ratio = float(
        np.max(np.abs(np.asarray(dimmed.amplitude)))
        / np.max(np.abs(np.asarray(plain.amplitude)))
    )
    assert ratio == pytest.approx(0.4, rel=1e-9)


def test_from_phase_applies_the_repository_phasor_sign() -> None:
    """`t = exp(+i phi)`, in one place.

    A caller writing `exp(-i phi)` gets a conjugated surface: a real DOE that
    focuses on the wrong side of the substrate and looks entirely plausible in any
    intensity. The classmethod exists so the sign is written once, and the twin is
    the conjugate, which sends every order the other way.
    """
    column = np.arange(SHAPE[1])
    phase = 2.0 * math.pi * 3.0 * column / SHAPE[1]  # a linear ramp: one order
    ramp = np.tile(phase, (SHAPE[0], 1))
    surface = DiffractiveSurface.from_phase(
        ramp, sample_pitch_m=PITCH_M, reference_surface=a_surface("doe")
    )
    assert np.allclose(np.asarray(surface.transmission), np.exp(1j * ramp))

    rays, _, _ = an_incident_bundle()
    forward, _ = diffractive_surface(rays, surface=surface)
    conjugated = DiffractiveSurface(
        transmission=np.exp(-1j * ramp),
        sample_pitch_m=PITCH_M,
        reference_surface=DOE_SURFACE,
    )
    backward, _ = diffractive_surface(rays, surface=conjugated)

    def brightest_direction(bundle) -> float:
        power = np.abs(
            np.asarray(bundle.amplitude) * np.asarray(bundle.measure_weight)
        ) ** 2
        return float(np.asarray(bundle.directions)[int(np.argmax(power)), 0])

    # A linear phase ramp is a single order, and the sign decides which side.
    assert brightest_direction(forward) == pytest.approx(
        -brightest_direction(backward), rel=1e-9
    )
    assert brightest_direction(forward) != 0.0


# ---------------------------------------------------------------------------
# 3. What it declares, and what it refuses
# ---------------------------------------------------------------------------


def test_the_interior_fields_limitations_are_declared_on_the_way_out() -> None:
    """The ticket's named risk: two undeclared approximations baked into clean rays.

    `RayBundle` has no `validity` field, so the declaration travels two ways -- in
    the emitted bundle's `optical_path_reference`, which survives a caller dropping
    the record, and structurally in the diagnostics. The interior field declares
    `surface_only` and `no_wavefront_curvature_term` (CHE-50), and both must appear.
    """
    surface, _ = a_binary_phase_grating(period_px=8)
    rays, _, _ = an_incident_bundle()
    outgoing, record = diffractive_surface(rays, surface=surface)

    assert record["interior_field_validity"] == [
        "no_wavefront_curvature_term",
        "surface_only",
    ]
    reference = outgoing.optical_path_reference or ""
    assert "full_field" in reference
    assert "no_wavefront_curvature_term" in reference
    assert "'doe'" in reference

    # The two typed records the parts produced travel verbatim rather than being
    # summarized, so nothing about the interior has to be re-derived.
    assert record["reconstruction"]["projection"] == "asm_consistent"
    assert record["sampling"]["selection"] == "exhaustive"
    assert record["sampling"]["measure_kind"] == "importance_weight"


def test_a_bundle_declared_on_another_surface_is_refused() -> None:
    """R10.1's shared boundary, made executable: this operation does not propagate.

    The incident bundle must already be expressed on the surface. Refused by
    `ray_to_scalar`'s own expectation check rather than by a restatement here, so
    there is one place that decides it.
    """
    surface, _ = a_binary_phase_grating(period_px=8)
    rays, _, _ = an_incident_bundle()
    elsewhere = an_incident_bundle(surface=a_surface("sensor", z_m=1.0e-3))[0]
    diffractive_surface(rays, surface=surface)  # on the surface: fine
    with pytest.raises(ContractError) as raised:
        diffractive_surface(elsewhere, surface=surface)
    assert raised.value.code == "FRAME_MISMATCH"


def test_a_real_transmission_is_refused() -> None:
    """An amplitude mask with an undeclared phase, and the phase is what diffracts."""
    with pytest.raises(ContractError) as raised:
        DiffractiveSurface(
            transmission=np.ones(SHAPE),
            sample_pitch_m=PITCH_M,
            reference_surface=DOE_SURFACE,
        )
    assert raised.value.code == "DTYPE_KIND_MISMATCH"
    assert "from_phase" in (raised.value.remedy or "")


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"transmission": np.ones(8, dtype=complex)}, "SHAPE_MISMATCH"),
        ({"transmission": np.full(SHAPE, np.nan, dtype=complex)}, "NON_FINITE"),
        ({"sample_pitch_m": (0.0, 0.25e-6)}, "UNIT_NOT_SI"),
        ({"sample_pitch_m": (0.25e-6,)}, "UNIT_NOT_SI"),
    ],
)
def test_an_unusable_surface_is_refused(kwargs: dict, code: str) -> None:
    fields = {
        "transmission": np.ones(SHAPE, dtype=complex),
        "sample_pitch_m": PITCH_M,
        "reference_surface": a_surface("doe"),
    }
    with pytest.raises(ContractError) as raised:
        DiffractiveSurface(**{**fields, **kwargs})
    assert raised.value.code == code


def test_an_unknown_model_is_refused() -> None:
    """Named, never inferred. One member today, and that is the honest count."""
    from operators import DIFFRACTIVE_MODELS

    assert DIFFRACTIVE_MODELS == ("full_field",)
    surface, _ = a_binary_phase_grating(period_px=8)
    rays, _, _ = an_incident_bundle()
    with pytest.raises(ContractError) as raised:
        diffractive_surface(rays, surface=surface, model="local_patch")  # type: ignore[arg-type]
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "model"


def test_the_medium_index_refusal_is_inherited_rather_than_restated() -> None:
    """R09's finding reaches here through the parts, which is the right place for it.

    When the ramp convention is settled this module needs no change, and that is
    the property being pinned: the refusal is not duplicated in the composition.
    """
    surface, _ = a_binary_phase_grating(period_px=8)
    rays, _, _ = an_incident_bundle()
    submerged = dataclasses.replace(
        rays,
        reference_surface=ReferenceSurface(name="doe", z_m=0.0, medium_index=1.336),
    )
    in_water = DiffractiveSurface(
        transmission=np.asarray(surface.transmission),
        sample_pitch_m=surface.sample_pitch_m,
        reference_surface=submerged.reference_surface,
    )
    with pytest.raises(ContractError) as raised:
        diffractive_surface(submerged, surface=in_water)
    assert raised.value.declaration == "reference_surface.medium_index"
    # The refusal is not restated here: no `ContractError` in this module names the
    # medium index, so when the ramp convention is settled this module needs no
    # change. (The module *docstring* mentions it, which is why this walks the
    # raise sites rather than the source text.)
    raises = [
        node
        for node in ast.walk(ast.parse(MODULE.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ContractError"
    ]
    assert raises, "the walk found no refusal at all, so it cannot fail"
    assert not any("medium_index" in ast.unparse(node) for node in raises)


# ---------------------------------------------------------------------------
# 4. The record, and what did not land
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_registry() -> Iterator[None]:
    saved = dict(registry._REGISTERED)
    registry._REGISTERED.clear()
    yield
    registry._REGISTERED.clear()
    registry._REGISTERED.update(saved)


def test_the_diffractive_surface_registers_as_a_physical_operator(
    isolated_registry: None,
) -> None:
    """Criterion 4. `ray_bundle -> ray_bundle`, and **never** as a coupler.

    The interior converts representation twice and the identity is still a physical
    state change, because an optical surface changes the state. That is the call
    `docs/architecture_principles.md` section 2 makes, and the `kind` field is
    where it is said.
    """
    descriptor = registry.register(
        OperationDescriptor(
            operation_id="O_DIFFRACTIVE_SURFACE",
            kind=OperationKind.PHYSICAL_OPERATOR,
            input="ray_bundle",
            output="ray_bundle",
            implementation="operators.diffractive_surface:diffractive_surface",
            approximation=(
                "the full-field model: every incident ray is accumulated coherently onto "
                "the surface's own grid, the complex transmission is applied once as a "
                "thin element, and the transmitted field is decomposed into the modes "
                "that leave. Exact for a thin, angle-independent transmission on one "
                "common plane; the interior field carries no exp(i k r^2 / 2R) "
                "wavefront-curvature term (CHE-50) and is valid at the surface with zero "
                "further propagation, both of which the emitted bundle declares"
            ),
            validity=(
                "the incident bundle must already be expressed on the surface; this "
                "operation does not propagate",
                "one common plane, i.e. a planar substrate; a conformal one has no such "
                "plane and needs the local-patch model",
                "air only until the ray<->wave ramp convention carries the refractive "
                "index (R09)",
                "the surface's grid is the reconstruction grid, so its pitch must "
                "represent the steepest wavelet ramp of both the incident and the "
                "transmitted spectrum",
            ),
            evidence=("tests/physics/test_diffractive_surface_full_field.py",),
            capabilities=None,
            derivative="forward_only",
        )
    )
    assert descriptor.kind is OperationKind.PHYSICAL_OPERATOR
    assert descriptor.kind is not OperationKind.COUPLER
    assert descriptor.input == descriptor.output == "ray_bundle"
    assert resolve("O_DIFFRACTIVE_SURFACE") is diffractive_surface


def test_the_avoided_diffractive_classes_did_not_land() -> None:
    """Criterion 6. Class delta +1, and a budget cannot record what was avoided."""
    defined = {
        node.name
        for module in sorted((SRC / "operators").rglob("*.py"))
        for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert defined == {"DiffractiveSurface"}
    for avoided in (
        "DiffractiveSurfaceBase",
        "FullFieldDiffractiveSurfaceSubclass",
        "PlanarDoeStepCoupler",
        "CascadeDiagnostics",
        "DiffractiveInteractionResult",
        "FullFieldParameters",
        "PrimarySampling",
        "DiffractiveModel",
    ):
        assert avoided not in defined, avoided


def test_no_composite_operator_framework_landed() -> None:
    """Criterion 2's other half: one composition, so it is written out.

    `docs/architecture_principles.md` permits a composite framework only if at
    least two production compositions immediately need it, and there is one. The
    check is that the module contains exactly one public function and no
    registry, pipeline or stage vocabulary -- the shapes a framework would take.
    """
    source = MODULE.read_text(encoding="utf-8")
    defined = {
        node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
    }
    for framework in ("pipeline", "stage", "compose", "Composite", "register", "chain"):
        assert not any(framework.lower() in name.lower() for name in defined), framework
    assert "There is one." in source


def test_the_cost_knob_is_reachable_and_the_two_routes_agree_on_node() -> None:
    """`reconstruction=` is exposed because this is where the cost lands.

    `DIRECT` is `O(N_rays x ny x nx)`; at 512^2 incident rays onto a 512^2 surface
    that is about 7e10 term evaluations, and without this argument a caller would
    have no escape hatch through this entry point. `KSPACE` is
    `O(N_rays + K log K)`.

    They are checked to agree where R07.2 says they must -- with the k-grid named,
    so every ray lands on a node -- rather than at the default oversampling, where
    R07.2's own budget puts them tens of percent apart.
    """
    from couplers import Reconstruction

    surface, _ = a_binary_phase_grating(period_px=8)
    rays, _, _ = an_incident_bundle()

    direct, direct_record = diffractive_surface(rays, surface=surface)
    kspace, kspace_record = diffractive_surface(
        rays,
        surface=surface,
        reconstruction=Reconstruction.KSPACE,
        kspace_grid_shape=SHAPE,
    )
    assert direct_record["reconstruction"]["reconstruction"] == "direct"
    assert kspace_record["reconstruction"]["reconstruction"] == "kspace"
    assert kspace_record["reconstruction"]["kspace"]["on_node_fraction"] == 1.0

    residual = float(
        np.max(np.abs(np.asarray(kspace.amplitude) - np.asarray(direct.amplitude)))
        / np.max(np.abs(np.asarray(direct.amplitude)))
    )
    assert residual < 1e-12, residual


def test_the_projection_is_fixed_rather_than_defaulted() -> None:
    """`SENSOR_OBLIQUITY` is a detector model and has no place inside a surface.

    R07.1 measured that it does not preserve the field. Offering it here would let
    a caller select an operator that loses a few percent off-axis at a place where
    nothing is being detected, so it is not an argument at all -- which is a
    stronger statement than a default, and this is the test that keeps it one.
    """
    import inspect

    parameters = set(inspect.signature(diffractive_surface).parameters)
    assert "projection" not in parameters
    assert {"reconstruction", "allow_gain", "count", "density", "draw"} <= parameters

    surface, _ = a_binary_phase_grating(period_px=8)
    rays, _, _ = an_incident_bundle()
    _, record = diffractive_surface(rays, surface=surface)
    assert record["reconstruction"]["projection"] == "asm_consistent"


def test_allow_gain_is_reachable_so_its_remedy_is_actionable() -> None:
    """An amplifying surface is refused, and the refusal's remedy works from here.

    A surface with `|t| > 1` is a modelling error rather than a knob, so the
    default refuses it. But `complex_transmission`'s remedy says "pass
    allow_gain=True", and a remedy a caller cannot act on through the entry point
    they used is worse than none.
    """
    amplifying = DiffractiveSurface(
        transmission=np.full(SHAPE, 1.5 + 0.0j),
        sample_pitch_m=PITCH_M,
        reference_surface=DOE_SURFACE,
    )
    rays, _, _ = an_incident_bundle()
    with pytest.raises(ContractError) as raised:
        diffractive_surface(rays, surface=amplifying)
    assert raised.value.code == "REPRESENTATION_INCONSISTENT"
    assert "allow_gain" in (raised.value.remedy or "")

    outgoing, _ = diffractive_surface(rays, surface=amplifying, allow_gain=True)
    assert outgoing.count > 0
