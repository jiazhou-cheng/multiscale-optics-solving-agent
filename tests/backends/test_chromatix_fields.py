"""The translation boundary: what crosses it, what is refused, and what it costs.

CHE-183 (R06.1) acceptance criteria 1, 2, 4 and 5. Criterion 3 -- no chromatix or
JAX object outside the package -- is `tests/backends/test_chromatix_boundary.py`,
because it is an AST walk over the whole tree rather than a statement about this
module.

The four claims here, in order:

1. a `complex128` field is **refused through the capability path**, not downcast;
2. pad width and crop state come back on the artifact, and a round trip through
   the boundary preserves pitch and extent exactly;
3. memory is estimable from the **padded** shape, and the estimate is checked
   against a measured run rather than asserted;
4. the package adds no class.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backends.chromatix import fields
from backends.chromatix.fields import (
    NATIVE_DTYPE,
    edge_energy_fraction,
    fourier_plane_pitch_m,
    from_native,
    native_state,
    padded_field_bytes,
    padded_shape,
)
from backends.chromatix.solver import propagate
from numerics import ArrayNamespace, ArrayState, DevicePlacement, DType
from representations import ContractError, ReferenceSurface, ScalarField

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "backends" / "chromatix"

WAVELENGTH_M = 0.532e-6
PITCH_M = (0.30e-6, 0.25e-6)

#: Names the reference implementation used for this job, none of which landed.
AVOIDED_NAMES = (
    "ChromatixAdapter",
    "get_adapter",
    "ChromatixWaveRequest",
    "ChromatixWaveResult",
    "ChromatixWaveFailure",
    "WaveHandoffError",
    "CarrierRemovedPropagation",
    "_BaselineError",
    "run_standalone",
)


def a_field(*, dtype: str = "complex64", shape: tuple[int, int] = (48, 64)) -> ScalarField:
    """A small confined field with an explicit, deliberately non-square pitch."""
    y = (np.arange(shape[0]) - shape[0] // 2) * PITCH_M[0]
    x = (np.arange(shape[1]) - shape[1] // 2) * PITCH_M[1]
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    waist = 2e-6
    u = np.exp(-(grid_x**2 + grid_y**2) / waist**2).astype(dtype)
    return ScalarField(
        u=u,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="source", z_m=0.0, medium_index=1.0),
    )


def a_model(**overrides: object) -> dict[str, object]:
    model: dict[str, object] = {"method": "asm", "pad_width": 8, "target_surface": "target"}
    model.update(overrides)
    return model


# ---------------------------------------------------------------------------
# 1. complex128 is refused, not absorbed
# ---------------------------------------------------------------------------


def test_a_complex128_field_is_refused_through_the_capability_path() -> None:
    """Criterion 1. The backend would ingest it and truncate it internally.

    `ScalarField.__init__` on **this** project's side preserves whatever dtype it
    is handed, so a `complex128` field is a real object a caller can build and
    hand over. The backend's own field constructor is
    `jnp.asarray(u, dtype=jnp.complex64)` unconditionally -- so absorbing the
    request would lose ~8 decimal digits of a phase of `k * OPL` at a boundary
    that recorded nothing.
    """
    field = a_field(dtype="complex128")
    assert field.state.dtype is DType.COMPLEX128, "the field itself must not pre-truncate"

    with pytest.raises(ValueError) as caught:
        native_state(field)
    assert getattr(caught.value, "code", None) == "LOSSY_DOWNCAST_REQUIRED"
    # The refusal names the measurement behind it rather than asserting a policy.
    assert "complex64" in str(caught.value)

    with pytest.raises(ValueError) as from_propagate:
        propagate(field, distance_m=10e-6, model=a_model())
    assert getattr(from_propagate.value, "code", None) == "LOSSY_DOWNCAST_REQUIRED"


def test_the_refusal_happens_before_the_backend_is_imported() -> None:
    """`native_state` is pure: it reads a state and negotiates against a table."""
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "import numpy as np\n"
        "from representations import ReferenceSurface, ScalarField\n"
        "from backends.chromatix.fields import native_state\n"
        "field = ScalarField(np.ones((4, 4), np.complex128), (1e-6, 1e-6), 5e-7,\n"
        "                    ReferenceSurface('s', 0.0, 1.0))\n"
        "try:\n"
        "    native_state(field)\n"
        "except ValueError as error:\n"
        "    print(error.code, 'chromatix' in sys.modules or 'jax' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, cwd=ROOT
    )
    assert result.stdout.strip() == "LOSSY_DOWNCAST_REQUIRED False"


def test_a_complex64_field_negotiates_into_the_backend_namespace() -> None:
    """The admissible case: dtype preserved, host residency preserved, namespace moved."""
    state = native_state(a_field())
    assert state.dtype is NATIVE_DTYPE
    assert state.namespace is ArrayNamespace.JAX
    assert state.device == a_field().state.device


# ---------------------------------------------------------------------------
# 2. Pad state travels, and the round trip is exact where it must be
# ---------------------------------------------------------------------------


def test_a_round_trip_preserves_pitch_and_extent_exactly() -> None:
    """Criterion 2, second half. Exact equality, not a tolerance.

    The backend stores `dx` in float32, so reading the pitch back out of it would
    move `extent_m` by ~6e-8 relative on every crossing. The boundary checks the
    native pitch against the declared one and carries the declared float64 value,
    which is why this can be `==` rather than `approx`.
    """
    source = a_field()
    out = propagate(source, distance_m=10e-6, model=a_model())

    assert out.sample_pitch_m == source.sample_pitch_m
    assert out.extent_m == source.extent_m
    assert out.shape == source.shape
    assert out.wavelength_m == source.wavelength_m
    assert out.frame == source.frame
    # The array namespace the caller handed in is the one they get back.
    assert isinstance(out.u, np.ndarray)
    assert out.state.dtype is NATIVE_DTYPE


def test_pad_width_and_crop_state_travel_with_the_field() -> None:
    """Criterion 2, first half. A shape is not a window without the pad width."""
    source = a_field()

    cropped = propagate(source, distance_m=10e-6, model=a_model(pad_width=8, crop=True))
    assert cropped.shape == source.shape
    assert cropped.padded is False
    assert cropped.pad_width == 8, "the width used is recorded even once it is cropped away"

    uncropped = propagate(source, distance_m=10e-6, model=a_model(pad_width=8, crop=False))
    assert uncropped.padded is True
    assert uncropped.pad_width == 8
    assert uncropped.shape == padded_shape(source.shape, 8)
    # The extent describes the array as it stands, padding included -- which is
    # exactly why the pad width has to be on the artifact for it to be undone.
    assert uncropped.extent_m != source.extent_m
    recovered = tuple(
        (n - 2 * uncropped.pad_width) * pitch
        for n, pitch in zip(uncropped.shape, uncropped.sample_pitch_m, strict=True)
    )
    assert recovered == pytest.approx(source.extent_m, rel=1e-12)


def test_a_backend_regrid_is_refused_rather_than_reported_as_the_declared_pitch() -> None:
    """The pitch check, driven directly: a native pitch nobody declared.

    Reached with a stand-in rather than a real propagation because the ASM path
    cannot produce it -- `output_dx` is not exposed. What the refusal means changed
    at CHE-208 (R06.3): it is no longer "the pitch moved" but "the pitch is not the
    one the operation predicted", which is why the declaration passed in here is
    the source pitch and the stand-in returns twice it.
    """
    source = a_field()
    requested = ArrayState(NATIVE_DTYPE, DevicePlacement.parse("cpu"), ArrayNamespace.NUMPY)
    regridded = SimpleNamespace(
        u=np.asarray(source.u), dx=np.asarray([[PITCH_M[0], PITCH_M[1] * 2.0]])
    )
    with pytest.raises(ContractError) as caught:
        from_native(
            regridded,
            source=source,
            requested=requested,
            expected_pitch_m=source.sample_pitch_m,
            reference_surface=source.reference_surface,
            validity=frozenset(),
            pad_width=0,
            padded=False,
        )
    assert caught.value.code == "REPRESENTATION_INCONSISTENT"

    # float32 storage rounding of the same pitch is not a regrid and passes.
    rounded = SimpleNamespace(
        u=np.asarray(source.u), dx=np.asarray(PITCH_M, dtype=np.float32).reshape(1, 2)
    )
    assert from_native(
        rounded,
        source=source,
        requested=requested,
        expected_pitch_m=source.sample_pitch_m,
        reference_surface=source.reference_surface,
        validity=frozenset(),
        pad_width=0,
        padded=False,
    ).sample_pitch_m == PITCH_M


# ---------------------------------------------------------------------------
# 2b. The declaration is the operation's, and it is not optional (CHE-208)
# ---------------------------------------------------------------------------


def test_a_declared_regrid_is_carried_when_the_backend_agrees() -> None:
    """R06.3 criterion 1's boundary half: a *predicted* change is not a refusal.

    The same stand-in that is refused above is accepted once the operation says
    the change is the one it expects -- and what comes back is the **declared
    float64** value, not the float32 the stand-in carried. This is the whole
    difference R06.3 makes: `focal_plane_transform` cannot exist while any change
    of sampling is an error.
    """
    source = a_field()
    requested = ArrayState(NATIVE_DTYPE, DevicePlacement.parse("cpu"), ArrayNamespace.NUMPY)
    declared = (PITCH_M[0], PITCH_M[1] * 2.0)
    regridded = SimpleNamespace(
        u=np.asarray(source.u), dx=np.asarray(declared, dtype=np.float32).reshape(1, 2)
    )

    out = from_native(
        regridded,
        source=source,
        requested=requested,
        expected_pitch_m=declared,
        reference_surface=source.reference_surface,
        validity=frozenset(),
        pad_width=0,
        padded=False,
    )
    assert out.sample_pitch_m == declared
    assert out.sample_pitch_m != source.sample_pitch_m
    # The declared float64 value, not the float32 the backend actually stored --
    # which differs from it, so this is a real distinction and not a tautology.
    stored = float(np.asarray(regridded.dx).reshape(-1)[1])
    assert stored != declared[1]
    assert out.sample_pitch_m[1] == declared[1]


def test_the_declaration_has_no_default_and_cannot_be_inferred() -> None:
    """R06.3's failing-open risk, pinned as a signature fact.

    A declaration the boundary could fill in for itself would make the check a
    tautology, and the errors it exists to catch -- a factor of `N`, of `n`, of
    `2 pi` -- all produce a native pitch that a self-filled declaration would
    match exactly.
    """
    source = a_field()
    requested = ArrayState(NATIVE_DTYPE, DevicePlacement.parse("cpu"), ArrayNamespace.NUMPY)
    native = SimpleNamespace(u=np.asarray(source.u), dx=np.asarray(PITCH_M).reshape(1, 2))
    with pytest.raises(TypeError, match="expected_pitch_m"):
        from_native(  # type: ignore[call-arg]
            native,
            source=source,
            requested=requested,
            reference_surface=source.reference_surface,
            validity=frozenset(),
            pad_width=0,
            padded=False,
        )

    with pytest.raises(ContractError) as caught:
        from_native(
            native,
            source=source,
            requested=requested,
            expected_pitch_m=(0.0, PITCH_M[1]),
            reference_surface=source.reference_surface,
            validity=frozenset(),
            pad_width=0,
            padded=False,
        )
    assert caught.value.code == "MISSING_DECLARATION"


# ---------------------------------------------------------------------------
# 2c. The Fourier-plane sampling relation, as pure arithmetic (CHE-208)
# ---------------------------------------------------------------------------

FOCAL_LENGTH_M = 20e-3


def test_the_fourier_pitch_is_lambda_f_over_n_times_the_extent() -> None:
    """`dx_out = lambda f / (n N dx_in)`, per axis, on a doubly asymmetric grid.

    The oracle is the discrete Fourier relation itself -- `df = 1/(N dx)` scaled
    onto position by `x = lambda f f_x / n` -- computed here in float64 from the
    definition rather than from the implementation. The grid is asymmetric in both
    sample count and pitch, so a transposed axis cannot pass.
    """
    shape = (48, 64)
    got = fourier_plane_pitch_m(
        PITCH_M,
        shape,
        wavelength_m=WAVELENGTH_M,
        focal_length_m=FOCAL_LENGTH_M,
        medium_index=1.0,
    )
    expected = tuple(
        WAVELENGTH_M * FOCAL_LENGTH_M / (1.0 * n * d)
        for n, d in zip(shape, PITCH_M, strict=True)
    )
    assert got == pytest.approx(expected, rel=1e-15)
    # The two axes differ, which is what makes the transposition check below mean
    # something: 48 x 0.30 um and 64 x 0.25 um have different extents.
    assert got[0] != got[1]


def test_a_transposed_declaration_is_a_different_number() -> None:
    """The falsifiable twin. Swapping the axes moves the answer well past tolerance."""
    shape = (48, 64)
    upright = fourier_plane_pitch_m(
        PITCH_M, shape, wavelength_m=WAVELENGTH_M, focal_length_m=FOCAL_LENGTH_M, medium_index=1.0
    )
    transposed = fourier_plane_pitch_m(
        (PITCH_M[1], PITCH_M[0]),
        shape,
        wavelength_m=WAVELENGTH_M,
        focal_length_m=FOCAL_LENGTH_M,
        medium_index=1.0,
    )
    for got, other in zip(upright, transposed, strict=True):
        assert abs(got / other - 1.0) > 0.1, "a transposed pitch has to be visible here"


def test_the_relation_inverts_itself_exactly() -> None:
    """Applying it twice returns the input pitch -- exactly, in float64.

    This is the arithmetic behind R06.4's round trip: the declared value a `+f`
    then `-f` pair predicts is the input's own pitch, so the round trip's pitch
    claim is an identity rather than a tolerance.
    """
    shape = (48, 64)
    once = fourier_plane_pitch_m(
        PITCH_M, shape, wavelength_m=WAVELENGTH_M, focal_length_m=FOCAL_LENGTH_M, medium_index=1.0
    )
    twice = fourier_plane_pitch_m(
        once, shape, wavelength_m=WAVELENGTH_M, focal_length_m=FOCAL_LENGTH_M, medium_index=1.0
    )
    assert twice == pytest.approx(PITCH_M, rel=1e-15)


def test_the_medium_index_enters_as_lambda_over_n() -> None:
    """`n` is not decoration: a water-immersion index halves nothing by accident."""
    shape = (48, 64)
    vacuum = fourier_plane_pitch_m(
        PITCH_M, shape, wavelength_m=WAVELENGTH_M, focal_length_m=FOCAL_LENGTH_M, medium_index=1.0
    )
    medium = fourier_plane_pitch_m(
        PITCH_M, shape, wavelength_m=WAVELENGTH_M, focal_length_m=FOCAL_LENGTH_M, medium_index=1.33
    )
    assert medium == pytest.approx(tuple(v / 1.33 for v in vacuum), rel=1e-15)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"focal_length_m": 0.0},
        {"focal_length_m": math.inf},
        {"medium_index": 0.0},
        {"wavelength_m": -1.0},
    ],
)
def test_the_relation_refuses_arguments_it_cannot_scale(kwargs: dict[str, float]) -> None:
    """Each argument is a physical quantity, and none of them has a sane zero."""
    call = {
        "wavelength_m": WAVELENGTH_M,
        "focal_length_m": FOCAL_LENGTH_M,
        "medium_index": 1.0,
        **kwargs,
    }
    with pytest.raises(ValueError):
        fourier_plane_pitch_m(PITCH_M, (48, 64), **call)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. The memory cost is the padded one
# ---------------------------------------------------------------------------


def test_padded_shape_and_bytes_are_the_cost_and_the_input_shape_is_not() -> None:
    """Criterion 4, first half: the arithmetic, and the size of the mistake it prevents.

    M1 measured a 256^2 input coming back at 1756^2. Sized from the input grid that
    workload is 0.5 MiB; sized from the padded one it is 23.5 MiB for a single
    array, and the transform pair holds several at once.
    """
    assert padded_shape((256, 256), 750) == (1756, 1756)
    assert padded_field_bytes((256, 256), 0) == 256 * 256 * 8
    assert padded_field_bytes((256, 256), 750) == 1756 * 1756 * 8
    assert padded_field_bytes((256, 256), 750) / padded_field_bytes((256, 256), 0) > 40

    with pytest.raises(ValueError):
        padded_shape((16, 16), -1)


@pytest.mark.parametrize("pad_width", [0, 8, 37])
def test_the_estimate_matches_a_measured_run(pad_width: int) -> None:
    """Criterion 4, second half. The prediction is checked against real bytes.

    An uncropped propagation returns the padded array itself, so
    `padded_field_bytes` is exactly its `nbytes` -- a prediction a run can falsify
    rather than a rule of thumb.
    """
    source = a_field()
    out = propagate(
        source, distance_m=10e-6, model=a_model(pad_width=pad_width, crop=False)
    )
    measured = np.asarray(out.u)
    assert measured.shape == padded_shape(source.shape, pad_width)
    assert measured.nbytes == padded_field_bytes(source.shape, pad_width)


# ---------------------------------------------------------------------------
# 4. The wraparound diagnostic
# ---------------------------------------------------------------------------


def test_edge_energy_notices_a_window_that_truncates_the_field() -> None:
    """A diagnostic, and it is used as one: it notices truncation, it does not gate.

    CHE-35 measured it moving by only 2x between a run carrying 1.4e-1 relative
    intensity error from wraparound and a correctly padded one, which is why
    `EDGE_ENERGY_REPORTING_THRESHOLD` is named a reporting threshold.
    """
    confined = a_field()
    assert edge_energy_fraction(confined) < fields.EDGE_ENERGY_REPORTING_THRESHOLD

    filling = ScalarField(
        u=np.ones((48, 64), np.complex64),
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="source", z_m=0.0, medium_index=1.0),
    )
    assert edge_energy_fraction(filling) > fields.EDGE_ENERGY_REPORTING_THRESHOLD
    # A one-sample axis has no border to speak of, and the answer is 0 rather than
    # a division by zero.
    tiny = ScalarField(
        u=np.ones((2, 2), np.complex64),
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="source", z_m=0.0, medium_index=1.0),
    )
    assert edge_energy_fraction(tiny) == 0.0
    assert math.isfinite(edge_energy_fraction(confined))


# ---------------------------------------------------------------------------
# 5. Class delta
# ---------------------------------------------------------------------------


def test_the_package_defines_no_class() -> None:
    """Criterion 5, as an absence claim over the source rather than over exports.

    `scripts/class_budget.py` counts nested classes too, so this walks the AST for
    the same reason: a class defined inside a function would satisfy any check made
    on the module namespace.
    """
    defined = [
        f"{path.relative_to(ROOT)}::{node.name}"
        for path in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    ]
    assert defined == []


def test_none_of_the_avoided_names_came_back() -> None:
    """The five types and the standalone route the ticket names as not landing."""
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(PACKAGE.rglob("*.py"))
    )
    tokens = {
        node.id if isinstance(node, ast.Name) else node.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Name | ast.FunctionDef | ast.AsyncFunctionDef)
    }
    assert sorted(tokens & set(AVOIDED_NAMES)) == []
    assert not (PACKAGE / "baseline.py").exists()
