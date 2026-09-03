"""The translation boundary: neutral `ScalarField` in, neutral `ScalarField` out.

CHE-183 (R06.1). Everything about turning a project field into something
Chromatix will accept, and turning the result back, lives here. Nothing in this
module propagates; `solver.py` is the physics and this is the border it crosses.

No classes. A translation is a pair of functions over two data models that
already exist -- `representations.ScalarField` on one side and the backend's own
field on the other -- and the reference implementation's `ChromatixAdapter`,
`ChromatixWaveRequest` / `ChromatixWaveResult` / `ChromatixWaveFailure` and
`WaveHandoffError` were five types wrapping arguments, a return value and a
`ValueError`.

complex64 is a capability, and the caller is told rather than downcast
-----------------------------------------------------------------------
The backend's own field constructor is `jnp.asarray(u, dtype=jnp.complex64)`,
unconditionally. Handing it a `complex128` array **with** `jax_enable_x64=True`
still yields `complex64`, so there is no `complex128` path at any device and this
project does not claim one. A boundary has two choices about that: absorb it, or
make it visible. Absorbing it is what the reference implementation did -- it
measured the loss and continued -- and the loss it measured is not a rounding
detail on a phase of `k * OPL`, where ~8 decimal digits is the difference between
a resolved wavefront and a plausible-looking one.

So `native_state` runs the request through `numerics.negotiate` against the
measured `M_WAVE_CHROMATIX` row, which refuses `complex128` with
`LOSSY_DOWNCAST_REQUIRED` and names the probe behind the refusal. A caller who
wants the truncation asks for it by handing a `complex64` field, which is the
same decision made where it can be seen.

Padding is the memory cost, and it is not the input shape
----------------------------------------------------------
`asm_propagate` pads before it transforms: M1 measured a 256^2 input coming back
at 1756^2. A caller sizing a workload from the input grid underestimates by two
orders of magnitude, so `padded_shape` and `padded_field_bytes` compute the cost
from the padded array, and the pad width and crop state travel back on the
returned `ScalarField` rather than being inferable from its shape.

Pitch crosses in float64 and is *checked* against the backend, not read from it
--------------------------------------------------------------------------------
The backend stores `dx` as `float32` under `jax_enable_x64=False`, which is the
only mode this adapter runs in. Reading the pitch back off the native field
would therefore inject a ~6e-8 relative rounding into `extent_m` on every round
trip, for a quantity the project can state exactly. `from_native` instead
compares the native pitch against a **declared** one and carries the declared
float64 value -- so a pitch the caller did not predict is a refusal, and a
float32 storage artifact is not silently promoted to a physical one.

The operation declares its output sampling; the boundary verifies it
---------------------------------------------------------------------
CHE-208 (R06.3). Until R06.3 the declaration was implicit and fixed: the result
had to come back on the *source* field's pitch. That is right for ASM, which
preserves sampling, and it is a hard stop for a Fourier-transforming operator,
which must change it. So `expected_pitch_m` is now an argument, with no default:
`propagate` passes the source pitch and behaves exactly as before, while
`focal_plane_transform` passes `fourier_plane_pitch_m(...)`, the analytic value
computed in float64 before the call.

"Unexpected regrid" therefore stops meaning "any change at all" and starts
meaning "not the change that was declared", which is the stronger statement. The
argument is required and never defaulted or inferred: a declaration filled in
from the native field would make the check a tautology, and a factor-of-`N`
sampling error produces a physically plausible image at completely the wrong
scale.

Importing this module imports no backend. `import_backend` is the only route in,
it is called from inside functions, and `tests/backends/test_chromatix_boundary.py`
asserts that against `sys.modules` in a fresh interpreter.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from numerics import (
    ArrayState,
    DType,
    array_state,
    load_capabilities,
    negotiate,
    numpy_dtype,
    refusal,
    to_state,
)
from representations import ContractError, ReferenceSurface, ScalarField, ValidityFlag

__all__ = [
    "CAPABILITIES",
    "EDGE_ENERGY_REPORTING_THRESHOLD",
    "NATIVE_DTYPE",
    "PINNED_COMMIT",
    "PINNED_VERSION",
    "edge_energy_fraction",
    "fourier_plane_pitch_m",
    "from_native",
    "import_backend",
    "native_state",
    "padded_field_bytes",
    "padded_shape",
    "to_native",
]

#: The measured capability record this package executes within, loaded once at
#: module scope from `knowledge/capabilities/M_WAVE_CHROMATIX.json` (CHE-223 /
#: R03.6). It used to be `numerics.CHROMATIX_CAPABILITIES`; the data moved to the
#: knowledge pack because backend-free discovery cites the same measurement and
#: cannot import this package. Loading it imports no backend.
_CAPABILITIES = load_capabilities("M_WAVE_CHROMATIX")

#: The row of the capability pack this package executes within.
#: Cited by name; the measurement stays with its probe.
CAPABILITIES = _CAPABILITIES.component

#: The one storage dtype the backend has. Not a policy: see the module docstring.
NATIVE_DTYPE = DType.COMPLEX64

#: The build this adapter is written against. There is an unrelated package
#: literally named `chromatix` on PyPI, so "it imported" is not evidence that the
#: right one did; the import failure below says which one is meant.
PINNED_VERSION = "0.6.0"
PINNED_COMMIT = "d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee"

#: Above this fraction of `|u|^2` on the one-pixel border, the sampled window is
#: truncating the field and any power or second-moment metric on it is
#: window-limited.
#:
#: A **reporting** threshold and never a pass/fail gate on its own: CHE-35
#: measured it moving by only 2x between a run carrying 1.4e-1 relative intensity
#: error from wraparound and a correctly padded one. It notices truncation; it
#: does not certify padding.
EDGE_ENERGY_REPORTING_THRESHOLD = 0.05

#: How far the native `float32` pitch may sit from the declared `float64` one
#: before the difference is a physical change rather than a storage artifact.
#: float32 carries ~1.2e-7 relative, so 1e-6 admits the storage rounding with a
#: factor of eight in hand and rejects any real regrid.
#:
#: **Re-derived for the regridding path rather than carried over** (CHE-208). The
#: ASM comparison is one float32 quantity round-tripping, so one epsilon. A
#: Fourier-plane pitch is `dx * df * lambda * f`-worth of float32 arithmetic
#: inside the backend -- `df` from `dx`, then a multiply and a divide -- so the
#: accumulated relative error is a small multiple of 1.2e-7 rather than one of
#: them. Four epsilons is 4.8e-7; 1e-6 still clears it, with about a factor of two
#: in hand instead of eight. It is kept because a *real* regrid is never within a
#: factor of two of the declared value -- the errors this catches are factors of
#: `N`, of `n`, or of `2 pi` -- and tightening it to the measured margin would
#: make the gate a float32 noise detector.
_PITCH_STORAGE_RTOL = 1e-6


def import_backend() -> tuple[Any, Any, Any]:
    """Import `jax`, `jax.numpy` and `chromatix.functional`, with x64 pinned off.

    The only import route into this package, and it is called from inside
    functions rather than at module scope.

    `jax_enable_x64` is process-global mutable state and anything in the process
    may have turned it *on*, possibly as an import side effect that Python will
    not re-trigger. Under x64 the FFTs behind the propagation promote to
    `complex128` while the field storage stays `complex64`, so the numbers would
    silently depend on what else had been imported first. It is therefore set on
    every call, not checked.
    """
    try:
        import chromatix.functional as cf
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "chromatix and/or jax could not be imported. This adapter is written "
            f"against chromatix {PINNED_VERSION} @ {PINNED_COMMIT} "
            "(git+https://github.com/chromatix-team/chromatix.git); the PyPI package "
            f"literally named 'chromatix' is an unrelated namesquat. Underlying: {exc!r}"
        ) from exc

    jax.config.update("jax_enable_x64", False)  # type: ignore[no-untyped-call]
    return jax, jnp, cf


def native_state(field: ScalarField) -> ArrayState:
    """The array state `field` must be in to enter the backend, or a refusal.

    Pure: it reads the field's observed state and negotiates it against the
    measured capability row. No array is converted and no backend is imported, so
    a `complex128` request is refused before jax is in the process.

    Raises:
        ValueError: carrying `code='LOSSY_DOWNCAST_REQUIRED'` for a `complex128`
            field -- the backend will ingest it and truncate it internally, and
            this boundary's job is to make that visible rather than absorb it.
    """
    return negotiate(field.state, _CAPABILITIES)


def to_native(field: ScalarField) -> tuple[Any, ArrayState]:
    """Build the backend's field from `field`, and report the state it went in as.

    Returns `(native_field, state)`. `state` is what `negotiate` decided and is
    what `from_native` checks the output against -- a requested placement and an
    observed one are different facts, and a process-global JAX platform pin
    produces a successful host run for a caller who asked for a device.
    """
    jax, jnp, cf = import_backend()
    state = native_state(field)
    u = to_state(field.u, state)

    dy, dx = field.sample_pitch_m
    # `(1, 2)` and not a bare `(2,)`: the backend reads the outer axis as
    # wavelengths, so a bare pair for a monochromatic field is rejected as "number
    # of wavelengths does not match". `[dy, dx]` is the backend's own order --
    # `Field.extent` comes back as `(ny * dy, nx * dx)`, verified on a
    # deliberately non-square grid.
    pitch = jax.device_put(jnp.asarray([[dy, dx]]), next(iter(u.devices())))
    native = cf.Field.build(u, pitch, field.wavelength_m)
    return native, state


def from_native(
    native: Any,
    *,
    source: ScalarField,
    requested: ArrayState,
    expected_pitch_m: tuple[float, float],
    reference_surface: ReferenceSurface,
    validity: frozenset[ValidityFlag],
    pad_width: int,
    padded: bool,
) -> ScalarField:
    """Read the backend's field back into a neutral `ScalarField`.

    The returned array is in the namespace the caller's field arrived in, on the
    device the computation actually happened on. A NumPy caller therefore never
    receives a JAX buffer, and a JAX caller keeps its device residency instead of
    being pushed through the host to satisfy a type rule.

    Args:
        expected_pitch_m: the `(dy, dx)` the calling operation predicts, in
            float64 project units. Required and never defaulted: it is the
            declaration this boundary checks the backend against, and it is what
            the returned field carries. A sampling-preserving operation passes the
            source field's own pitch; a Fourier-transforming one passes
            `fourier_plane_pitch_m(...)`.

    Raises:
        ValueError: the output landed on a device other than the one the
            negotiation placed it on (`DEVICE_TRANSFER_NOT_PERMITTED`). A
            requested device is never reported as an actual one.
        ContractError: the backend returned a pitch that is not the declared one,
            or more than one wavelength's worth of pitch.
    """
    u = native.u
    observed = array_state(u)
    if observed.device.kind is not requested.device.kind:
        raise refusal(
            code="DEVICE_TRANSFER_NOT_PERMITTED",
            component=CAPABILITIES,
            message=(
                f"the propagation was placed on {requested.device} but the output array "
                f"landed on {observed.device}. The usual cause is a process-global JAX "
                "platform pin (jax_platform_name / JAX_PLATFORMS) set before this "
                "adapter got a say; any device claim for this run would be false."
            ),
            requested=requested.device,
            supported=[str(observed.device)],
            evidence=_CAPABILITIES.cited_evidence,
        )

    native_pitch = np.asarray(native.dx).reshape(-1)
    if native_pitch.size != 2:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"the backend returned {native_pitch.size} pitch value(s), not a single "
            "(dy, dx) pair. A multi-wavelength field has no single pitch for this "
            "boundary to declare.",
            declaration="sample_pitch_m",
        )
    declared = tuple(float(value) for value in expected_pitch_m)
    if len(declared) != 2 or not all(math.isfinite(v) and v > 0.0 for v in declared):
        raise ContractError(
            "MISSING_DECLARATION",
            f"expected_pitch_m={expected_pitch_m!r} is not a positive, finite (dy, dx) pair. "
            "The operation has to predict its own output sampling for this boundary to "
            "check anything; a declaration read back off the native field would make the "
            "check a tautology.",
            declaration="expected_pitch_m",
        )
    if not all(
        abs(float(got) - want) <= _PITCH_STORAGE_RTOL * want
        for got, want in zip(native_pitch, declared, strict=True)
    ):
        raise ContractError(
            "REPRESENTATION_INCONSISTENT",
            f"the backend returned a sample pitch of {tuple(float(v) for v in native_pitch)} m "
            f"where the operation declared {declared} m. That is a regrid the caller did not "
            f"predict, not float32 storage rounding (which is bounded by "
            f"{_PITCH_STORAGE_RTOL:g} relative), and an extent taken from the declared pitch "
            "would be wrong by the ratio.",
            declaration="expected_pitch_m",
        )

    return ScalarField(
        u=to_state(u, ArrayState(NATIVE_DTYPE, observed.device, source.state.namespace)),
        # The **declared** float64 pitch, checked against the backend above and
        # not read out of it: `dx` is float32 storage, and a round trip through it
        # would move every extent this project reports by ~6e-8 relative.
        sample_pitch_m=(declared[0], declared[1]),
        wavelength_m=source.wavelength_m,
        reference_surface=reference_surface,
        frame=source.frame,
        validity=validity,
        pad_width=pad_width,
        padded=padded,
    )


def fourier_plane_pitch_m(
    sample_pitch_m: tuple[float, float],
    shape: tuple[int, int],
    *,
    wavelength_m: float,
    focal_length_m: float,
    medium_index: float,
) -> tuple[float, float]:
    """`(dy, dx)` at the focal plane conjugate to a grid of `shape` at `sample_pitch_m`.

    The analytic sampling relation of an optical Fourier transform, per axis:

        dx_out = lambda * f / (n * N * dx_in)

    which is `df = 1 / (N dx)` -- the discrete Fourier relation -- scaled onto
    position by `x = lambda f f_x / n`. Pure float64 arithmetic: it imports no
    backend, takes no array, and is therefore both what an operator declares to
    `from_native` and what a test holds the backend to.

    It is *derivable* rather than merely *checkable* against the pinned build:
    `chromatix.functional.optical_fft` computes `L_sq = lambda * z / n` and
    `du = field.df * |L_sq|`, which is the same statement. That agreement is a
    consistency check between two implementations of one formula, so the formula
    is what gates; the comparison is evidence.

    `shape` is the grid **actually transformed**, not the caller's window. `N`
    is in the denominator, so padding an optical FT changes its output pitch --
    which is why `focal_plane_transform` refuses a pad width rather than choosing
    one.

    Args:
        sample_pitch_m: `(dy, dx)` of the input grid, in metres.
        shape: `(ny, nx)` of the grid being transformed.
        wavelength_m: vacuum wavelength, in metres.
        focal_length_m: focal length, in metres. The sign of the transform
            direction does not enter: `|f|` sets the scale either way.
        medium_index: real index of the surrounding medium. It enters as
            `lambda / n`, the wavelength in the medium.
    """
    if not (math.isfinite(focal_length_m) and focal_length_m != 0.0):
        raise ValueError(
            f"focal_length_m={focal_length_m!r} must be a finite, non-zero length in metres"
        )
    if not (math.isfinite(medium_index) and medium_index > 0.0):
        raise ValueError(f"medium_index={medium_index!r} must be positive and finite")
    if not (math.isfinite(wavelength_m) and wavelength_m > 0.0):
        raise ValueError(f"wavelength_m={wavelength_m!r} must be positive and finite")

    scale = float(wavelength_m) * abs(float(focal_length_m)) / float(medium_index)
    pitches = []
    for count, pitch in zip(shape, sample_pitch_m, strict=True):
        count = int(count)
        pitch = float(pitch)
        if count < 1 or not (math.isfinite(pitch) and pitch > 0.0):
            raise ValueError(
                f"a transformed axis needs at least one sample and a positive pitch, got "
                f"{count!r} samples at {pitch!r} m"
            )
        pitches.append(scale / (count * pitch))
    return (pitches[0], pitches[1])


def padded_shape(shape: tuple[int, int], pad_width: int) -> tuple[int, int]:
    """`(ny, nx)` of the array the propagation actually transforms.

    The backend pads *before* the transform pair and by `pad_width` on **each**
    side of **each** axis, so the transformed grid is `n + 2 * pad_width` per
    axis. This is the shape every cost below is taken on.
    """
    if pad_width < 0:
        raise ValueError(f"pad_width must be a non-negative sample count, got {pad_width!r}")
    ny, nx = (int(value) for value in shape)
    return (ny + 2 * pad_width, nx + 2 * pad_width)


def padded_field_bytes(shape: tuple[int, int], pad_width: int) -> int:
    """Bytes of **one** padded complex64 field array.

    The unit of memory cost, not the whole cost: the transform pair holds several
    arrays of this size at once, so a workload estimate is a small multiple of
    this and never a fraction of it. Stated as one array because that is the
    number a measured run can confirm exactly -- `padded_field_bytes(shape, p)` is
    the `nbytes` of an uncropped output, which
    `tests/backends/test_chromatix_fields.py` checks against a real propagation.

    The point of the function is the `pad_width`: at a 256^2 input and the
    pad width the M1 configuration needed, this is two orders of magnitude above
    what the input shape suggests.
    """
    ny, nx = padded_shape(shape, pad_width)
    return ny * nx * int(numpy_dtype(NATIVE_DTYPE).itemsize)


def edge_energy_fraction(field: ScalarField) -> float:
    """Fraction of `|u|^2` sitting on the one-pixel border of the sampled window.

    The observable finite-window indicator: a value far above the floor means the
    window truncates the field, so power and second-moment metrics taken on this
    grid are window-limited. A **diagnostic**; see
    `EDGE_ENERGY_REPORTING_THRESHOLD` for why it does not gate.
    """
    xp = field.xp
    intensity = xp.abs(field.u) ** 2
    total = float(xp.sum(intensity))
    if total <= 0.0 or min(field.shape) < 3:
        return 0.0
    border = float(
        xp.sum(intensity[0, :])
        + xp.sum(intensity[-1, :])
        + xp.sum(intensity[1:-1, 0])
        + xp.sum(intensity[1:-1, -1])
    )
    return border / total
