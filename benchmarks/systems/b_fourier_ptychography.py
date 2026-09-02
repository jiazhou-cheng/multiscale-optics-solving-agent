"""Fourier ptychography, forward model only, against a fully analytic oracle.

CHE-213 (R06.8). Run it:

    ./run.sh python -m benchmarks.systems.b_fourier_ptychography

The 4f benchmark (R06.7) proves the primitives compose into a textbook system
whose answer is closed form. This proves something that one cannot: that the same
primitives express a **computational-imaging** system, where the physics is a
parameterized *family* of measurements, the illumination angle is a swept
variable, and the observable is an intensity rather than a field.

**Forward model only.** No reconstruction, no phase retrieval, no optimization,
no loss function, no pupil recovery. R06.9 is the differentiable-execution
question and it is deliberately separate.

The physical graph, as a plan
-----------------------------
::

    S_SOURCE_PLANE_WAVE         tilted coherent illumination, k_j
      -> O_COMPLEX_TRANSMISSION    complex object transmission
      -> O_FOCAL_PLANE_TRANSFORM   focal-plane transform, f1
      -> O_COMPLEX_TRANSMISSION    finite-NA pupil, at the Fourier plane
      -> O_FOCAL_PLANE_TRANSFORM   focal-plane transform, f2
      -> M_PSF                     intensity, normalization='raw'

repeated over a set of illumination wavevectors.

Those six ids are the plan this module hands to `runtime.Executor`, one node per
step, each with its own request -- and the sweep is a **plan parameter**, not a
different code path: an illumination angle is one number in step 0's request, and
the seven swept angles are seven runs of one plan. The benchmark imports no
operation; the ids are resolved through `operations.CATALOG` inside the executor.

`planning.routes` cannot enumerate this plan, because an operation may not repeat
in a route and this one transforms twice and masks twice. So the plan is written
down here and each consecutive edge is checked against
`planning.capability_graph()` before anything runs -- see `_semantic_chain`. The
two focal lengths and the two masks live in their own nodes' requests, which is
what the per-node request form is for: bound from one flat mapping keyed by
parameter name, this plan ran f1 on both legs and no pupil at all, and reported
every node `completed`.

Two structural points, neither assumed
---------------------------------------
**The pupil is applied at the Fourier plane, between the two transforms.** Not
through `ff_lens(NA=)`, which in the pinned build applies `circular_pupil` to the
*incoming* field -- the front focal plane, the wrong plane for a system stop.
R06.6 exists so the stop is a separately placed mask, and
`numerical_aperture_radius_m` gives its radius in the Fourier plane's own
coordinates.

**Tilt-as-spectral-shift is exact here, and it is exact because the object is a
thin element.** `O(x) exp(i k_j x)` has spectrum `O~(k - k_j)` exactly for a thin,
angle-independent transmission -- which is what `operators.complex_transmission`
implements and declares in its `approximation`. It would **not** hold for a thick
or angle-dependent sample, and the whole method rests on it, so it is said here
rather than left as an assumption of the arithmetic.

The oracle, and why it is a predicate
--------------------------------------
The object is three spatial-frequency lines:

    O(x) = c_0 + c_+ exp(+i 2 pi f_o x) + c_- exp(-i 2 pi f_o x)

with `|c_+| != |c_-|` and both complex. The asymmetry is deliberate and it is
load-bearing: for a **real** object -- an amplitude-only `1 + a cos(...)` -- a
`k_j` sign flip produces a *bit-identical* image intensity, because flipping the
illumination is then equivalent to `x -> -x` on a Hermitian spectrum. The
negative control the ticket asks for would silently not fire. A complex,
asymmetric object is what makes the sign of the illumination observable at all.

With illumination at bin `j` and an object line at bin `o * j_o`, the order sits
at bin `j + o j_o` of the Fourier plane, and whether it passes is arithmetic:

    passes  iff  |j + o j_o| <= NA / lambda * N dx   ==   R / dx_F

Everything downstream then follows in closed form. The two forward legs give
`U_img[k] = -(f1/f2) V_filtered[-k]` (R06.7's composition identity), the common
illumination carrier is a unit-modulus factor that `|.|^2` removes, and the image
intensity is

    I[k] = (f1/f2)^2 * | sum_{o in S} c_o exp(-i o theta_k) |^2 ,  theta_k = 2 pi j_o k / N

whose harmonic `m` has the **absolute** coefficient `sum_{o - o' = m} c_o conj(c_o')`
over the surviving set `S`. Three cases fall out, and they are the classic
synthetic-aperture statement:

* **three beams** -- a first harmonic *and* a second harmonic at `c_+ conj(c_-)`;
* **two beams** (the DC line plus one sideband) -- a first harmonic whose
  amplitude is that sideband's alone, and **no second harmonic**;
* **one beam** -- a flat intensity, no fringes: dark field.

So sweeping `k_j` across the angle where an order leaves the pupil produces a
**step**: the second harmonic goes from `|c_+ conj(c_-)|` to exactly zero at an
analytically known bin. That step is the gate. It tests the illumination
convention, the pupil placement, the pupil radius, the sampling chain and the
intensity normalization *simultaneously*, against arithmetic rather than against
another simulation, and a plausible-looking wrong answer cannot pass it.

No FFT appears anywhere in the oracle. The measurement reads harmonic amplitudes
with a DFT of the image intensity; the numbers it is compared against are
products and sums of the three object coefficients.

Nothing is peak-normalized
--------------------------
Every comparison is absolute, including the `(f1/f2)^2` prefactor and the object's
normalization constant. A peak-normalized metric cannot see a global scale error,
which is how a wrong amplitude convention survives; R11 records that as the trap
that let a pre-CHE-47 launch convention pass a frozen oracle.

The secondary model is diagnostic and is labelled so
-----------------------------------------------------
`ifft2(P * fft2(u))` in float64 NumPy, on a random complex phantom with no closed
form. It shares no code with the Chromatix path, so its agreement is real
evidence -- and it is still repository numerical code, so it decides nothing. Its
residual is reported with `oracle_kind='diagnostic'`.

Cost: CPU, 3.1 s of physics for both configurations (2 x 7 swept angles,
96 x 512). No GPU. Each measurement now runs two plans rather than one -- the
prefix to the Fourier plane, whose pitch sizes the pupil mask, and then the whole
graph -- so the wall time is roughly double that, plus 3.1 ms of executor overhead
per run. One `Executor` per configuration, because the environment fingerprint is
read once per `__enter__` at 7.5 ms.
"""

from __future__ import annotations

import cmath
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from backends.chromatix import fourier_plane_pitch_m
from benchmarks.record import control, describe_plan, gate, write_record
from operations import CATALOG
from operators import circular_aperture_amplitude, numerical_aperture_radius_m
from planning import ENTRY, capability_graph
from representations import ReferenceSurface, ScalarField
from runtime import Executor, PlanNode, normalize_plan

BENCHMARK_ID = "B-FP-FORWARD"
RECORDS = Path(__file__).resolve().parent / "records"

#: A NumPy array of unspecified dtype. An alias so the annotations stay readable.
Array = np.ndarray[Any, np.dtype[Any]]

WAVELENGTH_M = 0.532e-6
MEDIUM_INDEX = 1.0

#: 96 x 512 at 0.6 x 0.5 um. Long in `x` because that is the swept axis and the
#: sweep needs bins; asymmetric in both count and pitch so a transposed `(y, x)`
#: illumination is a different physical run rather than a relabelling -- the
#: Fourier plane's two pitches then differ by 4.4x and the pupil is 144 samples
#: across in `x` against 32 in `y`.
SHAPE = (96, 512)
PITCH_M = (0.6e-6, 0.5e-6)

#: NA 0.15 in air. At f1 = 20 mm that is a 3.000 mm stop, whose radius is 72.18
#: samples of this Fourier plane's `x` pitch. The cutoff therefore falls *between*
#: bins, which is what makes the pass/block step's location a prediction with no
#: adjustable part: order bin 72 is inside and 73 is outside.
NUMERICAL_APERTURE = 0.15

#: The object's spatial frequency, in bins of `1 / (N dx)`. 20 bins is a 12.8 um
#: period, 25.6 samples -- well sampled -- and it puts the second intensity
#: harmonic at bin 40, far below the grid's Nyquist at bin 256.
OBJECT_BIN = 20

#: The three object coefficients, before normalization: `c_0`, `c_+`, `c_-`.
#: `|c_+| != |c_-|` and both carry a phase. See the module docstring on why a real
#: object would make the `k_j` sign control silently pass.
OBJECT_DEPTH = 0.6
OBJECT_ASYMMETRY = 0.5
OBJECT_PHASE_PLUS_RAD = 0.7
OBJECT_PHASE_MINUS_RAD = 1.9

#: Illumination bins. `0` is on axis; `20` and `40` are comfortably inside the
#: pass band; `52` is the largest bin whose `+1` order (at 72) is still inside the
#: 72.18-sample cutoff and `53` is the first whose `+1` order (73) is outside, so
#: the pair straddles the analytic step; `65` is a two-beam case well past it; and
#: at `90` the DC line itself (bin 90) is outside the pupil and only the `-1` order
#: (bin 70) survives -- one beam, dark field, no fringes at all.
ON_GRID_BINS: tuple[float, ...] = (0.0, 20.0, 40.0, 52.0, 53.0, 65.0, 90.0)

#: Angles deliberately off the DFT frequency grid (criterion 7). `40.5` is well
#: inside the pass band, where leakage is the only effect; `52.5` straddles the
#: cutoff, where the `+1` order is *partly* transmitted and the harmonic amplitudes
#: land between the three-beam and two-beam predictions. Reported, never gated:
#: the closed form above is a statement about lines on the DFT grid.
OFF_GRID_BINS: tuple[float, ...] = (40.5, 52.5)

#: complex64 floor for a two-leg system carrying an intensity reading, derived:
#: float32 is 1.19e-7 relative, two FFT legs accumulate a small multiple of it,
#: and squaring the amplitude doubles the relative error. Eight epsilons is 9.5e-7;
#: 2e-6 is used for the harmonic amplitudes, which are differences of
#: same-magnitude terms. Measured worst residual across both configurations and
#: every swept angle is 1.1e-7, one epsilon. A real convention error here is O(1):
#: a blocked order that survives, or a second harmonic that should be exactly
#: zero -- every negative control below lands between 0.13 and 1.8.
COMPLEX64_FLOOR = 2e-6

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {"name": "unit_relay", "focal_length_1_m": 20e-3, "focal_length_2_m": 20e-3},
    {"name": "magnifying_relay", "focal_length_1_m": 20e-3, "focal_length_2_m": 40e-3},
)


# ---------------------------------------------------------------------------
# The object, as three coefficients and as a thin element
# ---------------------------------------------------------------------------


def _raw_intensity(measured: Any) -> np.ndarray[Any, np.dtype[Any]]:
    """An `M_PSF` node's result, widened to host float64.

    The widening is the one thing this wrapper adds and it is deliberate: the
    measurement keeps `|u|^2` in the field's own precision, which is complex64
    here, and the harmonic readings below sum over a large grid where float32
    accumulation loses digits they need. Squaring stays where the measurement puts
    it -- inside `measurements.psf`, in the field's precision -- so nothing is
    recomputed on the host.

    This is not a second intensity path, and it no longer even names one: `|U|^2`
    is computed by the `M_PSF` node of the plan, which the executor resolves to
    `measurements.psf`. `benchmarks/observables.py` was a second path, and R11.1
    landing `measurements/` is the condition under which its own docstring said it
    would be deleted; it was.
    """
    return np.asarray(measured.intensity, dtype=np.float64)


def object_coefficients() -> dict[int, complex]:
    """`{order: c_order}`, normalized so `max |O| <= 1`.

    The normalization is `|c_0| + |c_+| + |c_-|`, which bounds the modulus of the
    sum: `complex_transmission` refuses `A > 1` without `allow_gain`, and a
    passive object should not need the opt-in. The constant is carried into every
    analytic prediction rather than divided out, so the comparisons stay absolute.
    """
    raw = {
        0: 1.0 + 0.0j,
        +1: (OBJECT_DEPTH / 2.0) * cmath.exp(1j * OBJECT_PHASE_PLUS_RAD),
        -1: (OBJECT_ASYMMETRY * OBJECT_DEPTH / 2.0)
        * cmath.exp(-1j * OBJECT_PHASE_MINUS_RAD),
    }
    norm = sum(abs(value) for value in raw.values())
    return {order: value / norm for order, value in raw.items()}


def _require_resolvable_object(object_bin: int) -> None:
    """Refuse an object frequency the grid cannot carry, rather than alias it.

    The refusal is *here* and not in `operators.complex_transmission`, and that is
    a deliberate split. Whether an arbitrary mask array contains structure finer
    than the pitch is not decidable from the array -- the operator therefore
    *declares* the limitation in its `validity` and samples what it is given. This
    benchmark knows its object's frequency analytically, so it is the layer that
    can check it, and an aliased object frequency is exactly the failure that
    would come back as a plausible answer at a different frequency.

    The bound is on the **second intensity harmonic** at `2 * object_bin`, not on
    the object frequency alone: the object could be resolvable while the harmonic
    this benchmark reads is not.
    """
    nyquist_bin = SHAPE[1] // 2
    if not 0 < 2 * object_bin < nyquist_bin:
        raise ValueError(
            f"an object at bin {object_bin} puts its second intensity harmonic at bin "
            f"{2 * object_bin}, against this grid's Nyquist bin {nyquist_bin}. An object "
            "frequency past Nyquist aliases into a completely plausible answer at a "
            "different frequency, so it is refused rather than run."
        )


def _surface(name: str) -> ReferenceSurface:
    return ReferenceSurface(name=name, z_m=0.0, medium_index=MEDIUM_INDEX)


# ---------------------------------------------------------------------------
# The plan: node builders, the edge check, and the one call that executes
# ---------------------------------------------------------------------------
#
# Every node is `(operation_id, request)` and nothing here calls an operation. The
# `operators` names this module imports -- `circular_aperture_amplitude` and
# `numerical_aperture_radius_m` -- are mask builders rather than operations, which
# is what `operators/__init__.py` says where it declares `OPERATIONS`, and so is
# `fourier_plane_pitch_m`.


def _source_node(transverse_wavevector: tuple[float, float]) -> PlanNode:
    return (
        "S_SOURCE_PLANE_WAVE",
        {
            "shape": SHAPE,
            "sample_pitch_m": PITCH_M,
            "wavelength_m": WAVELENGTH_M,
            "reference_surface": _surface("front_focal"),
            "transverse_wavevector_rad_per_m": transverse_wavevector,
        },
    )


def _element_node(
    *, amplitude: Any = None, phase_rad: Any = None, target_surface: str | None = None
) -> PlanNode:
    """One thin element. Only the arguments actually given reach the request.

    `O_COMPLEX_TRANSMISSION` declares every argument optional, so an omitted
    `phase_rad` must be *absent* rather than present as `None`: the executor binds
    every optional argument it finds by name, and `phase_rad=None` would be handed
    to an operator whose own default is not `None`.
    """
    request: dict[str, Any] = {}
    if amplitude is not None:
        request["amplitude"] = amplitude
    if phase_rad is not None:
        request["phase_rad"] = phase_rad
    if target_surface is not None:
        request["target_surface"] = target_surface
    return ("O_COMPLEX_TRANSMISSION", request)


def _leg_node(focal_length_m: float, target: str, *, direction: str = "forward") -> PlanNode:
    return (
        "O_FOCAL_PLANE_TRANSFORM",
        {
            "focal_length_m": focal_length_m,
            "model": {"target_surface": target, "direction": direction},
        },
    )


def _measure_node() -> PlanNode:
    """The one intensity in the project, as the last node of the plan.

    `normalization='raw'` is in the request and not a default anywhere: the
    executor refuses a plan whose `M_PSF` node does not name one, which is the rule
    `tests/integration/test_executor.py` pins -- a runtime does not choose a
    physical parameter for a caller, and peak normalization is blind to exactly the
    global scale error every comparison here is absolute in order to catch.
    """
    return ("M_PSF", {"normalization": "raw"})


def _semantic_chain(plan: Sequence[PlanNode]) -> tuple[str, ...]:
    """The semantic types the plan passes through, refusing a step that cannot run.

    The part of `planning/` that answers a question about a hardcoded plan: each
    step must be in `capability_graph()[state]`, the operations that consume what
    the previous step produced, and the first must be a graph entry, which `ENTRY`
    (`None`) keys. Not `planning.routes`, which will not enumerate a plan that
    repeats an operation -- and this one repeats two.

    Raises:
        ValueError: a step does not consume what the step before it produced.
    """
    graph = capability_graph()
    produces = {descriptor.operation_id: descriptor.primary_output for descriptor in CATALOG}
    chain: list[str] = []
    state: str | None = ENTRY
    for index, (operation_id, _) in enumerate(normalize_plan(plan)):
        if operation_id not in graph.get(state, ()):
            raise ValueError(
                f"plan step {index} is {operation_id}, which does not consume "
                f"{state!r}. The operations that do are {list(graph.get(state, ()))}."
            )
        state = produces[operation_id]
        chain.append(state)
    return tuple(chain)


def _run(plan: Sequence[PlanNode], *, executor: Executor) -> Any:
    """Execute one plan and return what it produced. The whole physics path.

    Refuses to hand back the result of a run that did not complete, and names the
    node that stopped it: a refused node leaves `executor.result` at `None`, and a
    `None` read as an intensity is the fabricated result this project's rules are
    about.

    Raises:
        RuntimeError: the run refused or failed at some node.
    """
    record = executor.execute(plan)
    if record.status != "completed":
        stopped = [
            (node.operation_id, node.status, node.diagnostics)
            for node in record.nodes
            if node.status != "completed"
        ]
        raise RuntimeError(
            f"the plan {list(record.route)} did not complete ({record.status}): {stopped}"
        )
    return executor.result


def _grid_axes(executor: Executor) -> tuple[Array, Array]:
    """`(y, x)` of the source grid, read off a field the source plan produced.

    A one-node plan rather than a coordinate formula written here: the `n // 2`
    origin is `representations.ORIGIN_RULE`'s declaration, and a second copy of it
    in a benchmark is the kind of restatement that silently disagrees.
    """
    field = _run((_source_node((0.0, 0.0)),), executor=executor)
    y, x = field.coordinates()
    return np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)


def _wavevector_for_bin(bin_index: float) -> float:
    """`k_x` in **rad/m** for an illumination at `bin_index` of `1/(N dx)`.

    `k = 2 pi f`, and the `2 pi` is written here once. This is the conversion the
    negative control below gets wrong on purpose, and R06.5's whole ticket is
    about the fact that the backend spells the same argument name both ways.
    """
    return 2.0 * math.pi * bin_index / (SHAPE[1] * PITCH_M[1])


def _object_profile(coefficients: dict[int, complex], x_m: Array) -> Array:
    period_m = SHAPE[1] * PITCH_M[1] / OBJECT_BIN
    profile: Array = np.zeros(SHAPE[1], dtype=np.complex128)
    for order, value in coefficients.items():
        profile = profile + value * np.exp(2j * np.pi * order * x_m / period_m)
    return profile


def _object_node(coefficients: dict[int, complex], axes: tuple[Array, Array]) -> PlanNode:
    """The three-line object, as one thin-element node.

    The mask is sampled on `axes`, the source grid's own coordinates. Which
    illumination it modulates is the plan's business and not this node's -- the
    tilt lives in the source node's request -- and that separation is what makes an
    illumination sweep a sweep over one plan parameter.
    """
    _, x = axes
    profile = _object_profile(coefficients, x)
    grid = np.broadcast_to(profile[None, :], SHAPE)
    return _element_node(
        amplitude=np.abs(grid).copy(),
        phase_rad=np.angle(grid).copy(),
        target_surface="object",
    )


def _illumination_nodes(
    bin_index: float,
    coefficients: dict[int, complex],
    axes: tuple[Array, Array],
    *,
    transpose: bool = False,
) -> tuple[PlanNode, PlanNode]:
    """The first two steps: the tilted plane wave, and the object behind it."""
    wavevector = _wavevector_for_bin(bin_index)
    return (
        _source_node((wavevector, 0.0) if transpose else (0.0, wavevector)),
        _object_node(coefficients, axes),
    )


# ---------------------------------------------------------------------------
# The measurement path
# ---------------------------------------------------------------------------


def _stop(pitch_m: tuple[float, float], radius_m: float) -> Array:
    return circular_aperture_amplitude(
        SHAPE, sample_pitch_m=pitch_m, radius_m=radius_m, edge="hard"
    )


def _measure(
    bin_index: float,
    coefficients: dict[int, complex],
    *,
    executor: Executor,
    axes: tuple[Array, Array],
    f1: float,
    f2: float,
    radius_m: float,
    stop_at_the_image_plane: bool = False,
    transpose: bool = False,
    second_leg: str = "forward",
) -> tuple[Array, ScalarField]:
    """One illumination: the whole plan, returning `(intensity, image_field)`.

    Two plans are run, and the split is physical rather than mechanical: the pupil
    mask is sized in the **Fourier plane's own pitch**, and a radius in metres
    means nothing until the plane it sits in has a scale. So the prefix to the
    Fourier plane runs first, its pitch sizes the stop, the stop becomes step 3,
    and then the graph runs to the image and to the intensity. The image field is
    the plan without its `M_PSF` node; the intensity is the plan with it.

    `stop_at_the_image_plane` moves the mask from between the legs to after the
    second one, and note what that is: **the same nodes in a different order**. A
    negative control here is a reordered plan, not a second code path, so it cannot
    pass by quietly bypassing the composition the gate measured.
    """
    prefix: tuple[PlanNode, ...] = (
        *_illumination_nodes(bin_index, coefficients, axes, transpose=transpose),
        _leg_node(f1, "fourier"),
    )
    fourier = _run(prefix, executor=executor)
    if stop_at_the_image_plane:
        upto_image: tuple[PlanNode, ...] = (
            *prefix,
            _element_node(amplitude=1.0),
            _leg_node(f2, "image", direction=second_leg),
        )
        image = _run(upto_image, executor=executor)
        plan = (
            *upto_image,
            _element_node(
                amplitude=_stop(image.sample_pitch_m, radius_m), target_surface="image"
            ),
        )
        image = _run(plan, executor=executor)
    else:
        plan = (
            *prefix,
            _element_node(
                amplitude=_stop(fourier.sample_pitch_m, radius_m), target_surface="pupil"
            ),
            _leg_node(f2, "image", direction=second_leg),
        )
        image = _run(plan, executor=executor)
    measured = _run((*plan, _measure_node()), executor=executor)
    return _raw_intensity(measured), image


def _harmonics(measured: Array) -> dict[int, complex]:
    """`{m: coefficient}` of `I` at harmonics 0, 1 and 2 of the object frequency.

    Read off the centre row: both the object and the illumination vary in `x`
    only, so the image intensity is uniform in `y`.

    `S[m] = (1/N) sum_k I[k] exp(-2 pi i m k / N)`, and the closed form is written
    in terms of `exp(-i m theta_k)`, whose coefficient is therefore `conj(S[m
    j_o])`. Conjugated here so the analytic and measured coefficients are directly
    comparable *including their phase* -- which is what makes the fringe phase a
    checked quantity rather than a discarded one.

    The `(-1)^(m j_o)` factor is the origin bookkeeping, applied rather than left
    as a coincidence. `theta` is defined on the offset from the `n // 2` origin
    while `np.fft.fft` sums over the array index, and the two differ by
    `exp(i m pi j_o)` for even `N`. With the current even `OBJECT_BIN` the factor
    is 1 and this line changes nothing; with an odd one the measured first
    harmonic would come back negated, so the alternative to writing it is a gate
    that fails for a reason nobody would look for here.
    """
    row = measured[SHAPE[0] // 2]
    spectrum = np.fft.fft(row) / SHAPE[1]
    origin_sign = {m: (-1.0) ** (m * OBJECT_BIN) for m in (0, 1, 2)}
    return {
        m: origin_sign[m] * complex(np.conj(spectrum[m * OBJECT_BIN])) for m in (0, 1, 2)
    }


def _surviving_orders(bin_index: float, *, cutoff_bin: float) -> tuple[int, ...]:
    """Which object orders the stop passes. Arithmetic, and it reads no array."""
    return tuple(
        order for order in (-1, 0, +1) if abs(bin_index + order * OBJECT_BIN) <= cutoff_bin
    )


def _analytic_harmonics(
    surviving: tuple[int, ...], coefficients: dict[int, complex], *, scale: float
) -> dict[int, complex]:
    """`{m: scale * sum_{o - o' = m} c_o conj(c_o')}` over the surviving set.

    The whole primary oracle, in four lines and with no transform in it. `scale`
    is `(f1/f2)^2`, the amplitude factor of two forward legs squared, carried
    explicitly because nothing here is peak-normalized.
    """
    return {
        m: scale
        * sum(
            coefficients[o] * coefficients[o - m].conjugate()
            for o in surviving
            if o - m in surviving
        )
        for m in (0, 1, 2)
    }


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def _sweep(
    coefficients: dict[int, complex], *, executor: Executor, axes: tuple[Array, Array],
    f1: float, f2: float, radius_m: float, cutoff_bin: float,
) -> dict[float, dict[str, Any]]:
    """Every on-grid angle: measured harmonics beside their analytic values.

    Seven runs of one plan, differing in one number in step 0's request. That is
    what makes this a *parameterized family* rather than seven systems.
    """
    scale = (f1 / f2) ** 2
    results: dict[float, dict[str, Any]] = {}
    for bin_index in ON_GRID_BINS:
        measured, image = _measure(
            bin_index, coefficients, executor=executor, axes=axes,
            f1=f1, f2=f2, radius_m=radius_m,
        )
        harmonics = _harmonics(measured)
        surviving = _surviving_orders(bin_index, cutoff_bin=cutoff_bin)
        analytic = _analytic_harmonics(surviving, coefficients, scale=scale)
        results[bin_index] = {
            "surviving_orders": list(surviving),
            "beams": len(surviving),
            "measured": {str(m): [harmonics[m].real, harmonics[m].imag] for m in harmonics},
            "analytic": {str(m): [analytic[m].real, analytic[m].imag] for m in analytic},
            "residual": max(abs(harmonics[m] - analytic[m]) for m in analytic),
            "image_pitch_m": list(image.sample_pitch_m),
        }
    return results


def _pass_band_gate(sweep: dict[float, dict[str, Any]]) -> dict[str, Any]:
    worst = max(entry["residual"] for entry in sweep.values())
    return gate(
        "the_analytic_pass_band_harmonics_at_every_swept_angle",
        oracle="I[k] = (f1/f2)^2 |sum_{o in S} c_o exp(-i o theta_k)|^2, so harmonic m has "
        "coefficient sum_{o-o'=m} c_o conj(c_o') over the surviving set S, with S given by "
        "the predicate |j + o j_o| <= NA/lambda * N dx. Absolute, complex, and not "
        "normalized by anything",
        oracle_kind="closed_form",
        measured={
            str(bin_index): {
                "beams": entry["beams"],
                "surviving_orders": entry["surviving_orders"],
                "residual": entry["residual"],
            }
            for bin_index, entry in sweep.items()
        },
        expected=0.0,
        tolerance=COMPLEX64_FLOOR,
        tolerance_basis=(
            "two complex64 FFT legs and one squaring: eight float32 epsilons is 9.5e-7, "
            "and the harmonic coefficients are differences of same-magnitude products so "
            "the bound is taken at 2e-6. Worst measured residual across the sweep is "
            "reported above; the failures this catches -- a blocked order that survives, a "
            "second harmonic that should be exactly zero -- are O(1)"
        ),
        passed=worst < COMPLEX64_FLOOR,
    )


def _cutoff_step_gate(
    sweep: dict[float, dict[str, Any]], coefficients: dict[int, complex], *,
    f1: float, f2: float, cutoff_bin: float,
) -> dict[str, Any]:
    """The synthetic-aperture step: where the second harmonic goes to zero."""
    scale = (f1 / f2) ** 2
    second = {
        bin_index: abs(complex(*entry["measured"]["2"]))
        for bin_index, entry in sweep.items()
    }
    three_beam = scale * abs(coefficients[+1] * coefficients[-1].conjugate())
    #: The analytic step: the largest illumination bin whose `+1` order is still
    #: inside the cutoff. Nothing about this reads the simulation.
    last_three_beam_bin = math.floor(cutoff_bin) - OBJECT_BIN
    above = [b for b in sweep if b > last_three_beam_bin]
    below = [b for b in sweep if b <= last_three_beam_bin]
    return gate(
        "the_second_harmonic_steps_to_zero_at_the_analytic_cutoff",
        oracle="three beams give a second harmonic |c_+ conj(c_-)| (f1/f2)^2; two beams "
        "give exactly zero. The step is at the largest j with |j + j_o| <= R/dx_F",
        oracle_kind="closed_form",
        measured={
            "second_harmonic_by_bin": {str(b): second[b] for b in sorted(second)},
            "step_between_bins": [last_three_beam_bin, last_three_beam_bin + 1],
        },
        expected={
            "three_beam_second_harmonic": three_beam,
            "two_beam_second_harmonic": 0.0,
            "cutoff_bin": cutoff_bin,
        },
        tolerance=COMPLEX64_FLOOR,
        tolerance_basis=(
            "the two-beam second harmonic is the arithmetic zero of an elementwise multiply "
            "by 0, not a small number, so the step is a floor-to-signal jump of "
            f"{three_beam:.3e} rather than a threshold crossing"
        ),
        passed=(
            all(abs(second[b] - three_beam) < COMPLEX64_FLOOR for b in below)
            and all(second[b] < COMPLEX64_FLOOR for b in above)
            and three_beam > 1e3 * COMPLEX64_FLOOR
        ),
    )


def _dark_field_gate(sweep: dict[float, dict[str, Any]]) -> dict[str, Any]:
    """The one-beam case: a flat intensity, and the DC level is `|c_-|^2 (f1/f2)^2`."""
    single = [b for b, entry in sweep.items() if entry["beams"] == 1]
    return gate(
        "a_single_surviving_order_gives_a_flat_dark_field_intensity",
        oracle="one beam interferes with nothing: both harmonics are zero and the DC level "
        "is that order's |c|^2 alone",
        oracle_kind="closed_form",
        measured={
            str(b): {
                "first_harmonic": abs(complex(*sweep[b]["measured"]["1"])),
                "second_harmonic": abs(complex(*sweep[b]["measured"]["2"])),
                "dc": sweep[b]["measured"]["0"][0],
                "analytic_dc": sweep[b]["analytic"]["0"][0],
            }
            for b in single
        },
        expected=0.0,
        tolerance=COMPLEX64_FLOOR,
        tolerance_basis="the same complex64 harmonic floor; the fringes are arithmetically "
        "absent rather than small, and the DC level is checked against |c|^2 (f1/f2)^2 "
        "absolutely, so a dark field at the wrong brightness fails here too",
        passed=bool(single)
        and all(
            abs(complex(*sweep[b]["measured"][m])) < COMPLEX64_FLOOR
            for b in single
            for m in ("1", "2")
        )
        and all(
            abs(sweep[b]["measured"]["0"][0] - sweep[b]["analytic"]["0"][0])
            < COMPLEX64_FLOOR
            for b in single
        ),
    )


def _sampling_chain_gate(
    *, executor: Executor, axes: tuple[Array, Array], f1: float, f2: float, radius_m: float
) -> dict[str, Any]:
    """Criterion 6: the sampling chain is checked, not assumed, and recorded."""
    coefficients = object_coefficients()
    prefix: tuple[PlanNode, ...] = (
        *_illumination_nodes(0.0, coefficients, axes),
        _leg_node(f1, "fourier"),
    )
    fourier = _run(prefix, executor=executor)
    image = _run(
        (
            *prefix,
            _element_node(amplitude=_stop(fourier.sample_pitch_m, radius_m)),
            _leg_node(f2, "image"),
        ),
        executor=executor,
    )
    #: Recomputed here from the formula rather than by calling
    #: `fourier_plane_pitch_m` and comparing the result to itself.
    #: `focal_plane_transform` declares its output pitch with exactly that call, so
    #: `fourier.sample_pitch_m == fourier_plane_pitch_m(...)` cannot fail for any
    #: value of the formula -- it is repository code checked against the same call,
    #: which is the circularity AGENTS.md rules out of a deciding gate. The
    #: exact-equality check against the library value is kept *beside* it as what
    #: it actually is: the statement that the boundary carried the declaration
    #: through unchanged.
    from_scratch = tuple(
        WAVELENGTH_M * f1 / (MEDIUM_INDEX * count * pitch)
        for count, pitch in zip(SHAPE, PITCH_M, strict=True)
    )
    declared = fourier_plane_pitch_m(
        PITCH_M, SHAPE, wavelength_m=WAVELENGTH_M, focal_length_m=f1, medium_index=MEDIUM_INDEX
    )
    analytic_image = tuple(pitch * f2 / f1 for pitch in PITCH_M)
    spans = [radius_m / pitch for pitch in fourier.sample_pitch_m]
    return gate(
        "the_sampling_chain_at_every_plane",
        oracle="dx_F = lambda f1/(n N dx) recomputed from the formula; dx_img = dx f2/f1; "
        "the stop radius in Fourier samples is R/dx_F per axis",
        oracle_kind="closed_form",
        measured={
            "fourier_pitch_m": list(fourier.sample_pitch_m),
            "image_pitch_m": list(image.sample_pitch_m),
            "stop_radius_in_fourier_samples_yx": spans,
        },
        expected={
            "fourier_pitch_m": list(from_scratch),
            "image_pitch_m": list(analytic_image),
        },
        tolerance=1e-15,
        tolerance_basis="float64 on both sides. 1e-15 rather than exact equality because "
        "the recomputation groups the arithmetic differently from the library function -- "
        "(lambda f)/(n N dx) against (lambda f / n)/(N dx) -- and a 1-ulp associativity "
        "difference is not a sampling error",
        passed=(
            all(
                abs(got - want) <= 1e-15 * want
                for got, want in zip(fourier.sample_pitch_m, from_scratch, strict=True)
            )
            and fourier.sample_pitch_m == declared
            and all(
                abs(got - want) <= 1e-15 * want
                for got, want in zip(image.sample_pitch_m, analytic_image, strict=True)
            )
        ),
    )


def _off_grid_report(
    coefficients: dict[int, complex], *, executor: Executor, axes: tuple[Array, Array],
    f1: float, f2: float, radius_m: float, cutoff_bin: float,
) -> dict[str, Any]:
    """Criterion 7, reported and not gated.

    An illumination off the DFT frequency grid is not three lines on this grid: the
    Dirichlet kernel spreads each one over every bin. Two cases, and they behave
    differently, which is the point of exercising both:

    * `40.5` -- every line is far inside the pupil, so almost nothing is cut and
      the measured harmonics stay close to the three-beam prediction;
    * `52.5` -- the `+1` line straddles the cutoff, so it is *partly* transmitted
      and the harmonics land between the three-beam and two-beam predictions.
      There is no closed form for "half an order", which is exactly why this is a
      report rather than a gate.
    """
    scale = (f1 / f2) ** 2
    entries: dict[str, Any] = {}
    for bin_index in OFF_GRID_BINS:
        measured, _ = _measure(
            bin_index, coefficients, executor=executor, axes=axes,
            f1=f1, f2=f2, radius_m=radius_m,
        )
        harmonics = _harmonics(measured)
        # Bracketed by the two on-grid bins the case sits between, not by the
        # nearest one: `round(52.5)` is 52 under banker's rounding, which is a
        # three-beam bin, so a "nearest" field would silently duplicate the
        # three-beam prediction and hide the interval the measurement falls in.
        three = _analytic_harmonics((-1, 0, +1), coefficients, scale=scale)
        bracket = {
            float(edge): _analytic_harmonics(
                _surviving_orders(float(edge), cutoff_bin=cutoff_bin),
                coefficients,
                scale=scale,
            )
            for edge in (math.floor(bin_index), math.ceil(bin_index))
        }
        entries[str(bin_index)] = {
            "second_harmonic": abs(harmonics[2]),
            "three_beam_second_harmonic": abs(three[2]),
            "first_harmonic": abs(harmonics[1]),
            "three_beam_first_harmonic": abs(three[1]),
            "bracketing_on_grid_first_harmonics": {
                str(edge): abs(value[1]) for edge, value in bracket.items()
            },
            "bracketing_on_grid_surviving_orders": {
                str(edge): list(_surviving_orders(edge, cutoff_bin=cutoff_bin))
                for edge in bracket
            },
            "deviation_from_the_three_beam_prediction": max(
                abs(harmonics[m] - three[m]) for m in three
            ),
        }
    return gate(
        "off_grid_illumination_angles_are_exercised_and_reported",
        oracle="none: a line off the DFT grid is spread by the Dirichlet kernel and 'half "
        "a transmitted order' has no closed form",
        oracle_kind="diagnostic",
        measured=entries,
        expected="reported, not gated",
        tolerance=None,
        tolerance_basis="not a gate; recorded so R06.8's sweep does not rest on an "
        "unexamined assumption about grid-commensurate angles (R06.5 characterizes the "
        "underlying Dirichlet behaviour)",
        passed=True,
    )


def _diagnostic_model(
    *, executor: Executor, axes: tuple[Array, Array], f1: float, f2: float, radius_m: float
) -> dict[str, Any]:
    """The secondary Fourier-domain model, on a phantom with no closed form.

    `ifft2(P * fft2(u))` in float64 NumPy, with the project's `n // 2` centring
    applied explicitly. It shares no code with the Chromatix path, so its
    agreement is real evidence -- and it is repository numerical code, so it
    decides nothing and this entry is `diagnostic`.

    The relation it is compared against is R06.7's composition identity: the
    project runs two **forward** legs, so its image is `-(f1/f2)` times the
    *mirrored* filtered field, while `fft2` then `ifft2` returns it upright.
    """
    rng = np.random.default_rng(20260901)
    smooth = np.exp(
        -(
            (np.arange(SHAPE[0])[:, None] - SHAPE[0] // 2) ** 2 / (SHAPE[0] / 6.0) ** 2
            + (np.arange(SHAPE[1])[None, :] - SHAPE[1] // 2) ** 2 / (SHAPE[1] / 6.0) ** 2
        )
    )
    phantom = smooth * np.exp(1j * rng.uniform(-1.0, 1.0, SHAPE))
    phantom = phantom / np.max(np.abs(phantom))

    bin_index = 40.0
    wavevector = _wavevector_for_bin(bin_index)
    project_intensity, _ = _measure_phantom(
        phantom, executor=executor, axes=axes, wavevector=wavevector,
        f1=f1, f2=f2, radius_m=radius_m,
    )

    _, x = axes
    incoming = phantom * np.exp(1j * wavevector * x[None, :])
    stop = _stop(
        fourier_plane_pitch_m(
            PITCH_M, SHAPE, wavelength_m=WAVELENGTH_M, focal_length_m=f1,
            medium_index=MEDIUM_INDEX,
        ),
        radius_m,
    )
    spectrum = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(incoming)))
    filtered = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(spectrum * stop)))
    model_intensity = (f1 / f2) ** 2 * np.roll(
        np.flip(np.abs(filtered) ** 2, axis=(0, 1)), (1, 1), axis=(0, 1)
    )

    residual = float(
        np.max(np.abs(project_intensity - model_intensity)) / np.max(model_intensity)
    )
    return gate(
        "float64_numpy_fourier_domain_model_on_a_random_phantom",
        oracle="ifft2(P * fft2(u)) in float64 NumPy, mirrored and scaled by (f1/f2)^2 per "
        "the two-forward-leg composition identity",
        oracle_kind="diagnostic",
        measured={"relative_residual": residual, "illumination_bin": bin_index},
        expected=0.0,
        tolerance=None,
        tolerance_basis="not a gate: this is repository numerical code checking repository "
        "numerical code. Reported as evidence that the two independent paths agree",
        passed=True,
    )


def _measure_phantom(
    phantom: Array,
    *,
    executor: Executor,
    axes: tuple[Array, Array],
    wavevector: float,
    f1: float,
    f2: float,
    radius_m: float,
) -> tuple[Array, ScalarField]:
    """The same plan as `_measure`, with the phantom in place of the three lines."""
    prefix: tuple[PlanNode, ...] = (
        _source_node((0.0, wavevector)),
        _element_node(
            amplitude=np.abs(phantom).copy(),
            phase_rad=np.angle(phantom).copy(),
            target_surface="object",
        ),
        _leg_node(f1, "fourier"),
    )
    fourier = _run(prefix, executor=executor)
    plan: tuple[PlanNode, ...] = (
        *prefix,
        _element_node(
            amplitude=_stop(fourier.sample_pitch_m, radius_m), target_surface="pupil"
        ),
        _leg_node(f2, "image"),
    )
    image = _run(plan, executor=executor)
    return _raw_intensity(_run((*plan, _measure_node()), executor=executor)), image


# ---------------------------------------------------------------------------
# The negative controls
# ---------------------------------------------------------------------------


def _controls(
    coefficients: dict[int, complex], *, executor: Executor, axes: tuple[Array, Array],
    f1: float, f2: float, radius_m: float, cutoff_bin: float,
    sweep: dict[float, dict[str, Any]],
) -> list[dict[str, Any]]:
    scale = (f1 / f2) ** 2
    controls: list[dict[str, Any]] = []

    #: The reference angle for most controls: two beams, the `+1` order blocked,
    #: well past the cutoff so nothing about it is marginal.
    reference_bin = 65.0
    reference = _analytic_harmonics(
        _surviving_orders(reference_bin, cutoff_bin=cutoff_bin), coefficients, scale=scale
    )

    def residual_of(measured: Array, analytic: dict[int, complex]) -> float:
        harmonics = _harmonics(measured)
        return max(abs(harmonics[m] - analytic[m]) for m in analytic)

    # 1. k_j sign flipped: the passed orders swap, so the surviving sideband -- and
    #    with it the first-harmonic amplitude -- changes. This is *only* observable
    #    because the object is complex and asymmetric; for a real object the two
    #    intensities are identical, which is why the object is built the way it is.
    flipped, _ = _measure(
        -reference_bin, coefficients, executor=executor, axes=axes,
        f1=f1, f2=f2, radius_m=radius_m,
    )
    flipped_harmonics = _harmonics(flipped)
    mirrored = _analytic_harmonics(
        _surviving_orders(-reference_bin, cutoff_bin=cutoff_bin), coefficients, scale=scale
    )
    controls.append(
        control(
            "illumination_wavevector_sign_flipped",
            changed=f"k_j at bin {-reference_bin} instead of {reference_bin}",
            breaks_gate="the_analytic_pass_band_harmonics_at_every_swept_angle",
            measured={
                "first_harmonic": abs(flipped_harmonics[1]),
                "residual_against_the_unflipped_prediction": residual_of(flipped, reference),
            },
            reference={
                "unflipped_first_harmonic": abs(reference[1]),
                "mirrored_first_harmonic": abs(mirrored[1]),
            },
            broke=residual_of(flipped, reference) > 1e3 * COMPLEX64_FLOOR,
        )
    )

    # 2. The pupil at the image plane instead of the Fourier plane. A stop there
    #    selects no spatial frequencies at all, so the blocked order survives and
    #    the second harmonic reappears where the analytic set says zero.
    misplaced, _ = _measure(
        reference_bin, coefficients, executor=executor, axes=axes,
        f1=f1, f2=f2, radius_m=radius_m, stop_at_the_image_plane=True,
    )
    controls.append(
        control(
            "pupil_at_the_image_plane",
            changed="the same stop applied after leg two instead of between the legs",
            breaks_gate="the_second_harmonic_steps_to_zero_at_the_analytic_cutoff",
            measured={
                "second_harmonic": abs(_harmonics(misplaced)[2]),
                "residual": residual_of(misplaced, reference),
            },
            reference={"two_beam_second_harmonic": 0.0},
            broke=abs(_harmonics(misplaced)[2]) > COMPLEX64_FLOOR,
        )
    )

    # 3a. The pupil radius off by the medium index. NA = n sin(theta), so a radius
    #     computed in the wrong medium is a different aperture -- and a 1.33x error
    #     reads as a slightly tighter stop rather than as a bug.
    wrong_index_radius = numerical_aperture_radius_m(
        NUMERICAL_APERTURE, focal_length_m=f1, medium_index=1.33
    )
    wrong_index, _ = _measure(
        52.0, coefficients, executor=executor, axes=axes,
        f1=f1, f2=f2, radius_m=wrong_index_radius,
    )
    correct_at_52 = _analytic_harmonics(
        _surviving_orders(52.0, cutoff_bin=cutoff_bin), coefficients, scale=scale
    )
    controls.append(
        control(
            "pupil_radius_off_by_the_medium_index",
            changed="radius computed with medium_index=1.33 instead of 1.0",
            breaks_gate="the_second_harmonic_steps_to_zero_at_the_analytic_cutoff",
            measured={
                "second_harmonic": abs(_harmonics(wrong_index)[2]),
                "residual": residual_of(wrong_index, correct_at_52),
                "radius_m": wrong_index_radius,
            },
            reference={
                "correct_radius_m": radius_m,
                "three_beam_second_harmonic": abs(correct_at_52[2]),
            },
            broke=residual_of(wrong_index, correct_at_52) > 1e3 * COMPLEX64_FLOOR,
        )
    )

    # 3b. ...and off by 2 pi, which opens the pupil so wide that nothing is cut.
    wide, _ = _measure(
        reference_bin, coefficients, executor=executor, axes=axes,
        f1=f1, f2=f2, radius_m=2.0 * math.pi * radius_m,
    )
    controls.append(
        control(
            "pupil_radius_off_by_two_pi",
            changed="radius multiplied by 2 pi, the cycles/m vs rad/m confusion",
            breaks_gate="the_analytic_pass_band_harmonics_at_every_swept_angle",
            measured={
                "second_harmonic": abs(_harmonics(wide)[2]),
                "residual": residual_of(wide, reference),
            },
            reference={"two_beam_second_harmonic": 0.0},
            broke=residual_of(wide, reference) > 1e3 * COMPLEX64_FLOOR,
        )
    )

    # 4. The illumination angle read as a spatial frequency instead of an angular
    #    wavenumber: 2 pi times too small, so a 65-bin tilt becomes a 10-bin one.
    #: The same plan with one number in step 0's request divided by 2 pi. The
    #: object node is `_object_node` rather than `_illumination_nodes`, because the
    #: tilt here is deliberately *not* `_wavevector_for_bin`'s output.
    unconverted_plan: tuple[PlanNode, ...] = (
        _source_node((0.0, _wavevector_for_bin(reference_bin) / (2.0 * math.pi))),
        _object_node(coefficients, axes),
        _leg_node(f1, "fourier"),
        _element_node(
            amplitude=_stop(
                fourier_plane_pitch_m(
                    PITCH_M, SHAPE, wavelength_m=WAVELENGTH_M, focal_length_m=f1,
                    medium_index=MEDIUM_INDEX,
                ),
                radius_m,
            )
        ),
        _leg_node(f2, "image"),
        _measure_node(),
    )
    unconverted_intensity = _raw_intensity(_run(unconverted_plan, executor=executor))
    controls.append(
        control(
            "illumination_angle_read_as_spatial_frequency",
            changed="k_j divided by 2 pi, i.e. cycles/m handed over as rad/m",
            breaks_gate="the_analytic_pass_band_harmonics_at_every_swept_angle",
            measured={
                "second_harmonic": abs(_harmonics(unconverted_intensity)[2]),
                "residual": residual_of(unconverted_intensity, reference),
                "effective_bin": reference_bin / (2.0 * math.pi),
            },
            reference={"two_beam_second_harmonic": 0.0},
            broke=residual_of(unconverted_intensity, reference) > 1e3 * COMPLEX64_FLOOR,
        )
    )

    # 5. A transposed (y, x) in the illumination wavevector, and the mechanism
    #    measured rather than assumed. `_wavevector_for_bin` normalizes by the *x*
    #    window, so the transposed carrier is the same `k` on the `y` axis: it
    #    lands at `(k/2pi) lambda f1 / n = 2.702 mm`, which is 14.6 samples of the
    #    coarser `dy_F` and **inside** the 3.000 mm stop rather than outside it.
    #    What breaks is therefore not vignetting but the **order set**: on `k_y`
    #    all three lines sit at radius 2.70-2.83 mm and pass, where the bin-65
    #    `k_x` tilt the prediction is written for passes only two. The residual is
    #    a three-beam measurement against a two-beam analytic, and the carrier is
    #    also 0.625 bins off the `y` DFT grid so a few percent is clipped.
    transposed, _ = _measure(
        reference_bin, coefficients, executor=executor, axes=axes,
        f1=f1, f2=f2, radius_m=radius_m, transpose=True,
    )
    fourier_pitch = fourier_plane_pitch_m(
        PITCH_M, SHAPE, wavelength_m=WAVELENGTH_M, focal_length_m=f1,
        medium_index=MEDIUM_INDEX,
    )
    transposed_position_m = (
        _wavevector_for_bin(reference_bin) / (2.0 * math.pi)
    ) * WAVELENGTH_M * f1 / MEDIUM_INDEX
    controls.append(
        control(
            "transposed_illumination_wavevector",
            changed=f"the bin-{reference_bin} carrier put on k_y instead of k_x",
            breaks_gate="the_analytic_pass_band_harmonics_at_every_swept_angle",
            measured={
                "dc": _harmonics(transposed)[0].real,
                "residual": residual_of(transposed, reference),
                "carrier_position_on_y_m": transposed_position_m,
                "carrier_position_in_y_samples": transposed_position_m / fourier_pitch[0],
                "stop_radius_in_y_samples": radius_m / fourier_pitch[0],
            },
            reference={
                "two_beam_dc_for_a_bin_65_x_tilt": reference[0].real,
                "three_beam_dc": _analytic_harmonics(
                    (-1, 0, +1), coefficients, scale=scale
                )[0].real,
                "stop_radius_m": radius_m,
            },
            broke=residual_of(transposed, reference) > 1e3 * COMPLEX64_FLOOR,
        )
    )

    # 6. An object frequency past the grid's Nyquist, which must be **refused**
    #    rather than aliased into a plausible answer at a different frequency.
    refused = False
    message = ""
    try:
        _require_resolvable_object(SHAPE[1] // 2 + 7)
    except ValueError as exc:
        refused = True
        message = str(exc)
    controls.append(
        control(
            "object_frequency_past_nyquist_is_refused",
            changed=f"the object placed at bin {SHAPE[1] // 2 + 7}, past the Nyquist bin "
            f"{SHAPE[1] // 2}",
            breaks_gate="the run itself: it is refused before any field is built",
            measured={"refused": refused, "message": message},
            reference={"refused": True},
            broke=refused,
        )
    )

    # 7. The second leg run as an inverse transform: an upright image and a system
    #    that is not a relay. The harmonic amplitudes survive it -- an intensity
    #    reading is blind to the mirroring -- so what it breaks is the *sampling and
    #    composition* claim, which is recorded rather than glossed.
    upright_intensity, upright_image = _measure(
        reference_bin, coefficients, executor=executor, axes=axes,
        f1=f1, f2=f2, radius_m=radius_m, second_leg="inverse",
    )
    _, forward_image = _measure(
        reference_bin, coefficients, executor=executor, axes=axes,
        f1=f1, f2=f2, radius_m=radius_m,
    )
    difference = float(
        np.max(np.abs(np.asarray(upright_image.u) - np.asarray(forward_image.u)))
        / np.max(np.abs(np.asarray(forward_image.u)))
    )
    controls.append(
        control(
            "second_leg_as_an_inverse_transform",
            changed="model['direction']='inverse' on leg two",
            breaks_gate="the field-level composition identity; the intensity harmonics are "
            "blind to it, which is the point of recording it",
            measured={
                "field_difference": difference,
                "harmonic_residual": residual_of(upright_intensity, reference),
            },
            reference={"field_difference_if_identical": 0.0},
            broke=difference > 0.5,
        )
    )
    return controls


# ---------------------------------------------------------------------------
# What was executed, as a record
# ---------------------------------------------------------------------------


def _plan_record(plan: Sequence[PlanNode], *, executor: Executor) -> dict[str, Any]:
    """The full forward plan, its edge check, and the execution record of one run.

    Described from the `ExecutionRecord`'s own `node_requests` rather than from the
    plan this module wrote down, and the difference is the point: those are the
    arguments the **executor bound**, so a record showing f1 on one leg and f2 on
    the other, and different mask digests on the two elements, is evidence that the
    repeated operations really were independent nodes. That is the claim a single
    flat request could not support.
    """
    chain = _semantic_chain(plan)
    record = executor.execute(plan)
    if record.status != "completed":
        #: Refused here rather than written into the record, for the reason
        #: `b4f_ideal._plan_record` gives: `main()` counts failed gates and unbroken
        #: controls and would not notice a `plan.execution` block reading `failed`,
        #: so a resource trip on this run would produce a committed record claiming
        #: a failed run beside a printed "OK" and an exit code of 0.
        raise RuntimeError(
            f"the recorded plan {list(record.route)} did not complete ({record.status}): "
            f"{[(node.operation_id, node.status, node.diagnostics) for node in record.nodes]}"
        )
    steps = list(zip(record.route, record.node_requests, strict=True))
    legs = [
        entry["focal_length_m"]
        for identifier, entry in steps
        if identifier == "O_FOCAL_PLANE_TRANSFORM"
    ]
    return {
        "steps": describe_plan(steps, chain=chain),
        "semantic_chain": list(chain),
        "enumerable_by_planning_routes": False,
        "why_not_enumerable": (
            "planning.routes admits no repeated operation in a route, and this plan repeats "
            "O_FOCAL_PLANE_TRANSFORM and O_COMPLEX_TRANSMISSION. planning/graph.py records "
            "that as an open decision rather than an oversight; the edge check above is the "
            "part of planning that does apply to a plan a caller wrote down"
        ),
        "swept_parameter": (
            "transverse_wavevector_rad_per_m in step 0's request. The seven on-grid angles "
            "are seven runs of this one plan, which is what makes the sweep a parameterized "
            "family rather than seven systems"
        ),
        "execution": {
            "status": record.status,
            "route": list(record.route),
            "node_statuses": [node.status for node in record.nodes],
            "per_node_requests_recorded": len(record.node_requests),
            "per_node_focal_lengths_m": legs,
            "resource_failure": record.provenance["resources"]["resource_failure"],
        },
    }


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------


def run(configuration: dict[str, Any]) -> dict[str, Any]:
    f1 = float(configuration["focal_length_1_m"])
    f2 = float(configuration["focal_length_2_m"])

    _require_resolvable_object(OBJECT_BIN)
    coefficients = object_coefficients()
    radius_m = numerical_aperture_radius_m(
        NUMERICAL_APERTURE, focal_length_m=f1, medium_index=MEDIUM_INDEX
    )
    fourier_pitch = fourier_plane_pitch_m(
        PITCH_M, SHAPE, wavelength_m=WAVELENGTH_M, focal_length_m=f1, medium_index=MEDIUM_INDEX
    )
    #: The cutoff, in bins, two ways: as a stop radius in Fourier samples and as
    #: the spatial-frequency cutoff `NA/lambda` times the window. They are the same
    #: number by construction and both are recorded, because the agreement is the
    #: statement that the stop radius realizes the NA.
    cutoff_bin = radius_m / fourier_pitch[1]
    cutoff_from_frequency = (NUMERICAL_APERTURE / WAVELENGTH_M) * SHAPE[1] * PITCH_M[1]

    with Executor() as executor:
        axes = _grid_axes(executor)
        sweep = _sweep(
            coefficients, executor=executor, axes=axes, f1=f1, f2=f2,
            radius_m=radius_m, cutoff_bin=cutoff_bin,
        )
        sampling = _sampling_chain_gate(
            executor=executor, axes=axes, f1=f1, f2=f2, radius_m=radius_m
        )
        off_grid = _off_grid_report(
            coefficients, executor=executor, axes=axes, f1=f1, f2=f2,
            radius_m=radius_m, cutoff_bin=cutoff_bin,
        )
        diagnostic = _diagnostic_model(
            executor=executor, axes=axes, f1=f1, f2=f2, radius_m=radius_m
        )
        controls = _controls(
            coefficients, executor=executor, axes=axes, f1=f1, f2=f2,
            radius_m=radius_m, cutoff_bin=cutoff_bin, sweep=sweep,
        )
        #: The plan the whole benchmark is a family of, run once more so the record
        #: carries an `ExecutionRecord` of the canonical on-axis case.
        plan_record = _plan_record(
            (
                *_illumination_nodes(0.0, coefficients, axes),
                _leg_node(f1, "fourier"),
                _element_node(
                    amplitude=_stop(fourier_pitch, radius_m), target_surface="pupil"
                ),
                _leg_node(f2, "image"),
                _measure_node(),
            ),
            executor=executor,
        )

    gates = [
        sampling,
        _pass_band_gate(sweep),
        _cutoff_step_gate(sweep, coefficients, f1=f1, f2=f2, cutoff_bin=cutoff_bin),
        _dark_field_gate(sweep),
        gate(
            "the_stop_radius_realizes_the_declared_numerical_aperture",
            oracle="R / dx_F == (NA/lambda) N dx: the stop radius in Fourier samples equals "
            "the cutoff frequency in bins",
            oracle_kind="closed_form",
            measured=cutoff_bin,
            expected=cutoff_from_frequency,
            tolerance=1e-12,
            tolerance_basis="float64 arithmetic on both sides; the wavelength cancels, which "
            "is why the same stop realizes the same NA at any wavelength",
            passed=abs(cutoff_bin / cutoff_from_frequency - 1.0) < 1e-12,
        ),
        off_grid,
        diagnostic,
    ]

    return {
        "benchmark": BENCHMARK_ID,
        "ticket": "CHE-213",
        "configuration": configuration["name"],
        "produced_by": "benchmarks/systems/b_fourier_ptychography.py",
        "composition": [
            "S_SOURCE_PLANE_WAVE",
            "O_COMPLEX_TRANSMISSION",
            "O_FOCAL_PLANE_TRANSFORM",
            "O_COMPLEX_TRANSMISSION",
            "O_FOCAL_PLANE_TRANSFORM",
            "M_PSF",
        ],
        "execution_path": (
            "runtime.Executor.execute(plan) over a hardcoded plan of per-node requests, "
            "with the result read from Executor.result. The benchmark imports no operation: "
            "every step above is a catalog id resolved by operations.resolve inside the "
            "executor. planning.routes cannot enumerate this plan -- an operation may not "
            "repeat in a route and this one repeats two -- so the plan is written down here "
            "and every consecutive edge is checked against planning.capability_graph()"
        ),
        "plan": plan_record,
        "intensity_path": (
            "the M_PSF node of the plan, which the executor resolves to "
            "measurements.psf(field, normalization='raw').intensity -- the project's "
            "only |U|^2. R11.1 (CHE-197) landed measurements/ and this benchmark's own "
            "local implementation was deleted in the same change that pointed it here, "
            "which is what benchmarks/observables.py said would happen"
        ),
        "approximation_the_method_rests_on": (
            "O(x) exp(i k_j x) has spectrum O~(k - k_j) exactly, because the object is a "
            "thin, angle-independent transmission (operators.complex_transmission's declared "
            "approximation). The equivalence between tilting the illumination and shifting "
            "the object spectrum would not hold for a thick or angle-dependent sample"
        ),
        "parameters": {
            "wavelength_m": WAVELENGTH_M,
            "medium_index": MEDIUM_INDEX,
            "shape": list(SHAPE),
            "sample_pitch_m": list(PITCH_M),
            "focal_length_1_m": f1,
            "focal_length_2_m": f2,
            "numerical_aperture": NUMERICAL_APERTURE,
            "stop_edge": "hard",
            "stop_radius_m": radius_m,
            "fourier_plane_pitch_m": list(fourier_pitch),
            "image_plane_pitch_m": [pitch * f2 / f1 for pitch in PITCH_M],
            "stop_radius_in_fourier_samples_yx": [radius_m / p for p in fourier_pitch],
            "cutoff_bin": cutoff_bin,
            "object_bin": OBJECT_BIN,
            "object_coefficients": {
                str(order): [value.real, value.imag]
                for order, value in coefficients.items()
            },
            "on_grid_illumination_bins": list(ON_GRID_BINS),
            "off_grid_illumination_bins": list(OFF_GRID_BINS),
        },
        "sweep": {str(bin_index): entry for bin_index, entry in sweep.items()},
        "not_covered": [
            "no reconstruction of any kind: no phase retrieval, no Gerchberg-Saxton, no "
            "gradient descent over the object, no pupil recovery, no aberration estimation, "
            "no loss function",
            "LED-array geometry and calibration, shot noise and sensor response, partial "
            "coherence, sample thickness and multiple scattering, aberrated pupils",
            "overlap/redundancy analysis of the angle set (a reconstruction concern)",
        ],
        "gates": gates,
        "negative_controls": controls,
    }


def main() -> int:
    failed = 0
    for configuration in CONFIGURATIONS:
        record = run(configuration)
        path = write_record(
            record, path=RECORDS / f"{BENCHMARK_ID}-{configuration['name']}.json"
        )
        print(f"\n=== {BENCHMARK_ID} / {configuration['name']} ===")
        for bin_index, entry in record["sweep"].items():
            print(
                f"  bin {bin_index:>5}  orders {entry['surviving_orders']!s:<12} "
                f"{entry['beams']} beam(s)  residual {entry['residual']:.2e}"
            )
        for entry in record["gates"]:
            mark = "PASS" if entry["passed"] else "FAIL"
            print(f"  [{mark}] {entry['name']}  ({entry['oracle_kind']})")
            if not entry["passed"]:
                print(f"         measured {entry['measured']!r}")
                print(f"         expected {entry['expected']!r} +/- {entry['tolerance']!r}")
                failed += 1
        for entry in record["negative_controls"]:
            mark = "BROKE" if entry["broke_the_gate"] else "DID NOT BREAK"
            print(f"  [{mark}] control {entry['name']}")
            if not entry["broke_the_gate"]:
                print(f"         measured {entry['measured']!r} vs {entry['reference']!r}")
                failed += 1
        print(f"  record: {path.relative_to(Path(__file__).resolve().parents[2])}")

    if failed:
        print(f"\n{failed} gate(s) or control(s) did not hold.")
        return 1
    print("\nOK: every gate holds and every negative control breaks the gate it names.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
