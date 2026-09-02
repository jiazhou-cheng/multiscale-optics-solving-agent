"""The ideal coherent 4f relay, assembled from project primitives.

CHE-212 (R06.7). Run it:

    ./run.sh python -m benchmarks.systems.b4f_ideal

Two purposes, both required. It validates the physics of a **composition** --
per-operator tests cannot see a composition error, only a system can -- and it
demonstrates that a complete optical system is now expressible in this project's
public vocabulary rather than by calling a backend directly.

The system, as a plan
---------------------
::

    S_SOURCE_PLANE_WAVE       normal-incidence plane wave                 (R06.5)
      -> O_COMPLEX_TRANSMISSION   the object, at the front focal plane     (R06.6)
      -> O_FOCAL_PLANE_TRANSFORM  f1 -> Fourier plane, dx_F = lambda f1/(n N dx)
      -> O_COMPLEX_TRANSMISSION   the filter, at the Fourier plane         (R06.6)
      -> O_FOCAL_PLANE_TRANSFORM  f2 -> image plane, dx_img = dx f2/f1     (R06.4)

Run at `f1 == f2` and at `f1 != f2`, because several of the gates below are
trivially satisfied when the two are equal -- a magnification of -1 hides both the
`f1/f2` amplitude factor and the pitch change.

A plan through the executor, and no `System` class
--------------------------------------------------
That listing is not prose: it is the plan this module hands to `runtime.Executor`,
step by step, each step a catalogued operation id paired with its own request. The
benchmark imports no operation. `sources.plane_wave`, `operators.complex_transmission`
and `solvers.chromatix.focal_plane_transform` are named by id and resolved through
`operations.CATALOG`, so what composes the system is the catalog and the executor
rather than this file's import list.

Written down rather than enumerated, deliberately. `planning.routes` cannot
produce this plan -- no operation may repeat in a route, and this one transforms
twice and masks twice -- so the plan is hardcoded here and each consecutive edge is
checked against `planning.capability_graph()`, which is the part of planning that
does answer a question about it: does this operation consume what the one before it
produced. `_semantic_chain` is that check, and it runs before any physics.

**Each occurrence is an independent node.** The two focal-plane transforms carry
f1 and f2 in their own requests and the two masks carry their own amplitudes. That
is what the per-node request form exists for: bound from one flat mapping keyed by
parameter name, this plan ran f1 on both legs and an open pupil on both masks and
reported every node `completed` -- a different optical system, recorded as this
one. The record's `node_requests` is what that is verified from afterwards.

Still no `systems/` package, no composite-operator framework and no `Pipeline`.
`operators/` may not import `solvers/` under the dependency allowlist, so nothing
in `src/` holds this graph; what holds it is a plan, which is data, and the
executor, which is generic. Where the plan comes from -- this file today, an agent
later -- is the planner's question and not the executor's.

The one closed form that does most of the work
-----------------------------------------------
Two forward optical Fourier transforms compose into an exact statement. Each leg
carries the textbook `1/(i lambda f / n)` prefactor and the discrete relation
`DFT{DFT{U}}[k] = N U[-k]`, so with `dx_F = lambda f1 / (n N dx)`:

    U_img[k] = -(f1 / f2) * U_in[-k]        dx_img = dx * f2 / f1

Everything in that line is a separate claim the benchmark checks separately:

* the index mirroring is the **inversion** (criterion 1);
* `dx_img = dx f2/f1` with mirrored indices is the **magnification** `M = -f2/f1`
  in physical coordinates (criterion 2), and the pitch is an *independent*
  statement from it (criterion 4) -- a system can get the magnification right and
  the pitch wrong, and then every measured length is wrong by the same factor with
  nothing to reveal it;
* the `f1/f2` amplitude factor together with the pitch change conserves
  `discrete_power` exactly (criterion 7);
* the leading `-1` is `(1/i)^2`, a global pi that `|U|^2` cannot see. Both legs
  declare `carrier_removed_phase` for it, and the benchmark asserts the sign
  rather than comparing magnitudes.

Which oracles decide
--------------------
Every gate is closed-form Fourier optics: the discrete sampling relation, the
composition above, the Dirichlet kernel of a sampled boxcar, the Gaussian
transform pair, Jacobi--Anger's Bessel coefficients, and a pass/block predicate on
a stop radius. **No gate is another run of this repository's numerics.** The
`record.gate` entries carry `oracle_kind`, and a `diagnostic` entry may not decide
anything.

The direct `ifft2c(mask * fft2c(u))` NumPy model that CHE-144 used as a secondary
check is **not** run here, and that is stated rather than implied: the closed form
above is stronger than a differential check against another FFT, because it
predicts the amplitude factor, the sign and the pitch as well as the shape.
Adding a second numerical path would produce a number that agrees and decides
nothing.

Not covered, said plainly
--------------------------
The validity-envelope sweep over modulation frequency to the sampling limit --
CHE-144's most valuable output -- is explicitly optional in R06.7 and is **not
run**. Neither is a real or aberrated lens, ray-domain anything, partial
coherence, polarization, or a sensor model.

Cost: CPU, 2.8 s for both configurations on a 192 x 256 complex64 grid, plus the
executor's per-run overhead -- 3.1 ms of resolve, bind, sample and record against
milliseconds of physics per plan. One `Executor` is opened per configuration
rather than per plan, because the environment fingerprint is read once per
`__enter__` and costs 7.5 ms.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import jv

from benchmarks.record import control, describe_plan, gate, write_record
from operations import CATALOG
from operators import circular_aperture_amplitude, numerical_aperture_radius_m
from planning import ENTRY, capability_graph
from representations import ReferenceSurface, ScalarField
from runtime import Executor, PlanNode, normalize_plan
from solvers.chromatix import fourier_plane_pitch_m

BENCHMARK_ID = "B-4F-IDEAL"
RECORDS = Path(__file__).resolve().parent / "records"

#: A NumPy array of unspecified dtype. An alias so the annotations below stay
#: readable; nothing in this module is generic over dtype.
Array = np.ndarray[Any, np.dtype[Any]]

WAVELENGTH_M = 0.532e-6
MEDIUM_INDEX = 1.0

#: 192 x 256 at 0.6 x 0.5 um. Asymmetric in **both** count and pitch: the two axes
#: then have different extents and different Fourier pitches, so a transposed
#: `(y, x)` cannot pass any gate below, and a mask built for the wrong axis is a
#: shape error rather than a plausible answer.
SHAPE = (192, 256)
PITCH_M = (0.6e-6, 0.5e-6)

#: The complex64 floor for a two-leg system, derived rather than fitted. float32
#: carries 1.19e-7 relative; each leg is one FFT pair whose rounding scale is set
#: by its largest term, so two legs accumulate a small multiple of that. Four
#: epsilons is 4.8e-7, and 5e-7 is the bound used.
#:
#: The worst residual actually measured against it is the grating relay's
#: `image_residual` at 3.17e-7 -- 2.7 epsilons, a margin of 1.6x, not the comfortable
#: factor the round number suggests. The reason it is kept anyway is that every
#: failure this gate exists to catch is O(1) and not marginal: the same comparison
#: without the index mirroring is 1.0 and without the global pi is 2.0, both
#: recorded beside the residual. A tolerance whose margin is 1.6x over noise and
#: 10^6 under every real error is doing its job; if that margin ever matters, the
#: fix is a float64 path and not a wider number.
COMPLEX64_FLOOR = 5e-7

#: The stop, as a numerical aperture rather than as a radius, so the radius comes
#: from `numerical_aperture_radius_m` and the cutoff frequency `NA/lambda` is
#: available as an independent statement. NA 0.1 at f1 = 20 mm is a 2.000 mm stop,
#: which spans 24.06 samples of this Fourier plane's own x pitch -- a *discretized*
#: aperture, and the record says how many samples it is rather than leaving a
#: reader to assume it is smooth.
NUMERICAL_APERTURE = 0.1

#: Object frequencies for the filtering gate, in bins of `1/(N dx_in)`. 15 is
#: inside the analytic cutoff (24.06 bins) and 35 is outside it, both by a wide
#: enough margin that the predicate is not a boundary case -- the boundary case
#: itself is gated in `tests/physics/test_thin_element_spectrum.py`.
PASSBAND_BIN = 15
STOPBAND_BIN = 35
PASSBAND_DEPTH = 0.4
STOPBAND_DEPTH = 0.3

CONFIGURATIONS: tuple[dict[str, Any], ...] = (
    {"name": "unit_magnification", "focal_length_1_m": 20e-3, "focal_length_2_m": 20e-3},
    {"name": "magnifying_relay", "focal_length_1_m": 20e-3, "focal_length_2_m": 40e-3},
)


# ---------------------------------------------------------------------------
# The plan: four node builders, the edge check, and the one call that executes
# ---------------------------------------------------------------------------
#
# Every node below is `(operation_id, request)`. Nothing here calls an operation:
# the ids are resolved by `operations.resolve` inside the executor, which is why
# this module imports no `sources`, `operators` or `solvers` operation at all. The
# two `operators` names it does import -- `circular_aperture_amplitude` and
# `numerical_aperture_radius_m` -- are mask builders and are not operations;
# `operators/__init__.py` says so where it declares `OPERATIONS`, and they consume
# and produce no representation. `fourier_plane_pitch_m` is the same kind of thing.


def _surface(name: str) -> ReferenceSurface:
    return ReferenceSurface(name=name, z_m=0.0, medium_index=MEDIUM_INDEX)


def _source_node(transverse_wavevector: tuple[float, float] = (0.0, 0.0)) -> PlanNode:
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
    *,
    amplitude: Any = None,
    phase_rad: Any = None,
    target_surface: str | None = None,
) -> PlanNode:
    """One thin element. Only the arguments actually given reach the request.

    `O_COMPLEX_TRANSMISSION` declares every argument optional, so an omitted
    `phase_rad` must be *absent* from the request rather than present as `None`:
    the executor binds optional arguments it finds by name, and `phase_rad=None`
    would be passed through to an operator whose default is not `None`.
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


def _semantic_chain(plan: Sequence[PlanNode]) -> tuple[str, ...]:
    """The semantic types the plan passes through, refusing a step that cannot run.

    The part of `planning/` that answers a question about a hardcoded plan. Each
    step must appear in `capability_graph()[state]` -- the operations that consume
    the state the previous step produced -- and the first must be a graph entry,
    which is what `ENTRY` (`None`) keys. A route this rejects would still *execute*:
    the executor binds a port from the request and a mismatched representation
    would reach an operation that then fails somewhere inside a backend, so
    checking it here is what turns that into a refusal naming the step.

    Not `planning.routes`, and the reason is structural rather than a preference:
    `routes` will not enumerate a plan that repeats an operation, and this one
    transforms twice. The adjacency is still exactly right for checking a plan
    someone else wrote, which is what a hardcoded plan -- or a generated one -- is.

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

    Refuses to hand back a result from a run that did not complete, and names the
    node that stopped it. Without that a refused node would leave `executor.result`
    at `None` and the gate above would read a `None` as a physical answer -- which
    is the "fail plainly rather than fabricating" rule at this module's own
    boundary.

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
    origin is `representations.ORIGIN_RULE`'s declaration and a second copy of it
    in a benchmark is the kind of restatement that silently disagrees. Every mask
    below is sampled on these axes.
    """
    field = _run((_source_node(),), executor=executor)
    y, x = field.coordinates()
    return np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)


def _axes(field: ScalarField) -> tuple[Array, Array]:
    y, x = field.coordinates()
    return np.asarray(y, dtype=np.float64), np.asarray(x, dtype=np.float64)


def _mirror(array: Array) -> Array:
    """`a[-k]` on the `n // 2` origin, per axis.

    Index `j` maps to `2 * (n // 2) - j`, which for even `n` is `n - j` and so is
    `roll(flip(a), 1)`. Written once here because the whole inversion claim rests
    on it and `flip` alone -- `a[n - 1 - j]` -- is off by exactly one sample, a
    half-window phase ramp's worth of error that a symmetric input would not show.
    """
    mirrored: Array = np.roll(np.flip(array, axis=(0, 1)), (1, 1), axis=(0, 1))
    return mirrored


def _centroid_and_width(field: ScalarField) -> tuple[float, float, float, float]:
    """`(cy, cx, sigma_y, sigma_x)` of `|u|^2`, in metres, on the field's own axes."""
    intensity = np.abs(np.asarray(field.u)) ** 2
    y, x = _axes(field)
    total = float(intensity.sum())
    rows = intensity.sum(axis=1)
    columns = intensity.sum(axis=0)
    cy = float((rows * y).sum() / total)
    cx = float((columns * x).sum() / total)
    return (
        cy,
        cx,
        math.sqrt(float((rows * (y - cy) ** 2).sum() / total)),
        math.sqrt(float((columns * (x - cx) ** 2).sum() / total)),
    )


def _row_spectrum(field: ScalarField) -> Array:
    """`|DFT|` along `x` of the field's own centre row.

    For a field in a *position* domain -- the object or the image -- this is its
    line spectrum in bins. Bin indices survive both legs (leg two maps bin `j` to
    `-j`), so the same bin can be read at the object and at the image while the
    *physical* frequency it means changes with the pitch. That is the point of
    reading it this way, and it is why this must not be applied to a field that is
    already at the Fourier plane: `_fourier_row` is that case.
    """
    spectrum: Array = np.abs(np.fft.fft(np.asarray(field.u)[SHAPE[0] // 2]))
    return spectrum


def _fourier_row(field: ScalarField) -> Array:
    """`|u|` along the centre row of a field that is *already* at a Fourier plane.

    Read directly off the array, because the transform has happened: taking
    another DFT here would return the object and read as a spectrum that is
    exactly flat wherever the object was flat -- which is a plausible-looking wrong
    answer rather than an error.
    """
    row: Array = np.abs(np.asarray(field.u)[SHAPE[0] // 2])
    return row


# ---------------------------------------------------------------------------
# The objects, each one an `O_COMPLEX_TRANSMISSION` node
# ---------------------------------------------------------------------------
#
# An object is a plan node and not a field: it is the mask, and the plan the node
# sits in is what says which illumination it modulates. The masks are sampled on
# `axes`, the source grid's own coordinates, which every caller obtains from
# `_grid_axes`.


def _asymmetric_object(axes: tuple[Array, Array]) -> PlanNode:
    """Two unequal off-centre lobes under a two-axis phase carrier.

    Deliberately not a centred Gaussian: a symmetric input cannot detect an
    inversion failure and neither can a symmetric grid. The two lobes differ in
    position, width and weight, and the carrier makes the field complex, so
    `conj(U)` is a different field and the phasor control has something to break.
    """
    y, x = axes
    waist_m = 12e-6
    amplitude = np.exp(
        -(((y[:, None] - 18e-6) ** 2 + (x[None, :] - 25e-6) ** 2) / waist_m**2)
    )
    amplitude += 0.45 * np.exp(
        -(((y[:, None] + 30e-6) ** 2 + (x[None, :] + 8e-6) ** 2) / (0.45 * waist_m) ** 2)
    )
    amplitude /= amplitude.max()
    return _element_node(
        amplitude=amplitude,
        phase_rad=_asymmetric_object_phase(axes),
        target_surface="object",
    )


def _asymmetric_object_phase(axes: tuple[Array, Array]) -> Array:
    """The carrier of `_asymmetric_object`, as its own function.

    One expression rather than two: the phasor-sign control negates this phase, and
    the two copies the control used to carry were a restatement that could drift
    apart -- at which point the control would be comparing two different objects
    and would still "break the gate".
    """
    y, x = axes
    phase: Array = 3.0e5 * np.broadcast_to(x[None, :], SHAPE) + 1.0e5 * np.broadcast_to(
        y[:, None], SHAPE
    )
    return phase.copy()


def _gaussian_object(axes: tuple[Array, Array], waist_m: float) -> PlanNode:
    y, x = axes
    return _element_node(
        amplitude=np.exp(-((y[:, None] ** 2 + x[None, :] ** 2) / waist_m**2)),
        target_surface="object",
    )


def _slit_object(width_samples: int) -> PlanNode:
    """A boxcar `width_samples` wide in `x`, uniform in `y`.

    An integer sample count, and one that divides the grid: the sampled boxcar's
    transform is the Dirichlet kernel with **exact** zeros at bins that are
    multiples of `N / L`, which is a closed form rather than the continuous
    `sinc`'s approximation to it.
    """
    mask = np.zeros(SHAPE[1], dtype=np.float64)
    origin = SHAPE[1] // 2
    mask[origin - width_samples // 2 : origin + width_samples // 2] = 1.0
    return _element_node(
        amplitude=np.broadcast_to(mask[None, :], SHAPE).copy(), target_surface="object"
    )


def _sinusoidal_grating_object(
    axes: tuple[Array, Array], depth_rad: float, periods: int
) -> PlanNode:
    _, x = axes
    period_m = SHAPE[1] * PITCH_M[1] / periods
    profile = depth_rad * np.sin(2.0 * np.pi * x / period_m)
    return _element_node(
        phase_rad=np.broadcast_to(profile[None, :], SHAPE).copy(), target_surface="object"
    )


def _two_frequency_object(axes: tuple[Array, Array]) -> tuple[PlanNode, float]:
    """`(1 + a cos(2 pi f_a x) + b cos(2 pi f_b x)) / (1 + a + b)`, and its norm.

    An exact five-line spectrum on this grid, because both frequencies are integer
    bins of `1/(N dx)`. That is what makes the transmitted power fraction in
    criterion 7 an arithmetic statement rather than an integral.
    """
    _, x = axes
    window_m = SHAPE[1] * PITCH_M[1]
    profile = (
        1.0
        + PASSBAND_DEPTH * np.cos(2.0 * np.pi * PASSBAND_BIN * x / window_m)
        + STOPBAND_DEPTH * np.cos(2.0 * np.pi * STOPBAND_BIN * x / window_m)
    )
    norm = 1.0 + PASSBAND_DEPTH + STOPBAND_DEPTH
    return (
        _element_node(
            amplitude=np.broadcast_to((profile / norm)[None, :], SHAPE).copy(),
            target_surface="object",
        ),
        norm,
    )


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------


def _sampling_gates(
    fourier: ScalarField, image: ScalarField, *, f1: float, f2: float
) -> list[dict[str, Any]]:
    """Criteria 3 and 4: the two sampling relations, as independent statements."""
    analytic_fourier = fourier_plane_pitch_m(
        PITCH_M, SHAPE, wavelength_m=WAVELENGTH_M, focal_length_m=f1, medium_index=MEDIUM_INDEX
    )
    from_scratch = tuple(
        WAVELENGTH_M * f1 / (MEDIUM_INDEX * count * pitch)
        for count, pitch in zip(SHAPE, PITCH_M, strict=True)
    )
    analytic_image = tuple(pitch * f2 / f1 for pitch in PITCH_M)

    gates = [
        gate(
            "fourier_plane_pitch",
            oracle="dx_F = lambda f1 / (n N dx_in), per axis, recomputed here from the "
            "formula rather than by calling the same function the operator declares with",
            oracle_kind="closed_form",
            measured=list(fourier.sample_pitch_m),
            expected=list(from_scratch),
            tolerance=1e-15,
            tolerance_basis=(
                "float64 on both sides, at 1e-15 rather than exact equality because the "
                "recomputation groups the arithmetic differently from the library function "
                "-- (lambda f)/(n N dx) against (lambda f / n)/(N dx) -- and a 1-ulp "
                "associativity difference is not a sampling error. The separate exact "
                "equality against `fourier_plane_pitch_m` is kept beside it as what it "
                "actually is: the statement that the boundary carried the operator's own "
                "declaration through unchanged, which is a tautology about the formula and "
                "so cannot be the thing that decides"
            ),
            passed=(
                all(
                    abs(got - want) <= 1e-15 * want
                    for got, want in zip(fourier.sample_pitch_m, from_scratch, strict=True)
                )
                and fourier.sample_pitch_m == analytic_fourier
            ),
        ),
        gate(
            "image_plane_pitch",
            oracle="dx_img = dx_in f2 / f1, per axis",
            oracle_kind="closed_form",
            measured=list(image.sample_pitch_m),
            expected=list(analytic_image),
            tolerance=1e-15,
            tolerance_basis=(
                "float64 arithmetic on both sides; an independent claim from the "
                "magnification, because a system can invert correctly and still report "
                "every length wrong by f2/f1"
            ),
            passed=all(
                abs(got - want) <= 1e-15 * want
                for got, want in zip(image.sample_pitch_m, analytic_image, strict=True)
            ),
        ),
    ]
    return gates


def _frequency_axis_gate(
    f1: float, *, executor: Executor
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Criterion 3, second half, plus the numbers the negative controls reuse.

    A single spatial frequency enters as a tilted plane wave and must land at the
    sample whose offset from the origin is `j`, i.e. at `x_F = j lambda f1 /
    (n N dx_in)`. Measured two ways -- the peak sample and the intensity centroid
    -- because the first is what a discretized reading gives and the second is what
    survives an off-grid carrier, and the controls need both.
    """
    window_m = SHAPE[1] * PITCH_M[1]
    bin_index = PASSBAND_BIN
    wavevector = 2.0 * math.pi * bin_index / window_m

    fourier = _run(
        (_source_node((0.0, wavevector)), _leg_node(f1, "fourier")), executor=executor
    )
    intensity = np.abs(np.asarray(fourier.u)) ** 2
    peak = np.unravel_index(int(np.argmax(intensity)), SHAPE)
    _, x_axis = _axes(fourier)
    columns = intensity.sum(axis=0)
    centroid_m = float((columns * x_axis).sum() / columns.sum())
    concentration = float(intensity.max() / intensity.sum())

    analytic_m = bin_index * WAVELENGTH_M * f1 / (MEDIUM_INDEX * window_m)
    #: The mistake the control names: reading `k_x` in rad/m as if it were a
    #: spatial frequency in cycles/m puts the spot 2 pi times further out.
    two_pi_wrong_m = 2.0 * math.pi * analytic_m

    measured_m = float(x_axis[peak[1]])
    gates = [
        gate(
            "fourier_plane_frequency_axis",
            oracle="the sample at offset j is the spatial frequency j / (N dx_in), i.e. "
            "x_F = j lambda f1 / (n N dx_in)",
            oracle_kind="closed_form",
            measured={
                "peak_index_offset": [
                    int(peak[0]) - SHAPE[0] // 2,
                    int(peak[1]) - SHAPE[1] // 2,
                ],
                "peak_position_m": measured_m,
                "centroid_m": centroid_m,
                "peak_energy_fraction": concentration,
            },
            expected={"index_offset": [0, bin_index], "position_m": analytic_m},
            tolerance=COMPLEX64_FLOOR,
            tolerance_basis=(
                "the carrier sits on an integer DFT bin, so the whole plane wave lands on "
                "one output sample and the position is the declared pitch times an integer; "
                "the tolerance covers only the float32 storage of that pitch"
            ),
            passed=(
                int(peak[0]) - SHAPE[0] // 2 == 0
                and int(peak[1]) - SHAPE[1] // 2 == bin_index
                and abs(measured_m / analytic_m - 1.0) < COMPLEX64_FLOOR
                and concentration > 0.99
            ),
        )
    ]
    return gates, {
        "analytic_m": analytic_m,
        "measured_m": measured_m,
        "centroid_m": centroid_m,
        "concentration": concentration,
        "two_pi_wrong_m": two_pi_wrong_m,
        "wavevector": wavevector,
    }


def _composition_gates(
    f1: float, f2: float, *, executor: Executor, axes: tuple[Array, Array]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Criteria 1, 2 and 7 (open filter): the one exact closed form, three ways.

    Three states are compared -- the object, the Fourier plane and the image -- and
    each is read as the output of a **prefix of the one plan**, which is what makes
    them three planes of a single system rather than three separately assembled
    ones. The executor returns the last node's output, so a prefix is how an
    intermediate plane is asked for; the repeated legs cost milliseconds.
    """
    #: The open filter is the *same* operator with both factors at their identity,
    #: which is R06.6's whole design: an open pupil is not a special case that has
    #: to be skipped.
    plan: tuple[PlanNode, ...] = (
        _source_node(),
        _asymmetric_object(axes),
        _leg_node(f1, "fourier"),
        _element_node(amplitude=1.0),
        _leg_node(f2, "image"),
    )
    source = _run(plan[:2], executor=executor)
    fourier = _run(plan[:3], executor=executor)
    image = _run(plan, executor=executor)

    incoming = np.asarray(source.u)
    predicted = (-(f1 / f2) * _mirror(incoming)).astype(np.complex64)
    outgoing = np.asarray(image.u)
    scale = float(np.max(np.abs(predicted)))
    inversion_residual = float(np.max(np.abs(outgoing - predicted)) / scale)
    #: The falsifiable twin: the same comparison without the index mirroring. An
    #: upright image reads as 1.0 here, so the mirroring is the whole difference.
    upright_residual = float(
        np.max(np.abs(outgoing - (-(f1 / f2) * incoming).astype(np.complex64))) / scale
    )
    #: ...and without the global pi.
    unsigned_residual = float(
        np.max(np.abs(outgoing - ((f1 / f2) * _mirror(incoming)).astype(np.complex64))) / scale
    )

    before = _centroid_and_width(source)
    after = _centroid_and_width(image)
    magnification = -f2 / f1
    magnification_measured = [after[0] / before[0], after[1] / before[1]]
    width_ratio_measured = [after[2] / before[2], after[3] / before[3]]

    power_ratio = image.discrete_power() / source.discrete_power()

    gates = [
        gate(
            "image_is_minus_f1_over_f2_times_the_mirrored_input",
            oracle="U_img[k] = -(f1/f2) U_in[-k], from DFT{DFT{U}}[k] = N U[-k] and the "
            "1/(i lambda f / n) prefactor on each leg",
            oracle_kind="closed_form",
            measured={
                "residual": inversion_residual,
                "residual_without_mirroring": upright_residual,
                "residual_without_the_global_pi": unsigned_residual,
            },
            expected=0.0,
            tolerance=COMPLEX64_FLOOR,
            tolerance_basis=(
                "two FFT legs in complex64: four float32 epsilons is 4.8e-7. The two "
                "falsifiable twins land at O(1), so this is not a tolerance that could "
                "absorb a composition error"
            ),
            passed=(
                inversion_residual < COMPLEX64_FLOOR
                and upright_residual > 0.5
                and unsigned_residual > 0.5
            ),
        ),
        gate(
            "magnification",
            oracle="M = -f2/f1, as a feature's centroid and its second moment at both planes",
            oracle_kind="closed_form",
            measured={
                "centroid_ratio_yx": magnification_measured,
                "width_ratio_yx": width_ratio_measured,
            },
            expected={"centroid_ratio": magnification, "width_ratio": abs(magnification)},
            tolerance=1e-5,
            tolerance_basis=(
                "a ratio of two float32 second moments over the same window; 1e-5 is two "
                "orders above the 1.5e-7 measured and still far below any plausible "
                "magnification error, which would be a factor of f2/f1"
            ),
            passed=(
                all(abs(m / magnification - 1.0) < 1e-5 for m in magnification_measured)
                and all(abs(r / abs(magnification) - 1.0) < 1e-5 for r in width_ratio_measured)
            ),
        ),
        gate(
            "power_through_an_open_filter",
            oracle="the f1/f2 amplitude factor against the f2/f1 pitch change conserves "
            "sum |u|^2 dy dx exactly",
            oracle_kind="closed_form",
            measured=power_ratio,
            expected=1.0,
            tolerance=1e-6,
            tolerance_basis=(
                "a sum of squares in float32: two epsilons on the amplitude is ~2.4e-7 on "
                "the power, and the measured deviation is 2.36e-7 -- i.e. this gate runs at "
                "its derived floor with a factor of four in hand, not with orders. It is "
                "the gate that catches a normalization error nothing else here can see, "
                "which is why no comparison in this benchmark is normalized by its own peak"
            ),
            passed=abs(power_ratio - 1.0) < 1e-6,
        ),
        gate(
            "the_four_f_length_is_2f1_plus_2f2",
            oracle="each leg advances the declared surface by 2f",
            oracle_kind="closed_form",
            measured={
                "z_fourier_m": fourier.reference_surface.z_m,
                "z_image_m": image.reference_surface.z_m,
            },
            expected={"z_fourier_m": 2.0 * f1, "z_image_m": 2.0 * f1 + 2.0 * f2},
            tolerance=0.0,
            tolerance_basis="float64 addition of two declared focal lengths",
            passed=(
                fourier.reference_surface.z_m == 2.0 * f1
                and image.reference_surface.z_m == 2.0 * f1 + 2.0 * f2
            ),
        ),
    ]
    return gates, {
        "source": source,
        "fourier": fourier,
        "image": image,
        "plan": plan,
    }


def _transform_pair_gates(
    f1: float, f2: float, *, executor: Executor, axes: tuple[Array, Array]
) -> list[dict[str, Any]]:
    """Criterion 5: three known transform pairs, positions and amplitudes."""
    gates: list[dict[str, Any]] = []

    # -- a Gaussian returns a Gaussian of the analytically predicted waist -------
    #: 5 um: ten samples across the input waist, and 8.1 samples across the Fourier
    #: waist (`w_F / dx_F = N dx / (pi w0)`), so neither plane is undersampled.
    waist_m = 5e-6
    fourier = _run(
        (_source_node(), _gaussian_object(axes, waist_m), _leg_node(f1, "fourier")),
        executor=executor,
    )
    predicted_waist = tuple(
        WAVELENGTH_M * f1 / (math.pi * MEDIUM_INDEX * waist_m) for _ in range(2)
    )
    measured = _centroid_and_width(fourier)
    # For an amplitude exp(-r^2/w^2) the intensity is exp(-2r^2/w^2), whose second
    # moment is w/2. The relation w_F = lambda f / (pi n w0) is the Fourier
    # transform pair of a Gaussian, not a fit.
    measured_waist = (2.0 * measured[2], 2.0 * measured[3])
    gates.append(
        gate(
            "gaussian_transforms_to_a_gaussian_of_the_predicted_waist",
            oracle="w_F = lambda f1 / (pi n w0); intensity second moment is w/2",
            oracle_kind="closed_form",
            measured=list(measured_waist),
            expected=list(predicted_waist),
            tolerance=2e-2,
            tolerance_basis=(
                "2e-2 is **borrowed**, not derived for this case: it is the threshold the "
                "B1-WAVE Gaussian-spreading family justified for the same second-moment "
                "estimator, where window truncation dominated. Here it does not -- the "
                "Fourier waist is 15.7 (x) and 13.0 (y) waists inside the window and the "
                "measured error is 4e-8 -- so this bound is loose-but-safe rather than "
                "tight. It is kept because it still rejects every plausible error in the "
                "relation (a missing pi is 3.14x, a missing n is 1.33x at index 1.33), and "
                "tightening it to the measurement would be fitting"
            ),
            passed=all(
                abs(got / want - 1.0) < 2e-2
                for got, want in zip(measured_waist, predicted_waist, strict=True)
            ),
        )
    )

    # -- a slit gives the sampled boxcar's Dirichlet kernel ----------------------
    width_samples = 16
    row = _fourier_row(
        _run(
            (_source_node(), _slit_object(width_samples), _leg_node(f1, "fourier")),
            executor=executor,
        )
    )
    origin = SHAPE[1] // 2
    peak = float(row[origin])
    zero_bin = SHAPE[1] // width_samples  # exact zeros at multiples of N / L
    nulls = [
        float(row[origin + k] / peak) for k in (zero_bin, 2 * zero_bin, 3 * zero_bin)
    ]
    checked = (1, 3, 5, 7)
    dirichlet = {
        k: abs(
            math.sin(math.pi * k * width_samples / SHAPE[1])
            / math.sin(math.pi * k / SHAPE[1])
        )
        / width_samples
        for k in checked
    }
    lobes = {k: float(row[origin + k] / peak) for k in checked}
    null_analytic_m = [
        m * WAVELENGTH_M * f1 / (MEDIUM_INDEX * width_samples * PITCH_M[1]) for m in (1, 2, 3)
    ]
    gates.append(
        gate(
            "slit_gives_the_sampled_boxcars_dirichlet_kernel",
            oracle="|D(k)| / L with D(k) = sin(pi k L / N) / sin(pi k / N); zeros at "
            "k = m N / L, i.e. x_F = m lambda f1 / (n w)",
            oracle_kind="closed_form",
            measured={
                "null_amplitudes": nulls,
                "null_positions_m": null_analytic_m,
                "lobe_amplitudes": {str(k): lobes[k] for k in checked},
            },
            expected={"nulls": 0.0, "lobes": {str(k): dirichlet[k] for k in checked}},
            tolerance=1e-6,
            tolerance_basis=(
                "the sampled boxcar's transform is the Dirichlet kernel exactly, so the "
                "only error is complex64 storage. The continuous sinc would need a percent-"
                "level tolerance covering a known discretization effect, which is why the "
                "sampled form is the oracle"
            ),
            passed=(
                all(value < 1e-6 for value in nulls)
                and all(abs(lobes[k] - dirichlet[k]) < 1e-6 for k in checked)
            ),
        )
    )

    # -- a sinusoidal phase grating gives Bessel orders, and the image reproduces it
    depth_rad = 1.5
    periods = 8
    grating_plan: tuple[PlanNode, ...] = (
        _source_node(),
        _sinusoidal_grating_object(axes, depth_rad, periods),
        _leg_node(f1, "fourier"),
        _element_node(amplitude=1.0),
        _leg_node(f2, "image"),
    )
    grating = _run(grating_plan[:2], executor=executor)
    grating_fourier = _run(grating_plan[:3], executor=executor)
    row = np.asarray(grating_fourier.u)[SHAPE[0] // 2]
    origin = SHAPE[1] // 2
    orders = {n: complex(row[origin + n * periods]) for n in range(-3, 4)}
    order_residuals = {
        str(n): abs(orders[n] / orders[0] - jv(n, depth_rad) / jv(0, depth_rad))
        for n in orders
    }
    grating_image = _run(grating_plan, executor=executor)
    image_residual = float(
        np.max(
            np.abs(
                np.asarray(grating_image.u)
                - (-(f1 / f2) * _mirror(np.asarray(grating.u))).astype(np.complex64)
            )
        )
        / (f1 / f2)
    )
    gates.append(
        gate(
            "sinusoidal_phase_grating_gives_bessel_orders_and_relays_end_to_end",
            oracle="Jacobi-Anger: A_n / A_0 = J_n(m) / J_0(m) at x_n = n lambda f1 / "
            "(n_med Lambda), signed; and the image reproduces the object",
            oracle_kind="closed_form",
            measured={
                "order_residuals": order_residuals,
                "image_residual": image_residual,
                "order_positions_m": [
                    n * WAVELENGTH_M * f1 / (MEDIUM_INDEX * SHAPE[1] * PITCH_M[1] / periods)
                    for n in (1, 2, 3)
                ],
            },
            expected=0.0,
            tolerance=COMPLEX64_FLOOR,
            tolerance_basis=(
                "the FFT's rounding scale is set by its largest term, so the *absolute* "
                "residual on A_n/A_0 is a few float32 epsilons however small A_n is -- "
                "which is why the comparison is absolute. The signed ratio is asserted, "
                "because J_{-n} = (-1)^n J_n and a magnitude-only check would pass a "
                "mirrored spectrum"
            ),
            passed=(
                all(value < COMPLEX64_FLOOR for value in order_residuals.values())
                and image_residual < COMPLEX64_FLOOR
            ),
        )
    )
    return gates


def _filtering_gates(
    f1: float, f2: float, *, executor: Executor, axes: tuple[Array, Array]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Criteria 6 and 7 (with a filter): the stop removes exactly what it should.

    The stop's mask is sized in the **Fourier plane's own pitch**, which is only
    known once that plane exists -- so the plan is built in two pieces: the prefix
    up to the Fourier plane runs first, its pitch sizes the mask, and the mask
    becomes the fourth node of the plan that runs to the image. That ordering is a
    fact about the physics rather than an artifact of the executor: a pupil radius
    in metres means nothing until the plane it sits in has a scale.
    """
    object_node, norm = _two_frequency_object(axes)
    prefix: tuple[PlanNode, ...] = (
        _source_node(),
        object_node,
        _leg_node(f1, "fourier"),
    )
    source = _run(prefix[:2], executor=executor)
    fourier = _run(prefix, executor=executor)

    radius_m = numerical_aperture_radius_m(
        NUMERICAL_APERTURE, focal_length_m=f1, medium_index=MEDIUM_INDEX
    )
    cutoff_per_m = NUMERICAL_APERTURE / WAVELENGTH_M
    window_m = SHAPE[1] * PITCH_M[1]
    cutoff_bin = cutoff_per_m * window_m
    stop = circular_aperture_amplitude(
        SHAPE, sample_pitch_m=fourier.sample_pitch_m, radius_m=radius_m, edge="hard"
    )
    plan: tuple[PlanNode, ...] = (
        *prefix,
        _element_node(amplitude=stop, target_surface="fourier_stopped"),
        _leg_node(f2, "image"),
    )
    image = _run(plan, executor=executor)

    #: Which lines the analytic predicate keeps: |f| <= NA/lambda, i.e. |bin| <=
    #: cutoff_bin. Nothing about this reads the simulation.
    survives = {
        0: True,
        PASSBAND_BIN: cutoff_bin >= PASSBAND_BIN,
        STOPBAND_BIN: cutoff_bin >= STOPBAND_BIN,
    }
    spectrum = _row_spectrum(image)
    ratios = {
        str(bin_index): float(spectrum[bin_index] / spectrum[0])
        for bin_index in (PASSBAND_BIN, STOPBAND_BIN)
    }
    #: The Fourier coefficients of (1 + a cos + b cos)/norm are 1/norm at DC and
    #: a/(2 norm), b/(2 norm) at the sidebands, so the *ratio* to DC is a/2 and
    #: b/2 with the norm cancelling.
    expected_ratios = {
        str(PASSBAND_BIN): PASSBAND_DEPTH / 2.0 if survives[PASSBAND_BIN] else 0.0,
        str(STOPBAND_BIN): STOPBAND_DEPTH / 2.0 if survives[STOPBAND_BIN] else 0.0,
    }

    transmitted_fraction = image.discrete_power() / source.discrete_power()
    line_power = {
        0: 1.0,
        PASSBAND_BIN: 2.0 * (PASSBAND_DEPTH / 2.0) ** 2,
        STOPBAND_BIN: 2.0 * (STOPBAND_DEPTH / 2.0) ** 2,
    }
    expected_fraction = sum(
        power for bin_index, power in line_power.items() if survives[bin_index]
    ) / sum(line_power.values())

    gates = [
        gate(
            "the_stop_removes_exactly_the_frequencies_above_its_cutoff",
            oracle="pass iff |f| <= NA/lambda; the surviving sideband keeps its analytic "
            "amplitude ratio a/2 to the DC line",
            oracle_kind="closed_form",
            measured={
                "sideband_to_dc": ratios,
                "cutoff_bin": cutoff_bin,
                "stop_radius_m": radius_m,
                "stop_radius_in_fourier_samples": [
                    radius_m / fourier.sample_pitch_m[0],
                    radius_m / fourier.sample_pitch_m[1],
                ],
            },
            expected=expected_ratios,
            tolerance=1e-5,
            tolerance_basis=(
                "a ratio of two line amplitudes read off one complex64 DFT: a few float32 "
                "epsilons, with the blocked line at the arithmetic zero of an exact "
                "elementwise multiply by 0"
            ),
            passed=all(
                abs(ratios[key] - expected_ratios[key]) < 1e-5 for key in expected_ratios
            ),
        ),
        gate(
            "power_through_the_stop_is_the_analytically_transmitted_fraction",
            oracle="the transmitted fraction is the sum of |c_n|^2 over the surviving "
            "lines, over the sum over all of them",
            oracle_kind="closed_form",
            measured=transmitted_fraction,
            expected=expected_fraction,
            tolerance=1e-5,
            tolerance_basis=(
                "the object is an exact five-line spectrum on this grid, so the fraction is "
                "arithmetic on |c_n|^2; the residual is the same float32 power floor as the "
                "open-filter case, one order looser because two power sums are divided"
            ),
            passed=abs(transmitted_fraction / expected_fraction - 1.0) < 1e-5,
        ),
    ]
    return gates, {
        "source": source,
        "source_node": object_node,
        "fourier": fourier,
        "stop": stop,
        "cutoff_bin": cutoff_bin,
        "norm": norm,
        "expected_ratios": expected_ratios,
        "plan": plan,
    }


# ---------------------------------------------------------------------------
# The negative controls
# ---------------------------------------------------------------------------


def _controls(
    f1: float,
    f2: float,
    filtering: dict[str, Any],
    *,
    executor: Executor,
    axes: tuple[Array, Array],
) -> list[dict[str, Any]]:
    """Every one of CHE-144's controls, plus the one this boundary adds.

    Each is a *deliberately wrong* run whose only purpose is to break a gate that
    the correct run passes. A control that does not break its gate means the gate
    was not measuring what it claimed.

    A control is a **different plan**, not a different code path: what is changed
    is one node's request or one node's position, and the same executor runs it. So
    a control cannot pass by accidentally bypassing the composition the gate
    measured, which is what a hand-assembled wrong run can do.
    """
    controls: list[dict[str, Any]] = []
    object_node = _asymmetric_object(axes)
    source = _run((_source_node(), object_node), executor=executor)
    incoming = np.asarray(source.u)
    predicted = (-(f1 / f2) * _mirror(incoming)).astype(np.complex64)
    scale = float(np.max(np.abs(predicted)))

    def relay(element: PlanNode, *, second_leg: str = "forward") -> Array:
        image = _run(
            (
                _source_node(),
                element,
                _leg_node(f1, "fourier"),
                _element_node(amplitude=1.0),
                _leg_node(f2, "image", direction=second_leg),
            ),
            executor=executor,
        )
        return np.asarray(image.u)

    # 1. Phasor-sign flip, expressed where it can be: the object's phase is
    #    negated, which is exactly conj(U) for a real-amplitude element on a
    #    normal-incidence illumination.
    conjugated = _element_node(
        amplitude=np.abs(incoming),
        phase_rad=-_asymmetric_object_phase(axes),
        target_surface="object",
    )
    residual = float(np.max(np.abs(relay(conjugated) - predicted)) / scale)
    controls.append(
        control(
            "phasor_sign_flip",
            changed="the object's phase negated, i.e. the conjugate phasor convention",
            breaks_gate="image_is_minus_f1_over_f2_times_the_mirrored_input",
            measured=residual,
            reference=COMPLEX64_FLOOR,
            broke=residual > 0.5,
        )
    )

    # 2. A transposed axis: the same single-frequency carrier put on `k_y` instead
    #    of `k_x`. The grid is asymmetric, so this is not a relabelling.
    window_x_m = SHAPE[1] * PITCH_M[1]
    wavevector = 2.0 * math.pi * PASSBAND_BIN / window_x_m
    analytic_m = PASSBAND_BIN * WAVELENGTH_M * f1 / (MEDIUM_INDEX * window_x_m)
    transposed = _run(
        (_source_node((wavevector, 0.0)), _leg_node(f1, "fourier")), executor=executor
    )
    intensity = np.abs(np.asarray(transposed.u)) ** 2
    peak = np.unravel_index(int(np.argmax(intensity)), SHAPE)
    _, x_axis = _axes(transposed)
    measured_m = float(x_axis[peak[1]])
    controls.append(
        control(
            "transposed_axis",
            changed="the carrier moved from k_x to k_y",
            breaks_gate="fourier_plane_frequency_axis",
            measured={
                "peak_index_offset": [
                    int(peak[0]) - SHAPE[0] // 2,
                    int(peak[1]) - SHAPE[1] // 2,
                ],
                "x_position_m": measured_m,
            },
            reference={"index_offset": [0, PASSBAND_BIN], "x_position_m": analytic_m},
            broke=int(peak[1]) - SHAPE[1] // 2 != PASSBAND_BIN,
        )
    )

    # 3. The filter at the image plane instead of the Fourier plane. A stop there
    #    does not select spatial frequencies at all, so the out-of-band line
    #    survives -- and the resulting image is perfectly plausible.
    #: The same five nodes as the filtering plan, with the mask moved from between
    #: the legs to after the second one -- which is the whole of the change, and it
    #: is visible as a reordering of one plan rather than as different code.
    unfiltered: tuple[PlanNode, ...] = (
        _source_node(),
        filtering["source_node"],
        _leg_node(f1, "fourier"),
        _element_node(amplitude=1.0),
        _leg_node(f2, "image"),
    )
    unfiltered_image = _run(unfiltered, executor=executor)
    misplaced = _run(
        (
            *unfiltered,
            _element_node(
                amplitude=circular_aperture_amplitude(
                    SHAPE,
                    sample_pitch_m=unfiltered_image.sample_pitch_m,
                    radius_m=numerical_aperture_radius_m(
                        NUMERICAL_APERTURE, focal_length_m=f1, medium_index=MEDIUM_INDEX
                    ),
                    edge="hard",
                )
            ),
        ),
        executor=executor,
    )
    spectrum = _row_spectrum(misplaced)
    stopband_ratio = float(spectrum[STOPBAND_BIN] / spectrum[0])
    controls.append(
        control(
            "filter_at_the_image_plane",
            changed="the same stop applied after leg two instead of between the legs",
            breaks_gate="the_stop_removes_exactly_the_frequencies_above_its_cutoff",
            measured={"stopband_to_dc": stopband_ratio},
            reference=filtering["expected_ratios"][str(STOPBAND_BIN)],
            broke=abs(stopband_ratio - filtering["expected_ratios"][str(STOPBAND_BIN)]) > 1e-5,
        )
    )

    # 4. A 2 pi scale error in the frequency grid: `k_x` in rad/m read as a spatial
    #    frequency in cycles/m.
    correct = _run(
        (_source_node((0.0, wavevector)), _leg_node(f1, "fourier")), executor=executor
    )
    _, correct_x = _axes(correct)
    correct_m = float(
        correct_x[int(np.argmax(np.abs(np.asarray(correct.u)) ** 2) % SHAPE[1])]
    )
    wrong_prediction_m = 2.0 * math.pi * analytic_m
    controls.append(
        control(
            "two_pi_frequency_scale",
            changed="the Fourier-plane position predicted from k_x as if it were cycles/m",
            breaks_gate="fourier_plane_frequency_axis",
            measured={"position_m": correct_m},
            reference={"wrong_prediction_m": wrong_prediction_m},
            broke=abs(correct_m / wrong_prediction_m - 1.0) > 0.1,
        )
    )

    # 5. Grid-snapped vs continuous carrier placement. Not a wrong *model* -- a
    #    different case -- so what it breaks is the single-sample reading, while
    #    the centroid survives. R06.5 characterizes this; recording it here is what
    #    keeps R06.8's angle sweep from discovering it.
    off_grid = _run(
        (
            _source_node((0.0, 2.0 * math.pi * (PASSBAND_BIN + 0.5) / window_x_m)),
            _leg_node(f1, "fourier"),
        ),
        executor=executor,
    )
    off_intensity = np.abs(np.asarray(off_grid.u)) ** 2
    concentration = float(off_intensity.max() / off_intensity.sum())
    _, off_x = _axes(off_grid)
    off_columns = off_intensity.sum(axis=0)
    off_centroid_m = float((off_columns * off_x).sum() / off_columns.sum())
    off_analytic_m = (PASSBAND_BIN + 0.5) * WAVELENGTH_M * f1 / (MEDIUM_INDEX * window_x_m)
    controls.append(
        control(
            "carrier_off_the_dft_grid",
            changed="the carrier moved half a bin off the DFT frequency grid",
            breaks_gate="fourier_plane_frequency_axis (its peak_energy_fraction > 0.99 part)",
            measured={
                "peak_energy_fraction": concentration,
                "centroid_m": off_centroid_m,
                "centroid_relative_error": off_centroid_m / off_analytic_m - 1.0,
            },
            reference={
                "peak_energy_fraction_on_grid": 1.0,
                "dirichlet_half_bin": (2.0 / math.pi) ** 2,
                "analytic_position_m": off_analytic_m,
            },
            broke=concentration < 0.99,
        )
    )

    # 6. The second leg run as an inverse transform. This is the control this
    #    project's boundary adds: it yields an upright image and a system that is
    #    not a 4f relay, and nothing in the artifact says so -- the pitch, the
    #    power and the declared surface all still look right.
    inverse_residual = float(
        np.max(np.abs(relay(object_node, second_leg="inverse") - predicted)) / scale
    )
    controls.append(
        control(
            "second_leg_as_an_inverse_transform",
            changed="model['direction']='inverse' on leg two",
            breaks_gate="image_is_minus_f1_over_f2_times_the_mirrored_input",
            measured=inverse_residual,
            reference=COMPLEX64_FLOOR,
            broke=inverse_residual > 0.5,
        )
    )
    return controls


# ---------------------------------------------------------------------------
# What was executed, as a record
# ---------------------------------------------------------------------------


def _plan_record(plan: Sequence[PlanNode], *, executor: Executor) -> dict[str, Any]:
    """The full 4f plan, its edge check, and the execution record of one run of it.

    Recorded from the `ExecutionRecord` the executor returned rather than from the
    plan this module wrote down, and the difference matters: `node_requests` is
    what the *executor bound*, so a record showing f1 on one leg and f2 on the other
    is evidence that the two occurrences really were independent nodes. That is the
    claim the flat-request form could not support, and it is checked here rather
    than asserted -- `per_node_focal_lengths_m` below is read off the record.
    """
    chain = _semantic_chain(plan)
    record = executor.execute(plan)
    if record.status != "completed":
        #: Refused here rather than written into the record. `main()` counts failed
        #: gates and unbroken controls, and neither would notice a `plan.execution`
        #: block reading `failed` -- so a swap-growth trip on this last run would
        #: land a committed record claiming a failed run beside a printed "OK" and
        #: an exit code of 0. That is the misleading success AGENTS.md forbids, and
        #: this is the only `executor.execute` in the file that `_run` does not
        #: already guard.
        raise RuntimeError(
            f"the recorded plan {list(record.route)} did not complete ({record.status}): "
            f"{[(node.operation_id, node.status, node.diagnostics) for node in record.nodes]}"
        )
    #: Described from the record's own `node_requests` and not from `plan`, so the
    #: steps below are what the executor bound rather than what this file wrote.
    steps = list(zip(record.route, record.node_requests, strict=True))
    legs = [
        entry["focal_length_m"]
        for identifier, entry in zip(record.route, record.node_requests, strict=True)
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
    """Run every gate and every control for one `(f1, f2)`, and build its record.

    One `Executor` for the whole configuration. It is a context manager because it
    owns a memory sampling thread, and the shared-server swap-growth stop condition
    is evaluated at every node boundary of every plan run inside it -- so a
    benchmark that started swapping would stop rather than finish slowly.
    """
    f1 = float(configuration["focal_length_1_m"])
    f2 = float(configuration["focal_length_2_m"])

    with Executor() as executor:
        axes = _grid_axes(executor)
        composition, artifacts = _composition_gates(f1, f2, executor=executor, axes=axes)
        frequency, _ = _frequency_axis_gate(f1, executor=executor)
        filtering, filtering_state = _filtering_gates(f1, f2, executor=executor, axes=axes)

        gates = [
            *_sampling_gates(artifacts["fourier"], artifacts["image"], f1=f1, f2=f2),
            *frequency,
            *composition,
            *_transform_pair_gates(f1, f2, executor=executor, axes=axes),
            *filtering,
        ]
        controls = _controls(f1, f2, filtering_state, executor=executor, axes=axes)
        plan_record = _plan_record(filtering_state["plan"], executor=executor)

    return {
        "benchmark": BENCHMARK_ID,
        "ticket": "CHE-212",
        "configuration": configuration["name"],
        "produced_by": "benchmarks/systems/b4f_ideal.py",
        "composition": [
            "S_SOURCE_PLANE_WAVE",
            "O_COMPLEX_TRANSMISSION",
            "O_FOCAL_PLANE_TRANSFORM",
            "O_COMPLEX_TRANSMISSION",
            "O_FOCAL_PLANE_TRANSFORM",
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
        "parameters": {
            "wavelength_m": WAVELENGTH_M,
            "medium_index": MEDIUM_INDEX,
            "shape": list(SHAPE),
            "sample_pitch_m": list(PITCH_M),
            "focal_length_1_m": f1,
            "focal_length_2_m": f2,
            "numerical_aperture": NUMERICAL_APERTURE,
            "stop_edge": "hard",
            "fourier_plane_pitch_m": list(artifacts["fourier"].sample_pitch_m),
            "image_plane_pitch_m": list(artifacts["image"].sample_pitch_m),
            "stop_radius_in_fourier_samples_yx": [
                numerical_aperture_radius_m(
                    NUMERICAL_APERTURE, focal_length_m=f1, medium_index=MEDIUM_INDEX
                )
                / pitch
                for pitch in artifacts["fourier"].sample_pitch_m
            ],
            "declared_validity_at_the_image_plane": sorted(artifacts["image"].validity),
        },
        "not_covered": [
            "the validity-envelope sweep over modulation frequency to the sampling limit "
            "(CHE-144's optional output; not run)",
            "a direct ifft2c(mask * fft2c(u)) NumPy cross-model (not run: the closed form "
            "above predicts the amplitude factor, the sign and the pitch, which a "
            "differential FFT check does not)",
            "real or aberrated lenses, ray-domain anything, partial coherence, "
            "polarization, noise and sensor models",
        ],
        "gates": gates,
        "negative_controls": controls,
    }


def main() -> int:
    """Run both configurations, write their records, and report."""
    failed = 0
    for configuration in CONFIGURATIONS:
        record = run(configuration)
        path = write_record(
            record, path=RECORDS / f"{BENCHMARK_ID}-{configuration['name']}.json"
        )
        print(f"\n=== {BENCHMARK_ID} / {configuration['name']} ===")
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
