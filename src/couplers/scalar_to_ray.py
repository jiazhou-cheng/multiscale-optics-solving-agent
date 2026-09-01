"""`ScalarField -> RayBundle`: the angular spectrum, as rays the wavelet sum accepts.

CHE-189 (R08.1). The reverse of `ray_to_scalar`, and the half that makes the pair
a *representation* change rather than two loosely related operations: enumerate
every propagating mode of a field, hand the result to `ray_to_scalar`, and the
field comes back to round-off.

```python
couplers.scalar_to_ray(field, *, surface=None, count=None, density="uniform",
                       rng=None, launch_positions_xy_m=None)
    -> tuple[RayBundle, SamplingDiagnostics]
```

Implements SI S2 (eqs S1-S5) and Algorithm S2 of Cheng et al., ACS Photonics 2026
(DOI 10.1021/acsphotonics.6c00818).

**This is a quadrature scheme for an integral whose exact value is known** (SI eq
S2), not an approximation of the physics. That framing is what makes it testable:
enumerate every propagating bin and the estimator must collapse onto the
deterministic reference at dtype round-off. Only after that check passes is there
any point discussing sampling error.

The round trip is a test, not an operator
-----------------------------------------
There is no `ray_to_wave_to_ray` here and there will not be one. A ray -> wave ->
ray conversion with no physical transformation in between changes nothing about
the state; it is a representation-consistency check, and its home is
`tests/physics/test_scalar_to_ray.py`. Shipping it as an operation would advertise
a physical capability that is really a test fixture.

The measure, and where `1/p` lives
----------------------------------
The reference implementation emitted `amplitude = U~[m] / p[m]` -- the importance
weight folded into the amplitude -- and tagged the bundle
`reconstruction_normalization = "one_over_n"`. The new `RayBundle` separates the
two, for the same reason R05.2 moved the hexapolar area element off the amplitude:
a sampling weight is a property of *how the spectrum was sampled*, not of the
light, and a coupler has to be able to refuse an undeclared one. So:

* `amplitude` is `U~[m] exp(i k (d_u x_p + d_v y_p))` -- the modal amplitude with
  its launch-position phase, and **no** `1/p`;
* `measure_weight` is `1 / p[m]`, dimensionless, with
  `measure_kind = "importance_weight"`;
* `ray_to_scalar` multiplies the two and applies the `1/N` that
  `"importance_weight"` obliges, which is SI eq S5 exactly.

**`1/p` is therefore applied exactly once, structurally.** The risk this ticket
names is real -- `a = U~/p` already contains it and a reader of main-text eq 2
alone would apply it again in the sum -- and the separation is what removes the
opportunity rather than a comment asking people not to. The failure it would cause
is the recognizable one: a field that looks plausible, round-trips inexactly, and
fails no named test.

The centred DFT, and why the pairing is not cosmetic
----------------------------------------------------
`ifftshift` in, `fftshift` out, with the `1/(ny nx)` folded in so the coefficients
are the **modal amplitudes themselves** -- a plain sum of modes is the field, and
no stray inverse-DFT factor has to be remembered downstream. Zero frequency then
sits at index `n // 2`, which is `ScalarField`'s own origin rule. With the
ordinary un-centred transform the reconstruction picks up an `exp(-i pi m)` offset
per axis and misses the field entirely; it is the first thing to check if a round
trip fails.

Parseval then reads `sum |U~|^2 = (1 / (ny nx)) sum |u|^2`, which is why the
diagnostics report the field's discrete power beside the modal power rather than
claiming they are the same number.

The evanescent cut is strict, and R07.4 is why that matters
-----------------------------------------------------------
`radial < 1`, strictly. The grazing `d_n = 0` bin is singular for any `1/d_n`
factor, so it is excluded rather than silently included -- but a bin at
`radial = 1 - 1e-16` is *kept*, and that is deliberate. It is exactly the eight
Pythagorean-triple bins CHE-70 found at `d_n = 1.05e-8`, and this coupler and
`ray_to_scalar`'s grazing floor have to agree about them: this module emits them
and reports how many survived, and R07.4 decides whether a reconstruction can
carry them at the precision it was asked for. Two tickets, one mask, and the count
travels so the two can be checked against each other.

Discarded evanescent power is a **real loss** -- an evanescent mode has no
propagation direction to give a ray -- so it is reported as a named fraction. A
large fraction is the signature of a field that should not be turned into rays at
all.

Sampling is an input, not a side effect
---------------------------------------
Indices are drawn from an explicitly seeded generator and the kernel that turns
modes into rays is a pure function of them. Three properties follow structurally
rather than having to be engineered: bitwise determinism; one implementation
serving both a reference run and a gradient study, because there is no RNG inside
to differentiate through; and SI Algorithm S2's "sampled directions held fixed
during backpropagation" becoming the shape of the interface rather than a
`.detach()` someone has to remember.

`count=None` enumerates every propagating bin -- the deterministic exactness
limit, with no sampling error at all. Any other value draws from `rng`, which is
then required: an implicit seed is refused.

Enumeration is the zero-variance case of a **uniform** draw, and only of a uniform
one. Every bin is selected once regardless of `p`, so the `1/p` in the measure has
no compensating draw frequency; under a magnitude-proportional density the
reconstruction would be `sum_m U~[m] / (M p[m])` rather than the field. That pair
is refused rather than repaired by silently substituting the uniform density.

No intermediate representation
------------------------------
The reference implementation had a public `AngularSpectrum` dataclass. It is not a
boundary artifact -- nothing outside this module consumes one, and making it
public would add a third representation to a tree whose whole point is that there
are two. The decomposition is local variables inside one function.

This module imports no solver and no backend.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

from numerics import (
    ArrayState,
    DType,
    compute_dtype,
    device_of,
    dtype_of,
    matmul_precision_kwargs,
    namespace_of,
    numpy_dtype,
)
from representations import (
    ContractError,
    RayBundle,
    ReferenceSurface,
    ScalarField,
)

__all__ = [
    "SAMPLING_DENSITIES",
    "SamplingDensity",
    "SamplingDiagnostics",
    "scalar_to_ray",
]

#: Which density `p(d_u, d_v)` to draw spectral bins from.
#:
#: A `Literal` and not a `StrEnum`. The ticket writes it as an enum and budgets
#: the change at "+1 class"; an AST class count sees a `StrEnum` as a class, so
#: the two readings differ by one and this takes the stricter one -- the same
#: choice `couplers.ray_to_scalar.GrazingPolicy` and
#: `representations.MeasureKind` already make. Nothing here needs enum behaviour:
#: the value is compared against a table and reported as a string.
#:
#: `uniform`
#:     Uniform over propagating bins (`p_uni`). Assumes nothing about the
#:     spectrum, and is the density the exactness limit is measured through.
#: `magnitude`
#:     Proportional to `|U~|` (`p_mag`). The paper reports faster convergence for
#:     spectra concentrated in a single lobe and comparable rates for multilobed
#:     ones, so it is an option rather than an improvement.
SamplingDensity = Literal["uniform", "magnitude"]

SAMPLING_DENSITIES: tuple[SamplingDensity, ...] = ("uniform", "magnitude")

#: Axial direction cosine below which a surviving mode is *counted* as grazing.
#:
#: A reporting threshold, not a cut: nothing is excluded here. It is the reference
#: implementation's frozen band limit, so the number this module reports is
#: directly comparable with what `ray_to_scalar`'s grazing floor would exclude --
#: which is the point of reporting it at all (CHE-189 criterion 3).
GRAZING_REPORT_FLOOR = 1.0e-2


@dataclass(frozen=True)
class SamplingDiagnostics:
    """How a ray ensemble was obtained from a field, reported rather than judged.

    A class on rule 2, and the only class this module adds. It is the public
    record a caller reads back and a benchmark stamps: R08.1 criterion 3 and
    R08.2 criteria 1 and 4 are all statements about its fields, and the
    alternative -- a free-form mapping -- is the provenance dict R02.4 removed
    from `ScalarField`.

    It judges nothing. `evanescent_power_fraction` is a loss and
    `grazing_mode_count` is a hazard, and whether either matters is the caller's
    question; this record is what lets them ask it.
    """

    #: The field it came from.
    grid_shape: tuple[int, int]
    sample_pitch_m: tuple[float, float]
    wavelength_m: float

    #: The decomposition. `total_modes` is the whole grid; `propagating_modes`
    #: survived the strict `radial < 1` cut.
    total_modes: int
    propagating_modes: int
    evanescent_mode_count: int
    evanescent_power_fraction: float

    #: The modes the strict cut *keeps* which a `1/d_n` factor would find
    #: singular, and the smallest axial cosine among them. CHE-70's eight
    #: Pythagorean-triple bins are exactly this population, and
    #: `ray_to_scalar`'s grazing floor is what decides whether a reconstruction
    #: can carry them.
    grazing_mode_count: int
    grazing_report_floor: float
    min_axial_direction_cosine: float

    #: How modes were selected. `exhaustive` is the deterministic exactness
    #: limit; a stochastic draw records its seed so the ensemble is reproducible.
    selection: str
    density: str
    seed: int | None
    drawn_mode_count: int
    launch_position_count: int
    ray_count: int

    #: What the emitted bundle declares, restated so a record is self-contained.
    measure_kind: str
    reconstruction_normalization: str

    #: Power on both sides of Parseval's `1 / (ny nx)`, so neither is mistaken
    #: for the other. Both are **relative**, for the reason
    #: `couplers.ray_to_scalar.SCALE_NOTE` gives.
    field_discrete_power: float
    modal_power_sum: float

    compute_precision: str
    input_state: dict[str, str]
    output_state: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["grid_shape"] = list(self.grid_shape)
        record["sample_pitch_m"] = list(self.sample_pitch_m)
        return record


def _require_declared_surface(
    field: ScalarField, surface: ReferenceSurface | None
) -> ReferenceSurface:
    """The surface the rays are emitted on: the field's, checked against the caller's.

    Not an override, for the same reason `ray_to_scalar`'s is not: this coupler
    changes representation and does not propagate, so the rays leave exactly where
    the field is declared. The mirror-image check keeps a round trip honest --
    if either direction could relabel the surface, the pair would compose into a
    silent defocus.
    """
    if surface is not None and surface != field.reference_surface:
        raise ContractError(
            "FRAME_MISMATCH",
            f"the caller expects the surface {surface!r} but the field is declared on "
            f"{field.reference_surface!r}. This coupler changes representation and does "
            "not propagate, so the rays can only be emitted where the field already is.",
            declaration="surface",
            remedy=(
                "Propagate the field to the expected surface first, or pass the surface "
                "the field declares."
            ),
        )
    return field.reference_surface


def _density_over(
    xp: Any, amplitudes: Any, kind: SamplingDensity, real_np: Any
) -> Any:
    """Normalized probability over the propagating bins, aligned with `amplitudes`.

    A density that is zero where the spectrum is nonzero makes the estimator
    **inconsistent** rather than merely slow -- those modes are never drawn and no
    amount of `1/p` reweighting recovers them. That is refused rather than run,
    which is the difference between an estimator that converges to the wrong
    answer and one that converges slowly.
    """
    if kind not in SAMPLING_DENSITIES:
        raise ContractError(
            "MISSING_DECLARATION",
            f"density must be one of {list(SAMPLING_DENSITIES)}, got {kind!r}",
            declaration="density",
        )
    # No `count == 0` guard, and the reason is worth stating rather than leaving
    # as an omission: the DC bin has `radial = 0 < 1` on every grid, so the
    # propagating set is never empty however coarse the pitch or long the
    # wavelength. A branch for it would be a declared failure nothing can reach,
    # which is what `representations.CONTRACT_CODES` is enumerated to prevent.
    # `tests/physics/test_scalar_to_ray.py` pins the fact instead.
    count = int(amplitudes.shape[0])
    if kind == "uniform":
        return xp.full(count, 1.0 / count, dtype=real_np)

    magnitude = xp.abs(amplitudes)
    total = float(xp.sum(magnitude))
    if total <= 0.0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            "the spectral magnitude is identically zero, so a magnitude-proportional "
            "density is undefined",
            declaration="density",
        )
    density = (magnitude / total).astype(real_np)
    if bool(xp.any((magnitude > 0.0) & (density <= 0.0))):
        raise ContractError(
            "MISSING_DECLARATION",
            "the sampling density is zero on a bin where the spectrum is nonzero; the "
            "estimator would be inconsistent, not merely noisy -- those modes are never "
            "drawn and no 1/p reweighting recovers them",
            declaration="density",
        )
    return density


def _draw_indices(density: Any, count: int, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """Draw `count` spectral-bin indices from `density`.

    The draw is host work by construction: `numpy.random.Generator` is what pins
    the seed, and bitwise reproducibility across devices is worth more here than
    avoiding one copy of a probability vector. `np.asarray` is written out rather
    than left to happen inside `rng.choice`, so the host read is visible in the
    source and not a surprise in a profile.

    The renormalization after the widening is not cosmetic: NumPy requires `p` to
    sum to 1 within a tight tolerance, and a complex64 spectrum's float32 density
    does not once it is cast to float64.
    """
    if count <= 0:
        raise ContractError(
            "EMPTY_ENSEMBLE",
            f"the mode count must be positive, got {count}",
            declaration="count",
        )
    host = np.asarray(density, dtype=np.float64)
    host = host / host.sum()
    return rng.choice(host.size, size=count, p=host)


def _require_seed_matches(rng: np.random.Generator, seed: int | None) -> None:
    """A recorded seed must actually regenerate the ensemble, or not be recorded.

    `seed` is provenance: R08.2 criterion 1 is that a declared seed reproduces an
    identical ensemble, and a record naming a seed that does not is worse than a
    record naming none -- it reads as reproducible and is not. So when both are
    given the generator is checked to be at the *initial* state for that seed.

    A caller drawing several ensembles from one advanced generator is doing
    something legitimate; it simply cannot name a single seed for the second draw,
    and passing `seed=None` says so honestly.
    """
    if seed is None:
        return
    expected = np.random.default_rng(seed).bit_generator.state
    if rng.bit_generator.state != expected:
        raise ContractError(
            "MISSING_DECLARATION",
            f"seed={seed!r} was recorded, but the generator supplied is not at the "
            "initial state for it -- either it is a different seed or it has already "
            "been drawn from. The recorded seed would not regenerate this ensemble.",
            declaration="seed",
            remedy=(
                "Pass numpy.random.default_rng(seed) freshly, or omit seed: an "
                "unrecorded seed is honest, a wrong one is not."
            ),
        )


def scalar_to_ray(
    field: ScalarField,
    *,
    surface: ReferenceSurface | None = None,
    count: int | None = None,
    density: SamplingDensity = "uniform",
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    launch_positions_xy_m: Any = None,
) -> tuple[RayBundle, SamplingDiagnostics]:
    """Decompose `field` into propagating modes and emit them as rays.

    Parameters
    ----------
    field
        The scalar field to decompose. Its own reference surface is where the
        rays are emitted.
    surface
        The surface the caller expects the field to be on, checked and not
        applied. See `_require_declared_surface`.
    count
        `None` (the default) enumerates every propagating bin -- the
        deterministic exactness limit, with no sampling error. Any other value
        draws that many modes, and then `rng` is required.
    density
        `"uniform"` or `"magnitude"`. See `SamplingDensity`.
    rng, seed
        The generator to draw from, and the seed to record. Supply `rng` for a
        stochastic draw; `seed` is recorded in the diagnostics so an ensemble can
        be regenerated, and is **not** used to make a generator -- a function that
        silently constructs its own RNG is a function whose output depends on
        something the caller cannot see. When both are given the generator is
        checked to be at the initial state for that seed, because a recorded seed
        that does not regenerate the ensemble is provenance that lies.
    launch_positions_xy_m
        `(P, 2)` launch points on the surface, in metres. Each selected mode is
        emitted from each of them with the phase `exp(i k (d_u x_p + d_v y_p))`
        its position implies, giving `P * S` rays -- a budget set by the caller
        rather than by an incoming ray count, which is what stops the count
        growing multiplicatively across cascaded surfaces (SI Algorithm S1).
        `None` is a single launch point at the transverse origin.

    Returns
    -------
    The ray bundle and the diagnostics measured while producing it, for the same
    reason `ray_to_scalar` returns a pair: `RayBundle` carries physical state and
    no provenance mapping, and a record that a caller can drop is better than a
    field on the representation that quietly becomes load-bearing.
    """
    emitted_surface = _require_declared_surface(field, surface)

    ny, nx = field.shape
    dy, dx = field.sample_pitch_m
    xp = field.xp
    namespace = namespace_of(field.u)
    dot = matmul_precision_kwargs(namespace)

    complex_dtype = compute_dtype(field.state.dtype)
    precision = complex_dtype.precision
    real_dtype = precision.real_dtype
    real_np, complex_np = numpy_dtype(real_dtype), numpy_dtype(complex_dtype)

    # Centred DFT with the 1/(ny nx) folded in, so the coefficients *are* the
    # modal amplitudes and a plain sum of modes is the field.
    spectrum = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(field.u))).astype(complex_np) / (
        ny * nx
    )

    # Transverse direction cosines, d = lambda * f. Computed at the **widest real
    # dtype the namespace has**, not at the field's storage precision, because the
    # evanescent mask is a property of the grid and the wavelength -- both of which
    # arrive as Python floats -- and not of how the samples happen to be stored.
    # This is not hypothetical on the population criterion 3 is about: on the
    # CHE-70 grid the (30, 40) bins sit at `d_u^2 + d_v^2 = 1 - 1.1e-16`, which
    # float64 keeps and float32 rounds to >= 1, so a complex64 field would
    # otherwise emit a *different mode set* than the same field in complex128.
    # (Under JAX without x64 a float64 request comes back float32; the mask then
    # follows the namespace, which is a property of the execution environment and
    # is visible in `compute_precision`.)
    mask_np = numpy_dtype(DType.FLOAT64)
    direction_u = (
        xp.fft.fftshift(xp.fft.fftfreq(nx, d=dx)) * field.wavelength_m
    ).astype(mask_np)
    direction_v = (
        xp.fft.fftshift(xp.fft.fftfreq(ny, d=dy)) * field.wavelength_m
    ).astype(mask_np)
    grid_v, grid_u = xp.meshgrid(direction_v, direction_u, indexing="ij")
    radial = grid_u**2 + grid_v**2

    # Strict. The grazing d_n = 0 bin is singular for any 1/d_n factor and is
    # excluded; a bin one ulp inside is kept, which is the population R07.4 owns.
    propagating = radial < 1.0

    mode_power = xp.abs(spectrum) ** 2
    modal_power_sum = float(xp.sum(mode_power))
    evanescent_fraction = (
        float(xp.sum(mode_power[~propagating]) / modal_power_sum)
        if modal_power_sum > 0.0
        else 0.0
    )

    # Still at the mask precision here: the axial cosine of a near-grazing bin is
    # `sqrt` of a difference of nearly equal numbers, so computing it at the
    # storage precision would lose it for the same reason the mask would. The cast
    # to the field's precision happens once, on the emitted directions.
    transverse = xp.column_stack([grid_u[propagating], grid_v[propagating]])
    amplitudes = spectrum[propagating]
    axial = xp.sqrt(xp.clip(1.0 - xp.sum(transverse**2, axis=1), 0.0, None))

    probabilities = _density_over(xp, amplitudes, density, real_np)

    if count is None:
        # An exhaustive enumeration is the *zero-variance* case of a uniform draw:
        # every bin is selected exactly once, so `(1/M) sum_m U~[m] / p[m]` equals
        # `sum_m U~[m]` -- the field -- precisely when `p` is uniform. Under any
        # other density each bin still appears once while carrying `1/p[m]`, with
        # no compensating draw frequency, and the sum is not the field. That is a
        # silent wrong answer, so the pair is refused rather than resolved by
        # quietly substituting the uniform density.
        if density != "uniform":
            raise ContractError(
                "MISSING_DECLARATION",
                f"an exhaustive enumeration was requested with the {density!r} density. "
                "Enumeration selects every bin once regardless of p, so the 1/p in the "
                "measure has no compensating draw frequency and the reconstruction is "
                f"sum_m U~[m] / (M p[m]), not sum_m U~[m]. Only 'uniform' makes the two "
                "the same.",
                declaration="density",
                remedy=(
                    "Enumerate with density='uniform' for the exactness limit, or pass a "
                    "count and an rng to draw from the density you asked for."
                ),
            )
        indices = np.arange(int(amplitudes.shape[0]), dtype=np.int64)
        selection = "exhaustive"
    else:
        if rng is None:
            raise ContractError(
                "MISSING_DECLARATION",
                "a stochastic draw needs an explicitly seeded generator; this function "
                "will not make one for you, because a result that depends on state the "
                "caller cannot see is not reproducible even in principle",
                declaration="rng",
                remedy="Pass numpy.random.default_rng(seed), and pass the same seed.",
            )
        _require_seed_matches(rng, seed)
        indices = _draw_indices(probabilities, count, rng)
        selection = "stochastic"

    selected_transverse = transverse[indices]
    selected_axial = axial[indices]
    selected_amplitudes = amplitudes[indices]
    selected_probabilities = probabilities[indices]
    directions = xp.column_stack([selected_transverse, selected_axial]).astype(real_np)

    if launch_positions_xy_m is None:
        launch = xp.zeros((1, 2), dtype=real_np)
    else:
        launch = xp.asarray(launch_positions_xy_m, dtype=real_np)
        if launch.ndim != 2 or launch.shape[1] != 2:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"launch_positions_xy_m must be (P, 2), got {tuple(launch.shape)}",
                declaration="launch_positions_xy_m",
            )
    launch_count = int(launch.shape[0])
    mode_count = int(indices.size)

    # exp(i k (d_u x_p + d_v y_p)), the phase each launch point implies. Written
    # with an explicit complex dtype for the reason `ray_to_scalar._cis` gives:
    # scalar promotion must not decide a contract-visible dtype.
    projected = xp.matmul(launch, selected_transverse.T, **dot)
    launch_phase = xp.exp((field.wavenumber * projected).astype(complex_np) * 1j)

    amplitude = (selected_amplitudes[None, :] * launch_phase).reshape(-1)
    positions = xp.column_stack(
        [
            xp.repeat(launch[:, 0], mode_count),
            xp.repeat(launch[:, 1], mode_count),
            xp.full(launch_count * mode_count, emitted_surface.z_m, dtype=real_np),
        ]
    )

    rays = RayBundle(
        positions_m=positions,
        directions=xp.tile(directions, (launch_count, 1)),
        wavelength_m=field.wavelength_m,
        reference_surface=emitted_surface,
        frame=field.frame,
        # The modal amplitude with its launch phase, and **no** 1/p. The
        # importance weight is the measure below, and `ray_to_scalar` multiplies
        # them exactly once.
        amplitude=amplitude,
        optical_path_m=xp.zeros(launch_count * mode_count, dtype=real_np),
        optical_path_reference=(
            f"zero at the emitting surface {emitted_surface.name!r}; the accumulated "
            "path restarts here"
        ),
        measure_weight=xp.tile(1.0 / selected_probabilities, launch_count).astype(real_np),
        # This ensemble is a Monte-Carlo estimate of an integral -- of a *sum* of
        # modes when the selection is exhaustive, which is the same estimator with
        # zero variance -- so a coherent reconstruction from it owes the 1/N of SI
        # eq S5. `importance_weight` is what obliges `ray_to_scalar` to apply it.
        measure_kind="importance_weight",
    )

    grazing = axial < GRAZING_REPORT_FLOOR
    diagnostics = SamplingDiagnostics(
        grid_shape=(ny, nx),
        sample_pitch_m=(dy, dx),
        wavelength_m=field.wavelength_m,
        total_modes=int(spectrum.size),
        propagating_modes=int(amplitudes.shape[0]),
        evanescent_mode_count=int(spectrum.size) - int(amplitudes.shape[0]),
        evanescent_power_fraction=evanescent_fraction,
        grazing_mode_count=int(xp.sum(grazing)),
        grazing_report_floor=GRAZING_REPORT_FLOOR,
        min_axial_direction_cosine=float(xp.min(axial)),
        selection=selection,
        density=density,
        seed=seed,
        drawn_mode_count=mode_count,
        launch_position_count=launch_count,
        ray_count=rays.count,
        measure_kind=rays.measure_kind,
        reconstruction_normalization="one_over_n",
        field_discrete_power=field.discrete_power(),
        modal_power_sum=modal_power_sum,
        compute_precision=str(precision),
        input_state=field.state.as_dict(),
        output_state=ArrayState(
            dtype_of(amplitude), device_of(amplitude), namespace_of(amplitude)
        ).as_dict(),
    )
    return rays, diagnostics
