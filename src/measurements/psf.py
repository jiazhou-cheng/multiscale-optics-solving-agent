"""The point-spread function, as a measurement on a terminal `ScalarField`.

CHE-197 (R11.1). `measurements.psf(field, *, normalization) -> PsfResult`.

Why this is a measurement and not a coupler
-------------------------------------------
A coupler performs a physically meaningful change of *representation*.
`couplers.ray_to_scalar` is one: it carries declared assumptions about the OPL
reference, the phasor sign, the amplitude weighting, the sampling measure and the
handoff plane, and it refuses when they are not stated. Getting any of them wrong
produces a field that is wrong in a way no downstream check can name.

`ScalarField -> |u|^2` is not that. It changes no representation, consults no
convention it does not already hold, and cannot be gotten wrong in more than one
way. It is an **observable** of the terminal state, and the terminal physical
state stays the field. That is why CHE-36 removed `C_FIELD_TO_PSF` from the
reference implementation's coupler registry, and it is why this ticket does not
bring it back: a trivial observable sitting in the coupler list, complete with a
framework and a derivative mode it never had numerics for, made the category
unfalsifiable. **Do not add a coupler so that a graph terminates in a particular
artifact type.**

For the same reason `PSF` is not a representation here. It serializes nicely, it
has a pitch and a frame, and none of that is what makes something a
representation -- `representations/scalar.py` says so at the top, and this module
is the other half of that sentence.

Two implementations became one
------------------------------
The reference tree had two paths and they were not the same computation:

* `verification/psf_measurement.py:254 measure_psf` took `raw = np.abs(field.u)
  ** 2` -- **NumPy, on the host** -- and derived the peak, the window energy and
  the border fraction from it;
* `core/boundary.py:1508 PSF.from_complex_field` took `field_.xp.abs(field_.u) **
  2`, in the field's own namespace and precision, and that array became the
  returned intensity.

So for a complex64 GPU field the returned intensity was a float32 device array
while every scalar reported beside it came from a float64 host array of the same
data, computed by a second squaring after an implicit transfer. Nothing checked
they agreed, and for a well-conditioned field they nearly did -- which is the
worst case, because the disagreement would only surface where the numbers were
already hard to trust.

**This module keeps the namespace-preserving definition and derives everything
from that one array.** `|u|^2` is taken in exactly one place, in the field's own
namespace, device and precision, and the peak, the window energy, the border
fraction and the returned intensity are all reductions of it. R02.4's rule --
never fabricate float64 digits a producer did not have -- is the same rule, and
the host round trip the other path performed was invisible.

The normalization is the risk, so it is never a default
-------------------------------------------------------
`normalization` is required. An oracle comparison is only meaningful against a
stated scale, and an implicitly normalized PSF entering an oracle is the specific
failure this signature exists to prevent.

**Peak normalization is blind to a constant multiplicative error**, and that is
not a footnote: it is how the pre-CHE-47 launch-amplitude convention survived. A
propagated power of `7.0e-04` and one of `2.7e-24` both look identical once
divided by the peak, and every M3 oracle divided by the peak. An omitted per-ray
area weight under uniform sampling is exactly such a constant.

So the scale is not allowed to disappear. `raw_peak_intensity` and
`raw_window_energy` are recorded **whatever the normalization**, including under
`peak`, and they are what a caller checking R07's absolute scale must read.
Energy normalization does not fix this either -- it hides the same constant and
additionally makes the result depend on how much energy left the window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Literal, get_args

from numerics import ArrayState, array_state, numpy_dtype, xp_for
from representations import ContractError, Frame, ScalarField, require_finite

__all__ = [
    "COHERENCE_MODEL",
    "NORMALIZATION_DECLARATIONS",
    "PSF_INVARIANTS",
    "PSF_NORMALIZATIONS",
    "PsfNormalization",
    "PsfResult",
    "border_energy_fraction",
    "psf",
]

#: How the measured intensity is scaled. Always an explicit choice.
#:
#: A `Literal`, not a `StrEnum`, for the reason every other vocabulary in this
#: tree is one: `scripts/class_budget.py` counts a `StrEnum` as a class and R11.1
#: budgets +1, which `PsfResult` spends. The same call R08.1 made for
#: `SamplingDensity`.
PsfNormalization = Literal["raw", "peak", "energy"]

PSF_NORMALIZATIONS: tuple[PsfNormalization, ...] = get_args(PsfNormalization)

#: What each choice means, verbatim on the result, so a consumer reading only the
#: artifact can never be unsure whether a value is peak-normalized.
NORMALIZATION_DECLARATIONS: dict[PsfNormalization, str] = {
    "raw": (
        "raw: intensity = |u|^2 in the field's own amplitude units, unscaled. Not "
        "calibrated radiometric irradiance -- no step in this tree converts an "
        "amplitude to watts -- so it is comparable only against another number "
        "computed the same way."
    ),
    "peak": (
        "peak: intensity = |u|^2 / max(|u|^2) over the sampled window. "
        "Dimensionless, and max == 1 by construction. **Blind to any constant "
        "multiplicative error**; read raw_peak_intensity and raw_window_energy to "
        "see the scale this removed."
    ),
    "energy": (
        "energy: intensity = |u|^2 / (sum(|u|^2) dy dx) over the sampled window. "
        "Integrates to 1 over that window, so the value depends on the window: "
        "energy that left the grid is not in the denominator."
    ),
}

#: Monochromatic, fully coherent, scalar. Stated on every result, because an
#: analytic Airy comparison and a Fraunhofer oracle both assume exactly this, and
#: an incoherent or polychromatic aggregation would need a different comparison.
COHERENCE_MODEL = "monochromatic, fully coherent, scalar (single wavelength)"

#: The two invariants the retired `C_FIELD_TO_PSF` registry entry declared, kept
#: under their original names so the claim can be traced from the registry that
#: removed it to the type that now enforces it. They are executed in
#: `PsfResult.__post_init__`, not asserted by an edge.
PSF_INVARIANTS = ("nonnegative_intensity", "declared_psf_normalization")

_ZERO_ENERGY_REMEDY = (
    "A dark field has no peak and no total to divide by. Normalizing anyway yields "
    "NaN, which is refused one layer later as a non-finite intensity -- naming the "
    "symptom rather than the cause. Check that the propagation ran and that the "
    "input field was not all zeros."
)


@dataclass(frozen=True)
class PsfResult:
    """A measured PSF, plus everything its normalization removed.

    A class on rule 2: it is the public, serialized record a consumer reads back,
    and three of R11.1's acceptance criteria are statements about what it carries
    -- which normalization, at what scale, with how much energy on the border. The
    alternative is a free-form mapping, which is the provenance dict R02.4 removed
    from `ScalarField` for exactly this reason.

    The raw scale is kept deliberately. A normalized PSF alone cannot answer
    whether the upstream reconstruction conserved energy, and that question is the
    only check that can see a constant multiplicative error.
    """

    #: `(ny, nx)` non-negative intensity, in the field's own namespace and dtype.
    intensity: Any

    #: `(dy, dx)` of the field this measured. The measurement resamples nothing.
    sample_pitch_m: tuple[float, float]

    wavelength_m: float

    #: Which of `PSF_NORMALIZATIONS` was applied. Required, never inferred.
    normalization: PsfNormalization

    #: The prose form of the same fact, from `NORMALIZATION_DECLARATIONS`.
    normalization_declaration: str

    #: The factor the raw `|u|^2` was multiplied by. `1.0` for `raw`.
    scale_factor: float

    #: `max(|u|^2)` **before** scaling, in the field's own amplitude units.
    raw_peak_intensity: float

    #: `sum(|u|^2) dy dx` before scaling, over the sampled window only.
    raw_window_energy: float

    #: `(iy, ix)` of the maximum; the first in row-major order if tied.
    peak_index: tuple[int, int]

    #: `(y, x)` of the maximum in metres, on the field's own origin rule.
    peak_position_m: tuple[float, float]

    #: Fraction of the intensity on the one-pixel border of the window. A
    #: finite-window indicator, not a correctness claim.
    border_energy_fraction: float

    frame: Frame = dataclass_field(default_factory=Frame)

    coherence_model: str = COHERENCE_MODEL

    def __post_init__(self) -> None:
        """`PSF_INVARIANTS`, executed.

        `nonnegative_intensity` is not vacuous even though `psf()` only ever
        constructs this from `|u|^2`: this is a public frozen dataclass, and an
        amplitude stored where an intensity was expected is the one substitution
        that produces a plausible-looking map with negative values in it.

        Finiteness is checked here too, and `psf()` reaches it on its own inputs.
        Squaring halves the usable exponent range, so a **valid, finite** complex64
        field with an amplitude above `sqrt(3.4e38)` overflows `|u|^2` to `inf`;
        `peak` then scales by `1 / inf = 0` and every sample becomes `nan`. Nothing
        else here catches that -- `nan < 0` is `False`, so the non-negativity check
        passes -- and the result would come back with a `border_energy_fraction` of
        0.0, which is the reassuring value.
        """
        if self.normalization not in PSF_NORMALIZATIONS:
            raise ContractError(
                "MISSING_DECLARATION",
                f"normalization is {self.normalization!r}; the declared choices are "
                f"{list(PSF_NORMALIZATIONS)}. A PSF that does not state its scale cannot "
                "be compared with anything.",
                declaration="normalization",
            )
        if getattr(self.intensity, "ndim", None) != 2:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"a PSF is a 2-D (y, x) intensity map, got shape {self.intensity.shape}",
                declaration="intensity",
            )
        require_finite(self.intensity, name="intensity")
        xp = xp_for(array_state(self.intensity).namespace)
        if bool(xp.any(self.intensity < 0.0)):
            raise ContractError(
                "NEGATIVE_INTENSITY",
                "PSF intensity must be non-negative; a negative value means an amplitude "
                "was stored where an intensity was expected.",
                declaration="intensity",
            )
        pitch = tuple(float(value) for value in self.sample_pitch_m)
        if len(pitch) != 2 or not all(math.isfinite(v) and v > 0.0 for v in pitch):
            raise ContractError(
                "UNIT_NOT_SI",
                f"sample_pitch_m must be a positive (dy, dx) in metres, got {pitch!r}",
                declaration="sample_pitch_m",
            )
        object.__setattr__(self, "sample_pitch_m", pitch)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.intensity.shape[0]), int(self.intensity.shape[1]))

    @property
    def state(self) -> ArrayState:
        """Observed dtype, device and namespace of the intensity map."""
        return array_state(self.intensity)

    def coordinates(self) -> tuple[Any, Any]:
        """`(y, x)` coordinate vectors in metres, on the field's own origin rule.

        Delegates the rule to `Frame.origin_index`, exactly as
        `ScalarField.coordinates` does, so a measurement cannot quietly adopt a
        different centring than the field it measured. A half-pixel shift is a
        large fraction of an Airy radius at ordinary PSF sampling.
        """
        xp = xp_for(self.state.namespace)
        ny, nx = self.shape
        dy, dx = self.sample_pitch_m
        real = numpy_dtype(self.state.dtype.precision.real_dtype)
        y = (xp.arange(ny, dtype=real) - self.frame.origin_index(ny)) * dy
        x = (xp.arange(nx, dtype=real) - self.frame.origin_index(nx)) * dx
        return y, x

    def as_dict(self) -> dict[str, Any]:
        """The serializable record, for a benchmark bundle or a comparison report."""
        return {
            "measurement": "psf",
            "is_a_graph_edge": False,
            "intensity_definition": "|u|^2",
            "normalization": self.normalization,
            "normalization_declaration": self.normalization_declaration,
            "scale_factor": self.scale_factor,
            "raw_peak_intensity": self.raw_peak_intensity,
            "raw_window_energy": self.raw_window_energy,
            "raw_energy_units": (
                "the field's own amplitude units squared, times m^2. Not watts: nothing "
                "in this tree calibrates an amplitude radiometrically."
            ),
            "peak_index": list(self.peak_index),
            "peak_position_m": list(self.peak_position_m),
            "sample_pitch_m": list(self.sample_pitch_m),
            "pitch_source": "the measured field's own sample_pitch_m",
            "border_energy_fraction": self.border_energy_fraction,
            "coherence_model": self.coherence_model,
            "wavelength_m": self.wavelength_m,
            "shape": list(self.shape),
            "origin_rule": self.frame.origin_rule,
            "axis_order": self.frame.axis_order,
            "invariants_enforced": list(PSF_INVARIANTS),
            "invariants_enforced_by": "measurements.psf.PsfResult.__post_init__",
            "gradient_claim": "none. This measurement is forward_only.",
            "execution": self.state.as_dict(),
        }


def psf(field: ScalarField, *, normalization: PsfNormalization) -> PsfResult:
    """Measure `|u|^2` on a field. No propagation, no resampling, no new physics.

    The axes are the field's own `sample_pitch_m`. For a propagated field that
    must be the propagation's **output** pitch: a solver that reports one pitch in
    and another out has a real chance of writing the input pupil pitch onto the
    output artifact, and reading it here would rescale every distance this reports
    -- peak position, first-null radius -- by a constant while leaving the
    intensity map entirely plausible. That check belongs to whoever assembles the
    field, because it is the only place both numbers exist; this function measures
    what the field declares and says so in `pitch_source`.

    Absolute phase is deliberately not required: `|u|^2` is invariant under a
    global phase, so a carrier-removed field -- one that declares
    `carrier_removed_phase` -- is admissible here. The measurement records the
    coherence model it assumes rather than inferring that the phase was meaningful.
    """
    if normalization not in PSF_NORMALIZATIONS:
        raise ContractError(
            "MISSING_DECLARATION",
            f"normalization is {normalization!r}; the declared choices are "
            f"{list(PSF_NORMALIZATIONS)}. There is no default: an oracle comparison is "
            "only meaningful against a stated scale, and an implicitly normalized PSF "
            "entering one is the failure this signature exists to prevent.",
            declaration="normalization",
        )

    xp = field.xp
    dy, dx = field.sample_pitch_m

    # `|u|^2`, taken **once**, in the field's own namespace, device and precision.
    # Every number below is a reduction of this one array -- see the module
    # docstring for the two reference paths that did not agree on it.
    #
    # The cost of squaring in the field's own precision, stated because the choice
    # is argued at length above: squaring halves the usable exponent range. A
    # complex64 field whose amplitudes reach 1e-24 -- the low end of the scale range
    # that motivated CHE-47 -- has `|u|^2` flush to subnormal or zero, and a field
    # that underflows entirely is refused below as carrying no energy, which is the
    # right refusal for the wrong reason. Squaring on the host in float64 would move
    # that boundary and would reintroduce the transfer this module exists to remove;
    # a caller near either end of float32 should hand in a complex128 field.
    raw = xp.abs(field.u) ** 2
    raw_peak = float(xp.max(raw))

    # `float(sum) * dy * dx`, not `sum * dy * dx` inside the namespace, which is
    # what `ScalarField.discrete_power` does. The two agree to about 4e-8 relative
    # on a complex64 field and this one does not flush a small pitch to zero in
    # float32. They are not the same code and are not meant to be -- reusing
    # `discrete_power` would square `|u|` a second time, which is the round trip
    # the module docstring objects to -- but they are the same number, and
    # `test_psf.py` pins that they agree.
    raw_energy = float(xp.sum(raw)) * dy * dx

    if normalization == "raw":
        scale = 1.0
    elif normalization == "peak":
        if raw_peak <= 0.0:
            raise ContractError(
                "EMPTY_ENSEMBLE",
                "the field carries no energy, so it has no peak to normalize to",
                declaration="normalization",
                remedy=_ZERO_ENERGY_REMEDY,
            )
        scale = 1.0 / raw_peak
    else:
        if raw_energy <= 0.0:
            raise ContractError(
                "EMPTY_ENSEMBLE",
                "the field carries no energy, so it has no total to normalize to",
                declaration="normalization",
                remedy=_ZERO_ENERGY_REMEDY,
            )
        scale = 1.0 / raw_energy

    # Scaled on the intensity, not by rescaling the amplitude and squaring again:
    # that round trip would put a sqrt and a square between the field and the
    # number an oracle is compared to, for nothing.
    intensity = raw if scale == 1.0 else raw * scale

    flat_peak = int(xp.argmax(raw))
    ny, nx = field.shape
    peak_row, peak_col = divmod(flat_peak, nx)

    return PsfResult(
        intensity=intensity,
        sample_pitch_m=(dy, dx),
        wavelength_m=field.wavelength_m,
        normalization=normalization,
        normalization_declaration=NORMALIZATION_DECLARATIONS[normalization],
        scale_factor=scale,
        raw_peak_intensity=raw_peak,
        raw_window_energy=raw_energy,
        peak_index=(peak_row, peak_col),
        peak_position_m=(
            (peak_row - field.frame.origin_index(ny)) * dy,
            (peak_col - field.frame.origin_index(nx)) * dx,
        ),
        border_energy_fraction=border_energy_fraction(raw),
        frame=field.frame,
    )


def border_energy_fraction(intensity: Any) -> float:
    """Fraction of the intensity on the one-pixel border of the sampled window.

    The truncation indicator, so a PSF that ran off its grid is **visible** rather
    than quietly wrong: a normalized profile of a truncated PSF looks like a
    normalized profile.

    It notices window truncation and it does not certify padding, and the
    difference is measured rather than asserted -- CHE-35 watched this number move
    by only about 2x across a run carrying 1.4e-1 relative intensity error from
    wraparound. Read it as "energy is reaching the edge", never as "the window was
    big enough".

    A window narrower than three samples has no interior, so there is no border to
    distinguish and this returns 0.0 rather than 1.0: reporting total truncation
    for a 2-sample grid would be an artifact of the definition.
    """
    if getattr(intensity, "ndim", None) != 2:
        raise ContractError(
            "SHAPE_MISMATCH",
            f"a border is a property of a 2-D window; got shape "
            f"{getattr(intensity, 'shape', type(intensity).__name__)!r}",
            declaration="intensity",
        )
    xp = xp_for(array_state(intensity).namespace)
    total = float(xp.sum(intensity))
    if total <= 0.0 or min(intensity.shape) < 3:
        return 0.0
    border = (
        float(xp.sum(intensity[0, :]))
        + float(xp.sum(intensity[-1, :]))
        + float(xp.sum(intensity[1:-1, 0]))
        + float(xp.sum(intensity[1:-1, -1]))
    )
    return border / total
