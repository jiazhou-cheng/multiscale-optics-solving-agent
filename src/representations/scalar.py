"""One scalar-wave representation: `ScalarField`, a sampled complex amplitude.

CHE-176 (R02.4). Exactly one public scalar type, replacing the reference
implementation's `ComplexField` plus the separate wave-samples type. `PSF` is
**not** here: an observable derived from state is a measurement (R11), and
"it serializes nicely" is not what makes something a representation.

`u` is an **amplitude**. Intensity is `|u|^2`, and a real array handed in as `u`
is refused rather than read as a magnitude -- `|U|` has already discarded the
phase, and nothing downstream can recover it.

Pitch is explicit and required
------------------------------
There is no default and no inference from the array shape. M1 measured a 256x256
Chromatix input coming back as 1756x1756 after internal padding, so a shape is
not an extent and the number of samples says nothing about how big they are. The
pad state travels with the field for the same reason: `extent_m` describes the
array as it stands, and `pad_width` is what lets a consumer recover the window
the producer actually modelled.

`validity` is typed, and it is not provenance
---------------------------------------------
The reference implementation carried the CHE-50 no-curvature-term limitation in
`provenance["validity"]`, a string in a free-form dict. That is the right
instinct in the wrong home: `docs/architecture_principles.md` requires that a
provenance field never secretly determine physical interpretation, and the test
is simple -- if deleting the field changes how the numbers may be used, it is not
provenance. A declared limitation on where a field is valid is exactly that, so
it is a typed set here, and an unknown flag is refused.

A **set**, not one value, because the limitations are independent and the risk is
that expressing one hides the other. A carrier-removed field with no curvature
term declares both; a field with neither declares the empty set, which is the
default and the strongest statement this type can make.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

from numerics import ArrayState, array_state, numpy_dtype, xp_for
from representations.contracts import (
    ContractError,
    adopt_array,
    require_finite,
    require_positive_si,
)
from representations.geometry import PHASOR, Frame, ReferenceSurface

__all__ = [
    "VALIDITY_FLAGS",
    "VALIDITY_NOTES",
    "ScalarField",
    "ValidityFlag",
]

#: The declared-limitation vocabulary. Each flag is a statement about what this
#: field may be *used for*, not a note about how it was made.
ValidityFlag = Literal[
    "surface_only",
    "no_wavefront_curvature_term",
    "carrier_removed_phase",
]

VALIDITY_FLAGS: tuple[ValidityFlag, ...] = (
    "surface_only",
    "no_wavefront_curvature_term",
    "carrier_removed_phase",
)

#: What each flag costs a consumer, with the measurement behind it.
#:
#: Kept beside the vocabulary rather than in a docstring so a diagnostic can print
#: the consequence, not just the name. These are recorded findings, restated; none
#: of them was re-measured by R02.4.
VALIDITY_NOTES: dict[str, str] = {
    "surface_only": (
        "Valid at the declared reference surface and nowhere else: zero further "
        "propagation. Composing this field into a propagation is not a loss of "
        "accuracy, it is a different physical claim."
    ),
    "no_wavefront_curvature_term": (
        "Carries no exp(i k r^2 / 2R) wavefront-curvature term, because the wavelet sum "
        "that produced it is linear in the transverse coordinate (CHE-50). Measured "
        "consequence: about 1.2 rad of phase error against an exact spherical-wave "
        "reference at the 5-Airy-radius gate edge, while the intensity residual sits at "
        "1e-3 -- so |U|^2 will not warn a consumer who propagates it further."
    ),
    "carrier_removed_phase": (
        "The phase is relative to a removed linear carrier, not absolute. Two fields "
        "with different removed carriers may not be added or interfered without "
        "restoring them first."
    ),
}


@dataclass(frozen=True)
class ScalarField:
    """A sampled scalar complex field on a declared surface, in SI.

    A class on rules 1 and 2. Rule 1: the array, its pitch, its wavelength, the
    surface it sits on and its pad state are one physical object -- a pitch that
    does not belong to this array, or a shape read without the pad width, gives a
    physical extent that is wrong by a factor and looks entirely plausible. Rule
    2: it is the public model a wave solver produces and a measurement consumes.
    """

    #: `(ny, nx)` complex amplitude, in `(y, x)` order.
    u: Any

    #: `(dy, dx)` sample spacing in metres. Required, positional, no default:
    #: acceptance criterion 1 is that a field without an explicit pitch cannot be
    #: constructed at all.
    sample_pitch_m: tuple[float, float]

    #: Vacuum wavelength in metres. One value per field.
    wavelength_m: float

    #: The surface this field is sampled on.
    reference_surface: ReferenceSurface

    frame: Frame = field(default_factory=Frame)

    #: Declared limitations. Empty means none are declared -- which is a claim,
    #: not an absence of one.
    validity: frozenset[ValidityFlag] = frozenset()

    #: Samples of padding added on each side by the producer, in samples. `0` with
    #: `padded=False` is an unpadded field.
    pad_width: int = 0

    #: Whether `u` is still in its padded form. A propagation that padded and then
    #: cropped back reports `padded=False` with the width it used.
    padded: bool = False

    phasor: str = PHASOR

    def __post_init__(self) -> None:
        u = adopt_array(self.u, name="u", complex_=True)
        object.__setattr__(self, "u", u)
        if u.ndim != 2:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"u must be a 2-D (y, x) array, got shape {tuple(u.shape)}",
                declaration="u",
            )
        require_finite(u, name="u")

        pitch = tuple(float(value) for value in self.sample_pitch_m)
        if len(pitch) != 2:
            raise ContractError(
                "SHAPE_MISMATCH",
                f"sample_pitch_m must be (dy, dx), got {len(pitch)} value(s): "
                f"{self.sample_pitch_m!r}",
                declaration="sample_pitch_m",
            )
        object.__setattr__(
            self,
            "sample_pitch_m",
            (
                require_positive_si(pitch[0], name="sample_pitch_m[dy]"),
                require_positive_si(pitch[1], name="sample_pitch_m[dx]"),
            ),
        )
        object.__setattr__(
            self, "wavelength_m", require_positive_si(self.wavelength_m, name="wavelength_m")
        )

        if self.phasor != PHASOR:
            raise ContractError(
                "PHASOR_MISMATCH",
                f"phasor must be {PHASOR!r}, got {self.phasor!r}. The conjugate convention "
                "is invisible in |u|^2 and reverses the sign of every phase.",
                declaration="phasor",
            )

        flags = frozenset(self.validity)
        unknown = sorted(str(flag) for flag in flags - set(VALIDITY_FLAGS))
        if unknown:
            raise ContractError(
                "UNKNOWN_VALIDITY_FLAG",
                f"{unknown} is not a declared validity flag. The vocabulary is "
                f"{list(VALIDITY_FLAGS)}; a limitation nothing can branch on is the "
                "free-form provenance string this field exists to replace.",
                declaration="validity",
            )
        object.__setattr__(self, "validity", flags)

        pad_width = int(self.pad_width)
        if pad_width < 0:
            raise ContractError(
                "PAD_STATE_UNKNOWN",
                f"pad_width must be a non-negative sample count, got {pad_width!r}",
                declaration="pad_width",
            )
        if self.padded and pad_width <= 0:
            raise ContractError(
                "PAD_STATE_UNKNOWN",
                "the field is marked padded but declares no pad width, so the modelled "
                "window cannot be recovered from the array",
                declaration="pad_width",
            )
        object.__setattr__(self, "pad_width", pad_width)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.u.shape[0]), int(self.u.shape[1]))

    @property
    def extent_m(self) -> tuple[float, float]:
        """`(height, width)` of the array as it stands, in metres.

        Derived from the pitch, never from the shape alone -- and it describes the
        array *including* any padding. A consumer that wants the window the
        producer modelled subtracts `2 * pad_width` samples first.
        """
        ny, nx = self.shape
        dy, dx = self.sample_pitch_m
        return (ny * dy, nx * dx)

    @property
    def wavenumber(self) -> float:
        """Free-space wavenumber `k = 2 pi / lambda`, in rad/m."""
        return 2.0 * math.pi / self.wavelength_m

    @property
    def state(self) -> ArrayState:
        """Observed dtype, device and namespace of `u`. Never caller-declared."""
        return array_state(self.u)

    @property
    def xp(self) -> Any:
        return xp_for(self.state.namespace)

    def coordinates(self) -> tuple[Any, Any]:
        """`(y, x)` coordinate vectors in metres, on the `n // 2` origin.

        The one place the origin rule is applied to this type, delegating the rule
        itself to `Frame.origin_index` so a coupler cannot quietly adopt a
        different centring.

        Built in the field's own namespace, device and real precision, so a GPU
        complex64 field does not silently produce host float64 axes that every
        downstream operation then has to move or demote.
        """
        xp = self.xp
        ny, nx = self.shape
        dy, dx = self.sample_pitch_m
        real = numpy_dtype(self.state.dtype.precision.real_dtype)
        y = (xp.arange(ny, dtype=real) - self.frame.origin_index(ny)) * dy
        x = (xp.arange(nx, dtype=real) - self.frame.origin_index(nx)) * dx
        return y, x

    def discrete_power(self) -> float:
        """`sum |u|^2 dy dx` -- a **relative** quantity, not an SI power in watts.

        There is no radiometric normalization anywhere in this tree, so this number
        is only meaningful against another one computed the same way (before and
        after a propagation, at two ray densities). Returning a Python float
        synchronizes a GPU array, so this is for diagnostics and recorded metadata,
        never for the inside of a propagation loop.
        """
        xp = self.xp
        dy, dx = self.sample_pitch_m
        return float(xp.sum(xp.abs(self.u) ** 2) * dy * dx)
