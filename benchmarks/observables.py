"""`intensity(field)` -- the project's only `|U|^2`, and why it is here.

CHE-213 (R06.8) acceptance criterion 5, which requires that **exactly one**
intensity path exist in the tree and that where it lives be stated. This is it,
and this is where.

`|U|^2` at a plane is a **measurement**: an observable derived from state, not a
representation and not a coupler. That is the same call
`docs/architecture_principles.md` section 2 makes and the reason `C_FIELD_TO_PSF`
was retired -- a trivial observable is not a cross-representation handoff.

So it belongs in `measurements/`, which **R11 (CHE-163) has not landed.** R06.8
explicitly permits a benchmark to compute the observable locally provided it says
that it did, and this file is that statement. When `measurements/` lands, this
function moves there and this module is deleted. What must not happen in between
is a second implementation: `tests/benchmarks/test_records.py` counts them.

Normalization: **none.** The returned array is `|u|^2` in the field's own
relative amplitude units, and no comparison anywhere in `benchmarks/` is
normalized by its own peak. A peak-normalized metric cannot see a global scale
error, which is exactly how a wrong amplitude convention survives -- recorded on
R11 as the trap that let a pre-CHE-47 launch convention pass a frozen oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from representations import ScalarField

__all__ = ["intensity"]


def intensity(field: ScalarField) -> np.ndarray[Any, np.dtype[Any]]:
    """`|u|^2` on the field's own grid, as host float64.

    Promoted to float64 *after* the modulus, not before: the field's amplitude is
    complex64 and nothing is gained by widening it, but a sum over a large grid of
    float32 intensities loses digits the harmonic readings in R06.8 need.

    Returns a plain array rather than a `ScalarField`, because an intensity is not
    a complex amplitude and putting it in that type would be a field whose `u` has
    already discarded the phase -- the exact thing `ScalarField` refuses at
    construction.
    """
    magnitude: np.ndarray[Any, np.dtype[Any]] = np.abs(np.asarray(field.u)).astype(np.float64)
    return magnitude**2
