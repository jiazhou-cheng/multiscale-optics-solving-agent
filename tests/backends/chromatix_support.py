"""One small field builder shared by the chromatix boundary and solver tests.

A module rather than a fixture because two test files construct the same source
field and a fixture would have to live in a `conftest.py` that the rest of the
suite then loads for no reason.
"""

from __future__ import annotations

import numpy as np

from representations import ReferenceSurface, ScalarField

WAVELENGTH_M = 0.532e-6
#: Deliberately non-square in both count and pitch: an axis-symmetric fixture
#: cannot fail on a transposition.
SHAPE = (48, 64)
PITCH_M = (0.30e-6, 0.25e-6)


def a_scalar_field(
    *, dtype: str = "complex64", z_m: float = 0.0, medium_index: float = 1.0
) -> ScalarField:
    """A small confined Gaussian on a named source plane, with an explicit pitch."""
    y = (np.arange(SHAPE[0]) - SHAPE[0] // 2) * PITCH_M[0]
    x = (np.arange(SHAPE[1]) - SHAPE[1] // 2) * PITCH_M[1]
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    u = np.exp(-(grid_x**2 + grid_y**2) / (2e-6) ** 2).astype(dtype)
    return ScalarField(
        u=u,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=ReferenceSurface(name="source", z_m=z_m, medium_index=medium_index),
    )
