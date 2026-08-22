"""Cascaded planar DOE step — SI Algorithm S1 (CHE-26).

Propagating sampled secondary rays into the next surface and resampling there
makes the ray count grow multiplicatively per surface: ``P*S`` rays in becomes
``P*S`` times the next surface's secondary count. For **planar** surfaces the
paper gives an exact way out, because every ray crosses one common Cartesian
plane:

    1. accumulate ALL incident rays coherently into ``U_in`` on that plane;
    2. apply the complex DOE transmission once, ``U_out = U_in * U_DOE``;
    3. one global FFT gives the outgoing angular spectrum;
    4. resample a caller-specified budget of ``P`` launch positions times ``S``
       secondary rays from it.

The outgoing count is the budget, not a function of the incoming count.
Coherent interference at the plane survives because the accumulation in step 1
happens before the transmission in step 2 -- the order is the whole point, and
reversing it would apply the DOE to each ray in isolation.

This does **not** apply to conformal surfaces. Rays there strike different
local tangent planes with position-dependent frames and normals, so there is no
common plane to accumulate onto, and the patch size is instead bounded by the
curvature condition of :mod:`~couplers.curvature`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.boundary import (
    ComplexField,
    ContractCode,
    ContractError,
    RayBundle,
    ReferencePlane,
)
from couplers.ray_to_wave import Projection, ray_to_wave
from couplers.wave_to_ray import (
    SamplingDensity,
    decompose,
    draw_indices,
    enumerate_indices,
    sampling_density,
    spectrum_to_rays,
)

__all__ = ["CascadeDiagnostics", "planar_doe_step"]


@dataclass(frozen=True)
class CascadeDiagnostics:
    incident_ray_count: int
    outgoing_ray_count: int
    launch_count: int
    secondary_count: int
    incident_discrete_power: float
    transmitted_discrete_power: float
    evanescent_power_fraction: float
    propagating_modes: int
    density_kind: str
    enumerated: bool

    @property
    def count_growth(self) -> float:
        return self.outgoing_ray_count / max(self.incident_ray_count, 1)

    def as_dict(self) -> dict[str, Any]:
        return {
            "incident_ray_count": self.incident_ray_count,
            "outgoing_ray_count": self.outgoing_ray_count,
            "launch_count": self.launch_count,
            "secondary_count": self.secondary_count,
            "count_growth": self.count_growth,
            "incident_discrete_power": self.incident_discrete_power,
            "transmitted_discrete_power": self.transmitted_discrete_power,
            "evanescent_power_fraction": self.evanescent_power_fraction,
            "propagating_modes": self.propagating_modes,
            "density_kind": self.density_kind,
            "enumerated": self.enumerated,
        }


def planar_doe_step(
    bundle: RayBundle,
    doe_transmission: np.ndarray,
    *,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    plane: ReferencePlane,
    launch_positions_xy_m: np.ndarray,
    secondary_count: int | None = None,
    density_kind: SamplingDensity = SamplingDensity.UNIFORM,
    rng: np.random.Generator | None = None,
) -> tuple[RayBundle, ComplexField, CascadeDiagnostics]:
    """One planar ray -> field -> DOE -> field -> ray step (Algorithm S1).

    ``secondary_count=None`` enumerates every propagating bin, giving the
    deterministic limit with no sampling error. Any other value draws from
    ``rng``, which must then be supplied.

    Returns the outgoing bundle, the transmitted field (so a caller can check
    power without re-deriving it), and diagnostics.
    """
    transmission = np.asarray(doe_transmission)
    if transmission.shape != tuple(grid_shape):
        raise ContractError(
            ContractCode.SHAPE_MISMATCH,
            f"DOE transmission {transmission.shape} must match the plane grid {grid_shape}",
            declaration="doe_transmission",
        )
    if not np.iscomplexobj(transmission):
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "DOE transmission must be complex; a real array is an amplitude mask "
            "with an undeclared phase, not a transmission",
            declaration="doe_transmission",
        )

    # Step 1: accumulate every incident ray onto the common plane, BEFORE the
    # DOE is applied. Reversing this order would apply the DOE per ray and
    # destroy the interference the step exists to preserve.
    # normalization is left to the bundle's own declaration: a spectrally
    # sampled bundle needs the 1/N of SI eq S5, a traced one must not be
    # averaged, and the cascade cannot know which it was handed.
    incident_field, _ = ray_to_wave(
        bundle,
        grid_shape=grid_shape,
        sample_pitch_m=sample_pitch_m,
        plane=plane,
        projection=Projection.ASM_CONSISTENT,
    )

    # Step 2: one transmission, applied to the accumulated field.
    transmitted = ComplexField(
        u=incident_field.u * transmission,
        sample_pitch_m=incident_field.sample_pitch_m,
        wavelength_m=incident_field.wavelength_m,
        reference_plane=plane,
        frame=incident_field.frame,
        normalization=incident_field.normalization,
        provenance={**incident_field.provenance, "doe_applied": True},
    )

    # Steps 3-4: one global FFT, then resample a fixed budget.
    spectrum = decompose(transmitted)
    density = sampling_density(spectrum, density_kind)
    if secondary_count is None:
        indices = enumerate_indices(density)
    else:
        if rng is None:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "stochastic resampling requires an explicit seeded generator",
                declaration="rng",
            )
        indices = draw_indices(density, secondary_count, rng)

    outgoing = spectrum_to_rays(
        spectrum, indices, density, launch_positions_xy_m=launch_positions_xy_m
    )

    launches = np.asarray(launch_positions_xy_m, dtype=np.float64)
    diagnostics = CascadeDiagnostics(
        incident_ray_count=bundle.count,
        outgoing_ray_count=outgoing.count,
        launch_count=int(launches.shape[0]),
        secondary_count=int(indices.size),
        incident_discrete_power=incident_field.discrete_power(),
        transmitted_discrete_power=transmitted.discrete_power(),
        evanescent_power_fraction=spectrum.evanescent_power_fraction,
        propagating_modes=spectrum.propagating_count,
        density_kind=str(density_kind),
        enumerated=secondary_count is None,
    )
    return outgoing, transmitted, diagnostics
