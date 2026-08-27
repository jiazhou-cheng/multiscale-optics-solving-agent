"""The FULL_FIELD diffractive interaction — SI Algorithm S1 (CHE-26).

What this module is, in the package's own vocabulary (CHE-142): the
``FULL_FIELD`` **model** of the one diffractive interaction, not a coupler in
the representation-transition sense. Incident coherent rays meet a diffractive
surface and coherent rays come out; the two representation transitions inside
are its implementation, not its identity. ``LOCAL_PATCH``
(:mod:`couplers.patch`) is the same interaction at the other granularity, and
this one is its **shortcut** rather than its peer — SI S10, restated where the
models are declared in :mod:`couplers.interaction`. Reach it through
:func:`couplers.interaction.diffractive_interaction` with the model named;
:func:`planar_doe_step` below is that model's implementation and stays exported
because every shipped call site and committed record uses it.

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

Conventions this step declares
------------------------------
Three things change across the step that are invisible in an intensity, so they
are stated here and reported in the diagnostics rather than left to be inferred.

**Optical path length resets to zero.** The outgoing rays carry ``OPL = 0``,
because the accumulated path *restarts* at this plane: the incident path is
already folded into ``U_in``'s phase by the accumulation in step 1, and carrying
it forward as well would count it twice. The consequence a caller must know is
that the phase reference has been rebased -- an OPL measured after this step is
measured from this plane, not from wherever the incident bundle launched. The
reference plane on the returned bundle is what says where that is.

**Amplitude is a spectral amplitude, not the incident weight.** Each outgoing
ray carries ``U~[m] / p[m]``, the importance-weighted modal amplitude. Its
relationship to the incident rays' amplitudes runs through the accumulation and
the transmission; there is no per-ray correspondence to recover.

**Power is not conserved by default, and that is deliberate.** A lossy DOE
legitimately loses power. ``preserve_energy`` renormalizes the transmitted field
to the incident power and is **off by default**: on, it would hide a DOE losing
power for a real reason, which is the failure mode a conservation check exists
to catch. When it is on, the factor applied is reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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

__all__ = [
    "CascadeDiagnostics",
    "PrimarySampling",
    "planar_doe_step",
    "sample_primary_positions",
]


class PrimarySampling(StrEnum):
    """Where the outgoing rays are launched from on the DOE plane.

    The step needs ``P`` primary positions and ``S`` secondary directions per
    position. Directions come from the spectrum by importance sampling; the
    positions have to come from somewhere, and until CHE-95 the caller had to
    supply them.

    Note the asymmetry, because it is not obviously right: directions are drawn
    **by spectrum magnitude** and these positions are drawn **uniformly**. The
    paper does not justify the uniform choice, and this module did not adopt an
    argument it did not have -- ``UNIFORM_ON_GRID`` was offered because it is
    what the reference implementation does, not because it was established as the
    best estimator of the position integral.

    **CHE-120 supplied the argument, and for this route it favours uniform.**
    The variance-optimal density over a discrete set of positions is
    ``q_c ~ f_c``, where ``f_c`` is the modulus of the importance weight a ray
    launched from ``c`` carries (Cauchy-Schwarz; see
    :mod:`couplers.patch_positions` for the derivation). On **this** route every
    launch position resamples **one global spectrum**, and the only thing the
    position changes is the phase ``exp(i k (d_u x_p + d_v y_p))`` -- modulus 1.
    So ``f_c`` is the same at every position, ``q_c ~ f_c`` *is* the uniform
    density, and no reweighting of positions can reduce the dominant variance
    term here. The asymmetry is real and, on this FULL_FIELD route, correct.
    (The benchmark records call it RW-F; that label is theirs, and the model
    name is what the package uses.)

    It is **not** correct on the patch route, where each patch transforms its own
    window and ``f_c = ||U~_c||_1`` varies with how much aperture that window
    holds. There the same derivation gives a measured 1.44x, and
    :mod:`couplers.patch_positions` implements it. The two routes differ because
    one spectrum is shared and the other is not, which is the distinction this
    docstring previously lacked the argument to draw.
    """

    #: Uniform over the DOE's sample positions, without replacement where
    #: possible. What the reference implementation does.
    UNIFORM_ON_GRID = "uniform_on_grid"
    #: Keep each incident ray's own transverse position. Preserves whatever
    #: spatial structure the incident bundle had, at the cost of tying the
    #: outgoing count to the incoming one for the position axis.
    INCIDENT_POSITIONS = "incident_positions"


def sample_primary_positions(
    kind: PrimarySampling,
    *,
    count: int,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    bundle: RayBundle | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray[Any, Any]:
    """``(count, 2)`` launch positions in metres, on the plane's own origin rule.

    The origin rule matters and is easy to get wrong: coordinate zero sits at
    index ``n // 2``, matching :meth:`ComplexField.coordinates`. A sampler that
    used ``(n - 1) / 2`` would place every ray half a pitch off, which is
    invisible in an intensity and is a real phase error.
    """
    if count <= 0:
        raise ContractError(
            ContractCode.EMPTY_ENSEMBLE,
            f"primary position count must be positive, got {count}",
            declaration="count",
        )
    if kind is PrimarySampling.INCIDENT_POSITIONS:
        if bundle is None:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "INCIDENT_POSITIONS needs the incident bundle to read positions from",
                declaration="bundle",
            )
        positions = np.asarray(bundle.positions_m, dtype=np.float64)[:, :2]
        if positions.shape[0] < count:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"asked for {count} primary positions but the incident bundle has "
                f"{positions.shape[0]} rays; INCIDENT_POSITIONS cannot invent one",
                declaration="count",
                remedy="Lower the count, or use UNIFORM_ON_GRID.",
            )
        return positions[:count]

    if rng is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "UNIFORM_ON_GRID draws positions and so needs an explicit seeded generator",
            declaration="rng",
        )
    ny, nx = int(grid_shape[0]), int(grid_shape[1])
    dy, dx = float(sample_pitch_m[0]), float(sample_pitch_m[1])
    total = ny * nx
    replace = count > total
    flat = rng.choice(total, size=count, replace=replace)
    rows, cols = np.divmod(flat, nx)
    y = (rows - ny // 2) * dy
    x = (cols - nx // 2) * dx
    return np.column_stack([x, y]).astype(np.float64)


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
    #: How the primary launch positions were obtained. ``"caller"`` when the
    #: caller supplied them, otherwise the sampler that produced them.
    launch_source: str = "caller"
    #: ``sqrt(P_in / P_out)`` if ``preserve_energy`` was requested, else
    #: ``None``. Reported whenever applied, because a renormalization that is
    #: not visible in the record is indistinguishable from a DOE that happens
    #: to conserve power.
    energy_preservation_factor: float | None = None
    #: Half-width of the zero padding added per side before the transform, in
    #: samples. ``0`` when the caller's grid is used as given.
    pad_width: int = 0
    #: True when the collapsed single-ray mode was used instead of sampling.
    collapsed_to_mean_wavevector: bool = False
    #: The declared conventions, restated on every result so a consumer reading
    #: only the record still gets them.
    opl_convention: str = (
        "reset to 0 at this plane; the incident path is already in the "
        "accumulated field's phase and carrying it forward would double-count it"
    )
    amplitude_convention: str = (
        "importance-weighted spectral amplitude U~[m]/p[m]; no per-ray "
        "correspondence to the incident amplitudes survives the accumulation"
    )

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
            "launch_source": self.launch_source,
            "energy_preservation_factor": self.energy_preservation_factor,
            "pad_width": self.pad_width,
            "collapsed_to_mean_wavevector": self.collapsed_to_mean_wavevector,
            "opl_convention": self.opl_convention,
            "amplitude_convention": self.amplitude_convention,
        }


def planar_doe_step(
    bundle: RayBundle,
    doe_transmission: np.ndarray[Any, Any],
    *,
    grid_shape: tuple[int, int],
    sample_pitch_m: tuple[float, float],
    plane: ReferencePlane,
    launch_positions_xy_m: np.ndarray[Any, Any] | None = None,
    primary_sampling: PrimarySampling | None = None,
    primary_count: int | None = None,
    secondary_count: int | None = None,
    density_kind: SamplingDensity = SamplingDensity.UNIFORM,
    preserve_energy: bool = False,
    pad_width: int = 0,
    rng: np.random.Generator | None = None,
) -> tuple[RayBundle, ComplexField, CascadeDiagnostics]:
    """One planar ray -> field -> DOE -> field -> ray step (Algorithm S1).

    Every option below defaults to the behaviour this function had before
    CHE-95 added them, so an existing call is unchanged.

    ``secondary_count``
        ``None`` enumerates every propagating bin: the deterministic limit with
        no sampling error, and the gate the coupler protocol makes mandatory and
        first. Any value ``>= 2`` draws from ``rng``, which must then be
        supplied. A value ``<= 1`` selects the **collapsed** mode -- one
        outgoing ray along the power-weighted mean wavevector, with the whole
        spectrum's amplitude on it. That is a cheap preview, not an
        approximation with a stated error, and it is reported as
        ``collapsed_to_mean_wavevector``.

    ``launch_positions_xy_m`` / ``primary_sampling`` / ``primary_count``
        Supply positions directly, or ask for them. Exactly one, because a
        request that supplies both is ambiguous rather than redundant.

    ``preserve_energy``
        Renormalize the transmitted field to the incident power,
        ``U_out *= sqrt(P_in / P_out)``. **Off by default and it should stay
        off**: this is a policy, not physics, and a lossy DOE legitimately loses
        power. When on, the factor is reported so the record shows a
        renormalization happened.

    ``pad_width``
        Zero-pad the accumulated field by this many samples per side before the
        transform, giving a flat computational aperture around the DOE. ``0``
        uses the caller's grid as given. Padding changes the spectral sampling,
        not the physics: it interpolates the angular spectrum onto a finer grid.

    Returns the outgoing bundle, the transmitted field (so a caller can check
    power without re-deriving it), and diagnostics carrying every choice above
    plus the OPL and amplitude conventions.
    """
    if launch_positions_xy_m is not None and primary_sampling is not None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "launch_positions_xy_m and primary_sampling both specify where the "
            "outgoing rays launch from",
            declaration="launch_positions_xy_m",
            remedy="Supply exactly one.",
        )
    if launch_positions_xy_m is None and primary_sampling is None:
        raise ContractError(
            ContractCode.MISSING_DECLARATION,
            "no launch positions: supply launch_positions_xy_m, or a "
            "primary_sampling kind and primary_count",
            declaration="launch_positions_xy_m",
        )
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

    if primary_sampling is not None:
        if primary_count is None:
            raise ContractError(
                ContractCode.MISSING_DECLARATION,
                "primary_sampling needs primary_count",
                declaration="primary_count",
            )
        launch_positions_xy_m = sample_primary_positions(
            primary_sampling,
            count=primary_count,
            grid_shape=grid_shape,
            sample_pitch_m=sample_pitch_m,
            bundle=bundle,
            rng=rng,
        )
        launch_source = str(primary_sampling)
    else:
        launch_source = "caller"

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
    transmitted_u = incident_field.u * transmission

    energy_factor: float | None = None
    if preserve_energy:
        power_in = float(np.sum(np.abs(incident_field.u) ** 2))
        power_out = float(np.sum(np.abs(transmitted_u) ** 2))
        if power_out <= 0.0:
            raise ContractError(
                ContractCode.NON_FINITE,
                "preserve_energy cannot renormalize a transmitted field with zero "
                "power; the DOE extinguished everything",
                declaration="preserve_energy",
            )
        energy_factor = float(np.sqrt(power_in / power_out))
        transmitted_u = transmitted_u * energy_factor

    if pad_width:
        if pad_width < 0:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"pad_width must be non-negative, got {pad_width}",
                declaration="pad_width",
            )
        # Zero, not edge-clamp. A bounded DOE has no field outside it, and
        # continuing the edge value would invent one.
        transmitted_u = np.pad(transmitted_u, pad_width, mode="constant")

    transmitted = ComplexField(
        u=transmitted_u,
        sample_pitch_m=incident_field.sample_pitch_m,
        wavelength_m=incident_field.wavelength_m,
        reference_plane=plane,
        frame=incident_field.frame,
        normalization=incident_field.normalization,
        provenance={
            **incident_field.provenance,
            "doe_applied": True,
            "energy_preservation_factor": energy_factor,
            "pad_width": int(pad_width),
        },
    )

    # Steps 3-4: one global FFT, then resample a fixed budget.
    spectrum = decompose(transmitted)
    density = sampling_density(spectrum, density_kind)
    collapsed = False
    if secondary_count is None:
        indices = enumerate_indices(density)
    elif secondary_count <= 1:
        # The collapsed preview: one outgoing ray along the power-weighted mean
        # wavevector. Implemented by selecting the single bin nearest that mean
        # rather than by synthesising a direction off the spectral grid, so the
        # ray is still an actual mode of the transmitted field and every
        # downstream invariant (unit direction, propagating, on-grid) holds.
        collapsed = True
        amplitudes = np.asarray(spectrum.propagating_amplitudes())
        transverse = np.asarray(spectrum.transverse_directions())
        power = np.abs(amplitudes) ** 2
        total = float(power.sum())
        if total <= 0.0:
            raise ContractError(
                ContractCode.EMPTY_ENSEMBLE,
                "the transmitted field carries no propagating power to collapse",
                declaration="secondary_count",
            )
        mean_transverse = (power[:, None] * transverse).sum(axis=0) / total
        indices = np.array(
            [int(np.argmin(((transverse - mean_transverse) ** 2).sum(axis=1)))],
            dtype=np.int64,
        )
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
        launch_source=launch_source,
        energy_preservation_factor=energy_factor,
        pad_width=int(pad_width),
        collapsed_to_mean_wavevector=collapsed,
    )
    return outgoing, transmitted, diagnostics
