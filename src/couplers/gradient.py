"""Gradient characterization for the ray-wave coupler — SI S7.2 (CHE-28).

AGENTS.md forbids claiming a gradient across an untested boundary. This module
produces the test. It is expected to end in a *characterization*, not a
certification, and `benchmarks/coupler_protocol.yaml` keeps
``derivative.verified`` false unless the named evidence exists.

Two quantities, which must not be conflated
-------------------------------------------
1. the derivative of the **fixed-direction surrogate** -- what SI Algorithm S2
   actually computes;
2. the derivative of the **true objective** -- what an optimizer needs.

The paper states that the estimator "neglects gradients associated with changes
in the sampled secondary-ray directions". Measuring the gap between (1) and (2)
is the job here.

What the measurement found (CHE-28)
-----------------------------------
In the regime this repository implements -- a **fixed spectral grid** and a
fixed observation plane -- the gap is not detectable. Over 32 realizations at
N = 2048 the surrogate mean sits within 1 standard error of the true derivative
for both a linear and a quadratic objective, and stays inside 3 standard errors
across N from 512 to 32768.

The reason is structural rather than lucky: on a fixed spectral grid the
sampled direction is a property of the **bin**, not of the DOE parameter, so
there is no direction gradient to neglect. The paper's caveat describes a
formulation where the sampled wavevector can move continuously with the
parameter -- which is exactly the situation SI S7.3's Gumbel-Softmax relaxation
addresses.

What *is* measurable is the opposite: **detaching the sampling density is what
makes the estimator unbiased, not what biases it.** Letting the density track
the parameter -- the variant the paper's ``p.detach()`` avoids -- biases the
quadratic objective's gradient by 26 standard errors. Detaching drops precisely
the term that the omitted score-function term of the discrete sampling
distribution would have cancelled.

This does not promote any gradient claim. It is one parameter, one grid, one
wavelength, and two objectives; ``derivative.verified`` stays false.

Why no autodiff framework appears
---------------------------------
The surrogate is defined by *what is held fixed*: the sampled bin indices and
the sampling density. Holding those fixed and differencing the remainder is
exactly a finite difference evaluated at frozen indices -- so the surrogate
derivative can be computed directly, without a tape, and without introducing a
PyTorch-to-JAX handoff that AGENTS.md would require to stay forward-only. The
estimator is characterized as the mathematical object it is, rather than as one
framework's implementation of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

from couplers.contracts import (
    ComplexField,
    ContractCode,
    ContractError,
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
    "DifferentiabilityReport",
    "GradientProblem",
    "finite_difference_table",
    "surrogate_derivative",
    "true_derivative",
]

Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class GradientProblem:
    """A scalar objective on the field produced by a parameterized planar DOE.

    ``theta`` scales a fixed real phase mask, so the transmission is
    ``exp(i * theta * mask)`` -- one scalar knob, which keeps the directional
    derivative unambiguous.
    """

    incident: ComplexField
    phase_mask: np.ndarray
    objective: Objective
    objective_kind: Literal["linear", "quadratic"]
    #: Axial distance from the DOE to the observation plane. Must be nonzero for
    #: any intensity-based objective to depend on ``theta`` at all: at the DOE
    #: plane itself a pure-phase mask leaves ``|U|`` pointwise unchanged, so the
    #: true derivative of any intensity functional there is identically zero.
    #: Measured that way first, and the zero was the objective's fault rather
    #: than the estimator's.
    observation_distance_m: float = 0.0

    def __post_init__(self) -> None:
        mask = np.asarray(self.phase_mask, dtype=np.float64)
        object.__setattr__(self, "phase_mask", mask)
        if mask.shape != self.incident.shape:
            raise ContractError(
                ContractCode.SHAPE_MISMATCH,
                f"phase mask {mask.shape} must match the field {self.incident.shape}",
                declaration="phase_mask",
            )

    def transmitted(self, theta: float) -> ComplexField:
        return ComplexField(
            u=self.incident.u * np.exp(1j * theta * self.phase_mask),
            sample_pitch_m=self.incident.sample_pitch_m,
            wavelength_m=self.incident.wavelength_m,
            reference_plane=self.incident.reference_plane,
            frame=self.incident.frame,
            normalization=self.incident.normalization,
        )


def _reconstruct(bundle, shape, pitch, plane: ReferencePlane) -> np.ndarray:
    field, _ = ray_to_wave(
        bundle,
        grid_shape=shape,
        sample_pitch_m=pitch,
        plane=plane,
        projection=Projection.ASM_CONSISTENT,
    )
    return field.u


def _estimate(
    problem: GradientProblem,
    theta: float,
    indices: np.ndarray | None,
    density: np.ndarray | None,
    density_kind: SamplingDensity,
    *,
    detach_density: bool,
) -> float:
    """Evaluate the objective on the coupler's estimate of the field at ``theta``.

    ``indices`` and ``density`` are the frozen sampling state. ``detach_density``
    selects between the paper's estimator (density frozen with the indices) and
    the variant that lets the density move with ``theta``.
    """
    transmitted = problem.transmitted(theta)
    spectrum = decompose(transmitted)

    if indices is None:
        live_density = sampling_density(spectrum, density_kind)
        chosen = enumerate_indices(live_density)
        used_density = live_density
    else:
        chosen = indices
        used_density = density if detach_density else sampling_density(spectrum, density_kind)

    bundle = spectrum_to_rays(spectrum, chosen, used_density)

    plane = problem.incident.reference_plane
    distance = problem.observation_distance_m
    if distance:
        # Advance every ray geometrically to the observation plane. The path
        # travelled along the ray is distance / d_z, which is exactly the OPL it
        # accumulates. Directions are unchanged -- which is the point: they are
        # grid constants, not functions of theta.
        step_along_ray = distance / bundle.directions[:, 2]
        plane = ReferencePlane(name="observation plane", z_m=plane.z_m + distance)
        bundle = bundle._replace(
            positions_m=bundle.positions_m + bundle.directions * step_along_ray[:, None],
            optical_path_length_m=bundle.optical_path_length_m + step_along_ray,
            reference_plane=plane,
        )

    field = _reconstruct(
        bundle, problem.incident.shape, problem.incident.sample_pitch_m, plane
    )
    return problem.objective(field)


def true_derivative(problem: GradientProblem, theta: float, step: float) -> float:
    """Central difference of the objective on the *exact* field.

    Enumerating every propagating bin removes sampling error entirely (CHE-25's
    exactness limit), so this is the derivative of the true objective up to the
    finite-difference truncation error, which
    :func:`finite_difference_table` quantifies.
    """
    forward = _estimate(problem, theta + step, None, None, SamplingDensity.UNIFORM, detach_density=True)
    backward = _estimate(problem, theta - step, None, None, SamplingDensity.UNIFORM, detach_density=True)
    return (forward - backward) / (2.0 * step)


def surrogate_derivative(
    problem: GradientProblem,
    theta: float,
    step: float,
    *,
    count: int,
    rng: np.random.Generator,
    density_kind: SamplingDensity = SamplingDensity.MAGNITUDE,
    detach_density: bool = True,
) -> float:
    """Derivative of the fixed-direction surrogate (SI Algorithm S2).

    The sampled bin indices are drawn once at ``theta`` and held fixed across
    the difference -- which is precisely what "sampled wavevector and ray
    direction are treated as fixed during backpropagation" means. With
    ``detach_density=True`` the density is frozen with them, as in the paper's
    ``p.detach()``; with ``False`` it is allowed to track ``theta``, which is
    the variant the paper's ``.detach()`` avoids.
    """
    spectrum = decompose(problem.transmitted(theta))
    density = sampling_density(spectrum, density_kind)
    indices = draw_indices(density, count, rng)

    forward = _estimate(
        problem, theta + step, indices, density, density_kind, detach_density=detach_density
    )
    backward = _estimate(
        problem, theta - step, indices, density, density_kind, detach_density=detach_density
    )
    return (forward - backward) / (2.0 * step)


def finite_difference_table(
    problem: GradientProblem, theta: float, steps: tuple[float, ...]
) -> list[dict[str, float]]:
    """The true derivative at several step sizes.

    Required by the protocol so that a step-size artifact cannot be mistaken for
    a bias. A central difference carries an O(h^2) truncation error and an
    O(eps/h) round-off error, so the estimate is only trustworthy where the two
    are both small -- which the table makes visible rather than assumed.
    """
    rows = []
    for step in steps:
        value = true_derivative(problem, theta, step)
        rows.append({"step": float(step), "derivative": float(value)})
    for previous, current in zip(rows, rows[1:]):
        current["change_from_previous"] = abs(current["derivative"] - previous["derivative"])
    return rows


@dataclass(frozen=True)
class DifferentiabilityReport:
    """What was measured, in the form the benchmark schema expects."""

    claim: Literal[
        "not_verified",
        "characterized_biased",
        "characterized_unbiased_in_regime",
        "verified",
    ]
    estimator: str
    omitted_terms: list[str]
    objective_kind: str
    sampled_ray_count: int
    realizations: int
    true_derivative: float
    surrogate_mean: float
    surrogate_standard_error: float
    bias: float
    bias_in_standard_errors: float
    finite_difference_table: list[dict[str, float]]
    notes: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "estimator": self.estimator,
            "omitted_terms": self.omitted_terms,
            "objective_kind": self.objective_kind,
            "sampled_ray_count": self.sampled_ray_count,
            "realizations": self.realizations,
            "true_derivative": self.true_derivative,
            "surrogate_mean": self.surrogate_mean,
            "surrogate_standard_error": self.surrogate_standard_error,
            "bias": self.bias,
            "bias_in_standard_errors": self.bias_in_standard_errors,
            "finite_difference_comparison": {"table": self.finite_difference_table},
            "notes": self.notes,
        }


def characterize(
    problem: GradientProblem,
    *,
    theta: float = 1.0,
    step: float = 1e-4,
    count: int = 2048,
    realizations: int = 32,
    seed: int = 20260812,
    density_kind: SamplingDensity = SamplingDensity.MAGNITUDE,
    detach_density: bool = True,
) -> DifferentiabilityReport:
    """Measure the surrogate derivative's bias against the true derivative."""
    reference = true_derivative(problem, theta, step)
    table = finite_difference_table(problem, theta, (step * 10, step, step / 10))

    samples = [
        surrogate_derivative(
            problem,
            theta,
            step,
            count=count,
            rng=np.random.default_rng(seed + realization),
            density_kind=density_kind,
            detach_density=detach_density,
        )
        for realization in range(realizations)
    ]
    values = np.asarray(samples, dtype=np.float64)
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(values.size))
    bias = mean - reference
    bias_in_standard_errors = (
        abs(bias) / standard_error if standard_error > 0 else float("inf")
    )
    # The claim follows the measurement rather than being asserted in advance.
    # Neither outcome promotes anything: "unbiased in regime" is a scoped
    # observation, not a verified derivative.
    claim = (
        "characterized_unbiased_in_regime"
        if bias_in_standard_errors <= 3.0
        else "characterized_biased"
    )

    return DifferentiabilityReport(
        claim=claim,
        estimator=(
            "SI S7.2 fixed-direction estimator: sampled bin indices held fixed; "
            f"sampling density {'detached' if detach_density else 'live'}"
        ),
        omitted_terms=[
            "score-function term of the discrete sampling distribution",
            "gradients associated with changes in the sampled secondary-ray directions",
        ],
        objective_kind=problem.objective_kind,
        sampled_ray_count=count,
        realizations=realizations,
        true_derivative=reference,
        surrogate_mean=mean,
        surrogate_standard_error=standard_error,
        bias=bias,
        bias_in_standard_errors=bias_in_standard_errors,
        finite_difference_table=table,
        notes="",
    )
