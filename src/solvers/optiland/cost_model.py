"""A *measured* cost model for the Optiland trace, and the rule for using it (CHE-118).

Before M5.1 the only cost information this repository could give a planner was
the registry's scaling string, ``O(number_of_surfaces * number_of_rays)``, and
``OptilandAdapter.estimate()`` returning ``wall_time_s=None``. M0.4 scored that
honestly and the verdict is committed in ``estimate_accuracy.json``: *"NO
PREDICTION ... it means a planner cannot use this estimator to order work by
cost."* This module is the calibration that replaces the string.

Affine, not a power law
-----------------------
The fitted form is ``fixed_per_call_s + seconds_per_ray_per_surface * rays *
surfaces``. That shape is a measurement, not a modelling preference: after
CHE-118 removed the O(rays) host work from the trace, what is left is a fixed
per-call cost plus per-ray device work, and the two terms cross over inside the
range demo3 uses. A power-law fit over the same points returns an exponent of
0.19 at r^2 0.59 -- not a bad fit to a power law so much as evidence that this is
not one -- and an executor planning against that exponent would misprice every
chunk size. Both fits are in the committed record; the affine one holds to 1.9%
inside its domain.

Every number here is environment-bound, and the model says so
------------------------------------------------------------
These constants were measured on one GPU, one precision, one container image.
``core.performance`` refuses to compare records across environment fingerprints
for exactly this reason, and a cost model that would happily extrapolate across
what ``compare`` refuses to divide would be the same mistake wearing a different
name. So :func:`estimate_trace_seconds` checks the fingerprint and returns *no
prediction* rather than a portable-looking number when it does not match. It
does the same above the fitted domain, where the residual is a measured -44%.

Two things this model is not
----------------------------
* It is not a memory model. Nothing here predicts ``peak_memory_bytes``; that
  was not measured per chunk size and is left ``None`` rather than guessed.
* It is not calibrated on Optiland's own ray generation. It covers the
  propagation of a *caller-supplied* ray population, which is what
  ``trace_ray_batch`` does and what a graph node's cost is. A standalone pupil
  trace also builds its rays inside the solver, and that part is uncalibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.execution import CostEstimate

__all__ = [
    "CUDA_FP32_TRACE_COST",
    "TraceCostModel",
    "estimate_trace_seconds",
    "hexapolar_ray_count",
    "traced_surface_count",
]


@dataclass(frozen=True)
class TraceCostModel:
    """A fitted trace cost, with the environment it is only valid in attached."""

    fixed_per_call_s: float
    seconds_per_ray_per_surface: float
    #: Inclusive ray-count bounds the fit was taken over. Outside them the model
    #: is not evaluated, because outside them it was measured to be wrong.
    domain_rays: tuple[float, float]
    fitted_on_surfaces: int
    max_relative_error_in_domain: float
    #: ``core.performance.environment_fingerprint().sha256`` at calibration time.
    environment_sha256: str
    gpu_name: str
    device: str
    precision: str
    arm: str
    record: str
    issue: str

    def predict_s(self, *, rays: float, surfaces: int) -> float:
        return self.fixed_per_call_s + self.seconds_per_ray_per_surface * rays * surfaces

    def in_domain(self, rays: float) -> bool:
        return self.domain_rays[0] <= rays <= self.domain_rays[1]

    def as_dict(self) -> dict[str, Any]:
        return {
            "form": (
                "seconds = fixed_per_call_s + seconds_per_ray_per_surface "
                "* rays * surfaces"
            ),
            "fixed_per_call_s": self.fixed_per_call_s,
            "seconds_per_ray_per_surface": self.seconds_per_ray_per_surface,
            "domain_rays": list(self.domain_rays),
            "fitted_on_surfaces": self.fitted_on_surfaces,
            "max_relative_error_in_domain": self.max_relative_error_in_domain,
            "environment_sha256": self.environment_sha256,
            "gpu_name": self.gpu_name,
            "device": self.device,
            "precision": self.precision,
            "arm": self.arm,
            "record": self.record,
            "issue": self.issue,
        }


#: The one calibration this repository has, from
#: ``benchmarks/perf/records/optiland_trace_chunk_sweep.json``.
#:
#: Re-derived by re-running that sweep, never edited by hand:
#: ``tests/test_optiland_cost_model.py`` asserts these numbers against the
#: committed record, so a stale constant fails a test rather than quietly
#: mispricing a plan.
CUDA_FP32_TRACE_COST = TraceCostModel(
    fixed_per_call_s=0.013201215111627416,
    seconds_per_ray_per_surface=2.7152658731030526e-09,
    domain_rays=(1000.0, 1000000.0),
    fitted_on_surfaces=4,
    max_relative_error_in_domain=0.0186,
    environment_sha256="fb66d7be8b4ad129b5e05807f3af88645c4e8aff3559a03e6d76914ed0d29f11",
    gpu_name="NVIDIA RTX A6000",
    device="cuda",
    precision="fp32",
    arm="trace_ray_batch as shipped after CHE-118",
    record="benchmarks/perf/records/optiland_trace_chunk_sweep.json",
    issue="CHE-118 (M5.1)",
)


def hexapolar_ray_count(rings: int) -> int:
    """Rays Optiland's hexapolar pupil sampler produces for ``rings`` rings.

    ``1 + 3 N (N + 1)``: one central ray plus six per ring scaled by the ring
    index. Verified against every ring count in the committed
    ``scaling_ray_axis.json`` -- 16, 24, 32, 40, 48 and 64 rings give 817, 1801,
    3169, 4921, 7057 and 12481 rays, exactly -- rather than taken from the
    formula's plausibility. It is what lets ``estimate()`` know a ray count that
    the old note said it could not: *"the traced ray count returned by
    Optic.trace() is ... not known until after the solver call"*.

    Per field and per wavelength. The standalone path traces one of each.
    """
    if rings < 0:
        raise ValueError(f"rings must be >= 0, got {rings}")
    return 1 + 3 * rings * (rings + 1)


def traced_surface_count(prescription_surfaces: int) -> int:
    """Surfaces a trace actually walks, from the count in the prescription.

    The prescription's surfaces plus the image plane. A built ``Optic`` also
    carries an object surface, which sits at ``z = -inf`` for an object at
    infinity and is never traced -- that is the ``skip=1`` in
    ``trace_ray_batch``. Checked against the built demo3 system in the tests, so
    this arithmetic is not the place the convention drifts.
    """
    return prescription_surfaces + 1


def estimate_trace_seconds(
    *,
    rays: float,
    surfaces: int,
    model: TraceCostModel = CUDA_FP32_TRACE_COST,
    environment_sha256: str | None = None,
) -> CostEstimate:
    """Predicted trace wall time, or a refusal that says which check failed.

    ``environment_sha256`` defaults to reading the *current* environment, so a
    caller cannot get a prediction for a machine it is not on by omitting it.
    Pass it explicitly only to ask what this model would say somewhere else.

    ``peak_memory_bytes`` is always ``None``: memory was not measured across
    chunk sizes, and a guess there is worse than a gap, because a planner would
    size a batch with it.
    """
    if rays <= 0:
        raise ValueError(f"rays must be > 0, got {rays}")
    if surfaces <= 0:
        raise ValueError(f"surfaces must be > 0, got {surfaces}")

    if environment_sha256 is None:
        from core.performance import environment_fingerprint

        environment_sha256 = environment_fingerprint().sha256

    provenance = [
        f"Cost model: {model.record} ({model.issue}). Form: "
        f"{model.fixed_per_call_s:.6g} s per call + "
        f"{model.seconds_per_ray_per_surface:.6g} s per ray per surface, fitted on "
        f"{model.gpu_name} / {model.device} / {model.precision}, "
        f"{model.domain_rays[0]:.0f}-{model.domain_rays[1]:.0f} rays per call, to "
        f"within {model.max_relative_error_in_domain * 100:.1f}%.",
        "Covers propagation of a caller-supplied ray population only -- not "
        "Optiland's own ray generation, not the wave-side emitter, and not the "
        "reconstruction.",
    ]

    if environment_sha256 != model.environment_sha256:
        return CostEstimate(
            wall_time_s=None,
            peak_memory_bytes=None,
            solver_calls=1,
            confidence="unknown",
            notes=[
                *provenance,
                "NO PREDICTION: this environment's fingerprint "
                f"({environment_sha256[:12]}...) is not the one the model was "
                f"calibrated on ({model.environment_sha256[:12]}...). GPU, driver, "
                "container image and thread counts change speed without changing "
                "the answer, which is why `core.performance.compare` refuses a "
                "ratio across them; extrapolating a cost model across them would "
                "be the same error with a friendlier interface. Re-run "
                f"`{model.record}`'s sweep here to calibrate.",
                f"For reference only, not a prediction on this host: the model "
                f"would give {model.predict_s(rays=rays, surfaces=surfaces):.4f} s.",
            ],
        )

    if not model.in_domain(rays):
        below = rays < model.domain_rays[0]
        return CostEstimate(
            wall_time_s=None,
            peak_memory_bytes=None,
            solver_calls=1,
            confidence="unknown",
            notes=[
                *provenance,
                f"NO PREDICTION: {rays:.0f} rays per call is "
                f"{'below' if below else 'above'} the fitted domain "
                f"{model.domain_rays}. "
                + (
                    "Below it the fixed per-call term dominates and the model tends "
                    "to that constant, so the value below is a reasonable floor but "
                    "was not measured there."
                    if below
                    else "Above it cost per ray turns back up: the measured residual "
                    "at 4 M rays per call is -44%, so the model underpredicts and the "
                    "value below is a lower bound, not an estimate. Split the work "
                    "into calls inside the domain and the prediction applies again."
                ),
                f"Model value at this size: "
                f"{model.predict_s(rays=rays, surfaces=surfaces):.4f} s.",
            ],
        )

    return CostEstimate(
        wall_time_s=model.predict_s(rays=rays, surfaces=surfaces),
        peak_memory_bytes=None,
        solver_calls=1,
        # Not "high". The affine form holds to 1.2% across the domain, but it is
        # one sweep on one prescription on one card, and the per-surface term is
        # supported by four surfaces costing within a few percent of each other
        # rather than by a surface-count sweep.
        confidence="medium",
        notes=[
            *provenance,
            f"{rays:.0f} rays x {surfaces} surfaces, inside the fitted domain and on "
            "the environment the fit was taken on.",
            "peak_memory_bytes is None: memory was not measured across chunk sizes, "
            "and a planner would size a batch with a guess.",
        ],
    )
