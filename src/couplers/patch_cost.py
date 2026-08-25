"""A *measured* cost model for the patch emitter, and what it refuses (CHE-119).

``PatchWftCoupler.estimate()`` reported no wall time at all before M5.2. It was
right not to invent one -- but the consequence was recorded in M0.4's
``estimate_accuracy.json`` and it is the same consequence the ray adapter had: a
planner cannot order work by a cost estimate that has no cost in it.

Two terms, because the stage has two
------------------------------------
``fixed + per_patch * patches + per_secondary_ray * patches * S``.

That shape is read off the decomposition rather than chosen to fit. The padded
transform, the boolean gathers, the density and the cumulative sum are all
``O(pad^2)`` per patch and do not care how many modes are drawn from the result;
the search and the ray assembly are ``O(S)`` per patch. So a per-patch term and a
per-secondary-ray term, fitted together on a two-axis grid because fitting either
alone would attribute the whole stage to whichever axis moved.

Three things it will not do
---------------------------
* **Predict off its environment fingerprint.** The constants are one container
  image on one CPU at one thread count. ``core.performance.compare`` refuses to
  divide two records measured in different environments; a cost model that
  extrapolated across the same boundary would be that refusal with a friendlier
  interface.
* **Predict at a different pad.** ``pad_px`` is *held* at 301, not fitted. The
  per-patch term is ``O(pad^2 log pad)`` and every calibration point shares one
  pad, so the constant has absorbed it. A pad sweep would remove this limit and
  has not been run.
* **Predict below 16 patches.** The thread pool switches on at
  ``POOL_MIN_PATCHES`` and its per-call cost has not amortized just above that,
  so the region is two regimes rather than one. Measured error there is 32%,
  against 7% inside the domain.

None of this covers the reconstruction that consumes the bundle, which is
``O(rays x pixels)`` and is usually the larger number. ``estimate()`` keeps
saying so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.execution import CostEstimate

__all__ = [
    "CPU_COMPLEX128_EMITTER_COST",
    "PatchEmitterCostModel",
    "estimate_emitter_seconds",
]


@dataclass(frozen=True)
class PatchEmitterCostModel:
    """A fitted emitter cost, with everything it is only valid under attached."""

    fixed_s: float
    per_patch_s: float
    per_secondary_ray_s: float
    #: Inclusive patch-count bounds of the fit. Outside them it was measured wrong.
    domain_patches: tuple[int, int]
    #: Inclusive secondary-count bounds of the fit.
    domain_secondary: tuple[int, int]
    #: The one padded transform size the constants describe.
    pad_px: int
    max_relative_error_in_domain: float
    max_relative_error_excluded: float
    threads: int
    environment_sha256: str
    cpu_model: str
    record: str
    issue: str

    def predict_s(self, *, patches: int, secondary: int) -> float:
        return (
            self.fixed_s
            + self.per_patch_s * patches
            + self.per_secondary_ray_s * patches * secondary
        )

    def in_domain(self, *, patches: int, secondary: int, pad_px: int) -> bool:
        return (
            self.domain_patches[0] <= patches <= self.domain_patches[1]
            and self.domain_secondary[0] <= secondary <= self.domain_secondary[1]
            and pad_px == self.pad_px
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "form": (
                "seconds = fixed_s + per_patch_s * patches "
                "+ per_secondary_ray_s * patches * secondary_count"
            ),
            "fixed_s": self.fixed_s,
            "per_patch_s": self.per_patch_s,
            "per_secondary_ray_s": self.per_secondary_ray_s,
            "domain_patches": list(self.domain_patches),
            "domain_secondary": list(self.domain_secondary),
            "pad_px": self.pad_px,
            "max_relative_error_in_domain": self.max_relative_error_in_domain,
            "threads": self.threads,
            "environment_sha256": self.environment_sha256,
            "record": self.record,
            "issue": self.issue,
        }


#: The one calibration this repository has, from
#: ``benchmarks/perf/records/patch_emitter_cost_model.json``.
#:
#: Re-derived by re-running that section, never edited by hand:
#: ``tests/test_patch_wft.py`` asserts these numbers against the committed
#: record, so a stale constant fails a test rather than quietly mispricing a plan.
CPU_COMPLEX128_EMITTER_COST = PatchEmitterCostModel(
    fixed_s=0.023933936044637595,
    per_patch_s=0.000881039974083035,
    per_secondary_ray_s=1.890430222349601e-07,
    domain_patches=(16, 50),
    domain_secondary=(1_000, 20_000),
    pad_px=301,
    max_relative_error_in_domain=0.0728,
    max_relative_error_excluded=0.3157,
    threads=8,
    environment_sha256="fb66d7be8b4ad129b5e05807f3af88645c4e8aff3559a03e6d76914ed0d29f11",
    cpu_model="Intel(R) Xeon(R) Gold 6242R CPU @ 3.10GHz",
    record="benchmarks/perf/records/patch_emitter_cost_model.json",
    issue="CHE-119 (M5.2)",
)


def estimate_emitter_seconds(
    *,
    patches: int,
    secondary: int,
    pad_px: int,
    model: PatchEmitterCostModel = CPU_COMPLEX128_EMITTER_COST,
    environment_sha256: str | None = None,
) -> CostEstimate:
    """Predicted emitter wall time, or a refusal naming the check that failed.

    ``environment_sha256`` defaults to reading the *current* environment, so a
    caller cannot obtain a prediction for a machine it is not on by omitting it.

    ``peak_memory_bytes`` is left to the caller: this model was fitted on wall
    clock only, and the emitter's memory is dominated by the emitted rays rather
    than by anything measured here.
    """
    if patches <= 0 or secondary <= 0 or pad_px <= 0:
        raise ValueError(
            f"patches, secondary and pad_px must all be > 0; got {patches}, "
            f"{secondary}, {pad_px}"
        )

    if environment_sha256 is None:
        from core.performance import environment_fingerprint

        environment_sha256 = environment_fingerprint().sha256

    provenance = [
        f"Cost model: {model.record} ({model.issue}). "
        f"{model.fixed_s * 1e3:.1f} ms + {model.per_patch_s * 1e3:.3f} ms/patch + "
        f"{model.per_secondary_ray_s * 1e9:.1f} ns/secondary-ray, at "
        f"pad {model.pad_px}, {model.threads} host threads, complex128 on "
        f"{model.cpu_model}, to within "
        f"{model.max_relative_error_in_domain * 100:.1f}%.",
        "Covers the emitter only. The reconstruction that consumes this bundle is "
        "O(rays x pixels) and is usually the larger number -- see the notes on the "
        "estimate itself.",
    ]

    if environment_sha256 != model.environment_sha256:
        return CostEstimate(
            solver_calls=1,
            confidence="unknown",
            notes=[
                *provenance,
                "NO PREDICTION: this environment's fingerprint "
                f"({environment_sha256[:12]}...) is not the calibration's "
                f"({model.environment_sha256[:12]}...). CPU model, container image "
                "and thread counts change speed without changing the answer, which "
                "is why `core.performance.compare` refuses a ratio across them. "
                f"Re-run `{model.record}`'s sweep here to calibrate.",
                "For reference only, not a prediction on this host: the model would "
                f"give {model.predict_s(patches=patches, secondary=secondary):.4f} s.",
            ],
        )

    if not model.in_domain(patches=patches, secondary=secondary, pad_px=pad_px):
        reasons = []
        if pad_px != model.pad_px:
            reasons.append(
                f"pad_px {pad_px} is not the calibrated {model.pad_px}, and the "
                "per-patch term is O(pad^2 log pad) with the pad absorbed into the "
                "constant -- so this is not a small extrapolation, it is the wrong "
                "constant"
            )
        if not model.domain_patches[0] <= patches <= model.domain_patches[1]:
            reasons.append(
                f"{patches} patches is outside the fitted {model.domain_patches}; "
                f"below the lower bound the thread-pool transition makes the affine "
                f"form wrong by up to "
                f"{model.max_relative_error_excluded * 100:.0f}%"
            )
        if not model.domain_secondary[0] <= secondary <= model.domain_secondary[1]:
            reasons.append(
                f"{secondary} secondary rays a patch is outside the fitted "
                f"{model.domain_secondary}"
            )
        return CostEstimate(
            solver_calls=1,
            confidence="unknown",
            notes=[
                *provenance,
                "NO PREDICTION: " + "; ".join(reasons) + ".",
                "Model value at this size, for orientation only: "
                f"{model.predict_s(patches=patches, secondary=secondary):.4f} s.",
            ],
        )

    return CostEstimate(
        wall_time_s=model.predict_s(patches=patches, secondary=secondary),
        solver_calls=1,
        # Not "high": one pad, one prescription's DOE, one machine, and a 6%
        # in-domain residual on eight points.
        confidence="medium",
        notes=[
            *provenance,
            f"{patches} patches x {secondary} secondary rays at pad {pad_px}, inside "
            "the fitted domain and on the environment the fit was taken on.",
        ],
    )
