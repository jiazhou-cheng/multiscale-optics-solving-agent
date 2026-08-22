"""Intermediate / "Differentiable Ray Tracing" -- https://www.optiland.org/tutorials/differentiable-ray-tracing

Repo-owned reproduction of the autodiff tutorial: select the torch backend, set
the device, enable grad mode, build a parameterized singlet whose two radii and
back spacing are `torch` tensors, and drive them with Adam to minimize RMS spot
radius.

**This reproduction closes an open item in this repository's own knowledge
pack.** `usage_notes.md` listed "root-causing why the torch-backend
gradient tolerance (1.11e-03) is looser than the JAX-based solvers' tolerances"
as not-yet-done. The cause is the line this tutorial is the first upstream
material to show: ``be.get_precision()`` returns **32** by default under the
torch backend. `probes/gradient_probe.py` passes a ``dtype=torch.float64``
parameter, but the lens it is traced through is built in float32, so the trace --
and therefore the finite-difference reference -- is float32. Measured here at
the probe's own operating point:

    precision   eps=1e-3     eps=1e-4     eps=1e-5
    float32     1.32e-04     1.11e-03     3.26e-02     <- diverges: FD cancellation
    float64     6.24e-05     6.24e-07     6.28e-09     <- converges as O(eps^2)

The 1.11e-03 was finite-difference noise, not autodiff error. Under
``be.set_precision('float64')`` the directional derivative agrees with a centered
difference to 6e-9 and the error falls quadratically with step size, which is the
behaviour a correct reverse-mode gradient must show. Validation here is exactly
that convergence test plus the tutorial's own optimization monotonicity.

The backend is process-global and non-thread-safe, so ``run()`` restores the
numpy backend and float32 precision in a ``finally`` block.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from _optiland_harness import TutorialMeta, TutorialResult, standalone_main

TUTORIAL = TutorialMeta(
    slug="t10_differentiable_ray_tracing",
    title="Differentiable Ray Tracing",
    level="intermediate",
    url="https://www.optiland.org/tutorials/differentiable-ray-tracing",
    demonstrates=(
        "optiland.backend.set_backend('torch') + set_device + set_precision + "
        "get_precision + grad_mode.enable, torch tensors as surface radii and "
        "thicknesses, reverse-mode gradients through Optic.trace, and an Adam "
        "loop. Establishes that the torch backend defaults to float32."
    ),
    slow=True,
    needs_torch=True,
)

WAVELENGTH_UM = 0.55
EPD_MM = 25.0
# The probe operating point from benchmarks/probes/optiland/gradient_probe.py,
# reused so the two pieces of evidence are directly comparable.
PROBE_R0 = 1.6911
PROBE_NUM_RAYS = 64


def _singlet(r1, r2, t2, material_name="BK7"):
    """The tutorial's SingletConfigurable, as a factory."""
    import optiland.backend as be
    from optiland.materials import Material
    from optiland.optic import Optic

    lens = Optic()
    material = Material(material_name)
    lens.surfaces.add(index=0, radius=be.inf, thickness=be.inf)
    lens.surfaces.add(index=1, thickness=7.0, radius=r1, is_stop=True, material=material)
    lens.surfaces.add(index=2, radius=r2, thickness=t2)
    lens.surfaces.add(index=3)
    lens.set_aperture(aperture_type="EPD", value=EPD_MM)
    lens.fields.set_type(field_type="angle")
    lens.fields.add(y=0.0)
    lens.wavelengths.add(value=WAVELENGTH_UM, is_primary=True)
    return lens


def _probe_objective(radius):
    """`gradient_probe.py`'s objective: mean(x^2 + y^2) on ReverseTelephoto."""
    from optiland.samples.objectives import ReverseTelephoto

    lens = ReverseTelephoto()
    lens.surfaces.surfaces[1].geometry.radius = radius
    rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=PROBE_NUM_RAYS)
    return (rays.x**2 + rays.y**2).mean()


def run() -> TutorialResult:
    import torch

    import optiland.backend as be

    result = TutorialResult()
    original_backend = be.get_backend()
    try:
        be.set_backend("torch")
        be.set_device("cpu")
        default_precision = be.get_precision()
        result.record(
            backend=be.get_backend(),
            device=be.get_device(),
            default_precision=default_precision,
            supports_gpu=bool(be.supports_gpu),
            supports_gradients=bool(be.supports_gradients),
        )
        result.check_true(
            "torch_backend_defaults_to_float32",
            "invariant",
            int(default_precision) == 32,
            f"be.get_precision() == {default_precision} immediately after "
            "set_backend('torch'): single precision is the DEFAULT, not float64",
        )
        be.grad_mode.enable()
        result.check_true(
            "grad_mode_can_be_enabled",
            "invariant",
            bool(be.grad_mode.requires_grad),
            f"be.grad_mode.requires_grad = {be.grad_mode.requires_grad}",
        )

        # -- 1. precision is what governs the gradient/FD agreement ------------
        steps = (1e-3, 1e-4, 1e-5)
        convergence: dict[str, dict[str, float]] = {}
        traced_dtypes: dict[str, str] = {}
        for precision in ("float32", "float64"):
            be.set_precision(precision)
            dtype = torch.float32 if precision == "float32" else torch.float64
            r0 = torch.tensor(PROBE_R0, dtype=dtype, requires_grad=True)
            value = _probe_objective(r0)
            value.backward()
            grad_ad = float(r0.grad.item())
            traced_dtypes[precision] = str(value.dtype)
            entry = {"grad_autodiff": grad_ad, "objective": float(value.item())}
            for eps in steps:
                with torch.no_grad():
                    plus = float(_probe_objective(torch.tensor(PROBE_R0 + eps, dtype=dtype)).item())
                    minus = float(_probe_objective(torch.tensor(PROBE_R0 - eps, dtype=dtype)).item())
                grad_fd = (plus - minus) / (2.0 * eps)
                entry[f"grad_fd_eps_{eps:g}"] = grad_fd
                entry[f"rel_error_eps_{eps:g}"] = abs(grad_ad - grad_fd) / abs(grad_fd)
            convergence[precision] = entry
        result.record(gradient_convergence=convergence, traced_objective_dtype=traced_dtypes)

        f32 = convergence["float32"]
        f64 = convergence["float64"]
        result.check_true(
            "float32_finite_differences_do_not_converge",
            "analytic",
            f32["rel_error_eps_1e-05"] > f32["rel_error_eps_0.0001"] > f32["rel_error_eps_0.001"],
            "relative error GROWS as the step shrinks: "
            f"{f32['rel_error_eps_0.001']:.2e} (1e-3) -> {f32['rel_error_eps_0.0001']:.2e} (1e-4) "
            f"-> {f32['rel_error_eps_1e-05']:.2e} (1e-5). Classic single-precision "
            "cancellation, not a bad gradient.",
        )
        result.check_close(
            "float32_reproduces_the_recorded_1p11e_minus_3_probe_number",
            "reference",
            f32["rel_error_eps_0.0001"],
            1.11e-3,
            rel=0.05,
        )
        ratios = [
            f64["rel_error_eps_0.001"] / f64["rel_error_eps_0.0001"],
            f64["rel_error_eps_0.0001"] / f64["rel_error_eps_1e-05"],
        ]
        result.record(float64_error_reduction_per_decade=ratios)
        result.check_true(
            "float64_finite_differences_converge_quadratically",
            "analytic",
            all(ratio > 30.0 for ratio in ratios),
            "relative error falls by "
            f"{ratios[0]:.0f}x then {ratios[1]:.0f}x per decade of step size "
            f"({f64['rel_error_eps_0.001']:.2e} -> {f64['rel_error_eps_0.0001']:.2e} -> "
            f"{f64['rel_error_eps_1e-05']:.2e}), i.e. the O(eps^2) truncation of a centered "
            "difference against an exact derivative",
        )
        result.check_true(
            "float64_gradient_is_four_orders_more_accurate_than_float32",
            "analytic",
            f64["rel_error_eps_1e-05"] < 1e-7
            and f32["rel_error_eps_1e-05"] / f64["rel_error_eps_1e-05"] > 1e4,
            f"best relative error {f64['rel_error_eps_1e-05']:.2e} (float64) vs "
            f"{f32['rel_error_eps_1e-05']:.2e} (float32)",
        )
        result.check_true(
            "float64_precision_actually_changes_the_traced_dtype",
            "invariant",
            traced_dtypes == {"float32": "torch.float32", "float64": "torch.float64"},
            f"traced objective dtype per precision: {traced_dtypes}. A float64 "
            "parameter tensor alone is NOT enough -- the lens is built with "
            "be.array, which follows the global precision.",
        )

        # -- 2. the tutorial's Adam optimization -------------------------------
        be.set_precision("float64")
        params = {
            "r1": torch.tensor(70.0, dtype=torch.float64, requires_grad=True),
            "r2": torch.tensor(-70.0, dtype=torch.float64, requires_grad=True),
            "t2": torch.tensor(70.0, dtype=torch.float64, requires_grad=True),
        }
        optimizer = torch.optim.Adam(list(params.values()), lr=0.2)

        def loss_fn():
            lens = _singlet(params["r1"], params["r2"], params["t2"])
            rays = lens.trace(Hx=0.0, Hy=0.0, wavelength=WAVELENGTH_UM, num_rays=32)
            return torch.sqrt((rays.x**2 + rays.y**2).mean())

        history = []
        for _ in range(100):
            optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            optimizer.step()
            history.append(float(loss.item()))
        result.record(
            adam_loss_history_every_25=[history[i] for i in (0, 24, 49, 74, 99)],
            adam_initial_rms_mm=history[0],
            adam_final_rms_mm=history[-1],
            adam_best_rms_mm=min(history),
            optimized_r1_mm=float(params["r1"].item()),
            optimized_r2_mm=float(params["r2"].item()),
            optimized_t2_mm=float(params["t2"].item()),
        )
        result.check_true(
            "adam_reduces_the_rms_spot_radius",
            "invariant",
            history[-1] < history[0],
            f"RMS spot radius {history[0]:.6f} -> {history[-1]:.6f} mm over 100 Adam steps "
            f"(best {min(history):.6f} mm)",
        )
        result.check_true(
            "adam_improvement_is_substantial_not_marginal",
            "invariant",
            history[-1] < 0.5 * history[0],
            f"final/initial = {history[-1] / history[0]:.4f}",
        )
        result.check_finite("adam_loss_history_finite", history)
        result.check_true(
            "gradients_reach_every_declared_variable",
            "invariant",
            all(p.grad is not None and np.isfinite(float(p.grad.item())) for p in params.values()),
            "r1, r2 and t2 all received a finite gradient: autodiff flows through "
            "both surface curvature and axial spacing",
        )
        result.check_true(
            "optimized_singlet_is_still_physically_sensible",
            "analytic",
            float(params["r1"].item()) > 0.0
            and float(params["r2"].item()) < 0.0
            and float(params["t2"].item()) > 0.0,
            f"r1={float(params['r1'].item()):.4f} > 0, r2={float(params['r2'].item()):.4f} < 0 "
            f"(biconvex retained), back spacing {float(params['t2'].item()):.4f} > 0",
        )
    finally:
        be.set_precision("float32")
        be.set_backend(original_backend)
    result.note(
        "optiland.backend state is process-global and not thread-safe. This module "
        "restores the entering backend and float32 precision in a finally block so a "
        "pytest session that imports it does not leak the torch backend into other "
        "tests."
    )
    return result


if __name__ == "__main__":
    raise SystemExit(standalone_main(TUTORIAL, run))
