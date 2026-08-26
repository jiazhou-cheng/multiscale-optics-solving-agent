# M2 bidirectional coupler protocol

CHE-21 freezes `M2-COUPLER-CPU-V1`, the execution and reporting contract for
`L2-COUPLER-01`, before any coupler physics is written — mirroring what CHE-12
did for the M1 baselines. The machine-readable form is
[`coupler_protocol.yaml`](coupler_protocol.yaml). It used to extend the M1
baseline protocol; CHE-106 (M1.1) retired that protocol once the B1 families
expressed its content executably, so the rules this contract once inherited are
now declared in full in `coupler_protocol.yaml` itself, unchanged.

The scientific source of truth is Cheng, Gao, Shao, Mao, Milster and Fan,
*A Differentiable Ray–Wave Framework for Hybrid Refractive–Diffractive System
Modeling and Optimization*, ACS Photonics 2026, DOI
[`10.1021/acsphotonics.6c00818`](https://doi.org/10.1021/acsphotonics.6c00818),
together with its Supporting Information. Equation and algorithm labels used
below are the paper's own.

## What M2 inherits

M1 verified two engines **independently** and deliberately proved that neither
branch could reach the other. M2 tests the transformation between their
representations. Three inherited facts constrain the design before any code is
written:

1. **There is no coupler implementation to audit.** `couplers/base.py` is a
   44-line `Protocol` with zero numerics, `C_RAY_TO_WAVE` is a registry claim
   nothing executes, and `C_WAVE_TO_RAY` does not exist. AGENTS.md's
   "characterize the legacy `ray_wave`/`ray_ewave` code first" rule has no
   local subject: the reference implementation lives in an external, unvendored
   repository. M2 therefore authors against the paper, not against code.
2. **Optiland's `opd_native` sign and reference plane are unverified.** M1
   recorded this explicitly and refused to use it as an oracle. A coupler must
   not inherit a phase sign from it.
3. **`asm_propagate` returns padded arrays.** Pitch and extent bookkeeping must
   be explicit at every coupler boundary; M1 recorded a 256² input growing to a
   1756² output.

## Coupler boundary contract

| Contract item | Frozen `L2-COUPLER-01` convention | Verification state |
|---|---|---|
| SI conversion | Coupler core is SI-only. mm→m and µm→m conversion happens in adapters; the core neither accepts nor emits native solver units | Enforced by contract validation |
| Axes/order | Field arrays are `(y, x)`; ray arrays are flat parallel arrays of equal length | Inherited from M1, re-asserted |
| Origin | Array index `n//2` is coordinate zero on each spatial axis | Inherited from M1 |
| Frame | Right-handed Cartesian, propagation along `+z`, plane normal `n̂` declared per reconstruction plane | Protocol decision; manufactured check required |
| Handedness | Right-handed; no silent reflection or permutation | Negative test required (axis transpose) |
| Wavelength | Monochromatic, metres. Each simulation is evaluated at a single wavelength, as the paper states | Frozen scope |
| Reference plane | Every `ComplexField` declares its axial coordinate; every `RayBundle` declares its reference plane. A coupler refuses inputs whose planes are unstated | Enforced by contract validation |
| OPL | Metres, along the ray, with the reference declared by the caller. `opd_native` is **not** an admissible OPL source | Frozen; follows M1 limitation |
| Phasor | Time convention `exp(-i ω t)`, spatial factor `exp(+i k z)`, ray wavelet phase `exp(+i k · OPL)` (SI Figure S1b). Recorded verbatim, never inferred | Inherited from M1; cross-direction check required |
| Amplitude | Complex amplitude. Optiland `intensity` is a ray weight and is **not** an admissible complex amplitude without a declared, tested conversion | Frozen; follows M1 |
| Projection factor | Ray→wave applies `⟨n̂, d̂⟩` per wavelet (main-text eq 2). Omitting it is a required negative test | Required evidence |
| Importance weight | Wave→ray applies `a = Ũ(k_u,k_v) / p(k_u,k_v)` (SI eq S4). Omitting it biases the estimator and is a required negative test | Required evidence |
| Evanescent modes | `k_u² + k_v² > k²` discarded (SI S2). Discarded power reported as a named loss term | Required evidence |
| Normalization | The `1/N` factor of SI eq S3/S5 is declared explicitly per call; discrete power is reported on both sides of every transformation | Protocol decision |
| Sampling | Record incident ray count, patch size `D`, secondary-ray count `N`, zero-pad factor `q`, and the sampling density identifier | Required in provenance |
| Validity — curvature | Tangent-plane approximation bounded by `ε_curv ≤ arcsin(D / 2R)` (SI eq S9) | Executable precondition required (CHE-27) |
| Validity — planar | Planar patches have no intrinsic upper size bound (SI S2); the full-field patch rule applies | Recorded |
| Dtype/device | `float64` / `complex128` on CPU for the reference core. Comparisons against Chromatix are performed at that engine's `complex64` and the downcast is recorded | Protocol decision |
| Determinism | Bitwise for a given `(seed, configuration)`. Not evidence of accuracy | Separate claim |
| Derivatives | `not_verified` for both directions. The estimator is knowingly biased (SI S7.2) | Frozen default; see below |

## Engine-agnostic core

The coupler core is the physics under test, so it must not import Optiland or
Chromatix. `coupler_protocol.yaml` declares this as a forbidden-import rule
checked both statically and at runtime, exactly as M1 checked branch
independence. The benchmark *driver* may import either engine — to build inputs
and to supply an independent wave oracle — but the core may not.

The reason is diagnostic, not stylistic: if the core imported an engine, a
coupler defect could be masked by, or misattributed to, engine behaviour, and
M1's independence evidence would no longer bound the search.

## Sampling is an input, not a side effect

The core takes **pre-drawn spectral indices as an argument**; drawing them is a
separate, seeded step. Three properties follow directly, rather than having to
be engineered later:

- bitwise determinism is trivial, because the core is a pure function;
- the same core runs under NumPy for the reference and under a
  differentiable array backend for the gradient study, with no second
  implementation to keep in sync;
- the "sampled directions are held fixed during backpropagation" behaviour of
  SI Algorithm S2 is structural — the directions are inputs, so nothing
  differentiates through the draw by accident.

## Stochastic evidence

M1 baselines were analytic and used no RNG; a single number could be compared
to a single oracle. `C_WAVE_TO_RAY` is a Monte Carlo estimator, so
*deterministic* and *accurate* become two claims requiring two kinds of
evidence. `coupler_protocol.yaml` requires all four of:

1. **Exactness limit.** Enumerate every propagating bin with the importance
   weight applied; the estimator must collapse to the deterministic reference
   at dtype round-off. This removes sampling as an excuse before any stochastic
   claim is made.
2. **Unbiasedness.** The ensemble mean over ≥ 32 independent seeds must agree
   with the deterministic reference within 3 standard errors — where the
   tolerance *is* the measured standard error, not a chosen constant.
3. **Convergence order.** Fit the RMS field error against `N` over ≥ 5 sweep
   points and gate on the fitted exponent (`−0.5 ± 0.1`), not on the error at a
   single `N`.
4. **Variance by sampling density.** Report estimator variance for uniform and
   spectral-magnitude sampling at matched `N` (paper Figure 4).

Three things are explicitly forbidden: reporting a single realization as the
result, selecting the best of several seeds, and tuning `N` or the realization
count after seeing the metric.

## Differentiability

The default claim is `not_verified` and the protocol keeps it there unless
named evidence exists. The estimator of SI S7.2 detaches the sampling density
and holds sampled ray directions fixed, so it is a deliberately biased
estimator of the true derivative — the paper says so directly. M2 measures two
separate quantities and must not conflate them:

- whether the estimator is a **correct derivative of the fixed-direction
  surrogate** (expected exact), and
- how far that surrogate derivative is from the **derivative of the true
  objective** (the bias).

A promotion to `verified` requires directional finite differences at multiple
step sizes, a reported bias magnitude with its measurement regime, and an
explicit list of the omitted terms — recorded in the bundle, for one named
direction and parameter at a time.

## Measurement, artifacts, and failure

Measurement rules are the M1 baseline rules unchanged, and since CHE-106 (M1.1)
retired that protocol they are declared directly in `coupler_protocol.yaml`
rather than inherited: `./run.sh` only, two untimed warmups, seven timed
repeats, median as the primary statistic with minimum and p95, recorded CPU
affinity and thread counts with no claim of isolation, and
`resource.getrusage` peak memory with a structured `unsupported` diagnostic when
unavailable. The M2 seed is `20260812`.

Beyond the base artifact set — `result.json`, `provenance.json`, `arrays.npz`,
`plot.png`, `tolerances.yaml`, `README.md` — a completed run emits `convergence.json` and
`ensemble_statistics.json`. The scientific fingerprint reuses the
`m1_bundle` volatile-key stripping, including the M1.8 lesson that per-case
wall-clock must never enter the hash: ensemble means over a fixed seed sequence
are physics and belong inside; per-realization runtime does not.

A failed run emits a structured `blocked` result. M2 adds one distinction M1
did not need: **`blocked`** means no well-defined number exists to compare,
**`failed`** means two well-defined numbers disagree, and a non-converged Monte
Carlo estimate is neither — it is reported with its standard error and its gate
is marked failed.
