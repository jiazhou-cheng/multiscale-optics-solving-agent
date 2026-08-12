# M1 baseline exit report — independent ray and wave branches

CHE-19 (M1.8). This report integrates evidence only. It does not combine the
two physical models, and neither branch executes a ray–wave coupler.

Generated from the bundles in `outputs/M1/`, produced by the commands in
[Exact commands](#exact-commands) on the environment in
[Environment](#environment).

**Verdict: both branches pass their scientific gates. M1 is recommended for
exit with five recorded limitations (L1–L5)** — one blocked wave case, one
protocol documentation gap, one uncommitted-tree caveat, and two
performance-interpretation caveats. All are described in
[Risks and known limitations](#risks-and-known-limitations); none was resolved
by loosening a tolerance.

This review also **found and fixed one real defect**: the wave branch did not
reproduce its scientific fingerprint. See
[Defect found and fixed during this review](#defect-found-and-fixed-during-this-review).

---

## Exact commands

```bash
./run.sh python benchmarks/level1/L1-RAY-01/run_all.py  --output-dir outputs/M1/ray
./run.sh python benchmarks/level1/L1-WAVE-01/run_all.py --output-dir outputs/M1/wave
./run.sh python benchmarks/verify_m1_independence.py
./run.sh pytest -q tests/test_optiland_adapter.py tests/test_chromatix_adapter.py tests/benchmarks
./run.sh python scripts/validate_package.py
```

Each branch command runs in its own process tree and internally executes three
stages: the standalone baseline, the analytic accuracy gate, and the scaling
section. `./run.sh` and the pinned `agent_solver` container are the only
execution path used.

Outcome of this review's run:

| Command | Result |
|---|---|
| `L1-RAY-01/run_all.py` | `status: complete`, `pass: true` |
| `L1-WAVE-01/run_all.py` | `status: complete`, `pass: true` (Case 3 blocked, non-gating) |
| `verify_m1_independence.py` | `status: passed`, 5/5 checks |
| required pytest selection | 73 passed |
| `scripts/validate_package.py` | 8 models, 10 couplers, all YAML and example graphs valid |
| full repository suite `pytest -q` | **163 passed, 2 xfailed, 1 xpassed** |

The two xfails (fdtdx gradient locks) and one xpass (sax circuit gradient) are
pre-existing, documented, and unrelated to M1.

## Environment

| Item | Value |
|---|---|
| Git commit | `ae7fac01b9c04e0b63aab8bf2277e6636804006a` |
| Dirty worktree | **`true`** — see limitation L3 |
| Python | 3.12.13 |
| Platform | `Linux-6.8.0-84-generic-x86_64-with-glibc2.41`, `x86_64` |
| CPU | Intel(R) Xeon(R) Gold 6242R @ 3.10 GHz |
| Logical CPUs / observed affinity | 80 / 80 (**not pinned**; affinity is recorded as an observation, never claimed as isolation) |
| Thread counts | `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` all `unset` |
| Seed | `20260811` (recorded; both branches are analytic and use no RNG) |

Environment fingerprints, which gate every performance comparison:

- ray `5e30b1cae496994db1f137200fe5c0c8849a222e37a4418ceeb71ead3906eac1`
- wave `746f1ad381b63c2dfa1275ec51ec56d3e17098bd6bde98139b72df2c45e34622`

The two branches carry different fingerprints because they pin different
engine stacks. **Performance numbers are only ever compared within a matching
fingerprint**; the bundle builder refuses to publish a regression envelope
whose fingerprint differs from the run that produced it.

---

# Ray branch — L1-RAY-01 (Optiland)

**Protocol `M1-BASELINE-CPU-V2`** · engines `optiland 0.6.0`, `numpy 2.2.6` ·
dtype `float64`, device `cpu`
Scientific fingerprint `43dab1eedf5ca8fcd6a2674bcc6fb58020933aec8ad8618ad49583826cfc7236`

## Accuracy — PASS

Accuracy is evaluated and must pass **before** any performance number is
accepted; the bundle builder raises before reaching the performance section if
any gate fails.

| Case | Metric | Observed | Tolerance |
|---|---|---|---|
| Free space | max position error | `0.0 m` | `1e-10` scaled |
| Free space | max direction-norm error | `0.0` | `1e-12` |
| Paraxial thin lens | focal-intercept relative error | `4.44e-18` | `1e-6` |
| Paraxial thin lens | max direction-norm error | `2.22e-16` | `1e-12` |
| Catalog lens (Edmund #45-362) | EFL relative error | `3.02e-06` | `1e-2` |
| Catalog lens | BFL relative error | `1.39e-05` | `1e-2` |
| Catalog lens | axial OPL error | `7.11e-18 m` | `1e-10` scaled |
| Catalog lens | max direction-norm error | `5.55e-16` | `1e-12` |

Oracles: closed-form free-space geometry, ABCD thin-lens equations, SCHOTT
N-BK7 dispersion with thick-lens EFL/BFL, and the Edmund Optics catalog
reference. Tolerances: `benchmarks/level1/L1-RAY-01/tolerances.yaml`.

**Convention negative test: detected.** A deliberate mm-for-metre scale error
is rejected by the same evaluator gates.

## Performance — recorded, pending human review

Compile/import and setup time are reported **separately** from steady-state
solver time. The throughput denominator is the *actual traced ray count*,
never the requested sampling density.

| Sampling | Traced rays | Import | Setup | Trace median | Throughput | Peak RSS |
|---|---|---|---|---|---|---|
| 8 | 217 | 2.92 s | 0.064 s | 8.527 ms | 25,449 rays/s | 425 MiB |
| 16 | 817 | 2.90 s | 0.063 s | 12.786 ms | 63,900 rays/s | 425 MiB |
| 32 | 3,169 | 3.09 s | 0.067 s | 29.670 ms | 106,808 rays/s | 430 MiB |
| 64 | 12,481 | 3.15 s | 0.067 s | 105.961 ms | 117,789 rays/s | 455 MiB |

Seven timed repeats after two warmups; raw per-repeat samples are in
`outputs/M1/ray/raw_timing_samples.json`. The regression envelope
(`relative_increase_max = 0.25` on the steady trace median, matching
fingerprint required) is recorded with status
`recorded_pending_human_review` — it is a baseline to review, not a gate that
has yet been exercised against a regression.

## Determinism

Bitwise identical canonical scientific arrays and exact summary metrics across
seven repeats at every sampling level. The standalone baseline reports
`status: passed`, `deterministic: true`.

## Corruption rejection

Fixture `scaling/scientific_artifacts/sampling_8.npz`, mutated by appending
bytes; the evaluator exits `2` with a hash mismatch. A benchmark that cannot
notice a corrupted artifact cannot certify one.

## Artifacts

`result.json`, `provenance.json`, `bundle_manifest.json`,
`raw_timing_samples.json`, `tolerances.yaml`, `plot.png`, `scaling.png`,
`accuracy_plot.png`, plus the full `standalone/` and `scaling/` subtrees.
Every component file is SHA-256 hashed in `bundle_manifest.json`. Both
`result.json` and `provenance.json` validate against `benchmarks/schemas/`.

---

# Wave branch — L1-WAVE-01 (Chromatix)

**Protocol `M1-BASELINE-CPU-V1`** · engines `chromatix 0.6.0` @ commit
`d24bdf0022835bb8ce1cdcc6aeafbc7fcb39daee`, `jax 0.6.2`, `numpy 2.2.6` ·
dtype `complex64`, device `cpu`, `jax_enable_x64 = false`
Scientific fingerprint `b2d99bcc12874484050da0daeda46cbb96ef22f3d160bb573e1ab4ade260feb2`

## Accuracy — PASS on the two gated cases; one case blocked

The wave suite is a deliberate progression: each rung removes a class of
excuse before the next introduces it.

### Case 1 — exact homogeneous primitive · PASS

FFT-bin plane-wave eigenmodes, propagated **unpadded** (the mode is periodic;
zero-padding would manufacture an aperture edge the physics does not contain).
The oracle `u·exp(i k_z z)` is exact, so the tolerance is **derived from
float32 phase round-off, not chosen**:

| Accumulated phase `\|k_z z\|` | Phase error | Derived bound |
|---|---|---|
| 118 rad | `6.49e-06` rad | `7.04e-05` |
| 2952 rad | `3.26e-04` rad | `1.76e-03` |

Amplitude ratio, discrete power, and a `+z`/`−z` round trip are all exact to
`≤ 5e-7`. Eight records span an axis-asymmetric mode and `sinθ = 0.39`.

### Case 2 — ideal signed paraxial focusing · PASS

| Tilt (x) | Centroid error | FWHM(x) rel. error | Overlap vs independent float64 ASM |
|---|---|---|---|
| `−2e-3` | 0.0423 px | `9.49e-05` | `0.999999` |
| `0` | 0.0116 px | `9.78e-03` | `0.999999` |
| `+2e-3` | 0.0543 px | `1.01e-04` | `0.999999` |

Tolerances: centroid 0.1 input px, FWHM 2 %, sidelobe 5 %, overlap ≥ 0.99
(Fresnel) and ≥ 0.9999 (independent ASM). The focus lands at `+f·θ` with the
sign. Because Chromatix matches an independent float64 angular spectrum to
`1e-6`, the residual against the Fresnel oracle is attributed to the *oracle's*
paraxial approximation, not to the solver.

### Case 3 — high-NA vectorial · **BLOCKED**

See limitation L1.

### Negative perturbations — 4/4 detected

`case1_paraxial_dispersion`, `case2_lens_sign_flip`, `case2_axis_transpose`,
`case2_si_scale`. Each has an unperturbed control that passes.

## Scaling accuracy — PASS

Fixed-Gaussian radius and power gates hold across all three grids: measured
`D4σ` radius `1.00498887e-05 m` against the analytic `1.00498756e-05 m`
(relative `1.3e-6`), centroid `≈1e-7` input pixels.

## Performance — recorded, pending human review

| Input grid | Output shape | Steady median | Peak RSS | Deterministic |
|---|---|---|---|---|
| 64² | 192² | 8.772 ms | 315 MiB | yes |
| 128² | 384² | 20.549 ms | 334 MiB | yes |
| 256² | 1756² | 152.166 ms | 1182 MiB | yes |

Medians are **synchronized compiled steady-state** times, measured with
`time.perf_counter_ns` around `jax.block_until_ready(field_out.u)` — an
unsynchronized JAX timing would measure dispatch, not computation.

The cache policy differs from the ray branch and is recorded verbatim in each
case: *fresh process and fresh shape for the first call; seven following calls
reuse that process-local JAX compilation cache; first-call timing is never
mixed into steady samples.* Compilation is therefore reported separately
(`compile_plus_execute_seconds`, `import_seconds`, `setup_seconds`) rather
than amortised into the solver number — for the 64² case, 1.36 s of
compile-plus-execute against an 8.8 ms steady median.

The 256² case grows to a 1756² output because automatic padding is applied and
recorded; that padding, not the input grid, drives the memory jump.

## Corruption rejection

Fixture `scaling/complex_fields/gaussian_grid_64.npy`, mutated by appending
bytes; the evaluator exits `2` with a hash mismatch.

## Artifacts

Same root set as the ray branch, plus the analytic suite's
`error_attribution.json`, `reference.json`, `reference_fields.npz`, and
`solver_summaries.json` under `scaling/accuracy_gate/`. All schema-validated
and hashed.

---

## Independence evidence

`./run.sh python benchmarks/verify_m1_independence.py` → **passed**, all five
checks:

| Check | Result |
|---|---|
| `ray_source` | no `chromatix`, no `multiscale_optics_agent.couplers`, no `C_RAY_TO_WAVE`/`C_WAVE_TO_RAY` identifier in any ray entry-point source |
| `wave_source` | no `optiland`, no coupler import, no coupler identifier |
| `ray_bundle` | `forbidden_modules_loaded == []` at exit, in the bundle and both nested provenance files |
| `wave_bundle` | `forbidden_modules_loaded == []` at exit |
| `claim_audit` | 10/10 registry claims consistent with M1 evidence |

The check is both **static** (AST import analysis plus identifier scan of every
entry-point source) and **dynamic** (`sys.modules` inspected at process exit in
each emitted provenance file). Passing the ray branch therefore cannot mask a
wave failure, and neither branch can silently acquire a coupler dependency.

## Claim audit — solver cards and registry

All ten checks pass, confirming the registry claims nothing M1 did not
establish:

`ray_cpu_only`, `ray_float64_only`, `ray_gradient_unverified`,
`wave_cpu_only`, `wave_complex64_only`, `wave_scalar_only`,
`wave_gradient_unverified`, `ray_to_wave_experimental`,
`ray_to_wave_gradient_unverified`, `wave_to_ray_not_claimed`.

Explicitly **still unverified** after M1, in both the cards and the registry:
gradients through either engine, GPU/TPU execution, vector fields, chromatic
propagation, and every coupling direction.

## Reproducibility policy

Each branch publishes a **scientific fingerprint**: a SHA-256 over a canonical
projection containing the standalone stable summary, the accuracy section, the
canonical `.npz` array hashes, the tolerance file hashes, and the per-case
scientific array/field hashes.

Explicitly excluded as volatile: timestamps, run identifiers, output paths,
process IDs, wall-clock timing samples, peak-RSS observations, and the git
dirty flag. Re-running a branch must reproduce the fingerprint exactly;
`tests/benchmarks/test_m1_reproducibility.py` runs each branch twice and
asserts it.

### Defect found and fixed during this review

The first M1.8 reproduction run **failed**: the wave branch produced a
different scientific fingerprint on a second, otherwise identical run.

Diagnosis: the solver was not at fault. The wave scaling field hashes were
bitwise identical across both runs, and the ray branch reproduced exactly. The
projection was hashing `result["accuracy"]` wholesale, and the wave analytic
accuracy section embeds a per-case `solver.runtime_seconds`. Fifteen wall-clock
values were therefore inside a hash that the published policy already declared
free of "wall-clock runtime samples" — the implementation contradicted its own
stated exclusions, so the fingerprint tracked machine load rather than physics.

(Four further leaves also differed under a naive comparison, but those are
`nan != nan` artifacts in *perturbation* records where a measurement correctly
returns NaN. They serialize identically and never affected the hash.)

Fix: `m1_bundle.VOLATILE_KEYS` plus `_strip_volatile()` now remove wall-clock
and run-identity keys at any nesting depth before hashing, and the projection
publishes the stripped key list so the policy is auditable rather than merely
asserted. `tests/benchmarks/test_m1_bundle_projection.py` pins the behaviour in
milliseconds, including a case proving a genuine physics change still changes
the projection — stripping must not be so aggressive that it hides a real
difference.

No tolerance and no scientific value was altered. Both branches reproduce their
fingerprints after the fix.

---

## Risks and known limitations

### L1 — Wave Case 3 is blocked by a Chromatix defect (does not gate M1)

`chromatix.functional.high_na_ff_lens` does not produce a
sampling-independent focal field. Refining only the pupil sampling, with
wavelength, NA, focal length, output grid, and output pitch all fixed, moves
the `|E_z|` ring radius from 246 nm to 2536 nm (oracle: 197 nm) and `Iz/Ix`
from 0.126 to 0.366 (oracle: 0.150087) — a relative spread of **2.374**. The
independent Richards–Wolf quadrature oracle converges to `2e-14` over the same
comparison, so the non-convergence is in the solver, not the reference.

Root cause, from the pinned source: `s_z` is derived from `field.f_grid·λ/n`
(the frequency grid) rather than the pupil position grid, so `s_z ≈ 1` and the
`1/cosθ` obliquity Jacobian, the `exp(i k f cosθ)` defocus, and the
`zoom_factor` that sets the output scale all degenerate.

Reported as **blocked, not failed**: "failed" would imply a measured
disagreement between two well-defined numbers, and there is no converged
solver quantity to disagree with. No tolerance was changed. Recorded in
`knowledge/solvers/chromatix/solver_card.yaml` under `known_defective`.

### L2 — The two branches run different protocol versions, and V2 is undocumented

The ray branch declares `M1-BASELINE-CPU-V2` (amended by CHE-20 for
surface-shape and clear-aperture behaviour); the wave branch declares
`M1-BASELINE-CPU-V1`. Both schemas accept both values, and
`tests/test_m1_protocol.py` pins the enum — but `benchmarks/protocol.yaml` and
`benchmarks/M1_BASELINE_PROTOCOL.md` still describe only V1. **What V2 changed
is currently recorded only in the L1-RAY-01 README and design note, not in the
protocol document itself.**

Deliberately not fixed inside this review: amending a frozen protocol document
in the same pass that reviews compliance against it would defeat the point.
Recommended as a small follow-up before M2 consumes the protocol.

### L3 — The bundles were produced from a dirty worktree

`dirty_worktree: true`. The recorded commit `ae7fac01` therefore does not fully
describe the tree that produced these numbers, because the M1 implementation
work is not yet committed. The scientific fingerprints are still valid for
comparing *these* bundles against each other and against re-runs on this tree,
but the clean-checkout acceptance criterion is only fully satisfied once the
M1 work is committed and the two branch commands are re-run against that
commit. **This is a bookkeeping gap, not a scientific one.**

### L4 — Performance envelopes are baselines, not yet gates

Both branches record `regression_envelope.status =
recorded_pending_human_review` with `relative_increase_max = 0.25`. No
regression has been detected because no prior matching-fingerprint run exists
to compare against. The envelopes become meaningful on the second run in the
same environment.

### L5 — Machine is shared and unpinned

80 logical CPUs, no affinity pinning, thread-count environment variables
unset. Affinity is recorded as an observation and the report never claims core
isolation. Timing numbers should be read as same-machine relative figures, not
as absolute performance characterisations.

---

## What M2 should carry forward

1. **Do not build a coupler on the vector path.** Scalar CPU `asm_propagate` is
   the only Chromatix surface M1 validated. Vectorial focusing is defective
   (L1) and `thick_plano_convex_lens` is an ABCD/`ray_transfer` implementation
   rather than a surface-resolved wave model.
2. **No gradient is verified through either engine.** A PyTorch-to-JAX handoff
   remains `forward_only`, and the ray-to-wave coupler must not claim
   differentiability without its own directional-derivative evidence.
3. **The conventions that M1 pinned are the coupler's contract**: `(y, x)` axis
   order with index `n//2` as coordinate zero, `exp(-i ω t)` phasor with
   `exp(+i k z)` spatial factor, SI throughout, amplitude never intensity, and
   Optiland's mm/µm-to-SI conversion at the adapter boundary. Case 1 pins the
   wave side of these to float32 round-off; the ray side is pinned by the
   free-space and paraxial cases.
4. **`RealRays.opd` semantics are still unverified** (reference plane and sign).
   It is preserved as `opd_native` and must not be used as an absolute OPL
   oracle by a coupler without separate characterisation.
5. **Close L2 and L3** before M2 depends on the protocol or on a citable
   commit.
