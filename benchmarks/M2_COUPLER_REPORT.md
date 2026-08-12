# M2 exit report — bidirectional ray–wave coupler

CHE-29. This report integrates evidence only. Every number in it comes from a
run recorded in `outputs/M2/coupler` or from the test suite named beside it.

**Verdict: M2 is recommended for exit.** Both coupler directions are implemented
and characterized, all accuracy and stochastic gates pass, and no gate was
satisfied by loosening a tolerance. Six limitations are recorded (L1–L6), and
three findings changed what the milestone concluded rather than merely
confirming it.

M2 began from nothing: the M0 audit and a re-check at M2 open both found that
this repository contained **no coupler implementation at all** —
`couplers/base.py` was a 44-line `Protocol` with zero numerics, `C_RAY_TO_WAVE`
was a registry claim nothing executed, `C_WAVE_TO_RAY` did not exist, and the
four boundary artifacts existed only as enum members plus prose. The physics was
therefore authored from the paper (DOI
[`10.1021/acsphotonics.6c00818`](https://doi.org/10.1021/acsphotonics.6c00818)),
not from code. Its reference implementation is neither vendored nor executed
here, so nothing below cites it as evidence.

---

## Exact commands

```bash
./run.sh python benchmarks/level2/L2-COUPLER-01/run_benchmark.py --output-dir outputs/M2/coupler
./run.sh python benchmarks/level2/L2-COUPLER-01/evaluate.py outputs/M2/coupler
./run.sh python benchmarks/verify_m1_independence.py
./run.sh python scripts/validate_package.py
./run.sh pytest -q
```

| Command | Result |
|---|---|
| `L2-COUPLER-01/run_benchmark.py` | `status: complete`, accuracy and stochastic gates pass |
| `evaluate.py` on the clean bundle | exit `0` |
| `evaluate.py` on a mutated bundle | exit `2`, hash mismatch |
| `verify_m1_independence.py` | `status: passed`, 13/13 claim checks |
| `validate_package.py` | 8 models, 11 couplers, all YAML and example graphs valid |
| full suite `pytest -q` | **348 passed, 2 xfailed, 1 xpassed** |

The two xfails (fdtdx gradient locks) and one xpass (sax circuit gradient) are
pre-existing, documented, and unrelated to M2. M1 ended at 226 passed; M2 adds
122 tests.

## Environment

| Item | Value |
|---|---|
| Protocol | `M2-COUPLER-CPU-V1`, extending `M1-BASELINE-CPU-V1` |
| Python | 3.12.13 |
| Device / dtype | CPU, `float64` / `complex128` |
| Seed | `20260812` |
| RNG | `numpy.random.Generator(PCG64)` |
| Scientific fingerprint | `c928e4ca36c6dc1cdc6ed1b23f28edb0fed9ffb2b7211aa3df499fe0f5ed24b2` |
| Fingerprint reproduces | **yes**, bit-identical across two independent runs |
| `forbidden_modules_loaded` | `[]` — neither Optiland nor Chromatix reached the coupler core |

---

# What M2 established

## The coupler core is engine-agnostic

Checked statically (AST scan of every core module) and dynamically (a subprocess
asserting `sys.modules` contains neither engine after a reconstruction). The
reason is diagnostic rather than stylistic: M1 proved the two engines are
independently correct, and if the core could import one, a coupler defect could
be misattributed to engine behaviour and M1's evidence would stop bounding the
search.

## C_RAY_TO_WAVE — accuracy

| Case | Metric | Observed | Tolerance | Basis |
|---|---|---|---|---|
| Analytic tilted plane wave | max abs error | `7.82e-14` | `1e-9` | float64 round-off over the coherent sum |
| Multilobed enumeration limit | max abs error | `5.44e-15` | `8.80e-13` | dtype round-off, derived |
| Concentrated enumeration limit | max abs error | `8.88e-16` | `2.27e-13` | dtype round-off, derived |
| Chromatix ASM cross-check | relative residual | `2.77e-05` | `3.00e-04` | M1's own float32 figure, 5·ε₃₂ per radian |

The plane-wave oracle is the load-bearing one: SI Figure S1c means every ray in
a collimated bundle contributes the *same* plane wave once its OPL compensates
its launch position, so a single comparison pins the OPL handling, the `Δr`
ramp, the phasor sign and the projection convention simultaneously — removing
any one of them breaks it.

The Chromatix cross-check is an **independent implementation**: advancing rays
geometrically by 40 µm and reconstructing must agree with propagating the `z=0`
reconstruction through the M1-verified angular spectrum. The residual sits below
one ε₃₂ per radian of accumulated phase, which is asserted, so it is rounding
rather than disagreement.

## C_WAVE_TO_RAY — stochastic characterization

`C_WAVE_TO_RAY` is a Monte Carlo estimator, so *reproducible* and *accurate* are
two claims. All four required kinds of evidence pass, in the mandated order.

| Evidence | Observed | Gate | Why this order |
|---|---|---|---|
| Exactness limit | `5.44e-15` abs, 256/256 modes | `8.80e-13` | No sampling error at all, so a failure here is a transform defect and tuning `N` would be beside the point |
| Unbiasedness | mean error `2.02` vs SE `1.97` → **1.02 σ** | ≤ 3 σ | The tolerance *is* the measured standard error |
| Convergence | fitted exponent **`−0.4938`** | `−0.5 ± 0.1` | Fitted over a six-point sweep, never gated at one `N` |
| Variance by density | see below | reported | The size of the advantage is the property |

Variance at matched `N = 1024`, relative RMS:

| Spectrum | `p_uni` | `p_mag` | advantage |
|---|---|---|---|
| Concentrated (Gaussian) | `0.4642` | `0.0935` | **4.96×** |
| Multilobed (random) | `0.4864` | `0.4382` | 1.11× |

That is the paper's Figure 4 claim reproduced quantitatively: magnitude-
proportional sampling exploits spectral concentration and is merely comparable
to uniform without it.

## Round trip, cascade, and curvature

| Check | Observed |
|---|---|
| `wave → rays → wave`, enumeration limit | relative RMS `1.32e-15` |
| Same round trip with a mismatched phase sign | relative RMS **`1.40`** — detected |
| Cascade ray count, two planar DOEs in series | `256` then `256`, not `256 × 64` |
| Pure-phase DOE power ratio | `1.0000000000` |
| Curvature bound vs measurement, 12 cases | worst measured/bound `0.961`, all hold |

The mismatched pairing is what makes the round trip meaningful. A shared
convention error cancels between the two directions, so a round trip that cannot
be made to fail proves nothing.

## Negative controls — 5/5 detected

`phase_sign`, `oblique_ramp`, `axis_transpose`, `importance_weight`,
`launch_phase`. Each is run through the **shipping** implementation with one
term removed, not a parallel hand-written copy, and each has a passing
unperturbed control. The importance-weight omission is detected at **1594 σ** of
ensemble-mean bias.

Three blind spots are pinned as tests in their own right, because each would let
a negative control pass for the wrong reason:

- the projection factor is exactly `1` at normal incidence;
- the oblique ramp is inert for a single centred on-axis ray;
- under uniform sampling the omitted `1/p` weight is exactly a constant scale
  factor, so the bias test must run under `p_mag`.

---

## Three findings that changed the conclusion

### F1 — Main-text eq 2 and SI eq S5 are different operators

Eq 2 carries the factor `⟨n̂, d̂⟩`; eq S5, which derives the same wavelet sum as
an estimator of the angular-spectrum integral, does not. The paper does not flag
the difference. Measured: summing every propagating mode of a random 16×16 field
reproduces the field to `7.1e-15` **without** the factor and misses it by
**2.2 %** of peak amplitude **with** it, tracking the smallest `cos θ` on the
grid.

A representation change must preserve the field, so the coupler defaults to the
factor-free form and retains eq 2 as an explicitly named *sensor* model. Both
are implemented, both are tested against what each actually claims to reproduce,
and the choice is recorded in every reconstructed field's provenance. Choosing
silently would have cost a few percent off-axis under no test's name.

### F2 — The gradient estimator is not detectably biased here

The issue expected "biased, here is how much". In the regime this repository
implements — a **fixed spectral grid** and a fixed observation plane — the
surrogate mean sits within **0.21 σ** of the true derivative, and stays inside
3 σ across `N` from 512 to 32768 while the standard error falls ~8×.

The reason is structural: on a fixed spectral grid the sampled direction belongs
to the *bin*, not to the DOE parameter, so there is no direction gradient to
neglect. The paper's caveat describes the continuous-wavevector formulation that
SI S7.3's Gumbel–Softmax relaxation exists to address.

What *is* measurable is the reverse of the natural reading: **detaching the
sampling density is what makes the estimator unbiased, not what biases it.**
Letting `p` track the parameter biases the same gradient by **26 σ**, because
detaching drops precisely the term the omitted score-function term would have
cancelled.

This does not promote anything. See L4.

### F3 — The curvature bound is only observable above `sqrt(2 λ R)`

Not stated in the paper. A patch of width `D` resolves directions no finer than
`λ/D` while carrying a spread of `D/2R`, so the effect is spectrally visible
only when `D > sqrt(2 λ R)`. At `R = 10⁴ λ`, where the crossover is `141 λ`,
bound-to-measured ratios were `208` at `D = 100 λ`, `12.7` at `200`, and `1.15`
at `400`.

The bound stays correct throughout — it simply cannot be *confirmed tight* below
that width, and reading the 200:1 gap as a useless bound would be wrong.
Exposed as `curvature_observability_width()`.

Validating the bound also required measuring the **locally extracted** direction
rather than the extreme angle in the whole patch's spectrum. The global version
makes eq S9 look violated at every patch size, because that number is the
truncated aperture's edge diffraction: a flat `50 λ` patch already reaches
`0.02 rad` at its first sinc null, fifty times the curvature bound at
`R = 10⁵ λ`.

---

## Two defects found and fixed during M2

Both were caught by a *later* ticket exercising an *earlier* ticket's work,
which is the argument for building the round trip at all.

1. **Per-axis Nyquist.** CHE-24's grid check tested the direction *norm*, but the
   limit is per axis. A diagonal FFT bin has `|d| = √2 · λ/(2·pitch)` yet is
   exactly representable. The bug surfaced when CHE-26's round trip could not
   enumerate its own spectrum's corner modes.
2. **The `1/N` declaration was prose.** SI eq S5's factor was recorded on the
   bundle as text no component could act on, so the cascade reconstructed a
   field scaled by the mode count — 256× on a 16×16 grid. It is now structured
   data (`RayBundle.reconstruction_normalization`), set by `C_WAVE_TO_RAY` and
   honoured by the reconstruction, so the two cannot disagree.

A third, smaller correction: CHE-24's round-off bound counted only the OPL phase
and not the ramp phase across the output grid, which dominates on a wide grid.

---

## Claim audit

13/13 checks pass in `verify_m1_independence.py`. The single check
`wave_to_ray_not_claimed`, which asserted this coupler's **absence**, could no
longer hold once CHE-23 registered it. It was **replaced rather than deleted**,
by four narrower claims — `wave_to_ray_registered_experimental`,
`wave_to_ray_gradient_unverified`, `wave_to_ray_gradient_mode_is_surrogate`,
`wave_to_ray_declared_lossy` — so the audit became more specific rather than
weaker. Registration is not a capability claim.

`derivative.mode` for `C_WAVE_TO_RAY` is `surrogate`, not `native_autodiff`:
the estimator differentiates a fixed-direction surrogate of the objective, not
the objective itself, and declaring otherwise would overstate the method.

**Still unverified after M2**, in both cards and registry: any gradient through
either coupler, GPU/TPU execution, vector and polarized fields, chromatic
coupling, partial coherence, conformal-surface coupling, caustics, and the
end-to-end Optiland→Chromatix graph.

---

## Risks and known limitations

### L1 — The end-to-end graph is still blocked, by an M1 limitation

`L2-PSF-01` — the `M_RAY_OPTILAND → C_RAY_TO_WAVE → M_WAVE_CHROMATIX` path — is
**not implemented**, and M2 did not unblock it. Optiland's `opd_native` sign and
reference plane remain unverified, so no admissible optical path length can be
handed to the coupler from a real trace. The contract layer refuses the input
rather than guessing, because a wrong OPL *reference* is a harmless piston while
a wrong OPL *sign* conjugates the wavefront, and those are indistinguishable
downstream.

This is recorded in `benchmarks/manifest.yaml` as `implemented: false` with the
blocker named. **Characterizing Optiland's OPD convention is the single highest-
value piece of work for M3**: it is what stands between M2's verified couplers
and a working vertical slice.

### L2 — Fixed grid, one wavelength, one grid size

Every number here is at `16×16`, `1 µm` pitch, `500 nm`, `complex128`, CPU. The
convergence exponent, the variance advantage, and the gradient result are all
measured in that one configuration. Nothing suggests they are fragile, but
nothing here establishes grid-independence either.

### L3 — The curvature bound is enforced but not yet *used* by a coupler

`check_patch()` exists and is tested, but no coupler call site invokes it,
because M2 implemented no conformal path. It is a precondition waiting for a
caller.

### L4 — F2 is not a gradient certification

`derivative.verified` remains `false` for both couplers, asserted by a test. One
parameter, one grid, one wavelength, two objectives, 32 realizations, and no
optimization loop is a characterization. Promotion would additionally need the
continuous-wavevector regime where the paper's caveat does bite, and a gradient
exercised through an actual optimization.

### L5 — Bundles were produced from a dirty worktree

`dirty_worktree: true`, recorded against commit `c20aa224`. Identical in kind to
M1's L3: the fingerprints are valid for comparing these bundles and re-runs on
this tree, but the clean-checkout criterion is fully satisfied only once the M2
work is committed and the benchmark re-run against that commit. Bookkeeping, not
scientific.

### L6 — No performance envelope

The steady-state median is `0.622 s` for a 4096-ray round trip on a `16×16`
grid, peak RSS `843 MiB`. Recorded as an observation only: unlike M1 this
benchmark declares no regression envelope, so there is nothing yet to regress
against. The machine is shared and unpinned, as in M1, so timings are
same-machine relative figures.

---

## What M3 should carry forward

1. **Characterize Optiland's `opd_native`** against a known geometry. It is the
   one blocker on the vertical slice, and everything else in M2 is ready for it.
2. **The two projection conventions are a real choice, not a detail.** A coupler
   must use the field-preserving form; a sensor model may use eq 2. Any new
   surface that sums wavelets has to declare which.
3. **The exactness limit is the cheapest diagnostic in the system.** Before
   debugging any stochastic result, enumerate every bin: if that fails, the
   defect is deterministic and no amount of sampling work will help.
4. **Averaging too few realizations can manufacture a finding**, not merely
   widen an error bar. A fitted convergence exponent read `−0.58` at 8 seeds
   per point and `−0.48` at 64. The gating tests use 16; a benchmark can afford
   more.
5. **A round trip that cannot be made to fail proves nothing.** Keep the
   deliberately mismatched pairings alongside the passing ones.
6. **Conformal coupling needs the curvature precondition wired in**, and it
   cannot reuse Algorithm S1's cascade: there is no common plane to accumulate
   onto when rays strike position-dependent tangent frames.
