# Independent review — CHE-120 (M5.3): primary-position sampling and estimator variance

- **Date:** 2026-08-25
- **Reviewer:** independent read-only reviewer (`.claude/agents/code-reviewer.md`), separate from the implementing agent.
- **Scope:** `src/couplers/patch_positions.py`, `tests/test_patch_positions.py`,
  `src/couplers/patch.py`, `src/couplers/cascade.py`, `src/registry/couplers.yaml`,
  `knowledge/couplers/wave_to_ray/{card.yaml,failure_guide.md}`, and the
  already-committed `benchmarks/reports/2026-08/demo3_estimator_variance.md` and
  `benchmarks/manifest.yaml` claims.
- **Why required (AGENTS.md):** coupler behavior, estimator sampling density,
  numerical/convergence behavior, benchmark tolerances and cost claims.
- **Evidence supplied to the reviewer:** the diff; `pytest -q
  tests/test_patch_positions.py tests/test_patch_wft.py` → 70 passed; `pytest -q`
  over the architecture/registry/knowledge/coupler-contract subset → 164 passed;
  the declared unverified areas (GPU probe records not re-measured, `bias_ratio`
  independent-pixel assumption, density not exposed through the graph node,
  unattributed 13% inter-arm wall clock).

## Verdict

**No scientific blocker in the estimator.** Two mechanical/scope items were
raised as *must fix before merge*; both are resolved by this commit itself
(see "Disposition"). Five *should fix soon* items are recorded below and are
documentation/scope-precision defects, not result defects — none changes a
number in the report and none affects a shipped configuration.

## What the reviewer verified (so it is not re-litigated)

- **Weight formula.** `lambda_c = P/(D·pi_c)` with `P = realized` gives
  `E[F_hat] = (1/D)·sum_c g_c` for all three schemes. The full normalization
  chain was traced: `patch_secondary_rays` emits `coverage · lambda_p ·
  modal[picks]/density[picks]`, `ray_to_wave` divides by `bundle.count`,
  `StreamingReconstruction` divides by `total_rays`, and
  `demo3_hologram_lens.run_route` sets `n_patches = emitters.shape[0]` — the
  *realized* count. So the 926-of-1000 jittered arm is normalized by 926, not
  1000.
- **`_draw_stratified_cdf`:** `pi_c = P·q_c` is exact. The off-support snap is
  reachable only when `rng.random()` returns exactly `0.0`; the stated `2^-53`
  and the residual bias it introduces are both correct.
- **`_draw_jittered_grid`:** cells partition the candidate grid and `rng.choice`
  is with replacement, so `draws · conditional[c]` *is* the per-slot marginal.
  Skipping zero-mass cells is exact, because zero density occurs only where
  `window_sample_count_map == 0`, i.e. where the patch is identically zero.
- **`D` bookkeeping and coverage.** `candidate_index_grid` reproduces
  `plan_patches`' `rng.integers` range exactly; `coverage = D/patch_px²` is
  computed identically in the supplied-centres and drawn branches. Holding
  `D = 90601` while the importance draw never visits the 9,388 empty candidates
  is correct, not a leak.
- **Half-sample conventions.** `_box_sum`'s window matches `extract_patch`
  exactly including clip-to-array zero continuation, and `spectral_l1_map`
  reuses `extract_patch` and the emitter's own
  `fftshift(fft2(ifftshift(·)))/pad²`.
- **Default path (AC #4).** `center_weights=None` is genuinely *absent* from the
  emitter (a branch, not a `*1.0`); the probe's control arm routes through
  `plan_patches`' internal draw; `plan_positions`' uniform branch consumes the
  same rows-then-cols `rng.integers` stream. **The bitwise claim holds.**
- **Records describe this tree.** `./run.sh --no-build pytest -q
  tests/test_provenance_fingerprint.py` → **19 passed**, i.e.
  `test_every_stamped_record_still_describes_this_tree_code` accepts the CHE-120
  records against this working tree including the new module. The measured
  1.4685x was produced by exactly the code under review.
- **Report arithmetic re-derived:** `S* = sqrt(aB/(bA)) = 2.15e4` and the 0.10%
  objective penalty at `S = 20,000`; `6.71 × 1.162 × 1.471 = 11.5`;
  `1.494e9 × 3.350e-7 = 0.139 h`; `1.494e9/1.4709 = 1.016e9 → 0.094 h`; the
  three-rung out-of-sample predictions against the measured 0.4105; noise floor
  `3/420 = 0.00714` with the first rung at 3.17×. **AC #1, #2, #3, #5, #6 are
  supported by the records.**

## must fix before merge — disposition

1. **`.claude/agents/code-reviewer.md` bundled into a physics diff** (it also
   flips `permissionMode: plan` → `default`). — **Resolved:** excluded from the
   CHE-120 commit and left in the working tree for the owner to raise
   separately on its own merits.
2. **The module and its gate were untracked while HEAD already cited them** —
   HEAD's `benchmarks/probes/ray_wave/demo3_variance.py` imports
   `couplers.patch_positions`, and the report and manifest point at
   `tests/test_patch_positions.py`, so HEAD could not import. — **Resolved:**
   both files are explicitly `git add`ed in this commit, which is the commit
   whose purpose is to restore that consistency.

## should fix soon — recorded, not blocking

3. **`_draw_jittered_grid` silently biases when `count < len(cells)`,** and the
   comment claims the opposite. Cells `count..len(cells)-1` then get zero draws,
   so candidates inside them have `pi_c = 0` with `g_c != 0` — a support hole no
   weight can repair, and one that `support_power_fraction` checks on the
   *density* side but never on the *draw* side. Reachable only when the
   candidate grid's aspect ratio exceeds `count` (e.g. a 1024×8 DOE with
   `count = 64`). **Not reachable for demo2/demo3** (square grids, `count >> 1`),
   so no shipped number is affected. Fix: raise a `ContractError` or clamp
   `tiles_y` so `len(cells) <= count`, and delete the unbiasedness claim from
   that comment.
4. **"weights are exactly 1.0 for the uniform i.i.d. scheme" is false at
   demo3's own size.** `realized/(total·(count/total))` is `1.0` for the
   `(64, 361)` case the test pins, but `0.9999999999999999` for
   `(1000, 90601)`. Three docstrings state "exactly 1.0". The bitwise property
   is actually delivered by `center_weights=None`, and a 1-ulp scale is
   physically irrelevant, so this is documentation/test coverage rather than a
   result defect. Fix: return `np.ones(count)` in the uniform branch, or soften
   the docstrings and add `(1000, 90601)` to the test.
5. **`spectral_l1_map` sums over all `pad²` modes; the emitter's `f_c` is over
   propagating modes only.** They agree for demo3 (λ/2·pitch = 0.056, all modes
   propagate), so the docstring's "the spectrum the emitter will actually draw
   from" is true there and false in general. It would mis-rank positions for a
   short-pitch configuration — precisely the "compute it per configuration" use
   the card advertises. Fix: apply the propagating mask, or scope the docstring.
6. **"1.4413x is the ceiling on this axis" is over-scoped** in the report,
   `card.yaml` and `couplers.yaml`. The closed form keeps only the finite-`S`
   term, so `q ∝ ||U~_c||_1` is the exact optimum of the **B** term and merely a
   good choice for **A** — which is *why* the measured total 1.4685x legitimately
   exceeds it and why `A` fell 1.5849x. Small remaining headroom (A is 25% of V
   and already falls further than B), so this is wording, not a wrong number.
   Fix: say "ceiling on the `B` term", and qualify the M5-exit sentence to "at
   its optimum for the noise-dominated term".
7. **Two claim-precision items.** (a) `src/registry/couplers.yaml` says
   unbiasedness of *each* density is gated; the parametrization covers 7
   (density, draw) pairs but not `SPECTRAL_L1`, nor `WINDOW_ENERGY` under either
   stratified draw. (b) `demo3_variance_candidates.json` records
   `predicted_variance_ratio: 1.4475`, which is the *self-model* figure, not the
   1.4412 the report quotes against the exact `||U~_c||_1` model; only the
   `variance_model` string distinguishes them.

## Residual uncertainty

- **Anti-bias gate resolving power.** `tests/test_patch_positions.py` is a real
  oracle, not a self-comparison: it enumerates all 361 positions × all modes for
  the estimator's exact mean, its candidate set is pinned to `plan_patches` by
  the bitwise test, and its tolerance is a measured standard error. But its
  resolving power is a few percent — adequate for the named bug classes
  (`1/(Dq)` instead of `P/(Dπ)`, coverage folded in twice; both O(10x)) and not
  a fine bias bound. The exact algebra above, not the test, is what rules out a
  sub-percent bias.
- **A negligible RNG coupling in the probe's candidate arms.** `plan_positions`
  and `run_route` share `seed`, so patch 0's first secondary uniform is the same
  double that selected patch 0's position. One correlated ray in 2×10⁷, at any
  `(P, S)` with `S ≥ 1`. No action; recorded so it is not rediscovered.
- **Not run by the reviewer:** the GPU probes, the two pytest subsets the
  implementing agent already reported, the tutorial suite, the full suite. The
  `bias_ratio` speckle-correlation caveat and the unattributed 13% inter-arm
  wall-clock spread were accepted as the report's declared limitations.

## Regeneration after review

**Not required.** The review produced no behavior-affecting change — items 3–7
are documentation, scope-precision, and an unreachable-at-demo3 guard, and none
was applied in this commit. No estimator code changed after the records were
stamped, and `tests/test_provenance_fingerprint.py` (19 passed) confirms every
stamped record still describes this tree. The report's own regeneration pass
therefore stands, as does the conclusion that `PatchPlan.center_weights=None`
leaves the shipped estimator bitwise unchanged and triggers no re-measure.
