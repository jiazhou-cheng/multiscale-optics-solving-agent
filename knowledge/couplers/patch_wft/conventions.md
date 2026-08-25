# C_PATCH_WFT — conventions at the boundary

Nothing from the composed couplers is restated. Read
`knowledge/couplers/ray_to_wave/conventions.md` for the accumulation,
`.../wave_to_ray/conventions.md` for the resampling, and
`.../planar_doe_step/conventions.md` for the cascade — including its declared
plane (D1), its OPL rebasing (D2) and its spectral-amplitude outgoing convention
(D3). All of them bind here unchanged, because this coupler *is* that cascade
applied per patch.

What follows is only what patching adds.

## P1 — `patch_px` is odd, and an even request is refused

An even patch has no centre sample, so "centred on a ray" is undefined and the
odd-pad and even-`(pad − patch)` rules become jointly unsatisfiable.

The refusal is `SHAPE_MISMATCH` with a remedy naming the nearest odd sizes. It is
refused rather than rounded because the paper's own sizes — 40, 50, 100 — are all
even, so an even request is the *likely* one, and silently rounding hands back a
different operator than the caller asked for.

## P2 — `pad_factor` is a preference; `pad_px` is derived and reported

The planner raises the pad until clearance, centring and oddness all hold, then
reports what it used in `plan.pad_px`. On a 15-px grid with `patch_px = 5`,
`pad_factor` 1, 2 and 3 all derive `pad_px = 21`.

**Read `plan.pad_px`, never your own request.** A caller reasoning about the grid
it asked for is reasoning about a grid that does not exist. A pad that genuinely
violated clearance produces a plausible field wrong by 100%.

### The one exemption, and why it is not a relaxation

At one full-aperture patch, `pad_factor = 1` is the condition under which the
patch's mode set and the *unpadded* oracle's mode set are the same set. Padding
it — correct behaviour everywhere else — moves the modes off the oracle's grid
and the exactness anchor goes from `1.44474e-12` to O(1). The exemption is a
property of the limit. "Fixing" it changes the operator being compared.

## P3 — patch centres are snapped to the sample grid, over the dilated aperture

Two parts, both load-bearing.

**Snapped.** A continuous centre injects a sub-sample linear phase that no other
patch corrects, so the coherent sum converges to the wrong thing — cleanly, which
is why the resulting plateau (historically ~0.28) reads as a legitimate floor
rather than as a bug. This is a declared negative control on `B2-EQUIV`.

**Dilated.** The centre set covers the aperture *dilated by half a patch*, so
edge patches have a valid centre. Without the dilation the coverage is quietly
wrong at the boundary. `plan.centers_xy_m` is in **metres**, not pixels.

## P4 — the coverage correction is `A_draw / A_patch`, and its direction matters

A drawn subset of centres covers a fraction of the aperture, and the sum is
corrected by **drawn area over patch area** — `plan.coverage`, which is
`draw_positions / patch_px**2`. `patch_secondary_rays` multiplies the emitted
amplitude by it exactly once.

This is the convention most likely to be inverted without consequence, because it
is **exactly 1 at one full-aperture patch**. An inverted correction
(`A_patch/A_draw`) is a no-op there and passes every full-aperture test — which
is not hypothetical; it is how the real inversion survived here. A scale-invariant
metric (NCC) is blind to it by construction. Compare **power**.

`plan_patches` will not guess the direction for caller-supplied centres: the
correction is only unbiased for centres drawn uniformly over the *dilated*
aperture, and the density is not recoverable from the positions alone, so
`coverage_basis` is a required declaration rather than an inference.

## P5 — the launch phase is per patch, applied once, and NOT applied here

Patch `j`'s window offset `c_j` carries `exp(i k · c_j)`. For a single patch
centred on the origin `c_1 = 0`, so double-counting it is also a no-op at full
aperture — the same failure class as P4, with the same remedy: gate it on a
sub-aperture case.

**Where it is applied is the part that gets got wrong.** `couplers/patch.py`
applies no launch phase at all. The inter-patch phase relationship is carried by
`C_RAY_TO_WAVE`'s per-ray `Δr` ramp, because each emitted ray already knows the
position it launched from. So an explicit `exp(i k · c_j)` added in the emitter is
not a missing term — it *is* the double-count. `patch.py` calls this the thing
about that module most likely to be got wrong, and it is why this entry names the
applier rather than only the factor.

## P6 — there is no window taper, and adding one is not a refinement

The windows are indicators. `sum_j w_j = 1` is what makes the patch sum equal the
transmitted field, and any window below 1 removes field no other patch replaces.
So a taper does not smooth the convergence — it breaks the argument the
convergence rests on. Applying one is a declared negative control.

## P7 — patch granularity is a `RepresentationParameter`

It is *declared* not to change the answer beyond a stated budget. That is what
makes sweeping it a benchmark rather than a comparison: measuring how much a
parameter that should not change the answer does is itself the measurement, and
movement past the budget is a defect.

Expect roughly `P^-1/2` on a drawn decomposition and round-off on an enumerated
one. `B2-EQUIV` declares the tolerance and its basis.

## P8 — a score is not defined until the comparison grid is

Two distinct forms of the same rule, and the second is the one that surprises
people.

**The oracle's padding.** A discrete ASM is periodic. The same route scores
`8.8e-3` against a pad-200 oracle and `0.33` against a pad-101 one; neither is an
error in either implementation, and both are wraparound between two periods.
Every reported score names its oracle's pad.

**The comparison plane.** On a sub-aperture decomposition the patch pad grid is
not commensurate with the reconstruction grid, so at a large propagation distance
the ray sum is the *non-periodic* propagated field while the ASM is the
*periodic* one. Measured: `1.739e-15` at the DOE plane and `0.84` at
z = 1.26 mm, with **neither route wrong**. Choose the plane, not only the oracle.

## P9 — the substrate must be planar, and the bound for sizing one is stated

`Substrate.CONFORMAL` is refused with `MISSING_DECLARATION`. There is no common
plane to accumulate onto, so the operator is undefined rather than approximate.
To decompose a curved surface, size the patch with
`couplers/curvature.py`'s `eps_curv ≤ arcsin(D/2R)` — a bound that is independent
of the DOE phase profile.

## P10 — CPU and FP64 only, and that is a measurement gap not a preference

`core/capabilities.py::C_PATCH_WFT_CAPABILITIES` declares CPU/FP64 because no
CUDA or JAX path through this coupler has ever executed. Do not widen it from
this card.

Worth knowing where the time actually goes: the expensive half is the
`O(rays × pixels)` reconstruction inside `C_RAY_TO_WAVE`, which *is*
xp-parameterized and does run on CUDA. The patch transform is
`O(patches × pad² log pad)` and is not the bottleneck. If a device path is added,
it is measured first and declared after.
