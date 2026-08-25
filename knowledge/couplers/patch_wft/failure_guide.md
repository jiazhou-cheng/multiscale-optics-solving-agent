# C_PATCH_WFT — keyed by symptom

Every entry starts with what you would actually *see*, because that is what you
have when something is wrong. The cause comes second. An agent debugging a
convergence sweep that stops at 0.28 does not yet know it is looking at a
sub-sample linear phase.

Anything the composed couplers refuse, this coupler refuses:
`knowledge/couplers/ray_to_wave/failure_guide.md`,
`.../wave_to_ray/failure_guide.md` and `.../planar_doe_step/failure_guide.md`
bind here unchanged and are not repeated.

---

## Symptom index

| What you see | Go to |
| --- | --- |
| A refusal naming an odd size when you asked for an even one | [R1](#r1) |
| `plan.pad_px` is not the pad you asked for | [R2](#r2) |
| A refusal about the substrate | [R3](#r3) |
| The convergence sweep **plateaus** instead of falling | [S1](#s1) |
| More patches makes the answer *worse*, or no better | [S2](#s2) |
| The full-aperture anchor reads O(1) instead of ~1e-12 | [S3](#s3) |
| The same route scores 8.8e-3 one way and 0.33 the other | [S4](#s4) |
| Everything is exact at the DOE plane and ~0.84 at the sensor | [S5](#s5) |
| Sub-aperture is off by a constant factor; full aperture is fine | [S6](#s6) |
| A patch-count sweep changes the answer, and it is not supposed to | [S7](#s7) |
| Two of your negative controls will not fire | [S8](#s8) |
| Two routes agree beautifully and you want to call it validated | [S9](#s9) |
| The process is killed, or the machine starts swapping | [S10](#s10) |

---

## Refusals

These raise. They are the easy ones, and they are listed so a caller can tell a
refusal from a silent wrong answer.

### R1 — you asked for an even `patch_px` {#r1}

**Symptom.** `SHAPE_MISMATCH`, with a remedy naming the nearest odd sizes
(`Use 5 (or 3)`).

**Cause.** An even patch has no centre sample, so "centred on a ray" is
undefined, and the odd-pad and even-`(pad − patch)` rules become jointly
unsatisfiable.

**Why it is refused and not rounded.** The paper's own patch sizes are 40, 50 and
100 — all even — so an even request is the *likely* one, not the exotic one.
Rounding it silently hands back a different operator than the caller asked for.
An earlier `resolve_pad_px` did not refuse and looped forever instead.

**Do.** Pick the odd size the remedy names. Do not "fix" this by making the
resolver round.

### R2 — the pad you get is not the pad you asked for {#r2}

**Symptom.** `plan.pad_px` is 21 when you passed `pad_factor=1` on a 15-px grid
with `patch_px=5`. No error.

**Cause.** This is correct behaviour. `pad_factor` is a *preference*; the step
raises the pad until clearance, centring and oddness all hold, then reports what
it used. On that configuration `pad_factor` 1, 2 and 3 all derive 21.

**Do.** Read `plan.pad_px`. A caller that assumes its factor was honoured is
reasoning about a grid that does not exist — and a pad that actually violated
clearance produces a plausible field wrong by 100%.

### R3 — conformal substrate {#r3}

**Symptom.** `MISSING_DECLARATION` on `Substrate.CONFORMAL`.

**Cause.** A conformal substrate has position-dependent tangent frames and
normals, so there is no common plane to accumulate onto. The operator is
*undefined* there, not a worse approximation.

**Do.** Size a planar patch with the `couplers/curvature.py` bound
`eps_curv ≤ arcsin(D/2R)` — which is independent of the DOE phase profile — and
decompose. Do not widen the substrate declaration.

---

## Silent failures — it runs, and the answer is wrong

These are the entries worth loading this pack for. None of them raises, and all
of them produce a plausible field.

### S1 — the convergence sweep plateaus {#s1}

**Symptom.** The residual falls for the first few patch counts and then stops,
sitting at some fixed value (historically ~0.28) no matter how many more patches
you add.

**Cause.** Patch centres are not snapped to the sample grid. A continuous centre
injects a sub-sample linear phase that no other patch corrects, so the coherent
sum converges to the wrong thing — and it converges to it cleanly, which is why
the plateau looks like a legitimate floor.

**Detect.** Compare `plan.centers_xy_m` against integer multiples of the sample
pitch. This is a declared negative control on `B2-EQUIV`
(`grid-snapping-is-not-free`) and it fires on the enumerated instance.

**Do.** Snap centres to the sample grid, over the aperture *dilated* by half a
patch. The dilation is not optional: without it the edge patches have no valid
centre and the coverage is quietly wrong.

### S2 — more patches does not help {#s2}

**Symptom.** The residual is flat or rising across a patch-count ladder.

**Cause, most likely.** An apodization taper. Any window below 1 removes field
that no other patch replaces, so the partition-of-unity argument behind the
convergence relation is exactly what a taper breaks.

**Detect.** This coupler carries no apodization by design. If a taper has been
added anywhere in the path, the apodized ladder does not reach the correct
ladder's finest rung — that comparison is recorded in `B2-EQUIV`'s
`APODIZATION_BREAKS_THE_CONVERGENCE` diagnostic.

**Cause, second most likely.** S1. Check the centres first; it is cheaper.

**Do.** Remove the taper. A taper is not a refinement here.

### S3 — the full-aperture anchor is O(1) {#s3}

**Symptom.** One patch over the whole aperture, and the residual against the
float64 ASM reads 0.57 or similar instead of ~1e-12.

**Cause.** The full-aperture patch was padded. At one full-aperture patch,
`pad_factor=1` is the condition under which the patch mode set and the *unpadded*
oracle's mode set are the same set. Padding moves the modes off the oracle's grid.

**Do.** Keep the clearance exemption. It is a property of the limit, not a
relaxation, and "fixing" it by padding changes the operator being compared. The
measured anchor with the exemption is 1.44474e-12.

### S4 — the same route scores two very different numbers {#s4}

**Symptom.** 8.8e-3 against one oracle and 0.33 against another, with nothing
changed in the route.

**Cause.** The oracles had different padding. A discrete ASM is *periodic*, so a
score against one is undefined until its padding is stated. Neither number is an
error in either implementation; both are wraparound between two periods.

**Do.** Name the oracle's pad in every reported score. If you cannot say what pad
your oracle used, you do not have a measurement.

### S5 — exact at the DOE plane, ~0.84 at the sensor {#s5}

**Symptom.** An enumerated sub-aperture sum reproduces the field to 1.7e-15 at
z = 0 and disagrees with the independent ASM by 0.84 at z = 1.26 mm.

**Cause.** Commensurability, and **neither route is wrong**. A sub-aperture
patch's modes live on its own pad-21 grid, which is not commensurate with the
15-px reconstruction grid, so the ray sum is the *non-periodic* propagated field
while the ASM is the *periodic* one. They differ by the wrapped contributions. At
full aperture with `pad_factor=1` the two mode sets coincide exactly, which is
why the anchor *can* read 1.4e-12 and this comparison cannot.

**Do.** Choose the comparison plane, not only the oracle. A sub-aperture score at
large z against a periodic oracle is measuring commensurability, not the coupler.
This is S4 in a third coordinate: a score is not defined until the oracle's grid
is.

### S6 — sub-aperture is off by a constant factor {#s6}

**Symptom.** The sub-aperture sum has the right structure and the wrong scale.
Full aperture is fine. An NCC or any scale-invariant metric reads ~1.

**Cause.** The coverage correction — most likely inverted. The shipping
correction is `A_draw / A_patch` (`plan.coverage` = `draw_positions / patch_px²`,
applied once in `patch_secondary_rays`), so the inversion is `A_patch / A_draw`
and it shows up as a relative error of `|1/coverage² − 1|` — measured 0.995 on
the enumerated sub-aperture instance.

**Why it survives.** The correction is *exactly 1* at full aperture, so every
full-aperture test passes with it inverted. That is not hypothetical; it is how
the real inversion survived in this repository.

**Detect.** Compare power, not correlation. `B2-EQUIV`'s
`omit-coverage-correction` control fires on the enumerated sub-aperture instance
and is reported as a measured blindness — not a control — on the anchor.

### S7 — patch count changes the answer {#s7}

**Symptom.** Sweeping patch granularity moves the result by more than the
declared budget.

**Cause.** Patch granularity is a `RepresentationParameter`: it is *declared* not
to change the answer beyond a stated budget. Movement past that budget is a
defect, and measuring how much a parameter that should not change the answer does
is itself the benchmark.

**Do.** Read `B2-EQUIV`'s tolerance and its basis before deciding whether what
you are seeing is the budget or a bug. On a drawn decomposition, expect roughly
`P^-1/2` Monte Carlo behaviour; on an enumerated one, expect round-off.

### S8 — two controls will not fire {#s8}

**Symptom.** The coverage-correction and launch-phase controls report no
separation from the correct arm.

**Cause.** You are running them at full aperture, where both are inert: coverage
is exactly 1 and a patch centred on the origin has zero launch offset. A control
whose term is inert reports green and proves nothing.

**Do.** Run them on a sub-aperture case. This is why `B2-EQUIV` gates exactness
on the anchor and the controls on the sub-aperture instances, and why the anchor
records the inert mutations' values as a measured blindness.

### S9 — two routes agree, and you want to call that validation {#s9}

**Symptom.** RW-F and RW-P agree at NCC 0.9994 and it is tempting to gate on it.

**Cause.** A `CROSS_ROUTE` oracle forces category B4, and a B4 family may not
carry a gating tolerance. Two of our own routes agreeing is not evidence either
is right — if they share a convention error they agree *perfectly*.

**Do.** Report it as characterization and put the independent number beside it.
On demo2 the sub-aperture route against the independent float64 ASM reads
0.99941823 against the 0.99941808 cross-route number, and *that* coincidence is
the useful finding: the routes agree at exactly the level at which each
independently matches an oracle, so their agreement is not hiding a shared error.

For a system with no independent oracle at all — demo3, where the paper states no
conventional reference exists — use the noise-limited relation
`NCC(A,B) ≈ √(NCC(A,A′)·NCC(B,B′))`. Predicted 0.012943, measured 0.014713, ratio
1.1367. A ratio near 1 says the disagreement is the Monte Carlo noise each route
carries; a ratio well below 1 is a systematic difference between the routes.
It is still characterization.

### S10 — the machine starts swapping {#s10}

**Symptom.** A patch enumeration is killed, or system RAM collapses.

**Cause.** The reconstruction is `O(rays × pixels)` and the separable contraction
allocates `rays × n` factors. Enumerating a 33-px grid is 3.7 M rays and about
4 GB in one call — which pushed this shared machine into swap while CHE-96 was
being written.

**Do.** Keep gate-path tests small: `tests/test_patch_wft.py` is on a 15-px grid
(159 k rays, ~76 MB) for exactly this reason. Cost curves belong in a probe.
This is a shared server; see `AGENTS.md` for the memory-safety policy.
