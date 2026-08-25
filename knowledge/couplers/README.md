# Coupler knowledge packs

A coupler changes *representation*, so it carries physical assumptions that
belong to neither solver it joins. Each direction therefore gets its own pack,
in the same shape as a `knowledge/solvers/<name>/` pack:

```
<direction>/
  card.yaml     routing-critical facts, validation status, what is NOT verified
  theory.md             the governing equations, transcribed with their source labels
  conventions.md        units, axes, frame, phase sign, normalization, reference plane
  failure_guide.md      how it breaks, what the diagnostic must say
  source_manifest.yaml  per-source provenance
  probes/               executable checks
```

| Pack | Coupler ID | Transformation |
|---|---|---|
| [`ray_to_wave/`](ray_to_wave/) | `C_RAY_TO_WAVE` | ray bundle → complex field on a declared plane |
| [`wave_to_ray/`](wave_to_ray/) | `C_WAVE_TO_RAY` | complex field → ray bundle carrying complex amplitudes |
| [`planar_doe_step/`](planar_doe_step/) | `C_PLANAR_DOE_STEP` | ray bundle → ray bundle across one planar DOE |
| [`patch_wft/`](patch_wft/) | `C_PATCH_WFT` | ray bundle → ray bundle, per patch, summed coherently |

The last two are **composed** from the first two and from each other:
`C_PLANAR_DOE_STEP` is `C_RAY_TO_WAVE` then a transmission then `C_WAVE_TO_RAY`,
and `C_PATCH_WFT` is that cascade applied per patch — with the global step being
its special case at one full-aperture patch (SI S10), not a peer of it. Their
packs therefore *point* at what they compose rather than restating it, and their
capabilities are the **intersection** of what they compose. A composed pack that
duplicated a convention would create a second place to update, and the two would
drift.

Both are derived from
[`knowledge/papers/raywave_tracing/`](../papers/raywave_tracing/) (Cheng et al.,
ACS Photonics 2026, DOI `10.1021/acsphotonics.6c00818`). The paper's reference
implementation is **not vendored, pinned, or executed** by this repository, so
nothing here may cite it as evidence — only the paper's mathematics, and this
repository's own probes.

## The one-sentence version

A ray is a plane wavelet carrying a direction `d̂` and a phase `exp(+i k·OPL)`.
`C_RAY_TO_WAVE` sums those wavelets coherently onto a plane; `C_WAVE_TO_RAY`
decomposes a field into its propagating plane-wave modes and emits a Monte
Carlo sample of them as rays. The two directions are inverses of each other in
the limit of complete sampling, which is what makes a round trip a usable test.

## Reading order

1. `theory.md` — what the transformation is.
2. `conventions.md` — what the symbols mean in this repository's frame. This is
   where a coupler goes wrong in practice, not in the algebra.
3. `card.yaml` — what has actually been verified, and what has not.
4. `failure_guide.md` — before debugging a surprising result.

## What does not survive a representation change

This is a deliverable, not a caveat (CHE-112 / `B2-ROUNDTRIP`). **Every coupler
in this repository is declared `lossy: true`**, and an agent reasoning about a
chain of representations needs to know which quantities it may still ask about
downstream — because the answer to "what happened to my ray" is sometimes *there
is no such question any more*, and nothing raises when you ask it.

The table below is prose for the executable one:
`src/verification/families/b2_transitions.py::WHAT_DOES_NOT_SURVIVE`, which
`tests/test_b2_families.py` holds against the registry so a newly registered
coupler cannot arrive without a discard statement. Read that as the source of
truth; read this for the reasoning.

### `C_RAY_TO_WAVE` — ray bundle → field

* **Per-ray identity.** The field is an *accumulation*. No output sample
  corresponds to any one input ray, and there is no index that maps back.
* **The rays themselves.** Direction, position and optical path are consumed and
  are not recoverable from the field. A field plus a wish to know which ray
  contributed most is not a well-posed query.

*What does survive, and is asserted:* `phase_reference_consistency` and
`pupil_power_consistency`.

### `C_WAVE_TO_RAY` — field → ray bundle

* **Evanescent power**, discarded at the light cone. It is **accounted** rather
  than ignored: the discarded fraction is reported, and
  `evanescent_power_accounted` asserts that propagated + discarded = input.
  A coupler that silently dropped it would be indistinguishable from one that had
  no evanescent content.
* **Everything the Monte Carlo sample did not draw**, up to the variance the
  family measures. Enumerating every propagating bin is the exactness limit and
  has no sampling error at all; any smaller draw discards the rest, and *how
  much* is a measured quantity rather than an unknown.
* **The phase of the unsampled spectrum.** Not merely its amplitude — this is why
  a magnitude-sampled draw is not a low-pass version of the field.

*What does survive:* `evanescent_power_accounted`,
`importance_weight_applied`, `unit_direction_norm`.

### `C_PLANAR_DOE_STEP` — rays → rays across one planar DOE

* **The incident OPL reference.** Optical path is **rebased to zero** at every
  step, because the incident path is already inside the accumulated field's
  phase and carrying it forward would double-count it. So an absolute path length
  from before the step is *not comparable* with one from after it, and across two
  stacked steps the reference is rebased twice. This one is dangerous because the
  wrong version still looks like a diffraction pattern: the error scales with the
  incident path length, so it presents as a defocus that moves when the source
  moves.
* **Per-ray correspondence.** The outgoing amplitude is a spectral amplitude
  `U~[m]/p[m]`, not a transformed incident weight. Incident ray `i` does not
  become outgoing ray `i`, and there are deliberately different numbers of them.
  Code that indexes across the step by ray identity is wrong *even when the
  counts coincide*.

*What does survive, and is the reason this coupler exists:*
`outgoing_count_is_the_budget`. The outgoing count is the caller's `P × S`
budget and not a function of the incident count, so two planar DOEs in series
give 256 then 256 rays rather than 256 × 64. Without that, a multi-element
diffractive system is combinatorially unrunnable — which makes it a
*composability* invariant rather than a per-coupler one.

### `C_PATCH_WFT` — rays → rays, per patch

* **The global phase reference between patches**, as a *per-patch* quantity. What
  actually carries it is `C_RAY_TO_WAVE`'s per-ray `Δr` ramp — each emitted ray
  knows the position it launched from — and `patch.py` deliberately applies no
  launch phase of its own. So the phase relationship between patches is preserved
  through the rays, not through the coupler, and adding an explicit
  `exp(i k · c_j)` in the emitter double-counts it. The **amplitude** correction
  `A_draw / A_patch` is a real positive scalar and restores no phase at all.
* **Any structure finer than the patch**, which the tangent-plane approximation
  cannot represent.

*What does survive:* `patch_coverage_corrected`, plus everything the cascade
carries.

### The rule this table exists to support

A round trip through two of these couplers can look perfect while a shared
convention error cancels between the two directions. So: **a successful round
trip is not accepted unless a deliberately broken twin demonstrably fails**, and
that is enforced by the schema rather than by convention — `B2-ROUNDTRIP` makes
"the broken twin ran" a validity predicate, so a round trip that cannot be made
to fail reports `OUT_OF_VALIDITY` instead of a pass.

Measured, both directions at the enumeration limit: `5.31e-16` and `5.26e-16`,
against `1.414` for the mismatched-phase-sign twin and the transposed-axis twin.
The 1.414 is `sqrt(2)` — the exact distance between a field and its conjugate —
and getting it required replacing the original centred *real* Gaussian probe,
for which **both twins were exactly no-ops** and read identically to the correct
arm.
