# Coupler knowledge packs

`src/couplers/` holds **three kinds of operation**, and CHE-142 (M2.6) made the
distinction explicit here as well as in the code, because a pack that calls all
three "a coupler" cannot tell an agent which of the three it is reading about:

> **representation transition ≠ diffractive physical interaction ≠ propagation.**

* A **representation transition** changes what the light is *described by* —
  rays become a field, or a field becomes rays. `C_RAY_TO_WAVE`,
  `C_WAVE_TO_RAY`. These carry physical assumptions that belong to neither
  solver they join, which is why each direction gets its own pack.
* A **diffractive interaction** is *physics at a surface*: incident coherent
  rays meet a diffractive surface and coherent rays come out. It is **one**
  operation, and `C_PLANAR_DOE_STEP` and `C_PATCH_WFT` are **two granularities
  of it**, not two unrelated DOE steps — see [the interaction](#the-diffractive-interaction-and-its-models)
  below. They each contain two representation transitions plus a transmission,
  and that is their implementation rather than their identity.
* **Propagation** moves an existing representation between planes and changes
  neither the representation nor the physical content.
  `couplers/propagation.py::advance_bundle_to_plane`. It has no pack, because it
  introduces no convention the boundary artifacts do not already declare —
  which is itself the point: it is not a coupler and does not need one.

The executable form of that partition is `src/couplers/ontology.py`, held
against the registry and the package's exports by
`tests/test_diffractive_interaction.py`. Read that as the source of truth; the
prose here is the reasoning.

Each pack has the same shape as a `knowledge/solvers/<name>/` pack:

```
<direction>/
  card.yaml     routing-critical facts, validation status, what is NOT verified
  theory.md             the governing equations, transcribed with their source labels
  conventions.md        units, axes, frame, phase sign, normalization, reference plane
  failure_guide.md      how it breaks, what the diagnostic must say
  source_manifest.yaml  per-source provenance
  probes/               executable checks
```

| Pack | Coupler ID | Role | Transformation |
|---|---|---|---|
| [`ray_to_wave/`](ray_to_wave/) | `C_RAY_TO_WAVE` | representation transition | ray bundle → complex field on a declared plane |
| [`wave_to_ray/`](wave_to_ray/) | `C_WAVE_TO_RAY` | representation transition | complex field → ray bundle carrying complex amplitudes |
| [`planar_doe_step/`](planar_doe_step/) | `C_PLANAR_DOE_STEP` | diffractive interaction, model `full_field` | rays + surface → rays, one global field |
| [`patch_wft/`](patch_wft/) | `C_PATCH_WFT` | diffractive interaction, model `local_patch` | rays + surface → rays, per patch, summed coherently |
| [`generalized_snell/`](generalized_snell/) | `C_GENERALIZED_SNELL` | diffractive interaction, model `generalized_snell` | rays + surface → rays, no field ever formed |

## The diffractive interaction and its models

The bottom three rows are **one interaction**, `I_DIFFRACTIVE`, at three
granularities. All three registry rows declare that shared identity, and each
says how it relates to the others:

| Model | Where the field is formed | Regime |
|---|---|---|
| `full_field` | once, globally, on the one common plane | planar substrate, and a global field that fits in memory |
| `local_patch` | per patch, on its own local tangent plane | large or conformal surfaces; also any planar surface whose global field does not fit |
| `generalized_snell` | nowhere — no field is formed | reduced-order: one ray redirected by the local grating equation. Planar substrate only. CHE-143 (M2.7); see `knowledge/couplers/generalized_snell/` |

**`full_field` is the shortcut, not a peer.** SI S10 is explicit, about the
global aggregation `full_field` performs: *"For conformal DOEs, this global
aggregation before ray-DOE interaction is not applicable because rays intersect
different local tangent planes with position-dependent coordinate frames and
surface normals. We therefore retain the direct implementation."* The direct
implementation is `local_patch`; `full_field` is that model at **one
full-aperture patch**, and the identity is *measured* at 1.4e-12 relative field
error rather than asserted (`tests/test_patch_wft.py`).

The regimes **overlap** rather than partition. On a planar substrate both models
are valid and the choice is cost and variance, not correctness — SI Fig 3 gives
`local_patch` for large *planar* surfaces too, and SI Table S2 records the
4032×4032 Grating-Lens DOE as OOM on a 48 GB A6000 on the global route and
complete in 4.982 s at 11.492 GB on the patch route. On a conformal substrate
only `local_patch` is *applicable*, and `full_field` there is refused with
`MODEL_NOT_APPLICABLE` rather than approximated. `local_patch`'s conformal path
is itself refused with `MISSING_DECLARATION`, and the two codes must not be
collapsed: one says *never this model*, the other says *this model, once someone
builds it*.

One entry point performs the interaction with the model named explicitly, and
never inferred: `couplers.interaction.diffractive_interaction`.

### Why the two DOE packs point rather than restate

Both models are **composed** from the two representation transitions:
`full_field` is `C_RAY_TO_WAVE` then a transmission then `C_WAVE_TO_RAY`, and
`local_patch` is that cascade applied per patch. Their packs therefore *point* at
what they compose rather than restating it, and their capabilities are the
**intersection** of what they compose. A composed pack that duplicated a
convention would create a second place to update, and the two would drift.

Sharing an interaction identity does **not** share a capability. `C_PATCH_WFT`
is CPU-only and FP64-only and stays that way; grouping is an ontology statement,
and if it widened the narrower row the group would be advertising a CUDA patch
route that has never executed.

Both models are derived from
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

### `C_GENERALIZED_SNELL` — rays → rays, no field ever formed

* **Every diffraction order except the one requested.** One order per call, by
  construction: a caller wanting several says so several times, and any power
  a real surface would send to another order is simply not represented at all.
* **Any field, of any kind.** Unlike the other two models this one never forms
  a field even as an intermediate — there is nothing here to ask about the
  transmitted field's spatial structure beyond the single redirected ray's own
  direction, amplitude and phase.

*What does survive:* `unit_direction_norm`, and — unlike the other two models
— each incoming ray's own identity and OPL history, since this model redirects
existing rays rather than re-emitting fresh ones from a reconstructed
spectrum.

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
