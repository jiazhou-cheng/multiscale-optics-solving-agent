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
