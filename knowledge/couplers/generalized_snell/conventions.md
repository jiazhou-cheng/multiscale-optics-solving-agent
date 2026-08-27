# Conventions C_GENERALIZED_SNELL relies on or introduces

## Inherited from `core.boundary`, unchanged

- **Phasor:** `exp(-i omega t)`; the surface is built with `t = exp(+i phi)`
  via `DiffractiveSurface.from_phase`, the one place that sign is applied.
- **Axis order / origin:** the transmission grid is `(ny, nx)` with coordinate
  zero at index `n // 2`, matching `ComplexField.coordinates` and
  `couplers.patch.extract_patch`'s nearest-sample rule, which this model reuses
  for locating a ray's own sample.
- **Directions are unit vectors, always.** `RayBundle.__post_init__` enforces
  `|d| = 1` regardless of medium. The physical wavevector in a medium of index
  `n` is `n * k0 * d`, which is exactly the `n_i k0 d_t^in` term in the
  governing equation -- there is no separate "wavevector" representation to
  reconcile.

## What this model introduces

### The phase gradient is read from the complex transmission directly, not from `angle(t)` unwrapped and differenced

`d phi / du` is estimated as `angle(t[+1] * conj(t[-1])) / (2 du)` at the ray's
nearest sample, not by unwrapping `angle(t)` over the whole grid and
differencing the unwrapped array. The two methods agree wherever unwrapping
would succeed. The complex form is used because:

- it needs no global unwrap pass, so a ray's gradient estimate depends only on
  a small local stencil, matching the model's own "one ray, one local
  evaluation" identity;
- for a genuinely linear phase ramp it is **exact to round-off at any pixel
  pitch** -- `angle()` of a product of two exact unit-modulus complex numbers
  returns the true angle difference via `atan2`, with no dependence on how
  many samples the baseline spans, which is what the linear-ramp acceptance
  case needs.

It fails the same way any finite-difference phase gradient does: silently,
when the true phase change across the sampling baseline exceeds `pi`. See
`failure_guide.md`.

### The local phase value used for the OPL contribution is the wrapped principal value, not an unwrapped one

`phi(x, y) = angle(t(x, y))`, mod `2 pi`. This is not a shortcut -- a
phase-only diffractive surface (a real blazed grating, a Fresnel lens, a
metasurface phase mask) only ever specifies its phase mod `2 pi` in the first
place; the physical element does not know or care which `2 pi n` an unwrapped
convention would have added. Unwrapping here would invent information the
surface does not carry, not recover information it does.

### The OPL convention is additive, not reset-to-zero

`FULL_FIELD` and `LOCAL_PATCH` both reset outgoing OPL to zero at the
interaction plane, because both re-emit fresh rays from a reconstructed
spectrum and the phase reference has to rebase there. This model never
reconstructs anything -- each outgoing ray *is* the same incoming ray,
redirected -- so its own OPL history is preserved and the surface's local
contribution is added:

```
OPL_out = OPL_in + phi(x, y) / k0
```

under the repository's `exp(+i k z)` spatial-phase convention, so that
`exp(i k0 OPL_out)` picks up exactly the `exp(i phi)` the transmission declared.
Amplitude carries only `|t(x, y)|` -- the phase does not appear twice.

### `patch_px` is one declared transverse scale, reused for two purposes

`GeneralizedSnellParameters.patch_px` sizes both the local window
`single_order_dominance` transforms (same role as `LocalPatchParameters.patch_px`)
and, through `patch_px * sqrt(dy * dx)`, the transverse scale the
gradient-smoothness predicate checks local phase curvature against. This is
one declaration, not two dials that could disagree.

### The declared refractive indices execute here, and only here

`DiffractiveSurface.n_incident` / `n_transmitted` are refused by `FULL_FIELD`
and `LOCAL_PATCH` at dispatch (their transmission-and-transform math has no
interface factor); this model is the one place a declared index other than 1.0
is used, directly, in the governing equation.
