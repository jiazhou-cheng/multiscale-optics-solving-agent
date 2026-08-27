# The physics behind C_GENERALIZED_SNELL (CHE-143, M2.7)

## Tangential momentum matching

A diffractive surface with a spatially varying phase profile `phi(x, y)` adds a
position-dependent tangential wavevector to whatever crosses it. For an
incident ray of direction cosines `d_in` in a medium of index `n_i`, meeting
the surface at `(x, y)`, requesting diffraction order `m`:

```
k_t^out = n_i k0 d_t^in + m grad_t(phi)(x, y)
k_n^out = sqrt( (n_t k0)^2 - |k_t^out|^2 )
```

`k0 = 2 pi / lambda` is the free-space wavenumber, `n_t` the transmitted-side
index, and `grad_t` the in-plane (transverse) gradient of the phase. This is
the standard generalized Snell's law / local grating equation -- see
`source_manifest.yaml` for provenance; it is textbook physics, not this
repository's own derivation, and not the ray-wave paper's method.

## Why it reduces to what it should

- **Zero phase (`phi = 0` everywhere):** `grad_t(phi) = 0`, so
  `k_t^out = n_i k0 d_t^in`. Converting back to angles,
  `n_i sin(theta_in) = n_t sin(theta_out)` -- ordinary Snell's law. Tested
  directly, not assumed, because the equation is the same equation with the
  order term dropped, and a sign error in that term would not show up here.
- **Zero gradient and equal indices:** `k_t^out = k_t^in` and the transmitted
  cone matches the incident one exactly, so `d_out = d_in`: undeflected
  propagation. Also tested directly.
- **A linear phase ramp** `phi = 2 pi x / Lambda`: `grad_t(phi)` is the
  constant vector `(2 pi / Lambda, 0)` everywhere, so every ray is deflected to
  exactly `sin(theta_out) = (n_i sin(theta_in) + m lambda / Lambda) / n_t` --
  the one configuration where this model is not an approximation of anything,
  because a blazed grating genuinely does put (nearly) all its power in one
  order. This is the acceptance case, and it holds to floating-point round-off
  regardless of pixel pitch -- see `conventions.md` for why.

## What the model is a reduction of

`FULL_FIELD` and `LOCAL_PATCH` both form an actual field (globally or per
patch) and read off however many diffraction orders the surface actually
populates. This model assumes -- and its validity predicates measure whether
the assumption holds -- that exactly one order carries essentially all the
local power, so the field-forming step can be skipped entirely and the ray
simply redirected by the local grating vector. The assumption is testable
(`single_order_dominance`) and is exactly what fails once the surface's local
phase content spreads across more than one order (see `card.yaml`'s validity
section and the `disagrees_outside_the_smooth_limit` test).

## What this model does not claim

- **No amplitude physics beyond a declared complex transmission.** The
  outgoing amplitude is `|t(x, y)|` times the incident amplitude. There is no
  polarization, no vector diffraction efficiency model, no Fresnel-coefficient
  physics at the interface itself.
- **No multi-order emission.** One `m` per call. A metasurface that genuinely
  splits power across several orders needs several calls, one per order, each
  with its own declared amplitude scaling -- this model does not derive that
  scaling for you.
- **No gradient/autodiff claim.** See `card.yaml`'s `derivative` section.
