# C_PLANAR_DOE_STEP — conventions at the boundary

The two halves' conventions are **not** restated here. Read
`knowledge/couplers/ray_to_wave/conventions.md` for the accumulation and
`knowledge/couplers/wave_to_ray/conventions.md` for the resampling; both bind
unchanged. What follows is only what the composition adds.

## D1 — the declared plane is an input, not a choice

`config['plane_z_m']` is required. The step accumulates onto a plane the caller
declares and never picks one, because picking one would make the transmission's
location implicit.

**The step does not propagate the bundle to that plane.** It accumulates the
bundle exactly as given. If the incident bundle is referenced elsewhere, the
transmission is measured at the wrong place and nothing refuses it — the
reference-plane metadata is checked for consistency, not enforced by
propagation. Advance the ray state first: positions along each direction,
optical path by `n * arc length`. That advance is exact, and
`knowledge/couplers/ray_to_wave/card.yaml` derives why.

## D2 — the outgoing OPL reference is this plane, and it is zero

Outgoing rays carry `optical_path_length = 0`, with the reference rebased to the
declared plane.

This is the convention most likely to be got wrong silently, because the
alternative — carrying the incident OPL forward — produces a field that still
looks like a diffraction pattern. The error scales with the incident path length,
so it presents as a defocus that moves when the source moves rather than as an
obvious artifact.

Across two stacked steps the reference is rebased twice. An absolute OPL is
therefore meaningful only relative to the most recent step.

## D3 — the outgoing amplitude is a spectral amplitude

`U~[m]/p[m]`, not a transformed incident weight.

Two consequences a caller must hold:

* **No per-ray correspondence survives.** Incident ray `i` does not become
  outgoing ray `i`. There are deliberately different numbers of them, and code
  that indexes across the step by ray identity is wrong even when the counts
  coincide.
* **The importance division is load-bearing.** Dropping `1/p[m]` leaves every
  shape intact and multiplies the answer by the sampling density — largest
  exactly where the spectrum is peaked. `importance_weight_applied` is the
  invariant that catches it.

## D4 — power is not conserved, and that is the default

`preserve_energy=False`. A lossy DOE legitimately loses power, and renormalizing
to the incident power hides precisely the case a conservation check exists to
catch.

When `preserve_energy=True` the applied factor `sqrt(P_in / P_out)` is reported
in the diagnostics, so a record always shows that a renormalization happened
rather than presenting a normalized number as a measured one.

Evanescent bins are accounted for separately (`evanescent_power_accounted`), so a
power ledger balances without attributing evanescent content to loss.

## D5 — the transmission must be complex

A real array is an amplitude mask with an undeclared phase, not a transmission,
and it is refused (`MISSING_DECLARATION`) rather than promoted with an assumed
zero phase. Its shape must equal the plane grid exactly; a mismatch is
`SHAPE_MISMATCH`.

## D6 — sampling is declared, never inferred

* Exactly one of `launch_positions_xy_m` or `primary_sampling`. Supplying both
  is a **conflict**, refused, not resolved by precedence — a precedence rule
  would make one of the caller's two statements silently ineffective.
* `primary_sampling` requires `primary_count`.
* Any draw requires an explicit `seed`. The protocol requires an explicit seed
  rather than an implicit one, so a stochastic result is always reproducible from
  its record.
* `secondary_count=None` enumerates; `>= 2` samples; `<= 1` collapses to one ray
  on the power-weighted mean wavevector and is reported as
  `collapsed_to_mean_wavevector` so it cannot be mistaken for a converged result.

## D7 — capability is the intersection, and the registry is not the authority on it

`core/capabilities.py::C_PLANAR_DOE_STEP_CAPABILITIES` is authoritative. The step
accumulates through `couplers/ray_to_wave.py` and resamples through
`couplers/wave_to_ray.py`, so any device or dtype either half refuses is refused
here. CUDA is declared with a JAX namespace only; CPU accepts NumPy or JAX.

No CUDA execution of this coupler is covered by the default gate. The device
evidence that exists (`benchmarks/probes/records/planar_doe_step_device.json`) is
a CROSS_ROUTE / SHARES_CODE characterization and cannot decide correctness.
