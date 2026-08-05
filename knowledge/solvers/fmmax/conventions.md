# FMMAX conventions (pinned version `1.7.1`)

Every fact below was either read from `inspect.getdoc`/official docs, or
observed directly by running `knowledge/solvers/fmmax/probes/*.py` inside
the `agent_solver` container. None of it is copied from memory or from a
description of an older/different version.

## Units

FMMAX is **normalized-unit**, not literally SI: the official docs state
"the speed of light, vacuum permittivity, and vacuum permeability are all
1." Like Chromatix, it is scale-agnostic as long as `wavelength`, `dx`
(unit-cell vectors), and layer thicknesses are all expressed in the same
consistent length unit. **Implication for this project:** a future
`M_RCWA_FMMAX` adapter must convert the project's meter-valued
`period`/`wavelength` artifacts into whatever consistent scale is used for
a given call, and convert back on the way out (CLAUDE.md section 7). This
conversion boundary needs an explicit round-trip test, not an assumption.

## Time convention

Docs state the time-harmonic convention as `exp(-i*omega*t)`, matching this
project's canonical convention (CLAUDE.md section 7) as stated -- not
independently re-derived from source in this pass (unlike Chromatix, where
we derived the spatial sign convention from `inspect.getsource`). Treat
this as "stated by the vendor," not "independently verified," until a
manufactured traveling-wave test is added.

## Unit cell and grid

A unit cell is a parallelogram spanned by primitive lattice vectors `u`
and `v` (`fmmax.LatticeVectors(u=..., v=...)`); `fmmax.X`/`fmmax.Y` are the
standard axis-aligned unit vectors. Grid sample values are defined at the
half-cell offset `(du + dv) / 2` per the docs -- do not assume samples sit
at cell corners.

## Batching axis order (gotcha)

Per the official docs: most FMMAX quantities use **leading** batch axes,
but wave amplitudes/fields use a **trailing** batch axis instead. This is
an explicit exception to a single consistent axis-order rule -- a coupler
or adapter must check which category a given return value falls into
rather than assuming uniform placement. Not independently re-derived in
this pass; treat as vendor-stated pending a dedicated shape probe across
multiple return types.

## Scattering-matrix labeling (verified, and notably non-standard)

From `inspect.getdoc(fmmax.ScatteringMatrix)`, following [Whittaker 1999]:

```
a_N = s11 @ a_0 + s12 @ b_N
b_0 = s21 @ a_0 + s22 @ b_N
```

where `a` is forward-going and `b` is backward-going, `0` is the start
layer and `N` is the end layer. Concretely:

- **`s11`** = transmission (forward-going at start -> forward-going at end)
- **`s21`** = reflection (forward-going at start -> backward-going at start)

**This is the opposite of the common RF/photonic-circuit convention**
where `S11` is a reflection coefficient back into port 1 (see the official
docs' own warning: "notably different from photonic integrated circuit
conventions", also echoed in `docs/SOLVER_AND_COUPLER_CATALOG.md`'s
FMMAX/SAX comparison). A coupler bridging FMMAX's scattering matrix to
SAX's S-parameter dictionaries (which use `("port_in", "port_out")` tuple
keys, not `s11`/`s21` indices) must remap explicitly and test the remap,
not assume index correspondence.

## Reflection/transmission verified against an analytic oracle

`probes/fresnel_oracle_probe.py`: a bare ambient(n=1.0)/substrate(n=1.5)
interface at normal incidence, `approximate_num_terms=1` (homogeneous
limit, no real grating), gives:

| Quantity | FMMAX (`s21[...,0,0]`) | Analytic Fresnel |
|---|---|---|
| complex amplitude `r` | `+0.19999997 + 0.0j` | `-0.2` |
| reflectance `\|r\|^2` | `0.039999988` | `0.040000000` |

**Reflectance magnitude matches to ~3.0e-07 relative error** -- a genuine
independent analytic verification (CLAUDE.md section 3 rule 6). **The
complex amplitude's sign does not match the textbook convention**
(positive here vs. the textbook `r=(n1-n2)/(n1+n2)` being negative for
`n1<n2`). Do not assume FMMAX's phase reference agrees with a textbook or
another solver's phase convention -- reconcile and test explicitly before
coupling.

## Energy conservation: attempted, did NOT trivially close

Tried computing power transmittance from `s11` (`|s11|^2`, then scaled by
a naive index ratio `n_substrate/n_ambient`) and checking
`R + T_power == 1`. Result: `R=0.04`, naive `T_power=2.16`, sum `2.2` --
does **not** close to 1. `|s11|^2` alone was already `1.44`, i.e. greater
than 1, meaning FMMAX's raw modal amplitude is not a simple E-field
amplitude ratio and a naive index-ratio power correction is the wrong
formula. FMMAX exposes `fmmax.amplitude_poynting_flux` and
`fmmax.directional_poynting_flux` specifically for turning modal
amplitudes into physical power -- those, not raw `|amplitude|^2`, are
almost certainly the correct path to a closing energy-conservation check.
**Not yet done in this pass** -- do not claim energy conservation is
verified for FMMAX until a probe uses the Poynting-flux accessors.

## Energy conservation via Poynting flux: now closed (adapter work, 2026-07-30)

Follow-up to the "did NOT trivially close" section above, done while writing
`src/multiscale_optics_agent/adapters/fmmax_adapter.py`. Using
`fmmax.directional_poynting_flux` (not `fmmax.amplitude_poynting_flux`) on
the physical amplitude vectors, with the s11/s21 swap applied:

```python
a_N = s_matrix.s11 @ a_0 + s_matrix.s12 @ b_N   # transmission
b_0 = s_matrix.s21 @ a_0 + s_matrix.s22 @ b_N   # reflection
fwd_start, bwd_start = fmmax.directional_poynting_flux(a_0, b_0, s_matrix.start_layer_solve_result)
fwd_end, bwd_end = fmmax.directional_poynting_flux(a_N, b_N, s_matrix.end_layer_solve_result)
R = jnp.sum(-bwd_start.real) / fwd_start[0, 0].real
T = jnp.sum(fwd_end.real) / fwd_start[0, 0].real
```

closes `R + T` to `0.99999976` (~2.4e-7 residual) for the bare
ambient(n=1.0)/substrate(n=1.5) interface, and to `0.9999998584...`
(~1.4e-7 residual) for a small binary lamellar grating (9 Fourier orders,
18 modes with TE/TM). The sum over all mode entries (not just index 0)
matters for a real grating, since reflected/transmitted power spreads into
multiple diffraction orders; only the incident flux (`fwd_start[0, 0]`) is
guaranteed single-valued because the incident vector `a_0` is a one-hot
excitation of mode index 0. This contradicts the earlier
"not yet done"/"NOT YET DONE" status recorded elsewhere in this knowledge
pack (`failure_guide.md`, `capability_notes.md`, `solver_card.yaml`'s
`not_yet_probed` list) -- those files were not edited by this pass (outside
this adapter task's file-ownership scope) but should be updated by a human
or a future pass to reflect that this specific gap is now closed. Only the
*magnitude* (R, T) was checked this way; the complex amplitude's sign/phase
convention (see above) remains unreconciled.

## Numerical dtype

Default complex dtype observed is `complex64` (single precision), matching
the shared `agent_solver` JAX stack. No `complex128` path exercised.
