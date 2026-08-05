# SAX conventions (pinned version `0.18.2`)

Every fact below was either read from `inspect.getsource`/`inspect.getdoc`
on the installed package, or observed directly by running
`knowledge/solvers/sax/probes/*.py` inside the `agent_solver` container.
None of it is copied from the online docs without independent verification
-- see `source_manifest.yaml` for a case where the online docs example was
actually stale relative to the installed package.

## Port naming is NOT fixed -- it is global, settable state

The online docs example (`gdsfactory.github.io/sax`) uses port names like
`in0`/`out0`. The installed 0.18.2 package's own docstring examples instead
call `sax.set_port_naming_strategy("optical")` first, which switches the
naming convention used by built-in models (`sax.models.*`) to `o1`/`o2`/`o3`/`o4`
style names. `sax.get_port_naming_strategy()` reads the current setting back.

**Implication for a coupler/adapter:** never hardcode an assumed port-name
scheme. A `C_*` coupler that bridges SAX component/circuit output to
another data type MUST call `sax.get_port_naming_strategy()` (or set and
record the strategy itself) and use the returned names, not literals copied
from an example. This directly matches the existing
`knowledge/solver_cards/sax.yaml` warning "port order and propagation
directions agree across components" -- the warning is not hypothetical, it
is a real, observed, switchable behavior.

## S-matrix representations

SAX supports three S-matrix container types (`SDict`, `SDense`, `SCoo`),
selectable via `sax.circuit(..., return_type=...)`. All probes here use the
default `SDict`: a plain Python dict keyed by `(port_name, port_name)`
tuples, with complex JAX-array scalar values (not bare Python complex --
call `complex(x)` to convert, or JSON serialization fails; see
`probes/component_model_probe.py`).

## Phase / propagation sign convention (verified, matches chromatix/fmmax)

For the dispersive `straight` waveguide model:

```python
s = sax.models.straight(wl=1.55, wl0=1.55, neff=2.34, ng=3.4, length=10.0, loss_dB_cm=0.0)
t = s[("o1", "o2")]
# |t| == 1.0                          (lossless, as configured)
# angle(t) == 0.6080501910173587
# analytic 2*pi*neff*length/wavelength == 0.6080501910173624  (relative error 6.0e-15)
```

This confirms SAX's forward-propagation phase convention is
`exp(+i * 2*pi*n*L/wavelength)` -- the **same sign** as chromatix's spatial
`exp(+i k.r)` convention and consistent with this project's canonical
`exp(-i omega t)` time convention (CLAUDE.md section 7). This is a positive
cross-solver consistency finding: unlike FMMAX (see
`knowledge/solvers/fmmax/conventions.md`, which found a *sign* mismatch
against the textbook Fresnel convention for `stack_s_matrix`), SAX's
`straight` model phase checks out cleanly against the naive analytic
formula with no sign correction needed. This has only been checked for one
model (`straight`) and one backend -- do not generalize to every SAX model
without a targeted check.

## Ideal coupler: energy and reciprocity (verified exactly, to float precision)

```python
s = sax.models.coupler_ideal(wl=1.55, coupling=0.5)
# s[("o1","o4")] == 0.7071067811865476        (real -- "thru" port)
# s[("o1","o3")] == 0.7071067811865476j       (purely imaginary -- "cross" port, 90-degree phase shift)
# |thru|^2 + |cross|^2 == 1.0000000000000002  (energy conservation)
# s[("o4","o1")] == s[("o1","o4")]             (reciprocity, exact)
```

The 90-degree relative phase between the thru and cross ports of an ideal
lossless coupler is the standard convention and matches physical
expectation (unitarity of the coupler's 2x2 sub-block).

## Full netlist circuit assembly (verified against an independent analytic oracle)

A netlist-composed Mach-Zehnder interferometer (two `coupler_ideal` + two
`straight` waveguides with an path-length difference `dl`) was built via
`sax.circuit(netlist, models)` and evaluated. See `probes/circuit_probe.py`
for the full netlist dict. Comparing the assembled circuit's `in0 -> out0`
power transmission against the analytic MZI formula
`T = sin^2(pi * n_eff * dl / wavelength)`:

| Quantity | Value |
|---|---|
| SAX circuit `\|t_00\|^2` | 0.9770696282000273 |
| Analytic `sin^2(dphi/2)` | 0.977069628200026 |
| Relative error | 1.36e-15 |
| Energy conservation (`\|t_00\|^2 + \|t_01\|^2`) | 1.0000000000000002 |
| Reciprocity (`out0->in0 == in0->out0`) | exact |

This is a real, independent-oracle verification (CLAUDE.md rule 6) of the
full netlist-assembly path, not just an isolated component model -- a
stronger result than the chromatix/fmmax packs currently have for their
respective composed pipelines.

## Units

Wavelength arguments in the built-in models (`wl`, `wl0`) are used in
microns in every example encountered (matching the photonics-community
convention), and lengths (`length` in `straight`) are consistent with that
scale, but nothing in the type signatures enforces a unit -- like
chromatix, SAX is scale-agnostic as long as `wl`/`wl0`/`length` share a
consistent unit. An adapter must convert from the project's SI-meter
convention (CLAUDE.md section 7) explicitly.

## Numerical dtype

Values returned by `sax.models.*` are JAX arrays; observed dtype follows
JAX's default (complex64 unless `jax.config.update("jax_enable_x64", True)`
is set). Not independently pinned by SAX itself.

## Addendum (found while building `sax_adapter.py`, 2026-07-30)

- **Default port naming strategy is `"inout"`, not `"optical"`.** Calling
  `sax.get_port_naming_strategy()` immediately after `import sax` (before
  any call to `set_port_naming_strategy`) returns `"inout"`. Every example
  in this knowledge pack calls `sax.set_port_naming_strategy("optical")`
  first, which masked this. `sax.set_port_naming_strategy(strategy)`
  validates `strategy` against exactly `["optical", "inout"]` and raises
  `ValueError(f"Invalid port naming strategy: {strategy}")` for anything
  else -- a clean, adapter-catchable failure mode (used by
  `sax_adapter.py`'s failure test for an invalid port-naming request).
- **A bare (non-circuit) `sax.models.coupler_ideal(...)` SDict has mixed
  per-entry dtypes and is not the full dense port-pair matrix.** For a
  real-valued `coupling`, the "thru" entries (`o1->o4`, `o2->o3`, ...) come
  back as `float64` while the "cross" entries (`o1->o3`, `o2->o4`, ...)
  come back as `complex128` -- observed directly, not documented anywhere
  in SAX's own docs/docstrings. The dict also only has 8 of the
  16 possible 4-port pairs (no self-reflection terms, no direct `o1->o2`
  pass-through pairs). By contrast, the same `coupler_ideal` assembled
  inside `sax.circuit(...)` (see `probes/circuit_probe.py`) returns a
  **full** 16-entry dict (including self-terms like `("in0","in0")`) with
  **uniformly** `complex128` values. `sax_adapter.py` handles this by
  converting every entry through `complex(...)` regardless of source dtype
  and recording the raw observed dtype set separately in
  `ArtifactRecord.metadata["raw_dtypes_observed"]`.
- **A single point-check suggests `jax.grad` does propagate through an
  assembled `sax.circuit`'s matrix-solve**, contradicting the
  `not_yet_probed` assumption above that this was untested. At
  `coupling1=0.3, coupling2=0.5` (the two-coupler MZI from
  `probes/circuit_probe.py`, objective = `|S[("in0","out0")]|^2` w.r.t.
  `coupling1`), `jax.grad` agreed with a centered finite difference
  (`eps=1e-4`) to a relative error well under `1e-6`. At the symmetric
  point `coupling1=coupling2=0.5`, both the AD and FD gradients are
  individually ~1e-12/1e-17 (a genuine local extremum of the objective),
  so the *relative* error there is meaningless (division by ~0) -- this is
  a degenerate test point, not evidence of a bug. This one-point check is
  documented as an `xfail(strict=False)` regression probe in
  `tests/test_sax_adapter.py::test_gradient_through_assembled_circuit_not_yet_verified`,
  not as a promotion of `derivative.verified` to `true`: CLAUDE.md section
  6.2 requires multiple step sizes, a convergence table, and a
  deliberately ill-conditioned case before that claim can be made, and
  none of that has been done for the circuit-assembly path yet.
