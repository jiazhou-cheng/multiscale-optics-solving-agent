# FMMAX failure guide

Real issues hit while building this knowledge pack (2026-07-30), with
repairs. Add to this file rather than silently working around a new one.

## Historical repository URL redirects

**Symptom:** `https://github.com/mfschubert/fmmax` (recorded in an earlier
version of `knowledge/solver_cards/fmmax.yaml`) returns HTTP 301.

**Cause:** the project moved to the `invrs-io` GitHub org.

**Fix:** use `https://github.com/invrs-io/fmmax` (matches the docs domain
`invrs-io.github.io/fmmax`). GitHub's redirect means old clone/API URLs
still resolve, but stored references should be updated so they don't rot
further.

## `s21` is reflection, `s11` is transmission

**Symptom:** a coupler or probe written assuming RF/photonic-circuit
`S11`-is-reflection conventions silently computes transmission where it
expected reflection (or vice versa) -- no exception is raised, just a
physically wrong number that can still look plausible (e.g. `|s11|^2`
alone was `1.44`, an "obviously wrong for a naive reflectance" value that
would have been easy to miss if not cross-checked against an oracle).

**Fix:** always use `s21` for reflection and `s11` for transmission per
`inspect.getdoc(fmmax.ScatteringMatrix)` (see `conventions.md`), and verify
against an analytic case (as `probes/fresnel_oracle_probe.py` does) rather
than trusting variable names alone.

## Naive power-transmittance formula does not close energy conservation

**Symptom:** `R + (n_substrate/n_ambient) * |s11|^2` did not equal 1 for a
bare interface (got `2.2`, not `1.0`); `|s11|^2` alone was already `1.44`,
greater than 1.

**Cause:** FMMAX's raw modal amplitudes are not simple E-field amplitude
ratios in a form where `|amplitude|^2` (times a naive index ratio) equals
physical power. The correct path is almost certainly
`fmmax.amplitude_poynting_flux` / `fmmax.directional_poynting_flux`, which
exist specifically to convert modal amplitudes into physical power, but
this pass did not implement that conversion.

**Fix (follow-up, not yet done):** rebuild the energy-conservation check
using the Poynting-flux accessors instead of raw `|amplitude|^2`.

**Update (2026-07-30):** done in
`src/multiscale_optics_agent/adapters/fmmax_adapter.py`, using
`fmmax.directional_poynting_flux` on the physical amplitude vectors
(`a_0`/`b_0` at the start layer, `a_N`/`b_N` at the end layer, with the
s11/s21 swap applied). `R + T` closes to ~2.4e-7 residual for the bare
interface and ~1.4e-7 for a small lamellar grating. See `conventions.md`
for the exact formula and code.

## Homogeneous-limit oracle requires `approximate_num_terms=1`

To reduce a periodic RCWA solver to a plain Fresnel-interface check (no
real grating), use `approximate_num_terms=1` with
`fmmax.Truncation.CIRCULAR` in `generate_expansion`. Larger values
introduce genuine diffraction-order content that will not match a simple
two-index Fresnel formula, so don't raise this value when trying to
reproduce the oracle in `probes/fresnel_oracle_probe.py`.

## Python version

FMMAX itself only requires `>=3.10` (looser than Chromatix's `>=3.12` or
SAX/Optiland's `>=3.11`), but the shared `agent_solver` image uses
`python:3.12-slim` because other pinned solvers need it. No FMMAX-specific
incompatibility was observed at 3.12.
