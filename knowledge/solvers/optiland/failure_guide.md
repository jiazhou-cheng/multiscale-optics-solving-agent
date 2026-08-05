# Optiland failure guide

Real errors and surprises hit while building this knowledge pack
(2026-07-30), with repairs. Add to this file rather than silently working
around a new one.

## `pip install optiland` does not give you gradients or GPU support

**Symptom:** code that assumes `derivative.mode: native_autodiff` (as this
project's registry currently states for `M_RAY_OPTILAND`) just... doesn't
differentiate, or `optiland.backend.supports_gradients` reads `False`.

**Cause:** torch is not a declared dependency of the `optiland` PyPI
package (confirmed via `importlib.metadata.distribution('optiland').requires`
— matplotlib, numba, numpy, pandas, pyyaml, requests, scipy, seaborn,
tabulate, typing-extensions, vtk; no torch). The default backend is NumPy,
which has no autodiff.

**Fix:** install `torch` separately (already done in
`docker/requirements.txt` / `docker/Dockerfile`, from the CPU-only wheel
index) and call `optiland.backend.set_backend('torch')` before building or
tracing the lens you want gradients through.

## `TypeError: 'bool' object is not callable`

**Symptom:** raised by `be.supports_gpu()` or `be.supports_gradients()`.

**Cause:** these are plain module-level `bool` attributes, not functions,
despite reading like predicates.

**Fix:** use `be.supports_gpu` / `be.supports_gradients` directly, no
parentheses.

## `DeprecationWarning: Optic.surface_group is deprecated; use Optic.surfaces instead.`

**Symptom:** a warning (not an error) on `lens.surface_group` access.

**Cause:** real, observed API migration in progress within `0.6.0` itself —
the old name still works but is on its way out.

**Fix:** use `Optic.surfaces` in any new code. Treat the eventual removal
of `surface_group` as a staleness signal for this knowledge pack.

## Repository URL redirects (not broken, just not canonical)

**Symptom:** none functionally — `https://github.com/HarrisonKramer/optiland`
still works.

**Cause:** the project moved to a dedicated `optiland` GitHub org
(`https://github.com/optiland/optiland`), confirmed both by an HTTP 301
redirect and by the PyPI package's own `Project-URL` metadata.

**Fix:** cite the new URL in new documentation; the old one is fine to
follow but is not the canonical source of truth going forward.

## Ray count after `trace()` does not equal the requested `num_rays`

**Symptom:** `lens.trace(..., num_rays=16)` returns a `RealRays` object
with `.x.shape == (817,)`, not `(16,)`.

**Cause:** `num_rays` controls a pupil-sampling density parameter, not a
literal output count; aperture/vignetting/pupil geometry determines how
many rays actually survive tracing.

**Fix:** never hardcode an expected output array length from `num_rays`;
read `.x.shape` (or equivalent) from the returned object instead.

## Unverified: behavior when torch is requested but not installed

**Status:** NOT independently tested in this pass. This container always
has torch installed, so `optiland.backend.set_backend('torch')` always
succeeds here. Do not assume a specific failure mode (immediate
`ImportError` at `set_backend()` time vs. a deferred failure only when a
torch-specific operation actually runs) without testing a torch-less
environment directly. An attempted test using a Python import-blocker
(`sys.meta_path` with a legacy `find_module`/`load_module` finder) was
tried and found to **not work** on Python 3.12 (the legacy finder protocol
is silently ignored; `import torch` still succeeded) — that approach is a
dead end, not a valid negative result. A real test would need a second
Docker image built without torch installed.

## Python version

Optiland's `Requires-Python` is `>=3.10` per its wheel metadata — looser
than chromatix's `>=3.12` or sax's `>=3.11`. Not a source of conflict in
this repository's shared `python:3.12-slim` image, but worth knowing if
optiland is ever split into its own lighter environment.
