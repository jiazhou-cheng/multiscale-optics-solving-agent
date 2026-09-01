"""Declared illuminations: a problem statement becomes a representation.

`sources/` is a **new package**, landed by CHE-210 (R06.5) as a deliberate
architecture change with the owner's decision. A source maps a problem statement
into a representation, which is `docs/architecture_principles.md` section 2's
definition of a `solver` -- and its operations register as `solver`-kind for
exactly that reason -- but it has no external backend, and `solvers/<backend>/`
is organized per backend. `src/sources/plane_wave.py` records why each of the
alternative homes was worse.

The allowlist row is `sources/ -> problems, representations, numerics`. Only the
last two are exercised today: an illumination *declaration* belongs in
`problems/` when something needs one, and the edge is declared so that landing it
is not a second architecture change.

This package imports **no backend**. A plane wave is arithmetic on the project's
own grid; `chromatix.functional.plane_wave` is a cross-check, and one that
carries a `power=1.0` amplitude renormalization this tree does not want.

One module:

* `plane_wave` -- `plane_wave`, the single primitive (normal incidence is
  `k_t = (0, 0)`, not a second function), and
  `transverse_wavevector_from_angle`, the pure converter from `(theta, phi)`.

Not here: point sources, Gaussian beams as a source primitive, spectra and
chromatic fields, polarization, partially coherent illumination, and any physical
model of an illumination unit. A source here is an analytic field at a declared
surface, nothing more.
"""

from sources.plane_wave import plane_wave, transverse_wavevector_from_angle

__all__ = ["plane_wave", "transverse_wavevector_from_angle"]
