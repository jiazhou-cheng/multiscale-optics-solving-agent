"""The physically meaningful initialization of a representation.

`docs/architecture_principles.md` section 2 defines the term. In short: a
representation defines the *structure and conventions* of physical state at a
declared boundary, and a source defines how that state is *initialized* from
physical source parameters. A source therefore **does not consume an existing
physical representation -- it creates the initial state of one**, which makes it
the only operation in the graph with no input representation.

`sources/` is a **new package**, landed by CHE-210 (R06.5) as a deliberate
architecture change with the owner's decision. Its operations register as
`solver`-kind, because mapping a problem statement into a representation is what
that document calls a solver and there is no fifth operation kind; what separates
this package from `solvers/<backend>/` is that a source has no external backend,
so per-backend organization has nothing to organize. Section 3 records why each
alternative home was worse.

The allowlist row is `sources/ -> problems, representations, numerics`. Only the
last two are exercised today: a source *declaration* belongs in `problems/` when
something needs one, and the edge is declared so that landing it is not a second
architecture change. There is deliberately no edge to `solvers/`, `couplers/`,
`operators/` or `measurements/`: a source is upstream of everything that consumes
state.

Representation-independent as a package, representation-explicit per operation
---------------------------------------------------------------------------------
A source operation may initialize a `ScalarField`, a `RayBundle`, or any other
landed representation, and **which one it returns must be unambiguous in the
public API** -- in the signature's return type and in the descriptor's `output`.
So this package is not partitioned by representation and must not grow a
subpackage per representation, and no constructor here may return one of two
representations depending on its arguments.

Today the only landed operation initializes a `ScalarField`. A collimated
`RayBundle` source -- spatial sampling plus a common propagation direction -- is
the obvious second one, and it belongs beside `plane_wave` under this same rule
rather than under a `sources/wave/` and `sources/ray/` split.

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
