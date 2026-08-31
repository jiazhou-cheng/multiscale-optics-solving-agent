"""Physical state at a declared boundary.

A representation describes what the light *is* at some place, with every
convention it depends on made explicit and testable: units, axes, frame,
handedness, wavelength, phasor sign, polarization, coherence, normalization,
sampling, reference plane.

`representations/` may import `numerics/` and nothing else in this project. It
must not import a solver, a coupler, or a backend: the moment a representation
knows which package produced it, it has stopped being neutral ground, and the
reference implementation's two solver/coupler import cycles both started that
way.

Target content (R02): exactly one public ray representation and one scalar-field
representation. PSF is **not** here -- an observable derived from state is a
measurement, not a representation. Coherence is a stronger contract on the ray
representation, not a subtype of it.

Empty of code at R01 by intent, for the reason `numerics/__init__.py` gives.
"""
