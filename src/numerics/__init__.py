"""The lowest layer: precision, device and array-namespace policy.

`numerics/` is the one package that imports nothing else in this project. That is
its definition, not a convention -- `scripts/check_dependencies.py` gives it an
empty allowlist, so an import from here to any project package fails the gate.

It exists because precision and array policy need a home and `src/core/` is
banned: "core" names no domain, and a package that names no domain accumulates
whatever has no other home. That is how the reference implementation reached 110
classes in `core/`. `numerics/` names one job.

What belongs here: dtype ladders and the precision contract, device placement,
the numpy/jax/torch namespace dispatch, and the array-intake rules a
representation applies at construction.

What does not: anything with a physical unit or a physical boundary. A wavelength
is not a numeric policy. That is `representations/`.

Empty of code at R01 by intent -- R01 lands the gates, R02 lands the content.
Nothing here is a placeholder interface: the package is declared so the
dependency and class-budget gates have a real tree to walk, and it will be filled
by the ticket that has something true to put in it.
"""
