"""Cell parity: do the same kernel's `namespace x device x dtype` cells agree?

**What these tests answer, and what they do not.** A parity test here answers
"do the cells agree with each other". It does **not** answer "is the result
correct". Nothing in this package is an oracle, and the numpy-cpu leg every
subject compares against is **characterization**, not an independent
correctness gate -- it is this repository's own numerics run at a wider
precision on the host. `AGENTS.md`: "Repository numerical code must not be the
sole correctness oracle for the same numerical code." The analytic oracles that
*can* settle correctness live in `tests/physics/`; a green suite here means the
GPU and the host computed the same thing, including the case where both are
wrong.

**Why the package exists.** `docs/rewrite/reference_inventory.md` §8 open risk 1
records that device parity is outside the default gate, so "parity claims that
rest only on the default suite will be unfalsifiable for exactly the four
properties most likely to break". Every ticket that says "this ran on the
requested device, in the requested namespace, at the requested dtype" needs
somewhere for that claim to be falsified. CHE-244 (T0) is that instrument.

**What it is not allowed to become.** No records are written here and nothing
lives under `benchmarks/`: a parity cell is not a measurement anybody cites, and
giving it a record would make a fixture into an artifact with a fingerprint.
Cells are derived (`cells.cells_for`) rather than listed, tolerances are derived
in exactly one place (`cells.tolerance_for`), and placement is always *observed*
back off the buffer rather than trusted from the argument that requested it.

See `docs/architecture_principles.md` §"Parity evidence and correctness
evidence" for the rule these modules are the executable form of.
"""
