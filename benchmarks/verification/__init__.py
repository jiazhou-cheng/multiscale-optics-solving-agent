"""Verification harnesses: evidence about this project, run against upstream goldens.

CHE-238's overnight run. These are **not** benchmarks in the sense
`benchmarks/README.md` defines: a benchmark composes this project's public
primitives and gates itself on closed-form optics, and
`tests/benchmarks/test_records.py` enforces that for `benchmarks/systems/*.py`
only. A verification harness here does something different and narrower -- it
reads a *third-party* prescription or analysis, runs the same configuration
through this project's catalogued operations, and reports the difference.

Two consequences follow, and both are deliberate.

* **These modules import `optiland` directly, and a benchmark may not.** Reading
  a canonical `optiland.samples.*` prescription out of the sample class is the
  instruction CHE-239 gives (its §A.1), and the alternative -- hand-transcribing
  29 lens prescriptions -- would make the harness the thing under test. The
  no-direct-backend rule protects `benchmarks/systems/`, where the claim is
  "this project's vocabulary composes into a system"; here the claim is "this
  project's operation agrees with the tool it delegates to", which cannot be
  made without the tool.
* **Nothing here gates.** There is no closed-form oracle for "does our wrapper
  reproduce Optiland" -- both sides are the same numerics, which is exactly the
  circularity `AGENTS.md` names. A native-vs-native agreement is a *plumbing*
  regression: evidence that the translation did not lose or rename anything. A
  native-vs-repository difference is a physics comparison with an expected
  nonzero delta. Neither is a correctness gate on the physics.

Records go to `outputs/che-238-overnight/`, which is `.gitignore`d, because
`tests/unit/test_suite_shape.py` allows a committed JSON record in exactly two
trees and a verification record belongs to neither. The committed evidence is
`benchmarks/reports/2026-09/overnight_ray_wave_reproduction.md`, which carries
the same numbers, plus these scripts, which regenerate them.
"""
