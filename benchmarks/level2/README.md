# Level 2 — Two-to-Three-Model Orchestration

Implemented here:

* `L2-PSF-01` — Optiland → `C_RAY_TO_WAVE` → Chromatix, terminating at the
  propagated `ComplexField` with PSF as a measurement on it. Its
  `1.0e-3 fft_oracle_intensity_relative_l2` gate is **unmet** and carried into
  M4 as an explicit open limitation; see `benchmarks/manifest.yaml`'s
  `gate_disposition` and `L2-PSF-01/README.md`.
* `L2-COUPLER-01` — bidirectional ray-wave coupler characterization.

A Level 2 task passes only when the coupling boundary is independently tested.
Passing each model separately is not sufficient.

The three planned tasks that used to be listed here (`L2-META-01`,
`L2-GRATING-01`, `L2-THERMO-01`) named FMMAX, FDTDX and JAX-FEM, all removed by
CHE-87. They are recorded as intent in `benchmarks/roadmap.md`.
