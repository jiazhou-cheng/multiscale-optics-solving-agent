# Level 2 — Two-to-Three-Model Orchestration

Recommended implementation order:

1. `L2-PSF-01` — Optiland → ray-to-wave → Chromatix → PSF/sensor.
2. `L2-META-01` — FMMAX → unit-cell-to-surface → Chromatix.
3. `L2-GRATING-01` — FDTDX → near-to-far → mode/fiber metric.
4. `L2-THERMO-01` — FDTDX absorption → JAX-FEM heat → material/circuit response.

A Level 2 task passes only when the coupling boundary is independently tested; passing each model separately is insufficient.
