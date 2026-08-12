# Level 1 — One-Model Simulation

Implement in this order:

1. `L1-WAVE-01` — analytic Airy/Gaussian wave benchmark.
2. `L1-TMM-01` — Bragg mirror spectrum and gradient.
3. `L1-RAY-01` — sequential lens analysis and bounded optimization.
4. `L1-RCWA-01` — grating convergence and energy balance.
5. `L1-EM-01` — full-wave waveguide component.
6. `L1-PC-01` — photonic-crystal band/eigenmode task.

Level 1 calibrates adapters and verification. Do not begin a Level 2 graph until its constituent Level 1 models have passed their relevant tests.

`L1-RAY-01` is implemented under [`L1-RAY-01/`](L1-RAY-01/) as the CHE-17
Optiland-only analytic accuracy suite. It covers manufactured free-space
propagation, ideal paraxial focusing, and a documented Edmund Optics catalog
lens; bounded optimization remains outside this M1 accuracy issue.
