# Milestone 1 — Key Contributions

M1 established two **independently verified** physical baselines — a ray model
(Optiland) and a wave model (Chromatix) — with no coupling between them. The
contribution is not that the solvers run, but that what they may be trusted for
is now bounded by evidence.

**Verified baselines.** L1-RAY-01 checks Optiland against closed-form
free-space geometry, ABCD paraxial thin-lens equations, and the Edmund #45-362
catalog; worst relative error `1.4e-5`. L1-WAVE-01 checks Chromatix against
exact plane-wave eigenmodes and an independent float64 angular spectrum
(overlap `0.999999`). Both branches are bitwise deterministic across repeats.

**Tolerances derived, not chosen.** The wave Case 1 bound follows from float32
phase round-off. Nine convention perturbations — unit scale, lens sign flip,
axis transpose, paraxial dispersion — are each detected, each with a passing
unperturbed control.

**Enforced independence.** A static-plus-dynamic verifier proves neither branch
imports the other's engine or any coupler: AST scan of every entry-point
source, plus `sys.modules` inspected at process exit. A ray pass cannot mask a
wave failure.

**Reproducibility as a hash.** Each branch publishes a scientific fingerprint
over physics only, with wall-clock and run-identity keys stripped at any depth.
This caught a real defect — per-case runtimes were inside the hash, so it
tracked machine load rather than physics. Fixed and pinned by test.

**Honest negative results.** High-NA vectorial focusing is recorded as
*blocked*, with a root cause in Chromatix's pinned source (`s_z` derived from
the frequency grid, not the pupil grid) and an independent Richards–Wolf oracle
converging to `2e-14`. No tolerance was loosened to make it pass.

**What M2 inherits.** Scalar CPU `asm_propagate` is the only validated wave
surface. No gradient is verified through either engine. The pinned conventions
— `(y, x)` order, `exp(-iωt)`/`exp(+ikz)`, SI, amplitude never intensity — are
the coupler's contract.
