# C_PLANAR_DOE_STEP — what it refuses, and what it will not catch

## Refusals

Every precondition returns a `CouplerRunResult` with `status=FAILED`, a
`ContractCode`, and a remedy. `validate_request` and `transform` call the same
`PlanarDoeStepCoupler.diagnose`, so they cannot disagree about which requests are
acceptable — two parallel checklists is how a validator comes to bless a request
that then fails.

### Diagnosed before anything runs

These come from `diagnose`, so `validate_request` reports them without executing.

| Trigger | Code | Remedy |
| --- | --- | --- |
| No incident ray bundle on the source port | `MISSING_DECLARATION` | Supply the bundle on the default source port. |
| Source artifact is not a `ray_bundle` | `ARTIFACT_KIND_MISMATCH` | This edge consumes rays; convert upstream. |
| No DOE transmission | `MISSING_DECLARATION` | Supply it on the DOE port or as `config['doe_transmission_uri']`. |
| `config['plane_z_m']` absent | `MISSING_DECLARATION` | Declare the plane; the step does not choose one. |
| Both `launch_positions_xy_m` and `primary_sampling` | `MISSING_DECLARATION` | Supply exactly one. They are a conflict, not a precedence. |
| Neither position source | `MISSING_DECLARATION` | Declare where the outgoing rays launch. |
| `primary_sampling` without `primary_count` | `MISSING_DECLARATION` | Supply the count. |
| `uniform_on_grid`, or any other draw, without `config['seed']` | `MISSING_DECLARATION` | Supply an explicit seed; an implicit one is not reproducible. |
| Incident record lacks amplitude or OPL-with-reference | `OPL_REFERENCE_UNVERIFIED` | Declare both. A bundle without a stated OPL reference cannot be accumulated coherently. |

### Raised by the core, and flagged as undiagnosed

`couplers/cascade.py::planar_doe_step` enforces these itself. The node catches the
`ContractError` and returns a `FAILED` result carrying
**`undiagnosed_refusal: True`** — which is the useful part: it marks a request
`validate_request` would have accepted. A run that fails this way is a gap in
`diagnose`, not only a bad request, and the flag is what makes that visible.

| Trigger | Code |
| --- | --- |
| Transmission is real, not complex — an amplitude mask with an undeclared phase | `MISSING_DECLARATION` |
| Transmission shape ≠ plane grid | `SHAPE_MISMATCH` |
| More incident positions requested than exist | refused |

Evidence:
`tests/test_planar_doe_step.py::test_supplying_both_position_sources_is_a_conflict_not_a_precedence`,
`::test_asking_for_more_incident_positions_than_exist_is_refused`,
`::test_preserve_energy_is_off_by_default_and_reported_when_on`.

## Silent failures — what runs clean and is still wrong

These are the ones worth loading this pack for. None of them raises.

### S1 — a bundle referenced to the wrong plane

The step accumulates the bundle **as given** and does not propagate it. Handing
it a bundle referenced to a different plane measures the transmission somewhere
the DOE is not. The output is a perfectly plausible diffraction pattern.

*Detect:* advance the ray state explicitly and check that the result is
stationary. If moving the declared plane by `dz` and advancing the bundle by the
same `dz` changes the answer, one of the two is not being applied.

### S2 — reading OPL across the step

Outgoing OPL is zero **by convention**, rebased to this plane. Code that
differences an OPL measured downstream against one measured upstream is
differencing two origins. The error grows with the incident path, so it presents
as a defocus that tracks the source rather than as an artifact.

*Detect:* move the source and watch whether the focus moves with it in a way the
geometry does not predict.

### S3 — indexing outgoing rays by incident identity

There is no per-ray correspondence. The accumulation erases the incident
population, and the outgoing count is the caller's budget. Code that assumes ray
`i` maps to ray `i` will be wrong and will not crash whenever the counts happen
to match.

### S4 — a collapsed run read as a converged one

`secondary_count <= 1` returns one ray on the power-weighted mean wavevector
carrying the whole spectrum. It is a preview with **no stated error**. It is
reported as `collapsed_to_mean_wavevector` precisely so this is checkable — check
the diagnostic before believing a number.

### S5 — quoting an accuracy for a sampled run

Only the exactness limit (`secondary_count=None`, full enumeration) has evidence.
Unbiasedness, convergence exponent, and variance are **not established** for the
sampled step. A single realization is not evidence of accuracy, and the ledger
carries this as an open gap rather than an estimate.

### S6 — `preserve_energy=True` read as conservation

It is a renormalization, not a measurement. The factor is reported; a power
"agreement" obtained with it on says nothing about whether the step conserved
energy.

### S7 — a curved substrate

The planarity assumption is not a soft limit; off a plane the step is undefined
rather than approximate, because there is no common plane to accumulate onto. It
will still run and still produce a field.

*Remedy:* size a planar patch with the curvature bound in
`couplers/curvature.py`, and use `C_PATCH_WFT` for the patched route.

### S8 — treating `pad_width` as a convergence knob

Padding changes the spectral sampling, not the physics: it interpolates the
angular spectrum onto a finer grid. A result that keeps moving with `pad_width`
beyond interpolation is a defect, not an unconverged calculation.

## What no failure path here covers

* **Gradients.** The derivative is an inherited, deliberately biased surrogate
  (`C_WAVE_TO_RAY`'s fixed-direction estimator) and is `verified: false`. Nothing
  refuses a gradient request across this step; nothing certifies it either.
* **CUDA.** Declared, and not covered by the default gate. The only device
  evidence is a CROSS_ROUTE / SHARES_CODE characterization, which cannot decide
  correctness.
