# Single-layer anti-reflection coating at 550 nm

A flat glass substrate of refractive index **1.5168** sits in air. Light arrives
at **normal incidence** at a wavelength of **550 nm**.

A single-layer anti-reflection coating is to be applied, made of a material of
refractive index **1.38**, at the standard **quarter-wave optical thickness** for
550 nm.

## What to determine

1. The **normal-incidence reflectance of the bare substrate** (no coating).
2. The **normal-incidence reflectance with the quarter-wave layer applied**.
3. The **physical thickness of that layer**.

Both reflectances as fractions of incident power (so 4 % is `0.04`), not
percentages.

## What to submit

Write `submission.json` in your working directory:

```json
{
  "library": "<the import name of the package you used>",
  "uncoated_reflectance": <number>,
  "coated_reflectance": <number>,
  "coating_thickness_nm": <number>
}
```

The thickness in **nanometres**; the reflectances dimensionless. If you cannot
produce a result, write `{"library": "...", "error": "<what went wrong>"}`
instead of guessing.

> Check your answer against what the coating is *for*. A quarter-wave AR layer of
> this index on this substrate should reduce the reflectance substantially. If
> your coated and uncoated numbers come out nearly equal, the coating in your
> model is not doing anything, and the run having completed without error is not
> evidence that it is.
