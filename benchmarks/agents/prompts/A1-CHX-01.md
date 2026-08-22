# Diffractive spreading of a Gaussian beam

A monochromatic beam in vacuum has, at its waist, a transverse amplitude profile

```
A(x, y) = exp( -(x^2 + y^2) / w0^2 )
```

with **w0 = 5 µm**, at a wavelength of **532 nm**. The waist plane is at z = 0.

## What to determine

The **1/e² intensity radius** of the beam after it has propagated **100 µm** of
free space from the waist.

The 1/e² intensity radius is the radius at which the intensity has fallen to
1/e² of its on-axis value — the same convention `w0 = 5 µm` is quoted in above.

This is a diffraction problem: solve it by propagating the field, not by
substituting into a formula.

## What to submit

Write `submission.json` in your working directory:

```json
{
  "library": "<the import name of the package you used>",
  "beam_radius_um": <number>
}
```

In **micrometres**. If you cannot produce a result, write
`{"library": "...", "error": "<what went wrong>"}` instead of guessing.

> Sanity check: the answer must be larger than 5 µm. A beam that has propagated
> 100 µm has diffracted.
