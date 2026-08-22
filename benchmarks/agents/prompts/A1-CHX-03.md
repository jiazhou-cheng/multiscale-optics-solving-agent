# Lateral walk-off of a tilted beam

A monochromatic beam at **532 nm** propagates in vacuum. At the input plane its
amplitude is a Gaussian envelope

```
A(x, y) = exp( -(x^2 + y^2) / w^2 )      with w = 8 µm
```

carrying a linear phase tilt such that the beam travels at **5° from the z axis**,
tilted **toward positive x**. There is no tilt in y.

## What to determine

The **signed x coordinate of the intensity centroid** after the beam has
propagated **200 µm** of free space.

Both the magnitude and the **sign** are graded. A beam tilted toward +x arrives at
positive x; getting the magnitude right and the sign wrong is a failure, and so is
reporting a magnitude only.

## What to submit

Write `submission.json` in your working directory:

```json
{
  "library": "<the import name of the package you used>",
  "centroid_x_um": <signed number>
}
```

In **micrometres**. If you cannot produce a result, write
`{"library": "...", "error": "<what went wrong>"}` instead of guessing.

> Be careful with whatever convention you use to express the tilt. Angular
> wavenumber (radians per unit length) and spatial frequency (cycles per unit
> length) differ by a factor of 2π, and a beam displaced by 2π times too little —
> or in the wrong direction — will still propagate without any error being
> raised.
