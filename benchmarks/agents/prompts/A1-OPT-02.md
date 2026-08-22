# Focal shift caused by a plane-parallel plate

A beam in air converges toward a focus **100 mm** downstream of a reference
plane. A flat window is then inserted into the converging beam, with its front
face at that reference plane:

- thickness **10 mm** on axis,
- refractive index exactly **1.6**,
- both faces flat and parallel,
- clear aperture **2 mm** in diameter,
- wavelength **587.6 nm**.

Inserting the window moves the focus along the axis.

## What to determine

The **axial displacement of the focus** caused by the window.

Report it as a **signed** quantity, positive when the focus moves *further from*
the window (downstream, away from the incoming light) and negative when it moves
*toward* the window. Determine the sign from the physics; do not assume it.

## What to submit

Write `submission.json` in your working directory:

```json
{
  "library": "<the import name of the package you used>",
  "focal_shift_mm": <signed number>
}
```

In **millimetres**. If you cannot produce a result, write
`{"library": "...", "error": "<what went wrong>"}` instead of guessing.
