# Focal length of a thick plano-convex singlet

A single lens element sits in air. Light arrives collimated and parallel to the
optical axis.

- The first surface is spherical and convex toward the incoming light, with a
  radius of curvature of **25 mm**.
- The element is **4 mm** thick on axis.
- Its glass has a refractive index of exactly **1.5168** at the wavelength of
  interest, **587.6 nm**.
- The second surface is flat.
- The clear aperture is **10 mm** in diameter.

## What to determine

1. The **effective focal length** of the element.
2. The **back focal length** — the axial distance from the *rear vertex* of the
   element to the focus.

The element is thick enough that these two are not the same number.

## What to submit

Write `submission.json` in your working directory:

```json
{
  "library": "<the import name of the package you used>",
  "effective_focal_length_mm": <number>,
  "back_focal_length_mm": <number>
}
```

Both in **millimetres**. If you cannot produce a result, write
`{"library": "...", "error": "<what went wrong>"}` instead of guessing — a
reported failure is more useful than a fabricated number.
