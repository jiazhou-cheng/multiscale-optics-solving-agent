# First dark ring of a focused circular aperture

A circular aperture **40 µm** in diameter is uniformly illuminated at normal
incidence by a monochromatic plane wave at **532 nm**. An ideal aberration-free
lens of focal length **400 µm** immediately follows the aperture and brings the
light to a focus.

## What to determine

The **radius of the first dark ring** (the first zero of the intensity) in the
focal-plane intensity pattern, measured from the centre of the pattern.

Note the geometry before choosing a method: the focal plane is 400 µm away, which
is 10× the aperture diameter and 750× the wavelength. A propagation method valid
only for short distances relative to the aperture will not give the focal-plane
pattern here.

## What to submit

Write `submission.json` in your working directory:

```json
{
  "library": "<the import name of the package you used>",
  "first_null_radius_um": <number>
}
```

In **micrometres**, as a **radius** (not a diameter). If you cannot produce a
result, write `{"library": "...", "error": "<what went wrong>"}` instead of
guessing.
