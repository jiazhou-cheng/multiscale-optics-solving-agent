# L1-RAY-01 benchmark design

This design was frozen in the CHE-17 Linear work log before implementation.
All three cases use Optiland 0.6.0 on CPU/float64 and convert geometry from
millimetres to metres and wavelength from micrometres to metres at the
evaluation boundary.

CHE-20 amends the evidence contract to `M1-BASELINE-CPU-V2`. The three cases
and catalog prescription are unchanged. V2 adds authoritative spherical-front
sag, planar-rear, and clear-aperture classification checks plus complete
three-case overview and diagnostic figures. V1 remains the historical design
baseline.

## 1. Homogeneous free space

Manufactured normalized rays propagate from `z=0` to `z=100 mm` through
ideal air. The independent oracle is `t=(z_plane-z0)/N`, followed by
`x=x0+tL`, `y=y0+tM`, geometric path `t`, and OPL `n t` with `n=1`.
This isolates intersection, straight propagation, OPL accumulation, direction
preservation, SI conversion, and float64 roundoff.

## 2. Paraxial thin lens

An ideal `f=50 mm` paraxial surface receives five symmetric pupil heights at
each launch slope `-0.01`, `0`, and `+0.01 rad`; the reference plane is `z=f`.
The independent ABCD oracle is `u_after=u_before-y/f` and
`y_image=f*u_before`. Solver error is measured against that ideal model.
The physical paraxial-approximation error is listed separately and is not
attributed to Optiland.

## 3. Catalog lens

The highest-complexity case implements Edmund Optics TECHSPEC stock #45-362
from the [official product specification](https://www.edmundoptics.com/p/200mm-dia-x-500mm-fl-uncoated-plano-convex-lens/5832/):
N-BK7, `R1=+25.84 mm`, `R2=infinity`, center thickness `3.23±0.10 mm`,
`19 mm` clear aperture, `EFL=50.00 mm`, and `BFL=47.87 mm` at `587.6 nm`.
The convex face points toward the collimated input. Nine symmetric pupil
heights are traced at the same three launch slopes.

The [SCHOTT N-BK7 reference](https://www.schott.com/shop/advanced-optics/en/Optical-Glass/SCHOTT-N-BK7/c/glass-SCHOTT%20N-BK7%C2%AE)
provides the independent dispersion formula and `n_d=1.51680`. Thick-lens
equations provide EFL/BFL checks, and the axial chief-ray path supplies an
independent geometric-path/OPL oracle. Manufacturer values are reference
checks. Marginal spot and ray-minus-chief OPD arrays are deterministic
regression evidence only.

Reported error categories are solver/numerical, paraxial-model approximation,
prescription/reference uncertainty, finite sampling, and aperture/vignetting.
The cases progress from propagation, to ideal signed focusing, to real
spherical refraction/material/thickness/aperture/focus behavior.

The front-surface oracle evaluates spherical sag at every recorded primary-ray
intersection; the rear oracle checks a plane at the prescription back vertex.
Dedicated on-axis rays at `±9.4 mm` and `±9.6 mm` validate transmission just
inside and clipping just outside the `9.5 mm` semi-aperture without affecting
centroid or RMS statistics. Free-space is homogeneous propagation, not a
freeform surface; this benchmark contains no freeform optics.
