# Canonical optical-system prescriptions (CHE-56 / PB5)

How to define an optical system for this repository, and what the Optiland
adapter will and will not build from one.

Before PB5 the adapter had two construction paths: bundled samples came from
`optiland.samples.objectives`, and the adapter-owned `M3SingletRef` was a
hand-written sequence of `Optic.surfaces.add(...)` calls. Prescription *data* and
construction *procedure* were entangled, so a new system meant new code. Now
there is one typed, versioned prescription value and one generic builder.

```
OpticalSystemSpec  ->  build_optiland_system(spec)  ->  optiland.optic.Optic
(core/optical_system.py)   (solvers/optiland/builder.py)
```

| Concern | Module |
| -- | -- |
| The schema, its validation, and canonical normalization | `src/core/optical_system.py` (imports no solver) |
| Translation to the pinned Optiland API | `src/solvers/optiland/builder.py` |
| The named prescriptions the adapter supports | `src/registry/prescriptions.py` |
| Executable evidence for the admitted construction paths | `knowledge/solvers/optiland/probes/system_construction_probe.py` (+ `expected/system_construction_probe.json`) |
| Tests | `tests/test_optiland_canonical_prescriptions.py` |

## Units

These are fixed by the schema contract and asserted by tests. They are not free
to drift, and two of them are counter-intuitive.

| Quantity | Unit | Established by |
| -- | -- | -- |
| radius, thickness, object distance, EPD | mm | CHE-12 |
| curvature (alternative to radius) | 1/mm | CHE-56 |
| wavelength | µm | CHE-12 |
| **grating period** | **µm**, *not* the geometry unit | CHE-56 probe: reproduces `sin θ_m = mλ/d` exactly at three periods; the mm reading is wrong by 1000× |
| **groove orientation** | **radians** | CHE-56 probe: `PlaneGrating.grating_vector` takes `sin`/`cos` of the stored value |
| angular field coordinates | degrees | Optiland `AngleField` |
| even-asphere `coefficients[i]` | multiplies `r**(2*(i+1))` | CHE-56 probe: the series starts at **r²**, not r⁴ (the r⁴ reading is off by 1.1e-2 mm on the probe surface) |

## Defining a new prescription

```python
from core.optical_system import (
    ApertureSpec, CatalogMaterialSpec, EvenAsphereGeometrySpec, FieldSpec,
    GratingInteractionSpec, IdealMaterialSpec, OpticalSystemSpec,
    PlaneGeometrySpec, SphericalGeometrySpec, SurfaceSpec, WavelengthSpec,
)

spec = OpticalSystemSpec(
    name="MyDoublet",
    object_distance_mm=None,          # None = object at infinity
    surfaces=(
        SurfaceSpec(
            geometry=SphericalGeometrySpec(radius_mm=20.0),   # or curvature_per_mm=0.05
            thickness_mm=4.0,                                  # to the NEXT surface
            material=CatalogMaterialSpec(                      # the medium AFTER this surface
                name="N-BK7",
                expected_catalog_file="glass/schott/N-BK7.yml",
            ),
            is_stop=True,
        ),
        SurfaceSpec(geometry=SphericalGeometrySpec(radius_mm=-30.0), thickness_mm=45.0),
    ),
    aperture=ApertureSpec(value_mm=10.0),                      # EPD, mm
    fields=(FieldSpec(y_deg=0.0), FieldSpec(y_deg=3.0)),
    wavelengths=(
        WavelengthSpec(value_um=0.486),
        WavelengthSpec(value_um=0.5876, is_primary=True),
        WavelengthSpec(value_um=0.656),
    ),
)
```

Three conventions worth stating once:

* **`material` is the medium *after* the surface.** A lens is a surface carrying
  glass followed by a surface carrying air (`AirMaterialSpec()`, the default).
* **`thickness_mm` is the spacing to the next surface**, and on the last listed
  surface it is the distance to the image plane.
* **The object and image surfaces are not listed.** Phase 3 fixes them rather
  than parameterizing them: the object is a plane in air placed
  `object_distance_mm` before the first surface (`None` = infinity), and the
  image surface is a plane in air at zero further spacing. A curved object or a
  tilted image plane needs a new schema version, not a workaround.

To use it:

```python
result = OptilandAdapter().run(ModelRunRequest(
    run_id=..., node_id=...,
    config={"prescription": spec, "num_rays": 16},   # or spec.canonical_dict()
))
```

`config["prescription"]` accepts the object or its serialized mapping (parsed
through `OpticalSystemSpec.from_dict`, which checks the schema version).
`config["sample"]` names a registered prescription instead. Supplying both is a
conflict and is rejected. To make a prescription available by name, add it to
`_CANONICAL_PRESCRIPTIONS` in `registry/prescriptions.py` — but only with the
evidence that justifies calling it validated.

The `system` **input port** remains refused. It would carry an arbitrary solver
object with no typed contract; the canonical schema exists so that is
unnecessary.

## Phase 3 supported set

| Feature | Spec |
| -- | -- |
| Plane surface | `PlaneGeometrySpec()` |
| Spherical / conic surface | `SphericalGeometrySpec(radius_mm=… \| curvature_per_mm=…, conic=…)` |
| Even asphere | `EvenAsphereGeometrySpec(radius_mm=…, conic=…, coefficients=(…,))` |
| Refraction | `RefractiveInteractionSpec()` (the default) |
| Grating | `GratingInteractionSpec(order=…, period_um=…, groove_orientation_rad=…)` |
| Air | `AirMaterialSpec()` (the default) |
| Ideal constant index | `IdealMaterialSpec(refractive_index=…, absorption=…)` |
| Catalog glass | `CatalogMaterialSpec(name=…, catalog=…, expected_catalog_file=…)` |
| Aperture stop | `is_stop=True` on exactly one surface |
| EPD aperture | `ApertureSpec(value_mm=…)` |
| Angular fields | `FieldSpec(x_deg=…, y_deg=…)` |
| Wavelengths | `WavelengthSpec(value_um=…, is_primary=…)`, exactly one primary |

Rejected eagerly, with a coded `PrescriptionError`: decentres and tilts,
coordinate breaks, GRIN, freeform and arbitrary sag, phase surfaces beyond the
grating, coatings and BSDFs, `imageFNO`/`objectNA` apertures, object-height and
image-height fields, no stop or several stops, no primary wavelength or several,
a grating on an aspheric base (the pinned solver implements gratings as plane or
conic geometry classes), and any non-finite number. Adding one of these means a
new issue with its own capability probe — not a widened `Literal`.

## Why validation is strict

Optiland's `GeometryFactory.create` filters `**kwargs` down to the dataclass
fields of the geometry config it selected. A key that does not belong to the
chosen surface type is **discarded with no error at all** — measured in the
probe's `surface_kwargs_are_silently_filtered` case, where a coefficient list
handed to a spherical surface simply vanishes. So a prescription passed straight
through would turn a caller's typo into a silently different optical system. The
schema's `extra="forbid"`, and the builder assembling each surface type's exact
keyword set itself, are the only places that mistake can be caught.

Catalog glass has the same character. Optiland resolves a bare name by substring
filter over three columns, then ranks the survivors by Levenshtein distance and
returns the best row. `SK15` therefore selects HIKARI while `N-SK10` selects
SCHOTT, seven rows survive the filter for `SK15` before scoring, and `SK1`
resolves happily to `SK16`. A prescription that records only the name has not
pinned its glass, so `CatalogMaterialSpec` defaults to demanding an exact match
and optionally pins `expected_catalog_file`, turning a future catalog change into
an error rather than a quietly different trace.

## Determinism and fingerprints

Every collection is an ordered tuple, every model forbids undeclared fields, and
no validation or construction step consults a set, a mapping's ordering, a random
source, or the clock. Solver-side defaults that could change the constructed
system (field vignetting factors, field and wavelength weights, the wavelength
unit) are stated explicitly by the builder rather than inherited.

```python
spec.canonical_dict()   # key-sorted, JSON-safe, all defaults materialized
spec.canonical_json()   # strict JSON (allow_nan=False); a plane carries no radius, so no Infinity
spec.fingerprint()      # SHA-256 of the above
```

An omitted default and an explicitly written one normalize to the same digest;
one changed number does not. `run()` records
`diagnostics["prescription_fingerprint"]` and
`diagnostics["prescription_spec_version"]`, so two runs reporting the same digest
built the same optical system.

## Migration evidence

Both previously supported systems now come from canonical prescriptions, checked
against oracles that are independent of the new code:

* **`ReverseTelephoto`** — `Optic.to_dict()` is *equal* to
  `optiland.samples.objectives.ReverseTelephoto()`'s, name aside (geometry
  classes and parameters, resolved catalog files, coordinate systems, stop flags,
  interaction models, aperture, fields, wavelengths), and traces are element-wise
  identical at Hy = 0, 0.5 and 1.0. The bundled sample is retained as this
  oracle; it is no longer a construction path.
* **`M3SingletRef`** — reproduces `benchmarks/protocols/slice_protocol.yaml`'s frozen
  derived geometry (EFL, BFL, image-plane z, EPD, semi-aperture) and the
  hand-written builder's `to_dict()` and trace bit-for-bit. The M1
  `standalone_baseline` scientific-array SHA-256 and summary metrics are
  unchanged.

## Not migrated, and why

`adapters/optiland_benchmark_adapter.py` still constructs its three L1-RAY-01
systems directly: a free-space pair, a `surface_type='paraxial'` thin lens, and
the Edmund 45-362 catalog singlet. The thin lens needs the `paraxial`/thin-lens
interaction and the free-space cases use absolute `z=` placement, neither of
which is in the Phase 3 schema. Admitting them would mean claiming construction
features this ticket did not probe, against M1 fingerprints that must not move,
so they stay as they are and belong to a follow-up issue.
