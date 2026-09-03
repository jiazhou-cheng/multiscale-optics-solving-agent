"""The canonical production catalog: every landed operation, as one record each.

CHE-221 (R03.4). `CATALOG` is the **single canonical declaration** of what this
project can execute. `registry.py` derives its by-id index from this tuple at
import; nothing else declares an operation, and no test owns a production
descriptor any more.

**Production-complete and planner-ready, as of CHE-222 (R03.5).** Every landed
public operation has a record, and the mechanical gate in
`tests/operations/test_catalog.py` keeps that true. R03.4 landed this file with a
warning at the top that it was *not* planner-ready, because `input`/`output` were
single strings and a source's `input` named the representation it *produces* -- so
a graph built from these ports would have read a source as consuming what it emits.
R03.5 removed that: `inputs` is a tuple of ports with `()` for a graph entry,
`returns` is ordered with the primary result first, and `requires`/`optional` name
the arguments a caller must and may supply. The eight questions those fields answer
are listed in `operations/descriptors.py`, and each has a test.

`capabilities` was the last field with an eager coupling and CHE-223 (R03.6)
removed it: the measured evidence moved to `knowledge/capabilities/`, and a value
here is a component id checked for **shape** at construction and for **resolution**
by `tests/operations/test_capability_references.py`. **This file imports `numerics`
nowhere** -- it imports nothing in the project outside `operations/` -- which is the
point: constructing the catalog needs neither a backend nor the measured table.

Why the catalog lives here and needs no dependency change
---------------------------------------------------------
`scripts/check_dependencies.py::ALLOWED` gives `operations` one edge, to
`numerics`, and gives no implementation package an edge to `operations`. That is
deliberate and load-bearing: `operations -> backends` would end the one property
this package exists to provide, and `backends -> operations` would end it from the
other side, because listing the registry would then have loaded torch and JAX.

The escape is not a widening. `implementation` is already a `"module.path:attribute"`
string, so a catalog *inside* `operations/` needs no edge at all -- it names
`backends.optiland.solver:trace` without importing it, and `operations.resolve` is
still the only function in the package that imports anything.

Registration therefore stays **pulled, never pushed**. No implementation package
imports `operations`; each declares `OPERATIONS: tuple[str, ...]`, a tuple of
strings naming its public callables that are semantic operations, and the
completeness gate walks the catalog against those tuples in both directions.

A tuple, not a dict literal
---------------------------
`CATALOG` is a tuple keyed by nothing, and the by-id index is built from it with a
duplicate check. A dict literal keyed by `operation_id` would silently keep the
last of two entries sharing an id; the tuple form makes a duplicate id a
`ValueError` raised while `operations` is first imported, which is the same
refusal `register()` gave and for the same reason -- two descriptors under one id means
two answers to "what does this operation do", and last-write-wins makes which one
you get depend on nothing a reader can see.

The four argument tuples are checked against the code, not just written
------------------------------------------------------------------------
`inputs`, `requires`, `optional` and the arity of `returns` are all derivable from
`inspect.signature`, and `tests/operations/test_catalog_signatures.py` derives them
for all fifteen records and compares. So this file is a *checked* restatement of
the signatures rather than a hand-maintained one: renaming a parameter, giving one
a default, removing one or adding a required argument fails that gate.

That matters because a second source of truth beside a signature is precisely what
drifts, and two records here were already wrong before the check existed --
`S_SOURCE_PLANE_WAVE` and `S_RAY_OPTILAND` (now `SO_RAY_LAUNCH_TRACE`) each
declared a representation input their callables do not accept. `approximation`,
`validity` and `evidence` remain unguarded prose; nothing can derive those.

Two records may name one callable -- and none does any more
-----------------------------------------------------------
This section is kept and rewritten rather than deleted, because the arrangement it
described is gone and the reason is worth having next to the catalog it shaped.

It used to say: `S_WAVE_CHROMATIX` and `O_ASM_PROPAGATE` both resolve to
`backends.chromatix.solver:propagate`, and they are two records on purpose -- one
answers "what backend does this project drive, and in which measured capability
row", the other "what happens to the physical state". Their `kind`, `approximation`
and `validity` all differed, and the completeness gate was keyed on
`(implementation, kind)` to allow exactly that one case.

**CHE-224 (R15.1) removed the need for it.** The sentence names two questions, and
naming two questions is the diagnosis: `kind` was being asked both, so the only way
to answer both was two records over one function. `backend` on
`OperationDescriptor` now answers "which library executes this" and `kind` answers
only "what happens to physical state". `S_WAVE_CHROMATIX` is deleted;
`O_ASM_PROPAGATE` carries `backend="chromatix"` and the one sentence of its
`approximation` that the pair only said between them. The gate is keyed on
`implementation` alone, because no case needs two.

What that cost and what it bought: a planner enumerating routes over the catalog
used to see two candidates for one callable and had no field distinguishing them
except prose, and `capabilities` was cited twice for one measured row. Nothing was
lost -- the two records' `kind` values were `solver` and `physical_operator`, and
`solver` was never a statement about physical state.

What is deliberately absent from the catalog
--------------------------------------------
* `backends.optiland.launch:launch` -- not in `backends.optiland.__all__`, and
  `src/backends/optiland/__init__.py` records why: it takes native solver state (a
  constructed `Optic`) and is package-facing by construction. A public launch
  operation needs a neutral signature first.
* `backends.optiland.solver:configure_execution` and every other public name that
  is not a semantic operation -- mask builders, unit converters, diagnostics
  records, enums, declaration tables. This is why completeness is checked against
  `OPERATIONS` rather than against `__all__`: `couplers.__all__` has 20 names of
  which 2 are operations, and deriving coverage from it would demand a descriptor
  for `DrawRule`. That count is asserted rather than asserted-in-prose --
  `tests/operations/test_catalog.py::test_the_counts_this_justification_rests_on`
  -- because adding a name to `__all__` is what makes it drift, and this ticket's
  own `OPERATIONS` export did exactly that.

The prose below is migrated, not rewritten
------------------------------------------
Eleven of these records were defined in test fixtures, because no production
home existed. Their `approximation`, `validity` and `evidence` text is
reviewed physics and was moved verbatim, so a reviewer can diff the fixture text
against this file. **Two exceptions, both flagged on CHE-221 rather than made
silently:** `O_PROPAGATE_RAYS` and `O_DIFFRACTIVE_SURFACE` each carried a validity
line saying the reconstruction kernel refuses a non-unit medium index "until R09".
CHE-192 lifted that refusal and both production modules say so, so migrating those
two lines verbatim would have put a false validity claim into the canonical
catalog. They are corrected to the current fact and nothing else about either
record changed.
"""

from __future__ import annotations

from operations.descriptors import OperationDescriptor, OperationKind

__all__ = ["CATALOG"]

#: Every operation this project has landed, one record each.
#:
#: Ordered by package and then by id, which is a reading order and nothing else:
#: `find()` sorts by id and the index is a dict, so nothing depends on this
#: sequence.
CATALOG: tuple[OperationDescriptor, ...] = (
    # --- backends/optiland ---------------------------------------------------
    # The one composite record -- CHE-225 (R15.2). `trace` is `build_lens` ->
    # `launch` (materialize and declare the rays) -> `lens.trace` (refract through
    # every surface) -> `to_ray_bundle`, so it initializes state AND evolves it.
    # CHE-224 declared it `SOURCE` and that was a false claim, contradicted by this
    # record's own `approximation` below: "a surface interaction is refraction at a
    # real interface". `kind` is the TERMINAL stage and `composes` carries the
    # fusion; `operations/descriptors.py` holds the retraction and the reason the
    # honest decomposition is blocked on numbers rather than on taxonomy.
    OperationDescriptor(
        operation_id="SO_RAY_LAUNCH_TRACE",
        kind=OperationKind.PHYSICAL_OPERATOR,
        composes=(OperationKind.SOURCE, OperationKind.PHYSICAL_OPERATOR),
        inputs=(),
        returns=("ray_bundle",),
        implementation="backends.optiland.solver:trace",
        backend="optiland",
        requires=("setup", "source", "sampling", "execution"),
        optional=("aiming",),
        approximation=(
            "sequential geometric ray tracing: rays are plane wavelets, diffraction "
            "is not modelled, and a surface interaction is refraction at a real "
            "interface"
        ),
        evidence=("tests/physics/test_optiland_rays.py",),
        capabilities="M_RAY_OPTILAND",
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="O_RAY_TRACE",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("ray_bundle",),
        returns=("ray_bundle",),
        implementation="backends.optiland.solver:trace_rays",
        backend="optiland",
        requires=("setup", "execution"),
        approximation=(
            "the same sequential geometric ray trace as SO_RAY_LAUNCH_TRACE, over an "
            "externally supplied ensemble rather than a generated pupil fan: the "
            "geometry evolves and the optical path is composed onto the incoming one, "
            "while the complex amplitude and the sampling measure are the caller's and "
            "cross unchanged. The amplitude is a sidecar -- |a|^2 goes into the solver "
            "only so the clipping bookkeeping is meaningful, and what comes back is "
            "read for exactly one purpose, deciding which rays survived"
        ),
        validity=(
            "the bundle must declare itself on the first surface after the object "
            "surface, in that surface's medium; both are checked",
            "the bundle must be coherent: a missing amplitude, a missing optical path "
            "or an 'unverified' path reference is refused rather than composed onto",
            "no sampling and no illumination is declared, because the rays ARE the "
            "sampling; the setup's entrance pupil and reference wavelength are not "
            "what is traced here",
            "survival keeps the row and zeroes the amplitude, so the output aligns "
            "with the caller's arrays row for row -- but a trace in which NO supplied "
            "ray survives is refused rather than returned as an all-zero bundle",
        ),
        evidence=("tests/backends/test_optiland_bundle_trace.py",),
        capabilities="M_RAY_OPTILAND",
        derivative="forward_only",
    ),
    # The second composite -- CHE-226 (R16), and the first three-stage one. The
    # pinned solver's own spot analysis: `build_lens` -> Optiland generates its own
    # pupil fan from the declared field -> it refracts them through every surface ->
    # it reduces the intersections to a spot. So the record initializes state,
    # evolves it AND observes it, and `kind` is the terminal stage as CHE-225 defines
    # it. `inputs=()` with a bare `measurement` kind is refused by
    # `OperationDescriptor.__post_init__` -- only a source may begin a graph entry --
    # and that refusal is right: this operation consumes no upstream representation
    # because it makes its own rays. `M_SPOT_DIAGRAM` below is the *other* path, and
    # the pair is the point: one generates rays, one consumes them as supplied.
    OperationDescriptor(
        operation_id="SOM_SPOT_DIAGRAM",
        kind=OperationKind.MEASUREMENT,
        composes=(
            OperationKind.SOURCE,
            OperationKind.PHYSICAL_OPERATOR,
            OperationKind.MEASUREMENT,
        ),
        inputs=(),
        returns=("spot",),
        implementation="backends.optiland.analysis:spot_diagram",
        backend="optiland",
        requires=("setup", "source", "num_rings", "execution"),
        optional=("distribution", "coordinates", "reference"),
        approximation=(
            "the same sequential geometric ray trace as SO_RAY_LAUNCH_TRACE, reduced to "
            "spot statistics by the pinned solver's own analysis rather than by this "
            "project: intensity selects rays and never weights a moment, the centroid "
            "is an unweighted mean, and the two radii are unweighted moments about "
            "the centre the `reference` argument selects -- the CHIEF RAY by the "
            "solver's default, which off axis is not the centroid the same call "
            "reports (backends.optiland.analysis.NATIVE_SPOT_METRIC_DEFINITIONS, read "
            "from the pinned implementation). Diffraction is not modelled, so a "
            "geometric spot smaller than the Airy radius is a statement about the rays "
            "and not about the image"
        ),
        validity=(
            "infinite-conjugate angular sources only: a source declaring a finite "
            "object distance is refused with NotImplementedError rather than having its "
            "field angle reinterpreted as a direction, because at a finite distance "
            "that angle is a position",
            "exactly one field and one wavelength, because build_lens declares exactly "
            "one of each -- so 'all' means 'the one declared' and a multi-field "
            "aberration curve is NOT what this returns",
            "no RayBundle exists anywhere in this call: the rays are generated inside "
            "the solver and are not observable, which is the whole difference from "
            "M_SPOT_DIAGRAM",
        ),
        evidence=("tests/backends/test_optiland_analysis.py",),
        capabilities="M_RAY_OPTILAND",
        derivative="forward_only",
    ),
    # The second native analysis on the same path -- CHE-236 (R16.1), which R16
    # said would be additive: "a second analysis here is a second function and a
    # second record". `method` is an `optional` argument and NOT a dispatcher: all
    # three of the pinned solver's scalar PSF implementations return one intensity
    # map under one normalization, so the return type does not vary with it and one
    # record describes the callable. Three records for one physical measurement
    # would be the false claim.
    OperationDescriptor(
        operation_id="SOM_PSF",
        kind=OperationKind.MEASUREMENT,
        composes=(
            OperationKind.SOURCE,
            OperationKind.PHYSICAL_OPERATOR,
            OperationKind.MEASUREMENT,
        ),
        inputs=(),
        returns=("psf",),
        implementation="backends.optiland.analysis:psf",
        backend="optiland",
        requires=("setup", "source", "method", "num_rays", "execution"),
        optional=("strategy", "remove_tilt", "grid_size", "image_size", "pixel_pitch_m"),
        approximation=(
            "scalar diffraction from a geometrically traced wavefront: the pinned solver "
            "samples the pupil, refracts the rays through every surface to the image "
            "surface, builds a reference sphere by the selected strategy, extends each "
            "ray's final direction BACKWARD in the image-space medium to that sphere -- a "
            "geometric image-space retrace only, not a retrace through the surfaces -- and "
            "turns the accumulated optical path into an OPD in waves relative to the "
            "reference ray. Diffraction enters only in the propagation of that pupil, so "
            "the aberration content is entirely geometric and no ray in the pupil is "
            "diffracted by an aperture edge. Which propagation is the `method` argument: "
            "'fft' is zero-padded FFT of sqrt(I) exp(-2i pi W) on the circular normalized "
            "pupil mask, with Zemax-compatible grid sampling; 'mmdft' is the same pupil "
            "through a matrix DFT with explicitly controlled output sampling; 'huygens' is "
            "a coherent sum over the physical 3-D reference-sphere intersections to the "
            "actual image-surface geometry, with the 1/R and obliquity factors. THE THREE "
            "ARE NOT INTERCHANGEABLE AT COARSE SAMPLING -- measured on the R05 singlet at "
            "num_rays=32, fft peaks at 99.91 and huygens at 32x32 peaks at 82.70"
        ),
        validity=(
            "scalar and monochromatic: one field, one wavelength, no polarization. The "
            "vectorial classes the backend also ships are not reachable from here",
            "infinite-conjugate angular sources only, for the reason SOM_SPOT_DIAGRAM is: "
            "at a finite object distance a field angle is a position",
            "exactly one field and one wavelength, because build_lens declares one of each",
            "the normalization is Optiland's Strehl-percent convention and NOT this "
            "project's PsfNormalization vocabulary; 100.0 is the unaberrated peak of the "
            "same aperture, so the number is a Strehl ratio times 100 and is blind to any "
            "constant multiplicative error in the same way peak normalization is",
            "no complex pupil, no ScalarField and no wavefront crosses this boundary -- "
            "only an intensity map -- so the phasor sign the backend uses internally "
            "(exp(-2i pi W)) never has to be reconciled with this project's convention",
            "the sampled window only, as M_PSF: energy outside the returned grid was not "
            "measured, and unless grid_size is given the fft path's grid is DERIVED from "
            "num_rays rather than chosen -- and num_rays is reduced in the same step, so "
            "the record reports the pupil sampling that ran and not the one requested",
        ),
        evidence=(
            "tests/backends/test_optiland_psf.py",
            "tests/physics/test_native_psf_airy.py",
        ),
        capabilities="M_RAY_OPTILAND",
        derivative="forward_only",
    ),
    # --- backends/chromatix --------------------------------------------------
    #
    # One record over `propagate`, not two. `S_WAVE_CHROMATIX` was the second, and
    # CHE-224 (R15.1) deleted it: it existed to answer "what backend does this
    # project drive", which `backend` now answers as a field. Its `approximation`
    # said what the scalar model itself omits -- one complex amplitude per sample,
    # no polarization, no vectorial coupling, complex64 because the backend has no
    # other field storage -- and that sentence is carried below rather than dropped,
    # because it is a different claim from what the angular-spectrum kernel
    # approximates and the surviving record did not already make it.
    OperationDescriptor(
        operation_id="O_ASM_PROPAGATE",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("scalar_field",),
        returns=("scalar_field",),
        implementation="backends.chromatix.solver:propagate",
        backend="chromatix",
        requires=("distance_m", "model"),
        approximation=(
            "the exact (non-paraxial) angular spectrum in a homogeneous isotropic "
            "medium: no Fresnel approximation and no term dropped, but the sampled "
            "window is periodic, so power that leaves it wraps back in unless the "
            "grid is padded. Scalar throughout: one complex amplitude per sample, "
            "no polarization and no vectorial coupling, evaluated in complex64 "
            "because the backend has no other field storage"
        ),
        validity=(
            "z <= N pitch^2 / lambda, the transfer function's own sampling bound",
            "pitch <= lambda / (2 sin theta_max) for the field's own largest angle",
        ),
        evidence=("tests/physics/test_scalar_wave_propagation.py",),
        capabilities="M_WAVE_CHROMATIX",
        derivative="forward_only",
    ),
    # The paraxial sibling of O_ASM_PROPAGATE -- CHE-228 (R06.11). A second record
    # over a second callable rather than a third `method` on `propagate`, for two
    # reasons. `O_ASM_PROPAGATE`'s `approximation` above says "no Fresnel
    # approximation and no term dropped", so a paraxial method under that record
    # would make the record's own prose false and put `O_ASM_` on a run that is not
    # one; and one record per `implementation` is the rule since CHE-224 (R15.1), so
    # a second record needs a second callable either way.
    #
    # The two kernels are one substitution apart and it is measured, not asserted:
    # replacing `delay + 1.0` with `2.0` in `_carrier_removed_propagator` -- the
    # k_z -> n k0 limit -- reproduces the backend's Fresnel phase with a maximum
    # difference of exactly 0.0 in float32 over a 512^2 grid.
    OperationDescriptor(
        operation_id="O_FRESNEL_PROPAGATE",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("scalar_field",),
        returns=("scalar_field",),
        implementation="backends.chromatix.solver:fresnel_propagate",
        backend="chromatix",
        requires=("distance_m", "model"),
        approximation=(
            "the Fresnel (paraxial) transfer function in a homogeneous isotropic "
            "medium: exp(-i pi (lambda_0/n) z f^2), which is exactly "
            "O_ASM_PROPAGATE's carrier-removed kernel with the axial ratio "
            "k_z/(n k0) replaced by 1. One term IS dropped, and that is the whole "
            "difference from O_ASM_PROPAGATE: the phase error is "
            "n k0 z (1 - cos theta - sin^2(theta)/2), i.e. n k0 z sin^4(theta)/8 to "
            "leading order, at each direction cosine sin(theta) = lambda_0 f / n. "
            "The kernel carries NO exp(i k n z) factor, so unlike the angular "
            "spectrum there is no absolute-phase variant to choose: the result is "
            "always relative to a removed piston of carrier_phase_rad(lambda_0, z, "
            "n), the same constant 'asm_carrier_removed' removes. Sampling is "
            "preserved, because the transfer method is a convolution; the sampled "
            "window is periodic, so power that leaves it wraps back in unless the "
            "grid is padded. Scalar throughout: one complex amplitude per sample, "
            "no polarization and no vectorial coupling, evaluated in complex64 "
            "because the backend has no other field storage"
        ),
        validity=(
            "sin(theta_max) <= (lambda_0 / (n z))^(1/4) for the field's own largest "
            "direction cosine -- the angle at which the leading phase error reaches "
            "pi/4. On the Chromatix 101 tutorial's own grid (512^2, dx = 0.3 um, "
            "lambda_0 = 0.532 um, n = 1.33, z = 50 um) that is sin(theta) <= 0.299, "
            "i.e. 17.4 degrees, while the grid's own per-axis Nyquist is "
            "sin(theta) = 0.667, where the exact error is 25.5 rad, and its corner "
            "is 0.943, where it is 175 rad",
            "the returned field declares 'paraxial', and that flag is the only "
            "warning a consumer gets: the error is a phase error and |U|^2 does not "
            "show it. Measured on that same grid, a hard-edged square aperture "
            "differs from the exact angular spectrum by 2.3e-1 of peak intensity -- "
            "pad-independent, so it is the approximation and not wraparound -- while "
            "a soft-edged field on the identical grid differs by 4.9e-6",
            "z <= N pitch^2 / lambda, the transfer function's own sampling bound, "
            "the same one O_ASM_PROPAGATE carries",
            "carrier_removed_phase: the phase is relative to a removed piston, and "
            "two fields with different removed pistons may not be interfered "
            "directly. This one removes the same constant asm_carrier_removed does",
            "a tilted beam lands at z sin(theta), not z tan(theta): the kernel's "
            "group delay is lambda_0 z f / n, linear in spatial frequency. That is a "
            "property of the model rather than a defect in it, and it is the same "
            "sine-condition content O_FOCAL_PLANE_TRANSFORM's f sin(theta) records",
        ),
        evidence=("tests/physics/test_fresnel_propagation.py",),
        capabilities="M_WAVE_CHROMATIX",
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="O_FOCAL_PLANE_TRANSFORM",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("scalar_field",),
        returns=("scalar_field",),
        implementation="backends.chromatix.focal_plane:focal_plane_transform",
        backend="chromatix",
        requires=("focal_length_m", "model"),
        approximation=(
            "the ideal thin lens between its two focal planes: one optical Fourier "
            "transform, so spatial frequency maps linearly onto position and a plane "
            "wave at theta focuses at f sin(theta) rather than f tan(theta). No "
            "aberration, no thickness, no pupil, and the exp(i k n 2f) piston of the "
            "textbook relation is not carried"
        ),
        validity=(
            "the output grid is lambda f / (n N dx) per axis, so content beyond that "
            "window is not represented; padding is refused because it would change N",
            "carrier_removed_phase: the phase is relative to a removed piston, and two "
            "fields with different removed pistons may not be interfered directly",
        ),
        evidence=("tests/physics/test_focal_plane_transform.py",),
        capabilities="M_WAVE_CHROMATIX",
        derivative="forward_only",
    ),
    # --- sources ------------------------------------------------------------
    #
    # `inputs=()` on all three: a source consumes no upstream representation, which
    # is the whole of `docs/architecture_principles.md` §2's definition of one.
    # Until CHE-222 (R03.5) the schema could not say that, and all three of these
    # (well, the one that existed) declared `input="scalar_field"` -- the
    # representation they *produce*, named on both sides. `ENTRY_KINDS` is what
    # makes `()` a checked declaration: only a `source` may be a graph entry.
    #
    # `kind=OperationKind.SOURCE` since CHE-224 (R15.1). All three were `SOLVER`
    # before it, because the enum had no `SOURCE` member -- so the `S_` on these
    # three ids meant "source" while the `S_` on `S_RAY_OPTILAND` meant "solver",
    # a record CHE-225 (R15.2) then renamed to `SO_RAY_LAUNCH_TRACE`,
    # and `kind` could not tell a reader which.
    #
    # What a source consumes instead is in `requires`: a grid `shape`, a pitch, a
    # wavelength and a reference surface, plus the one geometric parameter that
    # distinguishes it. Those are not representations and never were, which is why
    # naming one on `inputs` was a false claim rather than an approximation.
    #
    # `capabilities=None` on all three is the honest citation: none of them imports
    # a backend, so none has a measured device/dtype row, and citing the chromatix
    # row would claim a measurement taken about something else.
    OperationDescriptor(
        operation_id="S_SOURCE_GAUSSIAN_BEAM",
        kind=OperationKind.SOURCE,
        inputs=(),
        returns=("scalar_field",),
        implementation="sources.gaussian_beam:gaussian_beam",
        requires=(
            "shape",
            "sample_pitch_m",
            "wavelength_m",
            "reference_surface",
            "waist_radius_m",
        ),
        optional=("center_m", "transverse_wavevector_rad_per_m", "amplitude"),
        approximation=(
            "an ideal monochromatic, fully coherent, scalar Gaussian beam **at its "
            "waist plane**: A exp(-rho^2 / w0^2) times exactly the carrier ramp "
            "plane_wave writes. At the waist the field is a real envelope times the "
            "carrier and nothing else -- no wavefront curvature, no Gouy phase, no "
            "w(z) -- so it is exact at its declared surface rather than paraxial. "
            "w0 is the 1/e AMPLITUDE radius, hence the 1/e^2 intensity radius; the "
            "competing exp(-rho^2 / (2 w0^2)) reading differs by sqrt(2) in the waist "
            "and produces a plausible beam of the wrong size with no downstream "
            "signature. The amplitude is a relative peak, not a radiometric power"
        ),
        validity=(
            "the waist plane only: the function takes no z argument, because an "
            "off-waist Gaussian is a paraxial solution and no ValidityFlag says "
            "'paraxial'",
            "|k_t| <= n k0 and |k_t| <= pi/d per axis, the same two carrier refusals "
            "plane_wave uses, from the shared sources._grid",
            "at least two samples across w0 per axis; an unresolved waist is refused, "
            "because it reads back as a beam of whatever size the grid can represent",
            "truncation is documented and NOT refused: a grid half-extent of 1.5 w0 "
            "leaves ~1e-4 of the power outside, 2 w0 ~1e-7, 1 w0 ~2e-2, and a "
            "truncated Gaussian rings when it is propagated",
            "the field is complex64; the envelope and the phase ramp are accumulated "
            "in float64 before the cast",
        ),
        evidence=(
            "tests/sources/test_gaussian_beam.py",
            "tests/physics/test_coherent_sources.py",
        ),
        capabilities=None,
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="S_SOURCE_PLANE_WAVE",
        kind=OperationKind.SOURCE,
        inputs=(),
        returns=("scalar_field",),
        implementation="sources.plane_wave:plane_wave",
        requires=("shape", "sample_pitch_m", "wavelength_m", "reference_surface"),
        optional=("transverse_wavevector_rad_per_m", "amplitude"),
        approximation=(
            "an ideal monochromatic, fully coherent, scalar plane wave sampled at a "
            "declared surface: A exp(i(k_y y + k_x x)) with k_t in rad/m. No spectral "
            "width, no partial coherence, no polarization, and no physical model of an "
            "illumination unit. The amplitude is a relative peak, not a radiometric "
            "power: chromatix's power= renormalization is deliberately not inherited"
        ),
        validity=(
            "|k_t| <= n k0, checked against the surface's own medium index: a larger "
            "value is an evanescent wave, not an illumination angle",
            "|k_t| <= pi/d per axis: past the grid's Nyquist limit the sampled ramp "
            "aliases and reads back as a different, entirely plausible angle",
            "the field is complex64, which is the one storage dtype of this project's "
            "wave path; the phase ramp is accumulated in float64 before the cast",
        ),
        evidence=(
            "tests/sources/test_plane_wave.py",
            "tests/physics/test_coherent_sources.py",
        ),
        capabilities=None,
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="S_SOURCE_SPHERICAL_WAVE",
        kind=OperationKind.SOURCE,
        inputs=(),
        returns=("scalar_field",),
        implementation="sources.spherical_wave:spherical_wave",
        requires=(
            "shape",
            "sample_pitch_m",
            "wavelength_m",
            "reference_surface",
            "source_position_m",
        ),
        optional=("amplitude", "converging"),
        approximation=(
            "the analytic spherical field of a point emitter, sampled on a plane: "
            "A (R_ref / R) exp(+/- i n k0 R) with R_ref = 1 m, exact at its declared "
            "surface. This is the project's wave-optics point source, as opposed to a "
            "single nonzero pixel, which is a delta whose spectrum is flat to the "
            "grid's Nyquist limit and therefore aliased by construction. The sign "
            "follows the project phasor: diverging is exp(+i n k0 R), converging is "
            "its exact complex conjugate, and conjugating the field swaps the two with "
            "no signature in any intensity. A is the amplitude at R = 1 m and is "
            "therefore dimensional, unlike plane_wave's and gaussian_beam's peaks"
        ),
        validity=(
            "|rho - rho_s| / R < lambda_0 / (2 n d) per axis, i.e. NA < lambda_0 / "
            "(2 d) independent of n: the local phase gradient of exp(i n k0 R) grows "
            "with lateral offset and is largest at the grid corner furthest from the "
            "source, and an under-sampled spherical wave reads back as a real ray at a "
            "smaller angle",
            "a source ON the sampled plane is refused outright, and so is any geometry "
            "whose smallest R on the grid is below one sample pitch: a 1/R evaluated "
            "inside a sample varies by orders of magnitude between adjacent samples",
            "the geometry/direction pairing is documented and NOT refused: "
            "converging=False with a downstream source is the conjugate field, "
            "travelling in -z, which this project's forward SPATIAL_FACTOR cannot carry",
            "no aperture argument: truncation composes through the thin-element "
            "operator, which is strictly more expressive than a hard disc here",
        ),
        evidence=(
            "tests/sources/test_spherical_wave.py",
            "tests/physics/test_coherent_sources.py",
        ),
        capabilities=None,
        derivative="forward_only",
    ),
    # --- operators ----------------------------------------------------------
    OperationDescriptor(
        operation_id="O_COMPLEX_TRANSMISSION",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("scalar_field",),
        returns=("scalar_field",),
        implementation="operators.transmission:complex_transmission",
        optional=("amplitude", "phase_rad", "target_surface", "allow_gain"),
        approximation=(
            "an infinitely thin element acting at the field's own reference surface: "
            "U_out = U_in * A * exp(i phi), elementwise. z_m does not advance, no "
            "propagation happens inside it, and there is no thickness, no multiple "
            "scattering, no polarization and no angular dependence of the "
            "transmission -- which is also what makes tilt-as-spectral-shift exact "
            "rather than approximate for this element"
        ),
        validity=(
            "the transmission is sampled on the field's own grid, so a mask with "
            "structure finer than the pitch is aliased rather than resolved",
            "A is a real non-negative modulus bounded by 1 unless gain is claimed",
            "surface_only fields are permitted: the element acts exactly at the "
            "surface where such a field is valid",
        ),
        evidence=(
            "tests/operators/test_transmission.py",
            "tests/physics/test_thin_element_spectrum.py",
        ),
        capabilities=None,
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="O_DIFFRACTIVE_SURFACE",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("ray_bundle",),
        returns=("ray_bundle", "diagnostics"),
        implementation="operators.diffractive_surface:diffractive_surface",
        requires=("surface",),
        optional=(
            "model",
            "reconstruction",
            "kspace_oversample",
            "kspace_grid_shape",
            "allow_gain",
            "order",
            "patch_px",
            "pad_factor",
            "window",
            "error_threshold_rad",
            "count",
            "density",
            "draw",
            "rng",
            "seed",
            "launch_positions_xy_m",
        ),
        approximation=(
            "the full-field model: every incident ray is accumulated coherently onto "
            "the surface's own grid, the complex transmission is applied once as a "
            "thin element, and the transmitted field is decomposed into the modes "
            "that leave. Exact for a thin, angle-independent transmission on one "
            "common plane; the interior field carries no exp(i k r^2 / 2R) "
            "wavefront-curvature term (CHE-50) and is valid at the surface with zero "
            "further propagation, both of which the emitted bundle declares"
        ),
        validity=(
            "the incident bundle must already be expressed on the surface; this "
            "operation does not propagate",
            "one common plane, i.e. a planar substrate; a conformal one has no such "
            "plane and needs the local-patch model",
            # Migrated with a correction, flagged on CHE-221: the fixture said "air
            # only until the ray<->wave ramp convention carries the refractive index
            # (R09)". CHE-192 put the `n` in and this composition inherited the fix
            # from its parts, which the production module's own docstring records.
            "the medium index comes from the parts: both couplers carry n in their "
            "ramps since CHE-192, so this composition is not restricted to air",
            "the surface's grid is the reconstruction grid, so its pitch must "
            "represent the steepest wavelet ramp of both the incident and the "
            "transmitted spectrum",
        ),
        evidence=("tests/physics/test_diffractive_surface_full_field.py",),
        capabilities=None,
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="O_PROPAGATE_RAYS",
        kind=OperationKind.PHYSICAL_OPERATOR,
        inputs=("ray_bundle",),
        returns=("ray_bundle",),
        implementation="operators.ray_propagation:propagate_rays",
        requires=("to",),
        optional=("phase_budget_rad",),
        approximation=(
            "exact rather than approximate: each ray advances along its own direction "
            "by the arc length s = dz / d_z and its optical path grows by n s, which "
            "changes each wavelet's constant phase by n k d_z dz -- precisely what a "
            "plane wave accumulates over the axial offset. Directions are unchanged, "
            "so nothing refracts, and the sampling measure is unchanged, because a "
            "plane wavelet's coefficient was fixed by the quadrature at the surface "
            "the rays were originally declared on"
        ),
        validity=(
            "one medium: the source and target surfaces must declare the same index, "
            "because two indices do not bound one medium",
            "the target must be perpendicular to the propagation axis",
            "every ray must reach the target; one that does not is refused, not dropped",
            "|d_z| must clear a floor derived from the phase the arc length would "
            "carry at the optical path's own precision",
            # Migrated with a correction, flagged on CHE-221: the fixture said the
            # reconstruction kernel "implements the n = 1 ramp and refuses n != 1, so a
            # bundle advanced through a medium cannot yet be reconstructed (recorded on
            # CHE-192)". CHE-192's follow-up put the `n` in and lifted the refusal.
            "correct for any medium index, and reconstructable in one: the ray-to-wave "
            "kernel's ramp carries n since CHE-192, so a bundle advanced through a "
            "medium is no longer refused downstream",
        ),
        evidence=("tests/physics/test_ray_propagation.py",),
        capabilities=None,
        derivative="forward_only",
    ),
    # --- couplers -----------------------------------------------------------
    OperationDescriptor(
        operation_id="C_RAY_TO_SCALAR",
        kind=OperationKind.COUPLER,
        inputs=("ray_bundle",),
        returns=("scalar_field", "reconstruction_diagnostics"),
        implementation="couplers.ray_to_scalar:ray_to_scalar",
        requires=("grid_shape", "sample_pitch_m"),
        optional=(
            "surface",
            "projection",
            "reconstruction",
            "kspace_oversample",
            "kspace_grid_shape",
            "grazing",
            "phase_budget_rad",
        ),
        approximation=(
            "each ray is a plane wavelet contributing a linear phase ramp across the "
            "whole surface, summed as a quadrature over the declared sampling "
            "measure. The sum is linear in the transverse coordinate, so the field "
            "carries no exp(i k r^2 / 2R) wavefront-curvature term (CHE-50) and is "
            "valid at the declared surface with zero further propagation. The scale "
            "omits the 1/(i lambda z) Kirchhoff prefactor, so U is i lambda z times "
            "the SI field and every reported power is relative"
        ),
        validity=(
            "the output grid must represent the steepest ramp, |d_t| <= lambda_0 / "
            "(2 n pitch) per axis; beyond it the reconstruction is refused",
            "the transverse ramp and the launch-ramp subtraction carry the surface's "
            "medium index; the optical path is already an optical one",
            "the bundle must declare its integration measure; 'undeclared' is refused",
            "fully coherent, scalar, monochromatic",
        ),
        evidence=("tests/physics/test_ray_to_scalar.py",),
        capabilities=None,
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="C_SCALAR_TO_RAY",
        kind=OperationKind.COUPLER,
        inputs=("scalar_field",),
        returns=("ray_bundle", "sampling_diagnostics"),
        implementation="couplers.scalar_to_ray:scalar_to_ray",
        optional=(
            "surface",
            "count",
            "density",
            "draw",
            "rng",
            "seed",
            "launch_positions_xy_m",
        ),
        approximation=(
            "the field is decomposed into plane-wave modes on its own grid and each "
            "selected mode becomes a ray. Evanescent modes are discarded -- they have "
            "no propagation direction to give a ray -- and the discarded power is "
            "reported. A stochastic selection is a Monte-Carlo estimator of the modal "
            "sum, so the emitted measure is an importance weight and the "
            "reconstruction owes a 1/N; an exhaustive enumeration is the same "
            "estimator with zero variance"
        ),
        validity=(
            "scalar, monochromatic, fully coherent",
            "the direction cosines are lambda_0 f / n and the evanescent cut is "
            "|k_t| < n k0, both on the surface's declared medium index",
            "the field's grid fixes the mode set; a mode finer than the pitch is not "
            "represented",
            "exhaustive enumeration is exact only under the uniform density",
        ),
        evidence=(
            "tests/physics/test_scalar_to_ray.py",
            "tests/physics/test_scalar_to_ray_estimator.py",
        ),
        capabilities=None,
        derivative="forward_only",
    ),
    # --- measurements -------------------------------------------------------
    OperationDescriptor(
        operation_id="M_PSF",
        kind=OperationKind.MEASUREMENT,
        inputs=("scalar_field",),
        returns=("psf",),
        implementation="measurements.psf:psf",
        requires=("normalization",),
        approximation=(
            "none in the reduction itself: intensity is |u|^2 exactly, and the "
            "declared normalization is an exact scaling of it. What the caller must "
            "know is that peak and energy normalization are both blind to a constant "
            "multiplicative error in the field, so the unscaled peak and window "
            "energy are recorded on every result"
        ),
        validity=(
            "monochromatic, fully coherent, scalar",
            "the sampled window only: energy that left the grid is not measured, and "
            "border_energy_fraction is the indicator for it",
        ),
        evidence=("tests/physics/test_psf.py",),
        capabilities=None,
        derivative="forward_only",
    ),
    OperationDescriptor(
        operation_id="M_SPOT_DIAGRAM",
        kind=OperationKind.MEASUREMENT,
        inputs=("ray_bundle",),
        returns=("spot",),
        implementation="measurements.spot:spot_diagram",
        approximation=(
            "none in the reduction itself: the coordinates are the bundle's own "
            "positions at its declared reference surface, untransformed, and the three "
            "metrics are exact moments of them. What the caller must know is which "
            "moments -- the centroid and the RMS radius are weighted by |a_i|^2 and "
            "referred to that centroid, the geometric radius is an unweighted maximum "
            "about it, and the sampling measure is deliberately NOT applied "
            "(measurements.spot.SPOT_WEIGHTING)"
        ),
        validity=(
            "the bundle must declare ray_splitting='unsplit'. A population containing "
            "split-ray descendants, and one whose splitting provenance is undeclared, "
            "are both refused -- with different codes -- because an unweighted moment "
            "over branches of one incident ray is a statistic of the branching, and "
            "nothing here infers provenance from the numbers",
            "the bundle must carry an amplitude: it is where the per-ray intensity and "
            "the survival of a ray both live, and weighting every row equally instead "
            "would measure clipped rays as delivered ones",
            "rays with |a|^2 = 0 are excluded as undelivered, so included_count may be "
            "below ray_count; a bundle in which no ray survived is refused rather than "
            "returned as a spot at the origin",
            "under a non-uniform sampling density this is a sampling-weighted moment of "
            "the irradiance rather than the irradiance moment; measure_kind is where a "
            "caller reads which it has",
        ),
        evidence=("tests/physics/test_spot_diagram.py",),
        capabilities=None,
        derivative="forward_only",
    ),
)
