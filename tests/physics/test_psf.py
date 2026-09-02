"""R11.1: the PSF as a measurement, with exactly one implementation.

CHE-197. `measurements.psf(field, *, normalization) -> PsfResult`.

The intensity arithmetic here is trivial and the tests are not about it. They are
about the three things that were actually wrong in the tree this replaces:

1. **There were two paths**, and they were not the same computation -- one took
   `|u|^2` in NumPy on the host and one took it in the field's own namespace, and
   nothing compared them.
2. **The normalization could hide a scale error.** Peak normalization is blind to
   any constant multiplicative factor, every M3 oracle divided by the peak, and
   that is how a propagated power of `7.0e-04` and one of `2.7e-24` both looked
   fine. The measurement has to be able to *see* the scale it removed, or it
   cannot serve as a check on R07.
3. **PSF was a representation**, which is what made a coupler to it look
   reasonable. It is an observable of a terminal state, and the state stays the
   field.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from ray_support import WAVELENGTH_M

from measurements import (
    COHERENCE_MODEL,
    NORMALIZATION_DECLARATIONS,
    PSF_INVARIANTS,
    PSF_NORMALIZATIONS,
    PsfResult,
    border_energy_fraction,
    psf,
)
from operations import CATALOG, OperationKind, registry, resolve
from representations import ContractError, Frame, ReferenceSurface, ScalarField

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PACKAGE = SRC / "measurements"

SHAPE = (32, 32)
PITCH_M = (0.30e-6, 0.25e-6)
SENSOR = ReferenceSurface(name="sensor", z_m=0.0, medium_index=1.0)

#: A deliberately anisotropic pitch. A square one would let a transposed axis, or
#: a `(dx, dy)` read as `(dy, dx)`, agree with the right answer everywhere.
GRID_Y, GRID_X = np.meshgrid(
    (np.arange(SHAPE[0]) - SHAPE[0] // 2) * PITCH_M[0],
    (np.arange(SHAPE[1]) - SHAPE[1] // 2) * PITCH_M[1],
    indexing="ij",
)


def a_field(
    *, scale: float = 1.0, dtype: str = "complex128", width_m: float = 2.0e-6
) -> ScalarField:
    """A confined Gaussian, optionally rescaled by a constant amplitude factor."""
    amplitude = scale * np.exp(-(GRID_X**2 + GRID_Y**2) / width_m**2)
    return ScalarField(
        u=amplitude.astype(dtype),
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=SENSOR,
    )


# ---------------------------------------------------------------------------
# 1. The arithmetic, and the one place it happens
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("normalization", PSF_NORMALIZATIONS)
def test_the_intensity_is_the_reference_expression_to_the_last_bit(
    normalization: str,
) -> None:
    """Criterion: the definitions are `|u|^2` and one of three scalings, exactly.

    The oracle is the expression itself, written out here in NumPy rather than
    read back from the module under test -- which is the whole content of a
    definitional check. `==` rather than `approx`: there is no approximation in
    any of these, so a tolerance would only hide a different formula.
    """
    field = a_field()
    u = np.asarray(field.u)
    dy, dx = field.sample_pitch_m
    raw = np.abs(u) ** 2
    expected_scale = {
        "raw": 1.0,
        "peak": 1.0 / float(raw.max()),
        "energy": 1.0 / (float(raw.sum()) * dy * dx),
    }[normalization]

    result = psf(field, normalization=normalization)  # type: ignore[arg-type]
    assert result.scale_factor == expected_scale
    assert np.array_equal(np.asarray(result.intensity), raw * expected_scale)
    assert result.raw_peak_intensity == float(raw.max())
    assert result.raw_window_energy == float(raw.sum()) * dy * dx


def test_peak_normalization_puts_the_peak_at_one_and_energy_integrates_to_one() -> None:
    """The two scalings say what they are named for, on the sampled window."""
    field = a_field()
    dy, dx = field.sample_pitch_m
    peak = psf(field, normalization="peak")
    energy = psf(field, normalization="energy")
    assert float(np.max(np.asarray(peak.intensity))) == pytest.approx(1.0, rel=1e-12)
    assert float(np.sum(np.asarray(energy.intensity))) * dy * dx == pytest.approx(
        1.0, rel=1e-12
    )


def test_the_measurement_resamples_nothing_and_moves_no_origin() -> None:
    """Axes are the field's own, on the field's own origin rule.

    `coordinates()` delegates to `Frame.origin_index`, the same call
    `ScalarField.coordinates` makes, so a measurement cannot quietly adopt a
    different centring. A half-pixel shift is a large fraction of an Airy radius
    at ordinary PSF sampling, and it is invisible in the intensity map.
    """
    field = a_field()
    result = psf(field, normalization="raw")
    assert result.shape == field.shape
    assert result.sample_pitch_m == field.sample_pitch_m
    assert result.frame == field.frame
    field_y, field_x = field.coordinates()
    psf_y, psf_x = result.coordinates()
    assert np.array_equal(np.asarray(psf_y), np.asarray(field_y))
    assert np.array_equal(np.asarray(psf_x), np.asarray(field_x))


def test_the_peak_is_reported_by_index_and_by_position() -> None:
    """A displaced maximum, so the origin rule is doing work rather than agreeing."""
    u = np.zeros(SHAPE, dtype=complex)
    row, col = SHAPE[0] // 2 + 3, SHAPE[1] // 2 - 5
    u[row, col] = 2.0
    field = ScalarField(
        u=u,
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=SENSOR,
    )
    result = psf(field, normalization="raw")
    assert result.peak_index == (row, col)
    assert result.peak_position_m == (3 * PITCH_M[0], -5 * PITCH_M[1])
    assert result.raw_peak_intensity == 4.0


# ---------------------------------------------------------------------------
# 2. The risk: peak normalization is blind to a constant
# ---------------------------------------------------------------------------


def test_a_global_scale_error_is_invisible_in_the_profile_and_visible_in_the_record() -> None:
    """**The reason `raw_peak_intensity` and `raw_window_energy` are always kept.**

    Two fields differing by a constant amplitude factor of `2^33` -- far larger
    than the omitted per-ray area weight that motivated CHE-47, and of the same
    kind -- produce **bit-identical** peak-normalized intensity maps. Every
    comparison an oracle can make on that map is therefore silent about a factor of
    `2^66` in power. (A power of two, so that "bit-identical" is a statement about
    this measurement rather than about how the scaling rounded.)

    That is not a hypothetical: it is how the pre-CHE-47 launch-amplitude
    convention survived a peak-normalized oracle.

    The record is where it shows. `raw_peak_intensity` and `raw_window_energy` are
    recorded before scaling and under *every* normalization, including `peak`, so
    a caller checking an upstream reconstruction's absolute scale has the number.
    """
    factor = 2.0**33
    faint = psf(a_field(scale=1.0), normalization="peak")
    bright = psf(a_field(scale=factor), normalization="peak")

    assert np.array_equal(np.asarray(faint.intensity), np.asarray(bright.intensity))
    assert faint.peak_index == bright.peak_index
    assert faint.border_energy_fraction == pytest.approx(bright.border_energy_fraction)

    assert bright.raw_peak_intensity / faint.raw_peak_intensity == factor**2
    assert bright.raw_window_energy / faint.raw_window_energy == pytest.approx(
        factor**2, rel=1e-12
    )
    # ...and the scale factor is the inverse of what it removed, so the two are
    # recoverable from each other rather than merely both present.
    assert bright.scale_factor * bright.raw_peak_intensity == pytest.approx(1.0)


@pytest.mark.parametrize("normalization", PSF_NORMALIZATIONS)
def test_the_raw_scale_is_recorded_under_every_normalization(normalization: str) -> None:
    """Including `energy`, which hides the same constant and adds a window
    dependence of its own."""
    field = a_field(scale=3.0)
    result = psf(field, normalization=normalization)  # type: ignore[arg-type]
    assert result.raw_peak_intensity > 0.0
    assert result.raw_window_energy > 0.0
    assert result.as_dict()["raw_peak_intensity"] == result.raw_peak_intensity
    assert "watts" in result.as_dict()["raw_energy_units"]


def test_a_consumer_can_never_be_unsure_whether_a_value_is_peak_normalized() -> None:
    """Criterion 2. The choice travels as a value *and* as the sentence for it."""
    for normalization in PSF_NORMALIZATIONS:
        result = psf(a_field(), normalization=normalization)  # type: ignore[arg-type]
        assert result.normalization == normalization
        assert result.normalization_declaration == NORMALIZATION_DECLARATIONS[normalization]
        assert result.as_dict()["normalization"] == normalization
        assert result.coherence_model == COHERENCE_MODEL
    assert "Blind to any constant" in NORMALIZATION_DECLARATIONS["peak"]


def test_the_normalization_has_no_default_and_an_unknown_one_is_refused() -> None:
    """An implicitly normalized PSF entering an oracle is the failure the required
    keyword exists to prevent."""
    with pytest.raises(TypeError):
        psf(a_field())  # type: ignore[call-arg]
    with pytest.raises(ContractError) as raised:
        psf(a_field(), normalization="normalised")  # type: ignore[arg-type]
    assert raised.value.code == "MISSING_DECLARATION"
    assert raised.value.declaration == "normalization"


# ---------------------------------------------------------------------------
# 3. Truncation, and the indicator that does not overclaim
# ---------------------------------------------------------------------------


def test_a_truncated_psf_is_visible_rather_than_quietly_wrong() -> None:
    """Criterion 3. A normalized profile of a truncated PSF looks like a profile.

    Two fields on the same grid: a Gaussian confined well inside the window, and a
    broad one filling it. Their peak-normalized maps are both smooth and both
    perfectly plausible; the border fraction is what separates them, by more than
    seven orders of magnitude.
    """
    contained = psf(a_field(width_m=1.0e-6), normalization="peak")
    spilling = psf(a_field(width_m=60.0e-6), normalization="peak")
    assert contained.border_energy_fraction < 1e-9
    assert spilling.border_energy_fraction > 1e-2
    assert float(np.max(np.asarray(spilling.intensity))) == pytest.approx(1.0)


def test_the_border_fraction_is_the_border_and_counts_each_corner_once() -> None:
    """The definition, against a hand-countable array: a uniform `n x n` window has
    `(4n - 4) / n^2` of its energy on the one-pixel border."""
    for n in (3, 5, 16):
        uniform = np.ones((n, n))
        assert border_energy_fraction(uniform) == pytest.approx((4 * n - 4) / n**2)
    # A dark window has no total to divide by, and a window narrower than three
    # samples has no interior -- reporting total truncation there would be an
    # artifact of the definition rather than a measurement.
    assert border_energy_fraction(np.zeros((8, 8))) == 0.0
    assert border_energy_fraction(np.ones((2, 9))) == 0.0


def test_a_dark_field_is_refused_rather_than_divided_by() -> None:
    """NaN would be caught one layer later as a non-finite intensity, which names
    the symptom rather than the cause."""
    dark = ScalarField(
        u=np.zeros(SHAPE, dtype=complex),
        sample_pitch_m=PITCH_M,
        wavelength_m=WAVELENGTH_M,
        reference_surface=SENSOR,
    )
    for normalization in ("peak", "energy"):
        with pytest.raises(ContractError) as raised:
            psf(dark, normalization=normalization)  # type: ignore[arg-type]
        assert raised.value.code == "EMPTY_ENSEMBLE"
        assert "propagation ran" in (raised.value.remedy or "")
    # `raw` has nothing to divide by, so it measures the dark field and says so.
    assert psf(dark, normalization="raw").raw_window_energy == 0.0


# ---------------------------------------------------------------------------
# 4. The invariants the retired coupler entry declared
# ---------------------------------------------------------------------------


def test_the_two_retired_registry_invariants_are_executed_here() -> None:
    """`nonnegative_intensity` and `declared_psf_normalization`, under their
    original names, enforced by the type rather than asserted by an edge.

    Non-negativity is not vacuous even though `psf()` only ever builds this from
    `|u|^2`: `PsfResult` is a public frozen dataclass, and an amplitude stored
    where an intensity was expected is the one substitution that yields a
    plausible-looking map with negative values in it.
    """
    assert PSF_INVARIANTS == ("nonnegative_intensity", "declared_psf_normalization")
    fields = {
        "intensity": np.ones((4, 4)),
        "sample_pitch_m": PITCH_M,
        "wavelength_m": WAVELENGTH_M,
        "normalization": "raw",
        "normalization_declaration": NORMALIZATION_DECLARATIONS["raw"],
        "scale_factor": 1.0,
        "raw_peak_intensity": 1.0,
        "raw_window_energy": 1.0,
        "peak_index": (0, 0),
        "peak_position_m": (0.0, 0.0),
        "border_energy_fraction": 0.75,
    }
    for override, code in (
        ({"intensity": np.array([[1.0, -1e-30], [1.0, 1.0]])}, "NEGATIVE_INTENSITY"),
        ({"normalization": "photons"}, "MISSING_DECLARATION"),
        ({"intensity": np.ones(4)}, "SHAPE_MISMATCH"),
        ({"sample_pitch_m": (0.0, 1e-6)}, "UNIT_NOT_SI"),
    ):
        with pytest.raises(ContractError) as raised:
            PsfResult(**{**fields, **override})  # type: ignore[arg-type]
        assert raised.value.code == code, override

    assert PsfResult(**fields).as_dict()["invariants_enforced"] == list(PSF_INVARIANTS)  # type: ignore[arg-type]


def test_the_intensity_stays_in_the_fields_own_precision() -> None:
    """The divergence between the two reference paths, pinned.

    `measure_psf` squared in NumPy on the host and `PSF.from_complex_field` squared
    in the field's namespace, so a complex64 field produced a float32 intensity
    beside float64 scalars computed from a second squaring of transferred data.
    One array now, and a complex64 field yields a float32 PSF -- R02.4's rule that
    nothing fabricates float64 digits a producer never had.
    """
    result = psf(a_field(dtype="complex64"), normalization="raw")
    assert np.asarray(result.intensity).dtype == np.float32
    assert psf(a_field(dtype="complex128"), normalization="raw").intensity.dtype == np.float64
    # The peak scalar is read off that same array, so it is exactly an element of
    # it rather than a more precise recomputation of one.
    assert result.raw_peak_intensity == float(np.max(np.asarray(result.intensity)))


# ---------------------------------------------------------------------------
# 5. Architecture: one path, one class, a measurement and not a coupler
# ---------------------------------------------------------------------------


def _module_sources() -> dict[Path, str]:
    return {
        path: path.read_text(encoding="utf-8")
        for path in sorted(SRC.rglob("*.py"))
        if "__pycache__" not in str(path)
    }


def _identifiers(source: str) -> set[str]:
    """Every name used as *code*: an identifier, an attribute, a definition.

    Deliberately not a substring search over the file. Both walks below are about
    whether a second implementation exists, and a docstring that names the path it
    replaced -- which is exactly what `measurements/psf.py` does, at length -- is
    the opposite of a violation.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name):
            found.add(node.id)
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            found.add(node.name)
        elif (isinstance(node, ast.keyword) and node.arg) or isinstance(node, ast.arg):
            found.add(node.arg)
    return found


def test_exactly_one_psf_path_exists_in_the_tree() -> None:
    """Criterion 1. Shipping both is how the ambiguity got here.

    The reference tree had `verification/psf_measurement.py:254 measure_psf` and
    `core/boundary.py:1508 PSF.from_complex_field`. Asserted structurally: one
    function returns a `PsfResult`, one module constructs one, and neither of the
    retired names exists anywhere in the new tree.
    """
    sources = _module_sources()
    constructing = sorted(
        path.relative_to(ROOT)
        for path, source in sources.items()
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "PsfResult"
    )
    assert constructing == [Path("src/measurements/psf.py")]

    retired = {"from_complex_field", "measure_psf", "PsfMeasurement", "C_FIELD_TO_PSF"}
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path, source in sources.items()
        for name in sorted(retired & _identifiers(source))
    ]
    assert offenders == []
    # ...and the walk would catch one.
    assert _identifiers("psf = PSF.from_complex_field(field)\n") & retired


def test_psf_is_not_a_representation() -> None:
    """Criterion 3, and the sentence `representations/scalar.py` already wrote.

    "It serializes nicely" is not what makes something a representation. The
    physical terminal state of a wave simulation is the `ScalarField`; the PSF is
    derived from it and the field is still there afterwards.
    """
    import representations

    assert not hasattr(representations, "PSF")
    assert "PsfResult" not in representations.__all__
    assert PACKAGE.exists() and not (SRC / "representations" / "psf.py").exists()
    # The field a measurement consumed is untouched by the measurement.
    field = a_field()
    before = np.asarray(field.u).copy()
    psf(field, normalization="peak")
    assert np.array_equal(np.asarray(field.u), before)


def test_measurements_imports_only_representations_and_numerics() -> None:
    """Criterion 5. `scripts/check_dependencies.py` is the gate; this is the
    statement about *this* package, so the boundary is legible where it is made."""
    allowed = {"measurements", "representations", "numerics"}
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                # The standard library this package uses, enumerated. `numpy` is
                # deliberately **not** here: the reference path this module replaced
                # squared `|u|` in NumPy on the host, and a gate that permits the
                # import permits that round trip back.
                assert root in allowed or root in {
                    "__future__",
                    "dataclasses",
                    "math",
                    "typing",
                }, f"{path.relative_to(ROOT)} imports {name!r}"


def test_the_class_delta_is_one() -> None:
    """Criterion 6. `PsfResult` is the only class; `PsfNormalization` is a
    `Literal`, because `scripts/class_budget.py` counts a `StrEnum` as a class and
    the stricter of the two readings is the one this tree takes."""
    defined = {
        node.name
        for path in sorted(PACKAGE.rglob("*.py"))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.ClassDef)
    }
    assert defined == {"PsfResult"}
    for avoided in ("PSF", "PsfMeasurement", "FraunhoferPsf", "ReferenceSphere",
                    "PupilAberration", "MetricDefinition", "AnalyticOracle"):
        assert avoided not in defined


def test_psf_registers_as_a_measurement_and_never_as_a_coupler() -> None:
    """Criterion 4. `scalar_field -> psf`: an observable derived from state.

    `psf` joins `SEMANTIC_TYPES` in R11.1's ticket, which is the discipline that
    vocabulary declares -- a type is added by the change that lands the boundary
    it names. It is added as the **output port of a measurement**, and the two
    tests below are what keep that from becoming the thing CHE-36 removed: no
    `coupler`-kind descriptor may name it on either port, because a coupler to an
    observable changes no representation and consults no convention it does not
    already hold.

    The descriptor used to be constructed here, inside a fixture that emptied the
    registry, because `measurements/` may not import `operations/` and there was no
    production registration site anywhere. CHE-221 (R03.4) put one *inside*
    `operations/`: the catalog names the implementation as a
    `"module.path:attribute"` string, so it needs no dependency edge in either
    direction, and the allowlist is unchanged. What is read below is the shipped
    record rather than a copy this file kept in step by hand.

    The uniqueness claim is now stronger than it was, and for a real reason: it
    used to read `find(kind=MEASUREMENT) == (descriptor,)` against a registry the
    fixture had just emptied, so it said "the one record this test registered".
    Against the shipped catalog it says the project has exactly one measurement,
    which is R11's criterion 1 and was previously unassertable.
    """
    descriptor = next(d for d in CATALOG if d.operation_id == "M_PSF")
    assert descriptor.kind is OperationKind.MEASUREMENT
    assert descriptor.input == "scalar_field"
    assert descriptor.output == "psf"
    assert registry.find(kind=OperationKind.MEASUREMENT) == (descriptor,), (
        "one measurement in the whole catalog, which is R11 criterion 1"
    )
    assert resolve("M_PSF") is psf


def test_no_catalogued_operation_consumes_or_mis_produces_the_observable() -> None:
    """R11 criterion 3, against the whole shipped catalog rather than a fixture.

    The construction-time rules in `operations/descriptors.py` make `C_FIELD_TO_PSF`
    unbuildable, and the two tests below assert that. This asserts the *catalog*
    obeys them, which is a different statement: it is the one that would fail if a
    future record slipped an observable onto a port some other way.
    """
    for record in CATALOG:
        assert record.input != "psf", record.operation_id
        if record.output == "psf":
            assert record.kind is OperationKind.MEASUREMENT, record.operation_id


# The two construction-error tests that used to sit here -- "only a measurement may
# produce an observable" and "nothing consumes an observable" -- moved to
# `tests/operations/test_descriptors.py` with CHE-221 (R03.4). Their subject is the
# schema's port validation, not the PSF, and the ticket's criterion 8 confines
# `OperationDescriptor(...)` construction to `tests/operations/`. R11 criterion 3 is
# still asserted here, by `test_no_catalogued_operation_consumes_or_mis_produces_the_
# observable` above, which is the stronger of the two statements: it holds against
# the whole shipped catalog rather than against a record a test built.


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_an_overflowing_field_is_refused_rather_than_returned_as_nan() -> None:
    """Squaring halves the exponent range, and `nan < 0` is `False`.

    A **valid, finite** complex64 field -- complex64 reaches 3.4e38, and the scale
    range that motivated CHE-47 spans twenty orders of magnitude -- with one
    amplitude above `sqrt(3.4e38)` overflows `|u|^2` to `inf`. Under `peak` the
    scale is then `1 / inf = 0`, `inf * 0` is `nan`, and every other check here
    passes: non-negativity does not fire on a NaN, and the border fraction of a
    NaN map comes out 0.0, which is the *reassuring* value.

    This is the one route by which a scale problem reached a consumer that
    recording the raw scale does not cover, because `raw_peak_intensity` is `inf`
    rather than a number. `require_finite` is the check, and the remedy string that
    promised it -- "refused one layer later as a non-finite intensity" -- is now
    true rather than aspirational.
    """
    u = np.zeros(SHAPE, dtype=np.complex64)
    u[1, 1] = np.complex64(2.0e19)
    u[2, 2] = np.complex64(1.0)
    field = ScalarField(
        u=u, sample_pitch_m=PITCH_M, wavelength_m=WAVELENGTH_M, reference_surface=SENSOR
    )
    assert bool(np.all(np.isfinite(np.asarray(field.u))))  # the *field* is fine

    with pytest.raises(ContractError) as raised:
        psf(field, normalization="peak")
    assert raised.value.code == "NON_FINITE"
    assert raised.value.declaration == "intensity"


def test_the_window_energy_is_the_fields_own_discrete_power() -> None:
    """Two computations of one number, so say that they agree.

    `raw_window_energy` is `float(sum(|u|^2)) * dy * dx`; `ScalarField.discrete_power`
    is `sum(|u|^2 * dy * dx)` inside the namespace. They are deliberately not the
    same code -- reusing `discrete_power` would square `|u|` a second time, the
    round trip this module objects to -- and this module's whole thesis is that two
    nearly-agreeing computations of one quantity are the worst case to leave
    unstated. So it is stated, and pinned, in both precisions.
    """
    for dtype, tolerance in (("complex128", 1e-15), ("complex64", 1e-6)):
        field = a_field(dtype=dtype)
        measured = psf(field, normalization="raw")
        assert measured.raw_window_energy == pytest.approx(
            field.discrete_power(), rel=tolerance
        )


def test_the_border_fraction_refuses_a_shape_that_has_no_border() -> None:
    """It is public, so a 1-D array is a refusal rather than an `IndexError` two
    lines later."""
    with pytest.raises(ContractError) as raised:
        border_energy_fraction(np.ones(8))
    assert raised.value.code == "SHAPE_MISMATCH"


def test_the_analytic_oracles_are_not_in_production() -> None:
    """Criterion 4 of the parent: oracles are evidence, not infrastructure.

    An Airy formula in `src/` would be a capability the project ships and would
    have to be maintained as one; under `tests/` it is what a comparison is made
    against. R11.2 puts them in `tests/physics/oracles.py`.
    """
    oracle_names = {
        "airy_first_null_radius_m",
        "airy_psf_on_grid",
        "fraunhofer_psf",
        "fit_reference_sphere",
        "pupil_aberration",
        "radial_profile",
        "first_null_comparison",
    }
    offenders = [
        f"{path.relative_to(ROOT)}: {name}"
        for path, source in _module_sources().items()
        for name in sorted(oracle_names & _identifiers(source))
    ]
    assert offenders == []


def test_a_measured_field_keeps_its_declared_validity_visible() -> None:
    """`|u|^2` is invariant under a global phase, so a field declaring
    `carrier_removed_phase` -- whose absolute phase is explicitly not physical --
    is admissible here, and the measurement does not thereby claim the phase was
    meaningful. It reports the coherence model it assumes and nothing more."""
    field = a_field()
    carrier_removed = ScalarField(
        u=field.u,
        sample_pitch_m=field.sample_pitch_m,
        wavelength_m=field.wavelength_m,
        reference_surface=field.reference_surface,
        validity=frozenset({"carrier_removed_phase"}),
    )
    plain = psf(field, normalization="peak")
    removed = psf(carrier_removed, normalization="peak")
    assert np.array_equal(np.asarray(plain.intensity), np.asarray(removed.intensity))
    assert removed.as_dict()["coherence_model"] == COHERENCE_MODEL

    # A global phase on the amplitude changes nothing at all.
    rotated = ScalarField(
        u=np.asarray(field.u) * np.exp(1j * 0.7),
        sample_pitch_m=field.sample_pitch_m,
        wavelength_m=field.wavelength_m,
        reference_surface=field.reference_surface,
    )
    # To round-off, not exactly: `|a e^{i phi}|^2` is computed as
    # `re^2 + im^2` on the rotated components, which is a different sum of the same
    # magnitude. The invariance is analytic; the residual is float64's.
    assert np.allclose(
        np.asarray(psf(rotated, normalization="raw").intensity),
        np.asarray(psf(field, normalization="raw").intensity),
        rtol=1e-15,
        atol=0.0,
    )


def test_the_frame_travels_with_the_result() -> None:
    """A PSF reported under a different origin rule than the field it measured is
    a half-pixel error nothing downstream can detect."""
    result = psf(a_field(), normalization="raw")
    assert result.frame.origin_rule == Frame().origin_rule
    assert result.as_dict()["origin_rule"] == result.frame.origin_rule
    assert result.as_dict()["axis_order"] == result.frame.axis_order
    assert math.isfinite(result.as_dict()["border_energy_fraction"])
