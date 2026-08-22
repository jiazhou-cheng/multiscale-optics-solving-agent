"""The patch-based local WFT, and the two SI S2 relations that gate it.

SI S2 (page S6) makes a claim that is directly testable, which is why this
module can be gated rather than only characterized:

    "For planar DOEs, there is no intrinsic upper bound on the patch size
    because the tangent-plane approximation is exact everywhere on the surface.
    As long as the ensemble of patches uniformly covers the DOE profile, the
    coherent sum of their responses converges to the full DOE response through
    coherent superposition, consistent with the linearity of the Fourier
    transform."

Two consequences, and both are asserted here:

1. **Full-aperture exactness** -- one patch as large as the whole aperture, all
   modes enumerated, reproduces an independent float64 ASM at round-off.
2. **Sub-aperture convergence** -- many smaller patches uniformly covering the
   aperture converge to the same field, at the Monte Carlo rate.

The second holds **at the DOE plane** and acquires a floor downstream. That is
not a defect in the relation; it is a boundary condition on reading it, and it
is measured below rather than asserted away.

On a curved substrate neither relation holds: every patch has its own tangent
frame and normal, and all that survives is the bound `eps_curv <= arcsin(D/2R)`.
The planar case gets a hard gate, the conformal case would get a bound, and the
second must not inherit the first's confidence. `Substrate.CONFORMAL` is refused.

This file is the **fast guard** the issue asks for: the exactness anchor and the
contract refusals only. The expensive convergence sweeps live in
`benchmarks/probes/`, out of the default suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.artifacts import ArtifactRecord
from core.boundary import ContractError, Frame, RayBundle, ReferencePlane
from core.execution import RunStatus
from core.graph import Severity
from core.specs import ArtifactKind
from couplers.base import CouplerRunRequest
from couplers.patch import (
    PatchPlan,
    Substrate,
    advance_bundle_to_plane,
    extract_patch,
    patch_secondary_rays,
    plan_patches,
    resolve_pad_px,
)
from couplers.patch_node import DOE_PORT, PatchWftCoupler
from couplers.ray_to_wave import Projection, ray_to_wave
from verification.asm_oracle import angular_spectrum_float64, compare_fields

pytestmark = pytest.mark.coupler

#: The anchor configuration, from the prototype measurement CHE-96 records:
#: a 33x33 random complex DOE at 6.3 um pitch, lambda = 0.7 um, z = 1.26 mm.
N = 33
PITCH_M = 6.3e-6
WAVELENGTH_M = 0.7e-6
Z_M = 1.26e-3
DOE_PLANE = ReferencePlane(name="doe", z_m=0.0)
SENSOR_PLANE = ReferencePlane(name="sensor", z_m=Z_M)


def _doe(seed: int = 20260822) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex128)


def _reconstruct(bundle, plane: ReferencePlane) -> np.ndarray:
    field, _ = ray_to_wave(
        bundle,
        grid_shape=(N, N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        plane=plane,
        projection=Projection.ASM_CONSISTENT,
    )
    return np.asarray(field.u)


# ---------------------------------------------------------------------------
# Relation 1 — full-aperture exactness. The anchor.
# ---------------------------------------------------------------------------

def test_a_full_aperture_patch_reproduces_the_independent_asm_at_roundoff() -> None:
    """The gate. One patch, every propagating mode, against a float64 ASM.

    The oracle is `verification/asm_oracle.angular_spectrum_float64` -- an
    independent implementation, not this module's own arithmetic rearranged.
    Using the patch route to check itself would be exactly the circular
    validation this project's rules forbid.
    """
    doe = _doe()
    plan = plan_patches(
        grid_shape=(N, N),
        sample_pitch_m=(PITCH_M, PITCH_M),
        patch_px=N,
        pad_factor=1,
        patch_count=None,
        substrate=Substrate.PLANAR,
    )
    assert plan.coverage == 1.0
    assert plan.curvature_bound_rad == 0.0, "planar means R = inf means bound 0, recorded"

    rays, diagnostics = patch_secondary_rays(
        doe,
        plan=plan,
        sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M,
        plane=DOE_PLANE,
        secondary_count=None,
    )
    assert diagnostics.enumerated
    assert diagnostics.propagating_modes == N * N, (
        "at 6.3 um pitch and 0.7 um wavelength every bin is propagating; a "
        "smaller count means the evanescent cut moved"
    )

    reconstructed = _reconstruct(advance_bundle_to_plane(rays, target=SENSOR_PLANE), SENSOR_PLANE)
    reference = angular_spectrum_float64(
        doe, wavelength_m=WAVELENGTH_M, sample_pitch_m=PITCH_M, z_m=Z_M
    )
    comparison = compare_fields(reconstructed, reference)
    assert comparison.raw_relative_field_error < 1e-11, (
        f"the full-aperture exactness anchor moved: "
        f"{comparison.raw_relative_field_error:.3e}. This is the relation the "
        "whole method rests on; investigate rather than widening the bound."
    )
    assert comparison.energy_residual < 1e-14


def test_the_advance_is_exact_rather_than_paraxial() -> None:
    """Advancing by arc length is not an approximation, and this shows why.

    A bundle advanced to a plane and reconstructed there must equal the same
    bundle reconstructed at the source plane and then propagated by the
    independent ASM. Both are float64, so any gap is the advance's own error.
    """
    doe = _doe()
    plan = plan_patches(
        grid_shape=(N, N), sample_pitch_m=(PITCH_M, PITCH_M), patch_px=N,
        pad_factor=1, patch_count=None, substrate=Substrate.PLANAR,
    )
    rays, _ = patch_secondary_rays(
        doe, plan=plan, sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M, plane=DOE_PLANE, secondary_count=None,
    )
    at_source = _reconstruct(rays, DOE_PLANE)
    propagated = angular_spectrum_float64(
        at_source, wavelength_m=WAVELENGTH_M, sample_pitch_m=PITCH_M, z_m=Z_M
    )
    advanced = _reconstruct(advance_bundle_to_plane(rays, target=SENSOR_PLANE), SENSOR_PLANE)
    assert compare_fields(advanced, propagated).raw_relative_field_error < 1e-11


# ---------------------------------------------------------------------------
# Relation 2 — sub-aperture. Exact in expectation; the estimator's rate is a probe.
# ---------------------------------------------------------------------------

def test_enumerating_every_patch_position_is_exact_not_merely_convergent() -> None:
    """The estimator's expectation, computed rather than sampled.

    Drawing centres is a Monte Carlo estimate of a finite sum over draw
    positions. Evaluating that sum exactly separates "is the estimator
    unbiased" from "how fast does it converge" -- and only the first is a gate.
    A biased estimator converges to the wrong answer, which a rate measurement
    alone cannot distinguish from slow convergence.

    Deliberately small. The reconstruction is O(rays x pixels) and the separable
    contraction allocates ``rays x n`` factors, so the same test at the anchor's
    33-px grid would be 3.7 M rays and ~4 GB in one call -- which pushed this
    shared machine into swap while it was being written. A fast guard has no
    business doing that; the cost curve belongs in a probe. At 15 px the whole
    enumeration is 159 k rays and about 76 MB.
    """
    small_n = 15
    patch_px = 5
    dilation = patch_px // 2
    rng = np.random.default_rng(20260822)
    doe = (
        rng.normal(size=(small_n, small_n)) + 1j * rng.normal(size=(small_n, small_n))
    ).astype(np.complex128)

    index = np.arange(-(small_n // 2 + dilation), small_n // 2 + dilation + 1)
    grid_y, grid_x = np.meshgrid(index, index)
    centers = np.column_stack([grid_x.ravel() * PITCH_M, grid_y.ravel() * PITCH_M])
    coverage = float(index.size**2 / (patch_px * patch_px))

    plan = PatchPlan(
        centers_xy_m=centers,
        patch_px=patch_px,
        pad_px=resolve_pad_px(
            grid_n=small_n,
            patch_px=patch_px,
            pad_factor=1,
            max_center_px=small_n // 2 + dilation,
        ),
        coverage=coverage,
        dilation_px=dilation,
        curvature_bound_rad=0.0,
    )
    rays, _ = patch_secondary_rays(
        doe, plan=plan, sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M, plane=DOE_PLANE, secondary_count=None,
    )
    field, _ = ray_to_wave(
        rays,
        grid_shape=(small_n, small_n),
        sample_pitch_m=(PITCH_M, PITCH_M),
        plane=DOE_PLANE,
        projection=Projection.ASM_CONSISTENT,
    )
    comparison = compare_fields(np.asarray(field.u), doe)
    assert comparison.raw_relative_field_error < 1e-12, (
        f"the sub-aperture estimator is biased: {comparison.raw_relative_field_error:.3e} "
        "with every draw position enumerated exactly once. The coverage "
        "correction A_draw / A_patch is the first thing to check -- inverting it "
        "is invisible on the full-aperture anchor, where the ratio is 1."
    )


# ---------------------------------------------------------------------------
# The three half-sample rules, each of which was a bug first
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("grid_n", "patch_px", "pad_factor", "max_center_px"),
    [
        (33, 11, 1, 0.0),
        (33, 11, 2, 0.0),
        (33, 11, 8, 0.0),
        (201, 101, 2, 0.0),
        (101, 51, 4, 0.0),
        (33, 11, 1, 21.0),
        (201, 101, 2, 150.0),
    ],
)
def test_the_derived_pad_satisfies_all_three_conditions(
    grid_n: int, patch_px: int, pad_factor: int, max_center_px: float
) -> None:
    """`pad_factor` is a preference; the pad is derived. Each condition was measured.

    **Clearance.** Sampling the spectrum at spacing ``1/(pad*dx)`` makes the
    reconstruction periodic with period ``pad*dx``, so every patch is
    accompanied by replicas at that spacing. A replica's support extends
    ``patch/2`` either side of its centre, and the farthest target the
    reconstruction window asks about is ``grid/2 + max_center`` from the patch,
    so the replica stays out iff

        pad > max_center + (grid + patch) / 2.

    Not ``pad >= grid + patch``: that is this same bound specialised to a plan
    whose patch centres are dilated over the full grid, which is what
    :func:`plan_patches` does (final case below) but not what a single
    origin-centred full-aperture patch needs. Over-padding the anchor is not
    conservative -- it moves the mode grid off the oracle's and the anchor reads
    0.57 instead of 1e-12.

    **Centring** (``pad - patch`` even) keeps the patch on the padded array's
    centre sample; odd puts it half a sample off and injects a linear phase.
    **Oddness** stops a mode landing exactly at Nyquist, which `ray_to_wave`
    refuses.
    """
    pad = resolve_pad_px(
        grid_n=grid_n,
        patch_px=patch_px,
        pad_factor=pad_factor,
        max_center_px=max_center_px,
    )
    assert pad > max_center_px + (grid_n + patch_px) / 2, "clearance"
    assert (pad - patch_px) % 2 == 0, "centring"
    assert pad % 2 == 1, "oddness"
    assert pad >= patch_px * pad_factor, "the caller's preference is a floor, not a cap"


def test_the_full_aperture_patch_is_the_one_exemption_from_clearance() -> None:
    """Clearance is a statement about sub-apertures, so the whole aperture is exempt.

    A patch that *is* the window has nothing outside it to be contaminated by,
    and the periodicity a pad would suppress is the same periodicity the
    unpadded reference ASM has. Exempting it is not a loosening: padding it
    moves the mode grid off the oracle's and the exactness anchor reads 0.57
    instead of 1.4e-12. The exemption is narrow -- it needs both a patch at
    least as wide as the grid and a centre pinned at the origin.
    """
    assert resolve_pad_px(grid_n=33, patch_px=33, pad_factor=1) == 33
    off_centre = resolve_pad_px(grid_n=33, patch_px=33, pad_factor=1, max_center_px=4.0)
    assert off_centre > 33, "a displaced patch is not exempt"


def test_a_dilated_plan_recovers_the_familiar_grid_plus_patch_clearance() -> None:
    """The strict rule is the general one specialised to how patches are placed.

    :func:`plan_patches` dilates centres over ``max(ny, nx) // 2 + patch // 2``,
    which substituted into the bound gives ``pad > grid + patch`` -- so the
    memorable form is recovered wherever it actually applies, and only the
    single centred patch is spared it.
    """
    grid_n, patch_px = 33, 11
    max_center_px = float(grid_n // 2 + patch_px // 2)
    pad = resolve_pad_px(
        grid_n=grid_n, patch_px=patch_px, pad_factor=2, max_center_px=max_center_px
    )
    assert pad >= grid_n + patch_px


@pytest.mark.parametrize("patch_px", [10, 40, 50, 100])
def test_an_even_patch_is_refused_rather_than_silently_rounded(patch_px: int) -> None:
    """An even patch has no centre sample, and the two parity rules then conflict.

    "Odd pad" and "even (pad - patch)" cannot both hold when `patch_px` is even.
    That is not an arithmetic accident: without a centre sample, "centred on a
    ray" is undefined, and the forced half-sample offset is the same error the
    parity rules exist to prevent.

    Refused rather than rounded, because the paper's patch sizes (40, 50, 100)
    are all even and a caller transcribing one should be told which value
    actually ran. The first version of `resolve_pad_px` looped forever on
    exactly these inputs.
    """
    with pytest.raises(ContractError, match="even"):
        resolve_pad_px(grid_n=201, patch_px=patch_px, pad_factor=2)


def test_a_pad_that_violates_clearance_produces_a_plausible_wrong_field() -> None:
    """The failure this guard exists for, demonstrated rather than described.

    An under-padded reconstruction does not raise and does not look wrong. It
    returns a field of the right shape and a comparable magnitude that is
    incorrect by 100%.
    """
    doe = _doe()
    patch_px = 11
    offsets = (np.arange(3) - 1) * patch_px * PITCH_M
    grid_y, grid_x = np.meshgrid(offsets, offsets)
    centers = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    def error_at(pad: int) -> float:
        plan = PatchPlan(
            centers_xy_m=centers, patch_px=patch_px, pad_px=pad, coverage=9.0,
            dilation_px=0, curvature_bound_rad=0.0,
        )
        rays, _ = patch_secondary_rays(
            doe, plan=plan, sample_pitch_m=(PITCH_M, PITCH_M),
            wavelength_m=WAVELENGTH_M, plane=DOE_PLANE, secondary_count=None,
        )
        return compare_fields(_reconstruct(rays, DOE_PLANE), doe).raw_relative_field_error

    # 33 clears the replicas for this tiling's extreme centre (-11p, so the
    # requirement is pad > 11 + 5.5 + 16.5 = 33); 23 does not.
    assert error_at(33) < 1e-12
    assert error_at(23) > 0.5, "an under-padded reconstruction should be badly wrong"


def test_an_even_pad_is_refused_by_the_nyquist_guard() -> None:
    """The mode at exactly `lambda / (2 * pitch)` is not representable."""
    doe = _doe()
    plan = PatchPlan(
        centers_xy_m=np.zeros((1, 2)), patch_px=11, pad_px=66, coverage=1.0,
        dilation_px=0, curvature_bound_rad=0.0,
    )
    rays, _ = patch_secondary_rays(
        doe, plan=plan, sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M, plane=DOE_PLANE, secondary_count=None,
    )
    with pytest.raises(ContractError):
        _reconstruct(rays, DOE_PLANE)


# ---------------------------------------------------------------------------
# Declarations and refusals
# ---------------------------------------------------------------------------

def test_a_conformal_substrate_is_refused_rather_than_approximated() -> None:
    """The exactness ladder does not extend to a curved surface, so neither does this."""
    with pytest.raises(ContractError, match="not implemented"):
        plan_patches(
            grid_shape=(N, N), sample_pitch_m=(PITCH_M, PITCH_M), patch_px=11,
            patch_count=None, substrate=Substrate.CONFORMAL,
        )


def test_a_patch_wider_than_the_curvature_budget_is_refused() -> None:
    """`check_patch` is a precondition, not a footnote.

    On a finite radius the tangent-plane approximation has a stated cost, and a
    patch that exceeds the caller's threshold is refused before anything runs.
    """
    with pytest.raises(ContractError):
        plan_patches(
            grid_shape=(N, N), sample_pitch_m=(PITCH_M, PITCH_M), patch_px=N,
            patch_count=None, substrate=Substrate.PLANAR,
            radius_m=1e-4, error_threshold_rad=1e-6,
        )


def test_extraction_continues_with_zero_not_with_the_edge_value() -> None:
    """A bounded DOE has no field outside it; clamping would invent structure."""
    doe = np.ones((N, N), dtype=np.complex128)
    patch = extract_patch(
        doe,
        center_xy_m=(float((N // 2) * PITCH_M), 0.0),
        patch_px=11,
        sample_pitch_m=(PITCH_M, PITCH_M),
    )
    assert patch.shape == (11, 11)
    assert np.any(patch == 0.0), "the half of the patch beyond the aperture must be zero"
    assert np.any(patch == 1.0), "the half inside it must not be"


def test_the_emitted_bundle_declares_its_normalization_and_reference() -> None:
    """`one_over_n` delegates SI eq S5's 1/(N_patches * S) to `ray_to_wave`.

    One place owns that factor. Two places owning it is how a field ends up
    scaled by a ray count.
    """
    plan = plan_patches(
        grid_shape=(N, N), sample_pitch_m=(PITCH_M, PITCH_M), patch_px=N,
        pad_factor=1, patch_count=None, substrate=Substrate.PLANAR,
    )
    rays, diagnostics = patch_secondary_rays(
        _doe(), plan=plan, sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M, plane=DOE_PLANE, secondary_count=None,
    )
    assert rays.reconstruction_normalization == "one_over_n"
    assert "zero at the patch plane" in (rays.optical_path_length_reference or "")
    assert diagnostics.apodization.startswith("none")
    assert diagnostics.substrate == "planar"


def test_a_ray_that_cannot_reach_the_target_plane_is_refused_not_dropped() -> None:
    """A bundle that quietly loses members produces a plausible field with missing power."""
    plan = plan_patches(
        grid_shape=(N, N), sample_pitch_m=(PITCH_M, PITCH_M), patch_px=N,
        pad_factor=1, patch_count=None, substrate=Substrate.PLANAR,
    )
    rays, _ = patch_secondary_rays(
        _doe(), plan=plan, sample_pitch_m=(PITCH_M, PITCH_M),
        wavelength_m=WAVELENGTH_M, plane=DOE_PLANE, secondary_count=None,
    )
    behind = ReferencePlane(name="behind", z_m=-Z_M)
    with pytest.raises(ContractError, match="travels away"):
        advance_bundle_to_plane(rays, target=behind)


# ---------------------------------------------------------------------------
# The graph node (CHE-96)
# ---------------------------------------------------------------------------
#
# The library is gated by the exactness anchor above. These check the *edge*:
# that a graph reaching for this operator gets the same refusals the library
# gives, and that the three derived quantities a caller cannot see -- the pad,
# the coverage basis, the evanescent loss -- come back in the record.

NODE_N = 15
NODE_GRID = (NODE_N, NODE_N)
NODE_PITCH = (6.3e-6, 6.3e-6)
NODE_PLANE = ReferencePlane(name="doe", z_m=0.0)


def _node_records(tmp_path, *, ray_count: int = 4):
    """An incident bundle with a declared OPL, plus a random phase DOE."""
    rng = np.random.default_rng(20260822)
    half = NODE_N // 2
    offsets = rng.integers(-half, half + 1, size=(ray_count, 2)) * NODE_PITCH[0]
    positions = np.column_stack([offsets[:, 0], offsets[:, 1], np.zeros(ray_count)])
    directions = np.tile(np.array([0.0, 0.0, 1.0]), (ray_count, 1))
    incident = RayBundle(
        positions_m=positions,
        directions=directions,
        wavelength_m=0.7e-6,
        reference_plane=NODE_PLANE,
        frame=Frame(),
        amplitude=np.ones(ray_count, dtype=np.complex128),
        optical_path_length_m=np.zeros(ray_count),
        optical_path_length_reference="zero at the DOE plane",
    )
    rays_uri = tmp_path / "rays.npz"
    source = incident.to_artifact_record(artifact_id="incident", uri=rays_uri)
    doe_uri = tmp_path / "doe.npy"
    np.save(doe_uri, np.exp(1j * rng.uniform(-np.pi, np.pi, size=NODE_GRID)))
    doe = ArtifactRecord(
        id="doe",
        kind=ArtifactKind.COMPLEX_FIELD,
        uri=str(doe_uri),
        shape=list(NODE_GRID),
        dtype="complex128",
    )
    return source, doe


def _node_config(**overrides):
    config = {
        "grid_shape": list(NODE_GRID),
        "sample_pitch_m": list(NODE_PITCH),
        "plane_z_m": 0.0,
        "patch_px": 5,
        "pad_factor": 2,
        "patch_placement": "incident_positions",
        "coverage_basis": "uniform_over_dilated_aperture",
    }
    config.update(overrides)
    return config


def _run(tmp_path, **overrides):
    source, doe = _node_records(tmp_path)
    return PatchWftCoupler().transform(
        CouplerRunRequest(
            run_id="che96",
            edge_id="patch_step",
            sources={"source": source, DOE_PORT: doe},
            config=_node_config(**overrides),
        )
    )


def test_the_node_runs_and_reports_what_the_caller_could_not_see(tmp_path) -> None:
    """Three derived quantities come back, because a consumer reads the record.

    The pad the edge actually used, the coverage basis it applied, and the
    evanescent modes it dropped are all decisions made inside the step that
    change the answer and are invisible in the output array. A diagnostics block
    that omitted them would leave a consumer unable to tell a converged run from
    a mis-padded one.
    """
    result = _run(tmp_path)
    assert result.status is RunStatus.SUCCEEDED, result.error_message
    payload = result.diagnostics["patch"]
    assert payload["substrate"] == "planar"
    assert payload["curvature_bound_rad"] == 0.0, "planar: established, not assumed"
    assert payload["apodization"].startswith("none")
    assert payload["reconstruction_normalization"] == "one_over_n"
    assert result.diagnostics["pad_used"] == payload["pad_px"]
    assert result.diagnostics["coverage_basis"] == "uniform_over_dilated_aperture"
    assert "double-count" not in result.diagnostics["opl_convention"]
    assert "reset to zero" in result.diagnostics["opl_convention"]


def test_the_node_raises_a_pad_that_would_alias_and_says_it_did(tmp_path) -> None:
    """`pad_factor` is a preference. Silently honouring it is the 100% error.

    The edge derives the pad from clearance, centring and oddness, and warns
    when the derived value differs from the requested one. Warning rather than
    refusing is deliberate: the caller asked for something unsafe and got
    something safe, which is the right outcome, but a caller who then reports
    "pad factor 2" would be reporting a run that did not happen.
    """
    result = _run(tmp_path, pad_factor=2)
    assert result.status is RunStatus.SUCCEEDED
    assert result.diagnostics["pad_used"] > result.diagnostics["pad_requested"]
    assert any("pad_factor" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"patch_px": 50}, "even"),
        ({"substrate": "conformal"}, "substrate"),
        ({"coverage_basis": None}, "coverage_basis"),
        ({"patch_placement": "drawn", "patch_count": 4}, "seed"),
        ({"patch_px": None}, "patch_px"),
    ],
)
def test_the_node_refuses_what_the_library_refuses(tmp_path, overrides, expected) -> None:
    """The edge and the library must not disagree about what is acceptable.

    Each of these is a refusal the library makes for a stated physical reason,
    re-asserted at the graph boundary. The even-patch case is the one a caller
    is most likely to hit -- the paper's own sizes are 40, 50 and 100 -- and it
    is refused rather than rounded so that a transcribed number cannot silently
    become a different one.
    """
    result = _run(tmp_path, **overrides)
    assert result.status is RunStatus.FAILED
    assert expected in (result.error_message or "") + str(result.error_type)


def test_validate_request_predicts_every_refusal_transform_makes(tmp_path) -> None:
    """Two checklists is how a validator comes to bless a request that then fails.

    Both call the same `diagnose`, so this asserts that arrangement holds rather
    than that the two happen to agree today.
    """
    source, doe = _node_records(tmp_path)
    request = CouplerRunRequest(
        run_id="che96",
        edge_id="e",
        sources={"source": source, DOE_PORT: doe},
        config=_node_config(patch_px=50),
    )
    node = PatchWftCoupler()
    report = node.validate_request(request)
    assert any(issue.severity is Severity.ERROR for issue in report.issues)
    assert node.transform(request).status is RunStatus.FAILED


def test_the_node_refuses_a_gradient_request(tmp_path) -> None:
    """`derivative.verified` is false and promotion did not change that."""
    source, doe = _node_records(tmp_path)
    report = PatchWftCoupler().validate_request(
        CouplerRunRequest(
            run_id="che96",
            edge_id="e",
            sources={"source": source, DOE_PORT: doe},
            config=_node_config(),
            require_gradients=True,
        )
    )
    assert any("GRADIENT_NOT_VERIFIED" in issue.code for issue in report.issues)


def test_the_cost_estimate_names_the_downstream_term_not_its_own(tmp_path) -> None:
    """This edge is cheap and the route it enables is not.

    The transform is `patches x pad^2 log pad`; the reconstruction that consumes
    the emitted bundle is `rays x pixels`, which at the paper's parameters is
    1.6e8 against 1e4. An estimate reporting only this edge's own work would
    understate the run by orders of magnitude, so the emitted ray count is
    surfaced as the quantity to budget against.
    """
    source, doe = _node_records(tmp_path)
    estimate = PatchWftCoupler().estimate(
        CouplerRunRequest(
            run_id="che96",
            edge_id="e",
            sources={"source": source, DOE_PORT: doe},
            config=_node_config(secondary_count=100, seed=0),
        )
    )
    joined = " ".join(estimate.notes)
    assert "DOWNSTREAM" in joined and "ray-pixel" in joined
