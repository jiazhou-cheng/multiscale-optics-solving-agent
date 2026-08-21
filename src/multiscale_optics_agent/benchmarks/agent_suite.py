"""V1 agent benchmark: can an agent turn a physics problem into a right answer? (CHE-71)

The existing benchmark registry (`benchmarks/manifest.yaml`) grades a **solver's
physics** and its whole value is reproducible fingerprints. This grades an
**agent's behaviour**, which is nondeterministic by nature. Putting a
nondeterministic score into that registry would spoil it, so this sits beside it
with its own ID space (`A1-*`) and its own runner. Design decision 4.

What a task is
--------------
A task hands a participant a natural-language physics problem and a workspace, and
asks for a `submission.json`. It does **not** hand over a preconstructed adapter
input, a solver choice, or an API call — choosing the tool and building the
simulation is the thing being measured.

Grading is then a two-stage question, and keeping the stages apart is the whole
point (§"Outcome"): *did it run* and *is it right* are different questions, and
this repository has two documented cases where the answer is "yes" and "no".

Three properties every task here has
------------------------------------
1. **An analytic oracle.** Every expected value is a closed form, verified against
   the pinned solver before the task shipped. Not a recorded solver output — a
   recorded output cannot tell a wrong answer from a wrong reference.
2. **Cheap.** The whole suite is seconds of solver time, so a trial count of 3+
   costs nothing on the grading side and the budget goes to the agent.
3. **No tutorial hints in the prompt.** The prompts state physics and required
   outputs. They never name a function, a module, or a tutorial.

Two tasks are deliberately traps
--------------------------------
`A1-OPT-03` and `A1-CHX-03` are cases where a plausible mistake produces code that
**runs perfectly and answers wrongly**, both measured on the pinned versions:

* Optiland's `ThinFilmStack.add_layer` takes micrometres while the AR-coating
  literature and the upstream tutorial talk in nanometres (CHE-57 finding on t07).
  A 1000x-too-thin quarter-wave layer gives R = 0.042164 against bare glass's
  0.042165 — the coating does nothing, no error is raised, and the number looks
  like a reflectance.
* Chromatix's `kykx` means *cycles per length* on `asm_propagate` and *radians per
  length* on `plane_wave` — same name, factor of 2*pi apart, and the displacement
  is opposite in sign to the parameter (CHE-57 finding on c06).

A taxonomy that collapsed "it ran" into "it worked" could not express either, and
they are the interesting failures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "SUITE_V1",
    "AgentTask",
    "CheckResult",
    "CheckSpec",
    "ContextPolicy",
    "Outcome",
    "Participant",
    "SuiteResult",
    "TaskResult",
    "TrialResult",
    "broken_participant",
    "command_participant",
    "grade",
    "main",
    "reference_participant",
    "run_suite",
    "task_by_id",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
SUITE_DIR = REPO_ROOT / "benchmarks_agent"


class ContextPolicy(StrEnum):
    """What the agent under test is allowed to see. Design decision 1.

    This is the most consequential knob in the whole harness, because it changes
    *what is being measured* rather than how well. With the knowledge pack in
    context the benchmark asks "can it follow our cards"; without it, "can it
    discover and use the tool". Both are legitimate and they are different
    benchmarks, so the policy is declared per task and recorded in every result —
    otherwise scores are not comparable across runs.
    """

    #: Problem statement only. Measures discovery: the agent must find the library
    #: and its API itself.
    COLD = "cold"
    #: Problem statement plus the solver card and minimal API examples. Measures
    #: whether the agent can use a documented tool correctly.
    WARM = "warm"
    #: WARM plus the conventions document, which names the unit and sign hazards.
    #: Measures whether the agent reads a warning it has been handed.
    GUIDED = "guided"


class Outcome(StrEnum):
    """Structured outcome codes. Design decision 5.

    Follows the repository's existing pattern (`ContractCode`, the precision
    codes): a code, a reason and a remedy, never a free-text string, so results
    aggregate.

    The ordering matters. Grading walks these in sequence and reports the *first*
    stage that failed, because a run that never produced a field cannot also be
    judged on its physics, and reporting the later failure would misattribute the
    cause.
    """

    #: Every check passed.
    PASS = "AGENT_PASS"
    #: Nothing was submitted inside the budget. Distinct from a crash: silence and
    #: a traceback need different follow-ups.
    FAIL_NO_SUBMISSION = "FAIL_NO_SUBMISSION"
    #: A submission exists but does not answer the question asked -- a required
    #: quantity is missing, or is not a number, or its declared unit is wrong.
    FAIL_PROBLEM_UNDERSTANDING = "FAIL_PROBLEM_UNDERSTANDING"
    #: The task's library was never used. The agent solved (or guessed) something
    #: else -- possibly with the right answer, which is still a benchmark failure
    #: because the task is about tool use.
    FAIL_TOOL_SELECTION = "FAIL_TOOL_SELECTION"
    #: The library was used but the setup was invalid: the solver refused it, or
    #: the declared configuration contradicts the problem statement.
    FAIL_SIMULATION_CONSTRUCTION = "FAIL_SIMULATION_CONSTRUCTION"
    #: The setup was accepted and the run raised, hung, or produced nothing.
    FAIL_TOOL_EXECUTION = "FAIL_TOOL_EXECUTION"
    #: **It ran, it produced numbers, and the numbers are wrong.** The one code
    #: the two trap tasks exist to produce.
    FAIL_PHYSICAL_RESULT = "FAIL_PHYSICAL_RESULT"
    #: The harness broke. Never charged to the agent, and never counted as a
    #: failed trial in the pass rate.
    FAIL_HARNESS = "FAIL_HARNESS"


#: What each outcome means for whoever reads the score, and what to do about it.
OUTCOME_REMEDY: dict[Outcome, str] = {
    Outcome.PASS: "nothing",
    Outcome.FAIL_NO_SUBMISSION: (
        "raise the budget, or check the participant wrote submission.json where the "
        "task said to"
    ),
    Outcome.FAIL_PROBLEM_UNDERSTANDING: (
        "the prompt may be ambiguous about the required outputs or their units; "
        "read the submission before blaming the agent"
    ),
    Outcome.FAIL_TOOL_SELECTION: (
        "under a cold policy this is the measurement, not a defect; under warm or "
        "guided it means the pack did not make the tool findable"
    ),
    Outcome.FAIL_SIMULATION_CONSTRUCTION: (
        "usually a schema or convention the pack does not state; candidate for a "
        "knowledge-pack addition"
    ),
    Outcome.FAIL_TOOL_EXECUTION: (
        "an API the pack documents wrongly, or an environment gap; check the "
        "captured stderr before anything else"
    ),
    Outcome.FAIL_PHYSICAL_RESULT: (
        "the interesting failure. The agent's code ran; its physics did not. Check "
        "the task's trap note first"
    ),
    Outcome.FAIL_HARNESS: "fix the harness; this trial is void, not failed",
}


@dataclass(frozen=True)
class CheckSpec:
    """One graded quantity, its oracle, and the strength of that oracle.

    ``kind`` follows CHE-57's classification so the two inventories can be read
    together. Every check in V1 is ``analytic``: the expected value is a closed
    form, verified against the pinned solver before the task shipped. A recorded
    solver output would be weaker in a way that matters -- it cannot distinguish a
    wrong answer from a wrong reference.
    """

    key: str
    description: str
    expected: float
    unit: str
    #: Relative tolerance. Stated per check with a reason in ``tolerance_basis``,
    #: never a house default.
    rtol: float
    tolerance_basis: str
    kind: str = "analytic"

    def evaluate(self, value: Any) -> CheckResult:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return CheckResult(
                spec=self, observed=None, passed=False,
                detail=f"{self.key} is {type(value).__name__}, not a number",
            )
        observed = float(value)
        if not math.isfinite(observed):
            return CheckResult(
                spec=self, observed=observed, passed=False,
                detail=f"{self.key} is not finite",
            )
        error = (
            abs(observed - self.expected) / abs(self.expected)
            if self.expected
            else abs(observed)
        )
        return CheckResult(
            spec=self,
            observed=observed,
            passed=error <= self.rtol,
            detail=f"relative error {error:.3e} against rtol {self.rtol:.3e}",
            relative_error=error,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "description": self.description,
            "expected": self.expected,
            "unit": self.unit,
            "rtol": self.rtol,
            "tolerance_basis": self.tolerance_basis,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class CheckResult:
    spec: CheckSpec
    observed: float | None
    passed: bool
    detail: str
    relative_error: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.spec.key,
            "expected": self.spec.expected,
            "observed": self.observed,
            "unit": self.spec.unit,
            "passed": self.passed,
            "relative_error": self.relative_error,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AgentTask:
    """One benchmark task: the problem, what it exercises, and how it is graded."""

    task_id: str
    title: str
    #: The import name the submission must show evidence of having used. This is
    #: what makes tool *selection* gradable separately from tool *use*.
    library: str
    context_policy: ContextPolicy
    #: What capability the task is meant to exercise, in one sentence. Read by a
    #: human deciding whether a failure is interesting.
    exercises: str
    checks: tuple[CheckSpec, ...]
    reference: Callable[[], dict[str, Any]]
    #: Non-empty only for the trap tasks: the plausible mistake, and the wrong
    #: number it produces. Printed next to a FAIL_PHYSICAL_RESULT.
    trap: str = ""
    notes: str = ""

    @property
    def prompt_path(self) -> Path:
        return SUITE_DIR / "prompts" / f"{self.task_id}.md"

    @property
    def expected_path(self) -> Path:
        return SUITE_DIR / "expected" / f"{self.task_id}.json"

    def prompt(self) -> str:
        return self.prompt_path.read_text()

    def context_files(self) -> tuple[Path, ...]:
        """The files a participant may read, per this task's declared policy."""
        if self.context_policy is ContextPolicy.COLD:
            return ()
        solver = REPO_ROOT / "knowledge" / "solvers" / self.library
        files = [solver / "solver_card.yaml", solver / "api_minimal_examples.md"]
        if self.context_policy is ContextPolicy.GUIDED:
            files.append(solver / "conventions.md")
        return tuple(path for path in files if path.exists())

    def required_keys(self) -> tuple[str, ...]:
        return tuple(check.key for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "library": self.library,
            "context_policy": str(self.context_policy),
            "exercises": self.exercises,
            "required_submission_keys": list(self.required_keys()),
            "checks": [check.as_dict() for check in self.checks],
            "trap": self.trap,
            "notes": self.notes,
            "prompt": str(self.prompt_path.relative_to(REPO_ROOT)),
        }


# --------------------------------------------------------------------------- #
# Reference implementations. Each one is the known-good solution, and each one
# was checked against its closed form before the task shipped -- see the
# `tolerance_basis` on every check for the measured agreement.
# --------------------------------------------------------------------------- #


def _optiland_numpy() -> Any:
    import optiland.backend as be

    be.set_backend("numpy")
    be.set_precision("float64")
    return be


def _reference_opt_01() -> dict[str, Any]:
    """Thick plano-convex singlet: EFL and BFL from a paraxial solve."""
    _optiland_numpy()
    from optiland.materials import IdealMaterial
    from optiland.optic import Optic

    lens = Optic()
    lens.surfaces.add(index=0, thickness=math.inf)
    lens.surfaces.add(
        index=1, radius=25.0, thickness=4.0,
        material=IdealMaterial(n=1.5168), is_stop=True,
    )
    lens.surfaces.add(index=2, radius=math.inf, thickness=45.737482)
    lens.surfaces.add(index=3)
    lens.set_aperture("EPD", 10.0)
    lens.fields.add(y=0.0)
    lens.wavelengths.add(0.5876, is_primary=True)
    return {
        "effective_focal_length_mm": float(lens.paraxial.f2()),
        "back_focal_length_mm": float(lens.paraxial.f2()) - 4.0 / 1.5168,
        "library": "optiland",
    }


def _reference_opt_02() -> dict[str, Any]:
    """Plane-parallel plate: focal shift from a real ray trace, not a formula."""
    import numpy as np

    _optiland_numpy()
    from optiland.materials import IdealMaterial
    from optiland.optic import Optic
    from optiland.rays import RealRays

    focus_mm, height_mm, thickness_mm, index = 100.0, 0.5, 10.0, 1.6
    lens = Optic()
    lens.surfaces.add(index=0, thickness=math.inf)
    lens.surfaces.add(
        index=1, radius=math.inf, thickness=thickness_mm,
        material=IdealMaterial(n=index), is_stop=True,
    )
    lens.surfaces.add(index=2, radius=math.inf, thickness=1.0)
    lens.surfaces.add(index=3)
    lens.set_aperture("EPD", 2.0)
    lens.fields.add(y=0.0)
    lens.wavelengths.add(0.5876, is_primary=True)

    slope = -height_mm / focus_mm
    axial = 1.0 / math.sqrt(1.0 + slope * slope)
    rays = RealRays(
        np.array([0.0]), np.array([height_mm]), np.array([0.0]),
        np.array([0.0]), np.array([slope * axial]), np.array([axial]),
        np.array([1.0]), np.array([0.5876]),
    )
    traced = lens.surfaces.trace(rays, skip=1)
    y, z = float(traced.y[0]), float(traced.z[0])
    crossing = z - y * float(traced.N[0]) / float(traced.M[0])
    return {"focal_shift_mm": crossing - focus_mm, "library": "optiland"}


def _reference_opt_03() -> dict[str, Any]:
    """Single-layer quarter-wave AR coating: reflectance with and without it.

    The whole task turns on one line: ``add_layer`` takes **micrometres**. The
    quarter-wave thickness is 99.64 nm, i.e. 0.09964 um, and passing 99.64 builds
    a stack 1000x too thick whose reflectance is indistinguishable from bare glass.
    """
    import numpy as np

    _optiland_numpy()
    from optiland.coatings import ThinFilmStack
    from optiland.materials import IdealMaterial

    air = IdealMaterial(n=1.0)
    substrate = IdealMaterial(n=1.5168)
    coating = IdealMaterial(n=1.38)
    design_nm = 550.0
    quarter_wave_um = design_nm / (4.0 * 1.38) / 1000.0

    bare = ThinFilmStack(air, substrate)
    coated = ThinFilmStack(air, substrate)
    coated.add_layer(coating, quarter_wave_um, "MgF2")
    def reflectance(stack: Any) -> float:
        return float(np.asarray(stack.reflectance_nm_deg(design_nm, 0.0)).ravel()[0])

    return {
        "uncoated_reflectance": reflectance(bare),
        "coated_reflectance": reflectance(coated),
        "coating_thickness_nm": design_nm / (4.0 * 1.38),
        "library": "optiland",
    }


def _chromatix_field(u: Any, pitch_um: float, wavelength_um: float) -> Any:
    import jax
    import jax.numpy as jnp

    # Pinned off, matching every other Chromatix path in this repository: a
    # process that flipped it would change every recorded number.
    jax.config.update("jax_enable_x64", False)
    import chromatix.functional as cf

    return cf.Field.build(
        jnp.asarray(u, dtype=jnp.complex64), jnp.asarray([[pitch_um, pitch_um]]), wavelength_um
    )


def _reference_chx_01() -> dict[str, Any]:
    """Gaussian beam diffraction: the 1/e^2 radius after 100 um of free space."""
    import chromatix.functional as cf
    import jax.numpy as jnp
    import numpy as np

    wavelength, waist, distance = 0.532, 5.0, 100.0
    grid, pitch = 512, 0.25
    axis = (np.arange(grid) - grid // 2) * pitch
    x, y = np.meshgrid(axis, axis, indexing="xy")
    field = _chromatix_field(np.exp(-(x**2 + y**2) / waist**2), pitch, wavelength)
    out = cf.asm_propagate(field, z=distance, n=1.0, pad_width=grid, mode="same")
    intensity = np.asarray(jnp.abs(out.u) ** 2).squeeze()
    # Second moment, which is the 1/e^2 radius for a Gaussian: I ~ exp(-2r^2/w^2)
    # has <x^2> = w^2/4.
    second_moment = float(((x**2) * intensity).sum() / intensity.sum())
    return {
        "beam_radius_um": 2.0 * math.sqrt(second_moment),
        "library": "chromatix",
    }


def _reference_chx_02() -> dict[str, Any]:
    """Airy pattern: the first dark ring of a focused circular aperture."""
    import chromatix.functional as cf
    import jax.numpy as jnp
    import numpy as np

    wavelength, diameter, focal = 0.532, 40.0, 400.0
    grid, pitch = 512, 0.5
    axis = (np.arange(grid) - grid // 2) * pitch
    x, y = np.meshgrid(axis, axis, indexing="xy")
    aperture = (np.hypot(x, y) <= diameter / 2).astype(np.complex64)
    field = _chromatix_field(aperture, pitch, wavelength)
    out = cf.ff_lens(field, f=focal, n=1.0)
    intensity = np.asarray(jnp.abs(out.u) ** 2).squeeze()
    output_pitch = float(np.asarray(out.dx).ravel()[0])
    row = intensity[grid // 2]
    index = grid // 2
    while index + 1 < grid and row[index + 1] < row[index]:
        index += 1
    return {
        "first_null_radius_um": (index - grid // 2) * output_pitch,
        "library": "chromatix",
    }


def _reference_chx_03() -> dict[str, Any]:
    """Tilted beam: the lateral displacement of the centroid after propagation.

    The tilt is built into the field as ``exp(+2i pi (sin(theta)/lambda) x)``,
    which is unambiguous. Reaching for ``plane_wave(kykx=...)`` or
    ``asm_propagate(kykx=...)`` instead is the trap: same parameter name, one in
    radians per length and one in cycles per length, and the displacement runs
    opposite to the parameter.
    """
    import chromatix.functional as cf
    import jax.numpy as jnp
    import numpy as np

    wavelength, waist, distance, tilt_deg = 0.532, 8.0, 200.0, 5.0
    grid, pitch = 1024, 0.2
    axis = (np.arange(grid) - grid // 2) * pitch
    x, y = np.meshgrid(axis, axis, indexing="xy")
    envelope = np.exp(-(x**2 + y**2) / waist**2)
    spatial_frequency = math.sin(math.radians(tilt_deg)) / wavelength
    tilt = np.exp(2j * math.pi * spatial_frequency * x)
    field = _chromatix_field(envelope * tilt, pitch, wavelength)
    out = cf.asm_propagate(field, z=distance, n=1.0, pad_width=grid, mode="same")
    intensity = np.asarray(jnp.abs(out.u) ** 2).squeeze()
    return {
        "centroid_x_um": float((x * intensity).sum() / intensity.sum()),
        "library": "chromatix",
    }


# --------------------------------------------------------------------------- #
# The V1 suite. Six tasks, three per library, all analytic-oracle-backed.
# --------------------------------------------------------------------------- #

SUITE_V1: tuple[AgentTask, ...] = (
    AgentTask(
        task_id="A1-OPT-01",
        title="Focal length of a thick plano-convex singlet",
        library="optiland",
        context_policy=ContextPolicy.COLD,
        exercises=(
            "recognising a paraxial-property question, choosing a ray-tracing "
            "package, transcribing a prescription into it, and reading back two "
            "derived quantities in the right units"
        ),
        checks=(
            CheckSpec(
                key="effective_focal_length_mm",
                description="effective focal length",
                expected=25.0 / 0.5168,
                unit="mm",
                rtol=1e-6,
                tolerance_basis=(
                    "R/(n-1) is exact for a single refracting surface in air; the "
                    "pinned solver reproduces it to 1e-13 relative (measured), so "
                    "1e-6 admits only a genuinely different answer"
                ),
            ),
            CheckSpec(
                key="back_focal_length_mm",
                description="back focal length from the rear vertex",
                expected=25.0 / 0.5168 - 4.0 / 1.5168,
                unit="mm",
                rtol=1e-6,
                tolerance_basis="EFL - t/n, exact for a plano rear surface",
            ),
        ),
        reference=_reference_opt_01,
        notes=(
            "The thick-lens correction is what makes this more than arithmetic: "
            "EFL - t/n differs from EFL by 2.64 mm, so an agent that reports the "
            "same number twice fails the second check and only the second check."
        ),
    ),
    AgentTask(
        task_id="A1-OPT-02",
        title="Focal shift caused by a plane-parallel plate",
        library="optiland",
        context_policy=ContextPolicy.COLD,
        exercises=(
            "building a real (non-paraxial) ray trace, injecting a converging ray, "
            "and finding an axis crossing from the traced state"
        ),
        checks=(
            CheckSpec(
                key="focal_shift_mm",
                description="axial displacement of the focus, positive away from the plate",
                expected=10.0 * (1.0 - 1.0 / 1.6),
                unit="mm",
                rtol=1e-3,
                tolerance_basis=(
                    "t(1 - 1/n) is the paraxial result; a real trace at h = 0.5 mm "
                    "into f = 100 mm gives 3.750048 mm (measured), 1.3e-5 relative "
                    "from the closed form, so 1e-3 covers any sane sampling while "
                    "still rejecting a sign error or a t/n answer"
                ),
            ),
        ),
        reference=_reference_opt_02,
        notes=(
            "The sign is graded. A plate in a converging beam moves the focus "
            "*away* from the plate; reporting -3.75 fails."
        ),
    ),
    AgentTask(
        task_id="A1-OPT-03",
        title="Single-layer anti-reflection coating at 550 nm",
        library="optiland",
        context_policy=ContextPolicy.WARM,
        exercises=(
            "thin-film modelling, and getting a length unit right when the "
            "literature's unit and the API's unit differ"
        ),
        checks=(
            CheckSpec(
                key="uncoated_reflectance",
                description="normal-incidence reflectance of the bare substrate",
                expected=((1.0 - 1.5168) / (1.0 + 1.5168)) ** 2,
                unit="fraction",
                rtol=1e-3,
                tolerance_basis=(
                    "the Fresnel formula at normal incidence is exact; the pinned "
                    "solver gives 0.04216456 against 0.04216471 analytic"
                ),
            ),
            CheckSpec(
                key="coated_reflectance",
                description="normal-incidence reflectance with the quarter-wave layer",
                expected=((1.5168 - 1.38**2) / (1.5168 + 1.38**2)) ** 2,
                unit="fraction",
                rtol=5e-3,
                tolerance_basis=(
                    "the single-layer quarter-wave formula is exact for a "
                    "non-absorbing film; the pinned solver gives 0.01283544 "
                    "against 0.01283537 analytic. 5e-3 is 300x tighter than the "
                    "gap to the bare-glass value the unit trap produces"
                ),
            ),
            CheckSpec(
                key="coating_thickness_nm",
                description="physical thickness of the quarter-wave layer",
                expected=550.0 / (4.0 * 1.38),
                unit="nm",
                rtol=1e-3,
                tolerance_basis="lambda/(4 n) is the definition of a quarter-wave layer",
            ),
        ),
        reference=_reference_opt_03,
        trap=(
            "MEASURED TRAP. `ThinFilmStack.add_layer` takes MICROMETRES while the "
            "quarter-wave thickness is naturally quoted in nanometres (99.64 nm). "
            "Passing 99.64 builds a layer 1000x too thick and the reflectance comes "
            "back 0.04216384 -- indistinguishable from the bare substrate's "
            "0.04216456, with no error raised and a number that looks like a "
            "reflectance. `coating_thickness_nm` is graded separately precisely so "
            "the report can tell a unit slip from a wrong design."
        ),
        notes="CHE-57 recorded the same unit hazard on upstream tutorial t07.",
    ),
    AgentTask(
        task_id="A1-CHX-01",
        title="Diffractive spreading of a Gaussian beam",
        library="chromatix",
        context_policy=ContextPolicy.COLD,
        exercises=(
            "choosing a wave-propagation package, sampling a field adequately, "
            "propagating it, and measuring a beam radius from an intensity map"
        ),
        checks=(
            CheckSpec(
                key="beam_radius_um",
                description="1/e^2 intensity radius after 100 um",
                expected=5.0 * math.sqrt(1.0 + (100.0 * 0.532 / (math.pi * 25.0)) ** 2),
                unit="um",
                rtol=2e-2,
                tolerance_basis=(
                    "w(z) = w0 sqrt(1 + (z/zR)^2) is exact for a paraxial Gaussian; "
                    "the pinned solver's second moment gives 6.040167 um against "
                    "6.039084 analytic (1.8e-4 relative). 2e-2 absorbs a different "
                    "but reasonable grid or radius definition while rejecting the "
                    "unpropagated waist (5.0 um, 17% low)"
                ),
            ),
        ),
        reference=_reference_chx_01,
        notes=(
            "The discriminating wrong answer is 5.0 um -- the input waist, which is "
            "what a run that propagated zero distance, or measured the input, "
            "returns. It is 17% away, so the tolerance separates them cleanly."
        ),
    ),
    AgentTask(
        task_id="A1-CHX-02",
        title="First dark ring of a focused circular aperture",
        library="chromatix",
        context_policy=ContextPolicy.COLD,
        exercises=(
            "recognising a focal-plane diffraction problem, choosing a "
            "Fourier-transforming propagation rather than a near-field one, and "
            "handling the output sampling that comes with it"
        ),
        checks=(
            CheckSpec(
                key="first_null_radius_um",
                description="radius of the first zero of the focal-plane intensity",
                expected=0.61 * 0.532 / (20.0 / math.hypot(20.0, 400.0)),
                unit="um",
                rtol=5e-2,
                tolerance_basis=(
                    "0.61 lambda / NA is the exact Airy first null. The pinned "
                    "solver's focal-plane pitch is 0.83 um, so the null lands "
                    "between samples and the measured value is 6.65 um against "
                    "6.4985 analytic -- 2.3%, which is a sampling limit rather than "
                    "a physics error. 5e-2 covers it and still rejects the "
                    "1.22 lambda/NA (2x) and 0.5 lambda/NA confusions"
                ),
            ),
        ),
        reference=_reference_chx_02,
        notes=(
            "Deliberately in the regime where a near-field method would wrap "
            "around: the tool choice is part of the task, not incidental to it."
        ),
    ),
    AgentTask(
        task_id="A1-CHX-03",
        title="Lateral walk-off of a tilted beam",
        library="chromatix",
        context_policy=ContextPolicy.WARM,
        exercises=(
            "representing an off-axis beam, propagating it, and getting both the "
            "magnitude and the sign of a transverse displacement right"
        ),
        checks=(
            CheckSpec(
                key="centroid_x_um",
                description="signed x displacement of the intensity centroid",
                expected=200.0 * math.tan(math.radians(5.0)),
                unit="um",
                rtol=2e-2,
                tolerance_basis=(
                    "z tan(theta) is exact geometry for a collimated beam. The "
                    "pinned solver gives +17.5017 um against +17.4977 analytic "
                    "(2.3e-4 relative). 2e-2 rejects z sin(theta) only if the angle "
                    "were large -- at 5 degrees they differ by 0.4%, so this check "
                    "deliberately does NOT claim to separate them, and the report "
                    "says so"
                ),
            ),
        ),
        reference=_reference_chx_03,
        trap=(
            "MEASURED TRAP. `kykx` means CYCLES per length on `asm_propagate` and "
            "RADIANS per length on `plane_wave` -- same parameter name, a factor of "
            "2*pi apart -- and the resulting displacement is OPPOSITE in sign to the "
            "parameter (CHE-57 finding on upstream example c06). Either mistake is "
            "off by 6.28x or by a sign, both far outside the tolerance, and neither "
            "raises."
        ),
        notes=(
            "The sign is the point. A magnitude-only answer, or one from the "
            "opposite convention, fails."
        ),
    ),
)


def task_by_id(task_id: str) -> AgentTask:
    for task in SUITE_V1:
        if task.task_id == task_id:
            return task
    raise KeyError(f"unknown task {task_id!r}; suite v1 is {[t.task_id for t in SUITE_V1]}")


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #


def grade(task: AgentTask, submission: dict[str, Any] | None, *, stderr: str = "") -> TrialResult:
    """Judge one submission, reporting the FIRST stage that failed.

    The staging is the substance of the taxonomy. A run that never produced a
    number cannot also be judged on its physics, and reporting the later failure
    would misattribute the cause -- so the checks below are ordered and the walk
    stops at the first one that fails.
    """
    started = time.time()
    if submission is None:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_NO_SUBMISSION,
            detail="no submission.json was produced", checks=(), stderr=stderr,
            wall_time_s=0.0,
        )
    if submission.get("error"):
        # The participant itself reports it could not run. Which stage that was is
        # its own claim, and is trusted only as far as the vocabulary allows.
        claimed = str(submission["error"])
        outcome = (
            Outcome.FAIL_SIMULATION_CONSTRUCTION
            if "construct" in claimed.lower()
            else Outcome.FAIL_TOOL_EXECUTION
        )
        return TrialResult(
            task_id=task.task_id, outcome=outcome, detail=claimed, checks=(),
            stderr=stderr, wall_time_s=0.0,
        )
    library = str(submission.get("library", "")).strip().lower()
    if library != task.library:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_TOOL_SELECTION,
            detail=(
                f"the submission declares library={library!r}; this task is about "
                f"{task.library!r}. A right answer from the wrong tool is still a "
                "failure of the thing being measured"
            ),
            checks=(), stderr=stderr, wall_time_s=0.0,
        )
    missing = [key for key in task.required_keys() if key not in submission]
    if missing:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_PROBLEM_UNDERSTANDING,
            detail=f"required quantities absent from the submission: {missing}",
            checks=(), stderr=stderr, wall_time_s=0.0,
        )
    results = tuple(check.evaluate(submission[check.key]) for check in task.checks)
    non_numeric = [r for r in results if r.observed is None]
    if non_numeric:
        return TrialResult(
            task_id=task.task_id, outcome=Outcome.FAIL_PROBLEM_UNDERSTANDING,
            detail="; ".join(r.detail for r in non_numeric),
            checks=results, stderr=stderr, wall_time_s=0.0,
        )
    failed = [r for r in results if not r.passed]
    outcome = Outcome.PASS if not failed else Outcome.FAIL_PHYSICAL_RESULT
    return TrialResult(
        task_id=task.task_id,
        outcome=outcome,
        detail=(
            "every check passed"
            if not failed
            else "; ".join(f"{r.spec.key}: {r.detail}" for r in failed)
        ),
        checks=results,
        stderr=stderr,
        wall_time_s=time.time() - started,
    )


@dataclass
class TrialResult:
    task_id: str
    outcome: Outcome
    detail: str
    checks: tuple[CheckResult, ...]
    stderr: str = ""
    wall_time_s: float = 0.0
    trial: int = 0

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "trial": self.trial,
            "outcome": str(self.outcome),
            "detail": self.detail,
            "remedy": OUTCOME_REMEDY[self.outcome],
            "checks": [check.as_dict() for check in self.checks],
            "wall_time_s": self.wall_time_s,
            "stderr_tail": self.stderr[-2000:] if self.stderr else "",
        }


@dataclass
class TaskResult:
    """Every trial of one task, and the pass *rate* rather than a pass/fail.

    Design decision 3. A single agent run is one realization of a stochastic
    process, so one lucky run must not read as a capability. The rate is reported
    even when the trial count is 1, so a reader always sees the denominator.
    """

    task: AgentTask
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def valid_trials(self) -> list[TrialResult]:
        """Trials the agent is accountable for. A harness failure is void."""
        return [t for t in self.trials if t.outcome is not Outcome.FAIL_HARNESS]

    @property
    def pass_rate(self) -> float | None:
        valid = self.valid_trials
        return (sum(t.passed for t in valid) / len(valid)) if valid else None

    def as_dict(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for trial in self.trials:
            counts[str(trial.outcome)] = counts.get(str(trial.outcome), 0) + 1
        return {
            "task": self.task.as_dict(),
            "trials": len(self.trials),
            "valid_trials": len(self.valid_trials),
            "passes": sum(t.passed for t in self.valid_trials),
            "pass_rate": self.pass_rate,
            "outcome_counts": counts,
            "results": [trial.as_dict() for trial in self.trials],
        }


@dataclass
class SuiteResult:
    participant: str
    context_policies: dict[str, str]
    trials: int
    tasks: list[TaskResult] = field(default_factory=list)
    started_unix: float = 0.0
    wall_time_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        valid = [trial for task in self.tasks for trial in task.valid_trials]
        rates = [task.pass_rate for task in self.tasks if task.pass_rate is not None]
        return {
            "suite": "v1",
            "id_space": "A1-* (deliberately disjoint from benchmarks/manifest.yaml)",
            "participant": self.participant,
            "declared_trials_per_task": self.trials,
            "context_policies": self.context_policies,
            "started_unix": self.started_unix,
            "wall_time_s": self.wall_time_s,
            "task_count": len(self.tasks),
            "total_valid_trials": len(valid),
            "total_passes": sum(trial.passed for trial in valid),
            "suite_pass_rate": (sum(rates) / len(rates)) if rates else None,
            "tasks_fully_passed": sum(1 for task in self.tasks if task.pass_rate == 1.0),
            "outcome_counts": {
                code: sum(1 for trial in valid if str(trial.outcome) == code)
                for code in sorted({str(trial.outcome) for trial in valid})
            },
            "results": [task.as_dict() for task in self.tasks],
        }


# --------------------------------------------------------------------------- #
# Participants. A participant turns a task and a workspace into a submission.
# --------------------------------------------------------------------------- #

#: A participant is called with (task, workspace) and returns (submission, stderr).
Participant = Callable[[AgentTask, Path], tuple[dict[str, Any] | None, str]]


def reference_participant(task: AgentTask, workspace: Path) -> tuple[dict[str, Any], str]:
    """Run the shipped reference solution.

    This exists to grade the **grader**. A harness whose reference solutions do not
    pass is measuring something other than what it claims, and no agent score from
    it means anything -- so the reference run is the first thing the opt-in suite
    does, and it is a hard gate.
    """
    submission = task.reference()
    (workspace / "submission.json").write_text(json.dumps(submission, indent=2))
    return submission, ""


def broken_participant(mode: str) -> Participant:
    """A participant that fails in exactly one declared way.

    Every outcome code needs a participant that produces it, or the taxonomy is
    decoration: a code nothing has ever emitted cannot be trusted to fire when it
    matters. Each mode below is a *plausible* mistake, and the two ``trap`` modes
    reproduce the measured unit and convention errors the trap tasks are built on.
    """

    def participate(task: AgentTask, workspace: Path) -> tuple[dict[str, Any] | None, str]:
        if mode == "silent":
            return None, ""
        if mode == "wrong_tool":
            submission: dict[str, Any] = {
                "library": "numpy",
                **{key: 0.0 for key in task.required_keys()},
            }
        elif mode == "missing_quantity":
            submission = {"library": task.library}
        elif mode == "non_numeric":
            submission = {
                "library": task.library,
                **{key: "about right" for key in task.required_keys()},
            }
        elif mode == "execution_error":
            submission = {
                "library": task.library,
                "error": "RuntimeError: the solver raised while tracing",
            }
        elif mode == "construction_error":
            submission = {
                "library": task.library,
                "error": "PrescriptionError: could not construct the system",
            }
        elif mode == "trap":
            submission = _trap_submission(task)
        else:  # pragma: no cover - guarded by the CLI's choices
            raise ValueError(f"unknown broken mode {mode!r}")
        (workspace / "submission.json").write_text(json.dumps(submission, indent=2))
        return submission, ""

    return participate


def _trap_submission(task: AgentTask) -> dict[str, Any]:
    """The answer the *measured* plausible mistake actually produces.

    Not an invented wrong number: each value below was produced by running the
    mistaken code against the pinned solver, so the taxonomy is exercised by the
    failure mode the task was designed around rather than by a placeholder.
    """
    if task.task_id == "A1-OPT-03":
        # add_layer given 99.638 (nanometres) where it wants micrometres.
        return {
            "library": "optiland",
            "uncoated_reflectance": 0.04216456,
            "coated_reflectance": 0.04216384,
            "coating_thickness_nm": 550.0 / (4.0 * 1.38),
        }
    if task.task_id == "A1-CHX-03":
        # The 2*pi confusion, with the sign the parameter actually produces.
        return {
            "library": "chromatix",
            "centroid_x_um": -200.0 * math.tan(math.radians(5.0)) / (2.0 * math.pi),
        }
    if task.task_id == "A1-OPT-01":
        # BFL reported as the EFL: the thick-lens correction omitted.
        efl = 25.0 / 0.5168
        return {
            "library": "optiland",
            "effective_focal_length_mm": efl,
            "back_focal_length_mm": efl,
        }
    if task.task_id == "A1-OPT-02":
        return {"library": "optiland", "focal_shift_mm": -10.0 * (1.0 - 1.0 / 1.6)}
    if task.task_id == "A1-CHX-01":
        return {"library": "chromatix", "beam_radius_um": 5.0}
    if task.task_id == "A1-CHX-02":
        # 1.22 lambda / NA -- the diameter formula used as a radius.
        return {
            "library": "chromatix",
            "first_null_radius_um": 1.22 * 0.532 / (20.0 / math.hypot(20.0, 400.0)),
        }
    raise KeyError(task.task_id)  # pragma: no cover


def command_participant(argv: Sequence[str], *, timeout_s: float = 1800.0) -> Participant:
    """Run an external agent CLI in the workspace and read its ``submission.json``.

    The contract is deliberately minimal so any agent can satisfy it: the prompt is
    written to ``prompt.md``, the permitted context files (per the task's declared
    policy) are copied into ``context/``, and the command is run with the workspace
    as its working directory. ``{prompt}`` and ``{workspace}`` in ``argv`` are
    substituted.

    **Not executed in the CHE-71 delivery run**, and the reason is recorded rather
    than glossed: the container has no agent CLI installed and no API credentials,
    so running an agent would need a decision about spending model tokens that
    belongs to whoever owns the budget. What is delivered is the harness plus the
    reference and negative participants that validate it.
    """

    def participate(task: AgentTask, workspace: Path) -> tuple[dict[str, Any] | None, str]:
        prompt_path = workspace / "prompt.md"
        prompt_path.write_text(task.prompt())
        context = workspace / "context"
        context.mkdir(exist_ok=True)
        for source in task.context_files():
            (context / source.name).write_text(source.read_text())
        command = [
            argument.replace("{prompt}", str(prompt_path)).replace(
                "{workspace}", str(workspace)
            )
            for argument in argv
        ]
        try:
            completed = subprocess.run(
                command, cwd=str(workspace), capture_output=True, text=True,
                timeout=timeout_s, check=False,
            )
            stderr = completed.stderr or ""
        except subprocess.TimeoutExpired:
            return None, f"the participant exceeded its {timeout_s} s budget"
        except FileNotFoundError as exc:
            return None, f"the participant command is not installed: {exc}"
        target = workspace / "submission.json"
        if not target.exists():
            return None, stderr
        try:
            return json.loads(target.read_text()), stderr
        except json.JSONDecodeError as exc:
            return {"error": f"submission.json is not valid JSON: {exc}"}, stderr

    return participate


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


def run_suite(
    participant: Participant,
    *,
    name: str,
    trials: int = 3,
    output: Path | None = None,
    tasks: Sequence[AgentTask] = SUITE_V1,
) -> SuiteResult:
    """Run every task ``trials`` times, sequentially, and record everything."""
    if trials < 1:
        raise ValueError(f"trials must be at least 1, got {trials}")
    started = time.time()
    suite = SuiteResult(
        participant=name,
        context_policies={task.task_id: str(task.context_policy) for task in tasks},
        trials=trials,
        started_unix=started,
    )
    for task in tasks:
        task_result = TaskResult(task=task)
        for trial in range(trials):
            workspace = (
                (output / "workspaces" / f"{task.task_id}_trial{trial}")
                if output
                else Path(os.environ.get("TMPDIR", "/tmp")) / f"a1_{task.task_id}_{trial}"
            )
            workspace.mkdir(parents=True, exist_ok=True)
            try:
                submission, stderr = participant(task, workspace)
                result = grade(task, submission, stderr=stderr)
            except Exception as exc:
                result = TrialResult(
                    task_id=task.task_id, outcome=Outcome.FAIL_HARNESS,
                    detail=f"{type(exc).__name__}: {exc}", checks=(),
                )
            result.trial = trial
            task_result.trials.append(result)
            print(
                f"[A1] {task.task_id} trial {trial}: {result.outcome}"
                + (f" — {result.detail}" if not result.passed else ""),
                flush=True,
            )
        suite.tasks.append(task_result)
    suite.wall_time_s = time.time() - started
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "results.json").write_text(
            json.dumps(suite.as_dict(), indent=2, default=str)
        )
    return suite


def write_expected() -> list[Path]:
    """Record each reference solution's output under ``benchmarks_agent/expected/``.

    Mirrors the CHE-57 tutorial pattern: a recorded output is a regression signal,
    **not** the oracle. The oracle is the closed form on every ``CheckSpec``, which
    is why re-recording cannot make a wrong answer pass.
    """
    written = []
    for task in SUITE_V1:
        record = task.reference()
        payload = {
            "task_id": task.task_id,
            "library": task.library,
            "recorded_reference_output": record,
            "checks": [check.as_dict() for check in task.checks],
            "graded_against": (
                "the closed forms in `checks`, not this recording. Re-recording "
                "cannot make a wrong answer pass."
            ),
        }
        task.expected_path.parent.mkdir(parents=True, exist_ok=True)
        task.expected_path.write_text(json.dumps(payload, indent=2, default=str))
        written.append(task.expected_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="v1", choices=["v1"])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument(
        "--participant", default="reference",
        help=(
            "'reference', 'broken:<mode>', or 'command:<argv...>' with {prompt} and "
            "{workspace} placeholders"
        ),
    )
    parser.add_argument(
        "--context-policy", default="per-task",
        help=(
            "informational only: the policy is declared per task and recorded in "
            "the results. Pass 'per-task' (the default) or a policy name to assert "
            "that every task uses it"
        ),
    )
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--write-expected", action="store_true")
    args = parser.parse_args(argv)

    if args.write_expected:
        for path in write_expected():
            print(f"wrote {path}")
        return 0

    try:
        tasks = (
            tuple(task_by_id(task_id) for task_id in args.task) if args.task else SUITE_V1
        )
    except KeyError as exc:
        # A mistyped --task is a usage error, not a traceback. The message already
        # lists the suite, so surface it rather than re-deriving one.
        parser.error(str(exc.args[0]))
        return 2
    if args.context_policy != "per-task":
        mismatched = [
            task.task_id for task in tasks if str(task.context_policy) != args.context_policy
        ]
        if mismatched:
            parser.error(
                f"--context-policy {args.context_policy!r} does not match the declared "
                f"policy of {mismatched}; the policy is a property of the task"
            )

    spec = args.participant
    if spec == "reference":
        participant, name = reference_participant, "reference"
    elif spec.startswith("broken:"):
        mode = spec.split(":", 1)[1]
        participant, name = broken_participant(mode), spec
    elif spec.startswith("command:"):
        participant = command_participant(spec.split(":", 1)[1].split())
        name = spec
    else:
        parser.error(f"unknown participant {spec!r}")
        return 2

    suite = run_suite(
        participant, name=name, trials=args.trials, output=args.output, tasks=tasks
    )
    record = suite.as_dict()
    print(
        f"\n[A1] participant={name} tasks={record['task_count']} "
        f"trials/task={args.trials} "
        f"passes={record['total_passes']}/{record['total_valid_trials']} "
        f"suite_pass_rate={record['suite_pass_rate']}"
    )
    for code, count in record["outcome_counts"].items():
        print(f"      {code}: {count}")
    return 0 if record["total_passes"] == record["total_valid_trials"] else 1


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
