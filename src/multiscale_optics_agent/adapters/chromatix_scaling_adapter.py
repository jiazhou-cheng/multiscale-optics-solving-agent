"""Narrow Chromatix CPU scaling capability for CHE-15.

External JAX and Chromatix imports remain inside this adapter. Analytic Gaussian
oracles, timing policy, persistence, and pass/fail decisions belong to the
benchmark runner rather than this solver-facing module.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import numpy as np

from multiscale_optics_agent.core.errors import AdapterDependencyError


@dataclass(frozen=True)
class ChromatixScalingRequest:
    n_grid: int
    physical_window_m: float = 64e-6
    waist_m: float = 10e-6
    wavelength_m: float = 532e-9
    refractive_index: float = 1.0
    z_over_rayleigh_range: float = 0.1
    device: str = "cpu"
    dtype: str = "complex64"
    padding_policy: str = "auto_transfer"

    def __post_init__(self) -> None:
        if self.n_grid <= 0:
            raise ValueError("n_grid must be positive")
        for name in ("physical_window_m", "waist_m", "wavelength_m", "refractive_index"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.z_over_rayleigh_range <= 0:
            raise ValueError("z_over_rayleigh_range must be positive")
        if (self.device, self.dtype, self.padding_policy) != (
            "cpu",
            "complex64",
            "auto_transfer",
        ):
            raise ValueError(
                "CHE-15 supports only CPU/complex64 with padding_policy='auto_transfer'"
            )

    @property
    def spacing_m(self) -> float:
        return self.physical_window_m / self.n_grid

    @property
    def rayleigh_range_m(self) -> float:
        return (
            np.pi * self.refractive_index * self.waist_m**2 / self.wavelength_m
        )

    @property
    def distance_m(self) -> float:
        return self.z_over_rayleigh_range * self.rayleigh_range_m


@dataclass(frozen=True)
class ChromatixScalingDependencies:
    jax: Any
    jnp: Any
    functional: Any
    compute_padding_transfer: Any
    chromatix_version: str
    chromatix_commit: str


@dataclass(frozen=True)
class ChromatixScalingOutput:
    field: np.ndarray
    output_spacing_m: tuple[float, float]


@dataclass
class ChromatixScalingSession:
    request: ChromatixScalingRequest
    dependencies: ChromatixScalingDependencies
    field_in: Any
    input_field: np.ndarray
    pad_width: int

    @property
    def input_shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.input_field.shape)

    @property
    def expected_output_shape(self) -> tuple[int, int]:
        size = self.request.n_grid + 2 * self.pad_width
        return (size, size)

    def propagate(self) -> ChromatixScalingOutput:
        field_out = self.dependencies.functional.asm_propagate(
            self.field_in,
            z=self.request.distance_m,
            n=self.request.refractive_index,
            pad_width=self.pad_width,
            mode="full",
        )
        self.dependencies.jax.block_until_ready(field_out.u)
        output = np.ascontiguousarray(
            np.asarray(self.dependencies.jax.device_get(field_out.u)).squeeze()
        )
        spacing = tuple(
            float(value)
            for value in np.asarray(
                self.dependencies.jax.device_get(field_out.dx)
            )
            .reshape(-1)
            .tolist()
        )
        if output.ndim != 2 or tuple(output.shape) != self.expected_output_shape:
            raise ValueError(
                f"Chromatix output shape {output.shape} does not match automatic "
                f"padding expectation {self.expected_output_shape}"
            )
        if output.dtype != np.complex64 or not np.all(np.isfinite(output)):
            raise ValueError(
                f"Chromatix scaling output must be finite complex64; got {output.dtype}"
            )
        return ChromatixScalingOutput(field=output, output_spacing_m=spacing)


def import_chromatix_scaling_dependencies() -> ChromatixScalingDependencies:
    """Import and pin the external CPU/complex64 dependency state for CHE-15."""

    try:
        import importlib.metadata
        import json

        import chromatix.functional as functional
        import jax
        import jax.numpy as jnp
        from chromatix.functional.propagation import compute_padding_transfer
    except Exception as exc:
        raise AdapterDependencyError(
            f"CHE-15 requires pinned Chromatix/JAX: {type(exc).__name__}: {exc}"
        ) from exc
    jax.config.update("jax_enable_x64", False)
    if jax.default_backend() != "cpu":
        raise AdapterDependencyError(
            f"CHE-15 requires JAX CPU; observed backend {jax.default_backend()!r}"
        )
    distribution = importlib.metadata.distribution("chromatix")
    commit = "unknown"
    for entry in distribution.files or []:
        if entry.name == "direct_url.json":
            with suppress(OSError, ValueError, KeyError, TypeError):
                commit = json.loads(entry.read_text())["vcs_info"]["commit_id"]
            break
    return ChromatixScalingDependencies(
        jax=jax,
        jnp=jnp,
        functional=functional,
        compute_padding_transfer=compute_padding_transfer,
        chromatix_version=distribution.version,
        chromatix_commit=commit,
    )


def prepare_chromatix_scaling_session(
    dependencies: ChromatixScalingDependencies,
    request: ChromatixScalingRequest,
) -> ChromatixScalingSession:
    """Build the fixed Gaussian input and automatic-padding propagation session."""

    coordinate = (
        np.arange(request.n_grid, dtype=np.float64) - request.n_grid // 2
    ) * request.spacing_m
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    input_field = np.exp(
        -(xx**2 + yy**2) / request.waist_m**2
    ).astype(np.complex64)
    pad_width = int(
        dependencies.compute_padding_transfer(
            request.n_grid,
            request.wavelength_m,
            request.spacing_m,
            request.distance_m,
        )
    )
    field_in = dependencies.functional.Field.build(
        dependencies.jnp.asarray(input_field, dtype=dependencies.jnp.complex64),
        dependencies.jnp.asarray([[request.spacing_m, request.spacing_m]]),
        request.wavelength_m,
    )
    return ChromatixScalingSession(
        request=request,
        dependencies=dependencies,
        field_in=field_in,
        input_field=input_field,
        pad_width=pad_width,
    )
