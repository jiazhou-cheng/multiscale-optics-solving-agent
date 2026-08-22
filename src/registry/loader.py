"""Load model and coupler specifications from versioned YAML registries."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

from core.errors import RegistryError
from core.specs import CouplerSpec, GraphSpec, ModelSpec


@dataclass(frozen=True, slots=True)
class Registry:
    models: dict[str, ModelSpec]
    couplers: dict[str, CouplerSpec]

    @classmethod
    def from_package(cls) -> Registry:
        root = files("registry")
        model_data = cls._read_yaml_resource(root.joinpath("models.yaml"))
        coupler_data = cls._read_yaml_resource(root.joinpath("couplers.yaml"))
        return cls._from_data(model_data, coupler_data)

    @classmethod
    def from_mapping(cls, model_data: Any, coupler_data: Any) -> Registry:
        """Build a registry from already-parsed data.

        Exists so a test can declare the component shapes it needs instead of
        borrowing the shipped registry as a bag of examples. That borrowing is
        what kept eight unimplemented couplers alive: deleting an entry nothing
        implements would have broken a test that only wanted its port types.
        The packaged registry states what this repository can execute; a fixture
        states what a test needs, and they are not the same claim.
        """
        return cls._from_data(model_data, coupler_data)

    @classmethod
    def from_files(cls, models_path: Path, couplers_path: Path) -> Registry:
        return cls._from_data(
            cls._read_yaml_path(models_path),
            cls._read_yaml_path(couplers_path),
        )

    @staticmethod
    def load_graph(path: Path) -> GraphSpec:
        data = Registry._read_yaml_path(path)
        try:
            return GraphSpec.model_validate(data)
        except Exception as exc:  # Pydantic formats the underlying schema details.
            raise RegistryError(f"Invalid graph YAML at {path}: {exc}") from exc

    @classmethod
    def _from_data(cls, model_data: Any, coupler_data: Any) -> Registry:
        try:
            model_items = model_data["models"]
            coupler_items = coupler_data["couplers"]
            models = [ModelSpec.model_validate(item) for item in model_items]
            couplers = [CouplerSpec.model_validate(item) for item in coupler_items]
        except (KeyError, TypeError, ValueError) as exc:
            raise RegistryError(f"Invalid registry data: {exc}") from exc

        model_map = {item.id: item for item in models}
        coupler_map = {item.id: item for item in couplers}
        if len(model_map) != len(models):
            raise RegistryError("Duplicate model IDs in registry")
        if len(coupler_map) != len(couplers):
            raise RegistryError("Duplicate coupler IDs in registry")
        return cls(models=model_map, couplers=coupler_map)

    @staticmethod
    def _read_yaml_path(path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return yaml.safe_load(handle)
        except OSError as exc:
            raise RegistryError(f"Unable to read YAML at {path}: {exc}") from exc

    @staticmethod
    def _read_yaml_resource(resource: Any) -> Any:
        try:
            return yaml.safe_load(resource.read_text(encoding="utf-8"))
        except OSError as exc:
            raise RegistryError(f"Unable to read packaged registry: {exc}") from exc
