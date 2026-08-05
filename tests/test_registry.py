from multiscale_optics_agent.registry.loader import Registry


def test_packaged_registry_loads() -> None:
    registry = Registry.from_package()
    assert "M_RAY_OPTILAND" in registry.models
    assert "M_WAVE_CHROMATIX" in registry.models
    assert "C_RAY_TO_WAVE" in registry.couplers


def test_registry_ids_are_unique() -> None:
    registry = Registry.from_package()
    assert len(registry.models) == len(set(registry.models))
    assert len(registry.couplers) == len(set(registry.couplers))
