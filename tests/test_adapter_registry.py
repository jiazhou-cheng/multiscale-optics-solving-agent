import pytest

from adapters.registry import (
    available_model_ids,
    get_adapter_for_model,
)
from core.errors import AdapterNotFoundError


def test_unknown_model_id_raises_not_found() -> None:
    with pytest.raises(AdapterNotFoundError):
        get_adapter_for_model("M_DOES_NOT_EXIST")


@pytest.mark.parametrize("model_id", sorted(available_model_ids()))
def test_discovered_adapter_spec_matches_model_id(model_id: str) -> None:
    adapter = get_adapter_for_model(model_id)
    assert adapter.spec.id == model_id
