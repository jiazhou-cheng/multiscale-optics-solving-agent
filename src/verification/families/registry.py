"""The one place that knows which benchmark families exist.

CHE-131 (M0.5.2). A registry rather than a module-scanning discovery mechanism,
for the reason ``core/capabilities.py`` gives about its own table: a
source-of-truth that is *discovered* agrees with itself by construction,
including about which entries exist at all. Registration is explicit, and
``tests/test_family_schema.py`` asserts that every family module the package
ships is actually imported here.

Family ids are unique. This replaces the ``A1-*`` id-space collision test, which
guarded an id space that is being retired.
"""

from __future__ import annotations

from verification.families.schema import BenchmarkCategory, BenchmarkFamily

__all__ = [
    "FAMILIES",
    "families_for_category",
    "family",
    "family_ids",
    "register",
]


_REGISTRY: dict[str, BenchmarkFamily] = {}


def register(fam: BenchmarkFamily) -> BenchmarkFamily:
    """Add a family, refusing a duplicate id.

    Returns the family so a module can write ``FOO = register(BenchmarkFamily(...))``
    and have one statement rather than two that can drift.
    """
    if fam.family_id in _REGISTRY:
        existing = _REGISTRY[fam.family_id]
        raise ValueError(
            f"duplicate family_id {fam.family_id!r}: already registered at version "
            f"{existing.family_version}. Two families with one id means a result "
            "cannot say which one produced it."
        )
    _REGISTRY[fam.family_id] = fam
    return fam


def family(family_id: str) -> BenchmarkFamily:
    try:
        return _REGISTRY[family_id]
    except KeyError:
        raise KeyError(f"no family {family_id!r}; registered: {sorted(_REGISTRY)}") from None


def family_ids() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def families_for_category(category: BenchmarkCategory) -> tuple[BenchmarkFamily, ...]:
    return tuple(f for _, f in sorted(_REGISTRY.items()) if f.category is category)


class _Families:
    """A read-only mapping view, so nothing mutates the registry by accident."""

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(_REGISTRY.values())

    def __len__(self) -> int:
        return len(_REGISTRY)

    def __contains__(self, key: object) -> bool:
        return key in _REGISTRY

    def __getitem__(self, key: str) -> BenchmarkFamily:
        return family(key)

    def values(self) -> tuple[BenchmarkFamily, ...]:
        return tuple(_REGISTRY[k] for k in sorted(_REGISTRY))

    def keys(self) -> tuple[str, ...]:
        return family_ids()

    def items(self) -> tuple[tuple[str, BenchmarkFamily], ...]:
        return tuple((k, _REGISTRY[k]) for k in sorted(_REGISTRY))

    def __repr__(self) -> str:
        return f"FAMILIES({', '.join(family_ids())})"


#: The registered families. Iterating yields :class:`BenchmarkFamily` objects.
FAMILIES = _Families()
