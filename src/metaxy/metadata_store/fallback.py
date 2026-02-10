"""Lazy fallback store list with per-entry resolution."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from metaxy.config import MetaxyConfig
    from metaxy.metadata_store.base import MetadataStore
    from metaxy.versioning.types import HashAlgorithm


class FallbackStoreList(Sequence["MetadataStore"]):
    """Sequence of fallback stores with per-entry lazy resolution.

    String entries are resolved to MetadataStore instances on first access.
    """

    def __init__(
        self,
        entries: Sequence[MetadataStore | str] | None = None,
        *,
        config: MetaxyConfig | None = None,
        parent_hash_algorithm: HashAlgorithm | None = None,
    ) -> None:
        self._entries: list[MetadataStore | str] = list(entries) if entries else []
        self._config = config
        self._parent_hash_algorithm = parent_hash_algorithm

    def _resolve(self, index: int) -> MetadataStore:
        entry = self._entries[index]
        if not isinstance(entry, str):
            return entry
        if self._config is None:
            raise ValueError(f"Cannot resolve fallback store '{entry}': no MetaxyConfig provided")
        store = self._config.get_store(entry)
        if self._parent_hash_algorithm is not None and store.hash_algorithm != self._parent_hash_algorithm:
            raise ValueError(
                f"Fallback store '{entry}' uses hash_algorithm='{store.hash_algorithm.value}' "
                f"but parent store uses '{self._parent_hash_algorithm.value}'. "
                f"All stores in a fallback chain must use the same hash algorithm."
            )
        self._entries[index] = store
        return store

    @overload
    def __getitem__(self, index: int) -> MetadataStore: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[MetadataStore]: ...
    def __getitem__(self, index: int | slice) -> MetadataStore | Sequence[MetadataStore]:
        if isinstance(index, slice):
            return [self._resolve(i) for i in range(*index.indices(len(self)))]
        return self._resolve(index)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[MetadataStore]:
        for i in range(len(self)):
            yield self._resolve(i)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FallbackStoreList):
            return self._entries == other._entries
        if isinstance(other, list):
            return self._entries == other
        return NotImplemented

    @property
    def all_resolved(self) -> bool:
        return all(not isinstance(e, str) for e in self._entries)
