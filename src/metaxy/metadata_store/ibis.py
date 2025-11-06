from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from typing_extensions import Self

from metaxy.metadata_store.base import MetadataStore
from metaxy.models.plan import FeaturePlan
from metaxy.provenance.ibis import IbisHashFn, IbisProvenanceTracker
from metaxy.provenance.tracker import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


class IbisMetadataStore(MetadataStore):
    def __init__(self, backend: Any, hash_functions: dict[HashAlgorithm, IbisHashFn]):
        self.backend = backend
        self.hash_functions = hash_functions

    @contextmanager
    def open(self) -> Iterator[Self]:
        try:
            yield self
        finally:
            pass

    def supports_native_tracker(self) -> bool:
        return True

    def create_tracker(self, plan: FeaturePlan) -> ProvenanceTracker:
        return IbisProvenanceTracker(plan, self.backend, self.hash_functions)
