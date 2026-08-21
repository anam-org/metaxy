from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from metaxy.metadata_store.compute_engine import ComputeEngine
from metaxy.metadata_store.io_handler import IOHandler
from metaxy.metadata_store.types import AccessMode
from metaxy.versioning import VersioningEngine
from metaxy.versioning.types import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.metadata_store.base import MetadataStoreConfig
    from metaxy.models.plan import FeaturePlan


class PolarsComputeEngine(ComputeEngine):
    """Compute engine for backends that use Polars for versioning (e.g. Delta, Lance).

    Has no persistent connection — handlers receive ``None`` and manage their own I/O.
    """

    def __init__(self) -> None:
        super().__init__()
        from metaxy.ext.polars.versioning import PolarsVersioningEngine

        self.versioning_engine_cls = PolarsVersioningEngine

    def _create_default_handlers(self) -> list[IOHandler[Any, Any]]:
        return []

    @property
    def connection(self) -> None:
        return None

    def open(self, mode: AccessMode) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    def validate_hash_algorithm_support(self, algorithm: HashAlgorithm) -> None:
        from metaxy.ext.polars.versioning import PolarsVersioningEngine
        from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError

        supported = PolarsVersioningEngine.supported_hash_algorithms()
        if algorithm not in supported:
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(a.value for a in sorted(supported, key=lambda a: a.value))}"
            )

    @contextmanager
    def create_versioning_engine(self, plan: FeaturePlan) -> Iterator[VersioningEngine]:
        from metaxy.ext.polars.versioning import PolarsVersioningEngine

        yield PolarsVersioningEngine(plan=plan)

    def display(self) -> str:
        return "PolarsComputeEngine()"

    @classmethod
    def config_model(cls) -> type[MetadataStoreConfig]:
        return MetadataStoreConfig
