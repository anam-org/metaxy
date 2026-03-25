"""PostgreSQL compute engine using Ibis with Polars versioning."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from metaxy.ext.ibis.engine import IbisComputeEngine
from metaxy.ext.ibis.metadata_store import IbisMetadataStoreConfig
from metaxy.ext.polars.versioning import PolarsVersioningEngine
from metaxy.metadata_store.exceptions import HashAlgorithmNotSupportedError
from metaxy.metadata_store.io_handler import IOHandler
from metaxy.models.plan import FeaturePlan
from metaxy.versioning.types import HashAlgorithm


class PostgreSQLEngine(IbisComputeEngine):
    """Compute engine for PostgreSQL backends using Ibis with Polars versioning."""

    versioning_engine_cls = PolarsVersioningEngine

    def __init__(
        self,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        auto_create_tables: bool = False,
        handlers: Sequence[IOHandler[Any, Any]] | None = None,
    ) -> None:
        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params for PostgreSQL")

        super().__init__(
            connection_string=connection_string,
            backend="postgres" if connection_string is None else None,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handlers=handlers,
        )

    def _create_default_handlers(self) -> list[IOHandler[Any, Any]]:
        from metaxy.ext.postgresql.handlers.native import PostgreSQLSQLHandler

        return [PostgreSQLSQLHandler(auto_create_tables=self._auto_create_tables)]

    def _create_hash_functions(self) -> dict:
        return {}

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    @contextmanager
    def create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        yield PolarsVersioningEngine(plan=plan)

    def validate_hash_algorithm_support(self, algorithm: HashAlgorithm) -> None:
        supported = PolarsVersioningEngine.supported_hash_algorithms()
        if algorithm not in supported:
            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm '{algorithm.value}' not supported. "
                f"Supported algorithms: {', '.join(a.value for a in sorted(supported, key=lambda a: a.value))}"
            )

    def display(self) -> str:
        from metaxy.metadata_store.utils import sanitize_uri

        location = self.connection_string or "postgresql"
        return f"PostgreSQLEngine(connection={sanitize_uri(location)})"

    @classmethod
    def config_model(cls) -> type[IbisMetadataStoreConfig]:
        from metaxy.ext.postgresql.metadata_store import PostgreSQLMetadataStoreConfig

        return PostgreSQLMetadataStoreConfig
