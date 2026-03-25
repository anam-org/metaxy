"""PostgreSQL metadata store — thin wrapper that composes PostgreSQLEngine + PostgreSQLSQLHandler."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from metaxy._decorators import experimental, public
from metaxy.ext.ibis.engine import (
    IbisStorageConfig,
    IbisStoreBackcompat,
)
from metaxy.ext.ibis.metadata_store import IbisMetadataStoreConfig
from metaxy.metadata_store.base import MetadataStore


@public
@experimental
class PostgreSQLMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for PostgreSQLMetadataStore.

    Inherits connection_string, connection_params, table_prefix, auto_create_tables from IbisMetadataStoreConfig.
    """

    auto_cast_struct_for_jsonb: bool = Field(
        default=True,
        description="Whether to encode/decode Struct columns to/from JSON on writes/reads. Metaxy system columns are always converted.",
    )


@public
@experimental
class PostgreSQLMetadataStore(IbisStoreBackcompat):
    """PostgreSQL metadata store factory.

    Returns a ``MetadataStore`` composed with a ``PostgreSQLEngine`` and
    ``PostgreSQLSQLHandler`` for JSON/Struct round-tripping.

    Example:
        <!-- skip next -->
        ```python
        from metaxy.ext.postgresql import PostgreSQLMetadataStore

        store = PostgreSQLMetadataStore(connection_string="postgresql://user:pass@localhost:5432/metaxy")

        with store:
            increment = store.resolve_update(MyFeature)
            store.write(MyFeature, increment.added)
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,  # noqa: ARG002
        *,
        connection_params: dict[str, Any] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        auto_cast_struct_for_jsonb: bool = True,  # noqa: ARG002
        table_prefix: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    def __new__(
        cls,
        connection_string: str | None = None,
        *,
        connection_params: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        auto_cast_struct_for_jsonb: bool = True,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        from metaxy.ext.postgresql.engine import PostgreSQLEngine
        from metaxy.ext.postgresql.handlers.native import PostgreSQLSQLHandler

        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params")

        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        handler = PostgreSQLSQLHandler(
            auto_create_tables=auto_create_tables,
            auto_cast_struct_for_jsonb=auto_cast_struct_for_jsonb,
        )
        engine = PostgreSQLEngine(
            connection_string=connection_string,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handlers=[handler],
        )

        prefix = table_prefix or ""
        location = connection_string or "postgresql"
        storage = [IbisStorageConfig(format="postgresql", location=location, table_prefix=prefix)]

        instance = IbisStoreBackcompat.__new__(cls)
        MetadataStore.__init__(
            instance,
            engine=engine,
            storage=storage,
            fallback_stores=fallback_stores,
            auto_create_tables=auto_create_tables,
            **kwargs,
        )
        return instance

    @classmethod
    def config_model(cls) -> type[PostgreSQLMetadataStoreConfig]:
        return PostgreSQLMetadataStoreConfig

    # -- backcompat properties (deprecated, will be removed in 0.2.0) --

    @property
    def _pg_handler(self):
        from metaxy.ext.postgresql.handlers.native import PostgreSQLSQLHandler

        self._engine._ensure_defaults()
        for handler in self._engine._handlers:
            if isinstance(handler, PostgreSQLSQLHandler):
                return handler
        raise RuntimeError("No PostgreSQLSQLHandler found")

    @property
    def _conn(self) -> Any:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("_conn")
        return self._ibis_engine._conn

    @property
    def auto_cast_struct_for_jsonb(self) -> bool:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("auto_cast_struct_for_jsonb")
        return self._pg_handler.auto_cast_struct_for_jsonb

    @auto_cast_struct_for_jsonb.setter
    def auto_cast_struct_for_jsonb(self, value: bool) -> None:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("auto_cast_struct_for_jsonb")
        self._pg_handler.auto_cast_struct_for_jsonb = value

    def transform_before_write(self, df: Any, feature_key: Any, table_name: str) -> Any:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("transform_before_write")
        return self._pg_handler.transform_before_write(self._ibis_engine.conn, df, table_name, feature_key)

    def _get_json_columns_for_struct(self, ibis_schema: Any) -> list[str]:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("_get_json_columns_for_struct")
        return self._pg_handler._get_json_columns_for_struct(ibis_schema)

    def _parse_json_to_struct_columns(self, pl_df: Any, feature_key: Any, json_columns: list[str]) -> Any:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("_parse_json_to_struct_columns")
        return self._pg_handler._parse_json_to_struct_columns(pl_df, feature_key, json_columns)

    def _validate_required_system_struct_columns(
        self,
        pl_df: Any,
        feature_key: Any,
        json_columns: list[str],
        *,
        require_all_system_columns: bool = False,
    ) -> None:
        from metaxy.ext.ibis.engine import _backcompat_warn_attr

        _backcompat_warn_attr("_validate_required_system_struct_columns")
        self._pg_handler._validate_required_system_struct_columns(
            pl_df,
            feature_key,
            json_columns,
            require_all_system_columns=require_all_system_columns,
        )
