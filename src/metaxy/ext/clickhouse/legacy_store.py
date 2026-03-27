from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from metaxy.ext.clickhouse.config import ClickHouseMetadataStoreConfig

from metaxy._decorators import public
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.ibis_compute_engine import (
    IbisStorageConfig,
    IbisStoreBackcompat,
)


@public
class ClickHouseMetadataStore(IbisStoreBackcompat):
    """[ClickHouse](https://clickhouse.com/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Example: Connection String
        ```py
        store = ClickHouseMetadataStore("clickhouse://localhost:8443/default")
        ```

    Example: Connection Parameters
        ```py
        store = ClickHouseMetadataStore(
            connection_params={
                "host": "localhost",
                "port": 8443,
                "database": "default",
            },
            hash_algorithm=HashAlgorithm.XXHASH64,
        )
        ```
    """

    def __init__(
        self,
        connection_string: str | None = None,  # noqa: ARG002
        *,
        connection_params: dict[str, Any] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        auto_cast_struct_for_map: bool = True,  # noqa: ARG002
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
        auto_cast_struct_for_map: bool = True,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        if connection_string is None and connection_params is None:
            raise ValueError("Must provide either connection_string or connection_params")

        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        from metaxy.ext.clickhouse.engine import ClickHouseEngine
        from metaxy.ext.clickhouse.handlers.native import ClickHouseSQLHandler

        handler = ClickHouseSQLHandler(
            auto_create_tables=auto_create_tables,
            auto_cast_struct_for_map=auto_cast_struct_for_map,
        )

        engine = ClickHouseEngine(
            connection_string=connection_string,
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handler=handler,
        )

        storage = [
            IbisStorageConfig(
                format="clickhouse",
                location=connection_string or "clickhouse",
                table_prefix=table_prefix or "",
            )
        ]

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
    def from_config(cls, config: MetadataStoreConfig, **kwargs: Any) -> MetadataStore:  # type: ignore[override]
        from metaxy.config import MetaxyConfig
        from metaxy.metadata_store.fallback import FallbackStoreList

        config_dict = config.model_dump(exclude_unset=True, exclude={"fallback_stores"})
        store = cast(MetadataStore, cls(**config_dict, **kwargs))
        fallback_store_names = config.model_dump(exclude_unset=True).get("fallback_stores", [])
        if fallback_store_names:
            store.fallback_stores = FallbackStoreList(
                fallback_store_names,
                config=MetaxyConfig.get(),
                parent_hash_algorithm=store.hash_algorithm,
            )
        return store

    @classmethod
    def config_model(cls) -> type[ClickHouseMetadataStoreConfig]:
        from metaxy.ext.clickhouse.config import ClickHouseMetadataStoreConfig

        return ClickHouseMetadataStoreConfig

    # -- backcompat properties (deprecated, will be removed in 0.2.0) --

    @property
    def _conn(self) -> Any:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_conn")
        return self._ibis_engine._conn
