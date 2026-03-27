"""DuckDB metadata store — thin wrapper that composes DuckDBEngine + IbisStorageConfig."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from metaxy.metadata_store.storage_handler import StorageHandler

if TYPE_CHECKING:
    pass

from metaxy._decorators import public
from metaxy.ext.metadata_stores.ducklake import (
    DuckDBPyConnection,
    DuckLakeAttachmentManager,
    DuckLakeConfig,
    MotherDuckCatalogConfig,
)
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.ibis import IbisMetadataStoreConfig
from metaxy.metadata_store.ibis_compute_engine import (
    DuckLakeStorageConfig,
    IbisComputeEngine,
    IbisSQLHandler,
    IbisStorageConfig,
    IbisStoreBackcompat,
)
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.models.types import FeatureKey
from metaxy.versioning.types import HashAlgorithm


@public
class ExtensionSpec(BaseModel):
    """DuckDB extension specification accepted by DuckDBMetadataStore."""

    name: str
    repository: str = "core"
    """Extension repository: `"core"` for official extensions, `"community"` for community extensions."""
    init_sql: Sequence[str] = ()
    """SQL statements to execute immediately after loading the extension."""


@public
class DuckDBMetadataStoreConfig(IbisMetadataStoreConfig):
    """Configuration for DuckDBMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.metadata_stores.duckdb.DuckDBMetadataStore"

        [stores.dev.config]
        database = "metadata.db"
        hash_algorithm = "xxhash64"
        ```
    """

    database: str | Path = Field(
        description="Database path (:memory:, file path, or md:database).",
    )
    config: dict[str, str] | None = Field(
        default=None,
        description="DuckDB configuration settings (e.g., {'threads': '4'}).",
    )
    extensions: Sequence[str | ExtensionSpec] | None = Field(
        default=None,
        description="DuckDB extensions to install and load on open. If only a string is provided, the `core` repository is assumed.",
    )
    ducklake: DuckLakeConfig | None = Field(
        default=None,
        description="DuckLake attachment configuration. Learn more [here](/integrations/metadata-stores/storage/ducklake.md).",
    )


def _normalise_extensions(
    extensions: Iterable[str | ExtensionSpec],
) -> list[ExtensionSpec]:
    """Coerce extension inputs into ExtensionSpec instances."""
    normalised: list[ExtensionSpec] = []
    for ext in extensions:
        if isinstance(ext, str):
            normalised.append(ExtensionSpec(name=ext))
        elif isinstance(ext, ExtensionSpec):
            normalised.append(ext)
        else:
            raise TypeError(f"DuckDB extensions must be strings or ExtensionSpec instances, got {type(ext).__name__}.")
    return normalised


@public
class DuckDBMetadataStore(IbisStoreBackcompat):
    """[DuckDB](https://duckdb.org/) metadata store using [Ibis](https://ibis-project.org/) backend.

    Example: Local File
        ```py
        store = DuckDBMetadataStore("metadata.db")
        ```

    Example: With extensions
        ```py
        store = DuckDBMetadataStore("md:my_database", extensions=["spatial"])
        ```
    """

    def __init__(
        self,
        database: str | Path = "",  # noqa: ARG002
        *,
        config: dict[str, str] | None = None,  # noqa: ARG002
        extensions: Sequence[str | ExtensionSpec] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        ducklake: DuckLakeConfig | None = None,  # noqa: ARG002
        table_prefix: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    # -- backcompat properties (deprecated, will be removed in 0.2.0) --

    @property
    def database(self) -> str:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("database")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine.database

    @property
    def extensions(self) -> list[ExtensionSpec]:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("extensions")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine.extensions

    def _open(self, mode: Any) -> None:
        """Backcompat: delegates to engine.open(). Patchable for tests."""
        self._engine.open(mode)

    def _close(self) -> None:
        """Backcompat: delegates to engine.close()."""
        self._engine.close()

    def _load_extensions(self) -> None:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_load_extensions")
        assert isinstance(self._engine, DuckDBEngine)
        self._engine._load_extensions()

    def _duckdb_raw_connection(self) -> DuckDBPyConnection:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("_duckdb_raw_connection")
        assert isinstance(self._engine, DuckDBEngine)
        return self._engine._duckdb_raw_connection()

    def preview_ducklake_sql(self) -> list[str]:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("preview_ducklake_sql")
        assert isinstance(self._engine, DuckDBEngine)
        handler = self._engine.handler if isinstance(self._engine.handler, DuckDBDuckLakeHandler) else None
        assert handler is not None, "DuckLake is not configured"
        return handler.preview_ducklake_sql()

    @property
    def ducklake_attachment(self) -> DuckLakeAttachmentManager:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("ducklake_attachment")
        assert isinstance(self._engine, DuckDBEngine)
        handler = self._engine.handler if isinstance(self._engine.handler, DuckDBDuckLakeHandler) else None
        assert handler is not None, "DuckLake is not configured"
        return handler.attachment_manager

    @property
    def ducklake_attachment_config(self) -> DuckLakeConfig:
        from metaxy.metadata_store.ibis_compute_engine import _backcompat_warn_attr

        _backcompat_warn_attr("ducklake_attachment_config")
        assert isinstance(self._engine, DuckDBEngine)
        handler = self._engine.handler if isinstance(self._engine.handler, DuckDBDuckLakeHandler) else None
        assert handler is not None, "DuckLake is not configured"
        return handler.ducklake_config

    def __new__(
        cls,
        database: str | Path = "",
        *,
        config: dict[str, str] | None = None,
        extensions: Sequence[str | ExtensionSpec] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        ducklake: DuckLakeConfig | None = None,
        table_prefix: str | None = None,
        **kwargs: Any,
    ) -> MetadataStore:
        auto_create_tables = kwargs.pop("auto_create_tables", None)
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            auto_create_tables = MetaxyConfig.get().auto_create_tables

        all_extensions = list(_normalise_extensions(extensions or []))
        ducklake_handler: DuckDBDuckLakeHandler | None = None

        prefix = table_prefix or ""
        storage: list[IbisStorageConfig]
        if ducklake is not None:
            ducklake_handler = DuckDBDuckLakeHandler(
                ducklake,
                auto_create_tables=auto_create_tables,
                store_name=kwargs.get("name"),
            )
            for ext in ducklake_handler.required_extensions():
                if ext.name not in {e.name for e in all_extensions}:
                    all_extensions.append(ext)
            storage = [DuckLakeStorageConfig(format="ducklake", location=str(database), table_prefix=prefix)]
        else:
            storage = [IbisStorageConfig(format="duckdb", location=str(database), table_prefix=prefix)]

        engine = DuckDBEngine(
            database=database,
            config=config,
            extensions=all_extensions,
            auto_create_tables=auto_create_tables,
            handler=ducklake_handler,
        )

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
    def from_config(cls, config: MetadataStoreConfig, **kwargs: Any) -> MetadataStore:
        from metaxy.config import MetaxyConfig
        from metaxy.metadata_store.fallback import FallbackStoreList

        assert isinstance(config, DuckDBMetadataStoreConfig)
        config_dict = config.model_dump(exclude_unset=True, exclude={"ducklake", "fallback_stores"})
        store = cast(MetadataStore, cls(ducklake=config.ducklake, **config_dict, **kwargs))
        fallback_store_names = config.model_dump(exclude_unset=True).get("fallback_stores", [])
        if fallback_store_names:
            store.fallback_stores = FallbackStoreList(
                fallback_store_names,
                config=MetaxyConfig.get(),
                parent_hash_algorithm=store.hash_algorithm,
            )
        return store

    @classmethod
    def config_model(cls) -> type[DuckDBMetadataStoreConfig]:
        return DuckDBMetadataStoreConfig


class DuckDBDuckLakeHandler(IbisSQLHandler):
    """Storage handler for DuckLake tables accessed through a DuckDB connection.

    Attachment happens via the ``on_connection_opened`` hook — the engine
    calls it after connecting, before any I/O.
    """

    def __init__(
        self,
        ducklake_config: DuckLakeConfig,
        *,
        auto_create_tables: bool = False,
        store_name: str | None = None,
    ) -> None:
        super().__init__(auto_create_tables=auto_create_tables)
        self._config = ducklake_config
        self._attachment = DuckLakeAttachmentManager(ducklake_config, store_name=store_name)

    # -- capability: only handles DuckLakeStorageConfig -----------------------

    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, DuckLakeStorageConfig)

    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:  # noqa: ARG002
        return isinstance(storage_config, DuckLakeStorageConfig)

    # -- lifecycle hook -------------------------------------------------------

    def on_connection_opened(self, conn: Any) -> None:
        """ATTACH the DuckLake catalog on the live DuckDB connection."""
        raw_conn = conn.con
        self._attachment._attached = False
        self._attachment.configure(raw_conn)

    # -- public DuckLake API -------------------------------------------------

    @property
    def ducklake_config(self) -> DuckLakeConfig:
        return self._config

    @property
    def attachment_manager(self) -> DuckLakeAttachmentManager:
        return self._attachment

    def preview_ducklake_sql(self) -> list[str]:
        return self._attachment.preview_sql()

    def required_extensions(self) -> list[ExtensionSpec]:
        """Extensions this handler needs loaded on the DuckDB connection."""
        exts = [ExtensionSpec(name="ducklake")]
        if isinstance(self._config.catalog, MotherDuckCatalogConfig):
            exts.append(ExtensionSpec(name="motherduck"))
        return exts


class DuckDBEngine(IbisComputeEngine):
    """Compute engine for DuckDB backends using Ibis."""

    def __init__(
        self,
        database: str | Path,
        *,
        config: dict[str, str] | None = None,
        extensions: Sequence[str | ExtensionSpec] | None = None,
        auto_create_tables: bool = False,
        handler: StorageHandler | None = None,
    ) -> None:
        self.database = str(database)
        self.extensions: list[ExtensionSpec] = _normalise_extensions(extensions or [])

        if "hashfuncs" not in {ext.name for ext in self.extensions}:
            self.extensions.append(ExtensionSpec(name="hashfuncs", repository="community"))

        connection_params: dict[str, Any] = {"database": self.database}
        if config:
            connection_params.update(config)

        super().__init__(
            backend="duckdb",
            connection_params=connection_params,
            auto_create_tables=auto_create_tables,
            handler=handler,
        )

    def open(self, mode: AccessMode) -> None:
        if mode == "r":
            db = self.connection_params.get("database", "")
            db = str(db) if db is not None else ""
            is_in_memory = db in {"", ":memory:"}
            scheme = urlsplit(db).scheme if db else ""
            is_windows_drive_path = len(scheme) == 1 and bool(Path(db).drive)
            is_local_file = bool(db) and not is_in_memory and (scheme == "" or is_windows_drive_path)
            is_remote = not (is_in_memory or is_local_file)
            if is_remote or (is_local_file and Path(db).exists()):
                self.connection_params["read_only"] = True
            else:
                self.connection_params.pop("read_only", None)
        else:
            self.connection_params.pop("read_only", None)

        super().open(mode)
        self._load_extensions()

    def close(self) -> None:
        super().close()

    def _create_hash_functions(self) -> dict:
        import ibis

        hash_functions = {}

        @ibis.udf.scalar.builtin
        def MD5(x: str) -> str:  # ty: ignore[empty-body]  # noqa: N802
            ...

        @ibis.udf.scalar.builtin
        def HEX(x: str) -> str:  # ty: ignore[empty-body]  # noqa: N802
            ...

        @ibis.udf.scalar.builtin
        def LOWER(x: str) -> str:  # ty: ignore[empty-body]  # noqa: N802
            ...

        def md5_hash(col_expr):  # noqa: ANN001, ANN202
            return LOWER(MD5(col_expr.cast(str)))

        hash_functions[HashAlgorithm.MD5] = md5_hash

        if "hashfuncs" in {ext.name for ext in self.extensions}:

            @ibis.udf.scalar.builtin
            def xxh32(x: str) -> int:  # ty: ignore[empty-body]
                ...

            @ibis.udf.scalar.builtin
            def xxh64(x: str) -> int:  # ty: ignore[empty-body]
                ...

            def xxhash32_hash(col_expr):  # noqa: ANN001, ANN202
                return xxh32(col_expr.cast(str)).cast(str)

            def xxhash64_hash(col_expr):  # noqa: ANN001, ANN202
                return xxh64(col_expr.cast(str)).cast(str)

            hash_functions[HashAlgorithm.XXHASH32] = xxhash32_hash
            hash_functions[HashAlgorithm.XXHASH64] = xxhash64_hash

        return hash_functions

    def get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    def _duckdb_raw_connection(self) -> DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDB connection is not open.")

        candidate = self._conn.con  # ty: ignore[unresolved-attribute]

        if not isinstance(candidate, DuckDBPyConnection):
            raise TypeError(f"Expected DuckDB backend 'con' to be DuckDBPyConnection, got {type(candidate).__name__}")

        return candidate

    def _load_extensions(self) -> None:
        if not self.extensions:
            return

        duckdb_conn = self._duckdb_raw_connection()
        for ext in self.extensions:
            duckdb_conn.install_extension(ext.name, repository=ext.repository)
            duckdb_conn.load_extension(ext.name)
            for sql in ext.init_sql:
                duckdb_conn.execute(sql)

    @property
    def sqlalchemy_url(self) -> str:
        return f"duckdb:///{self.database}"

    def display(self) -> str:
        from metaxy.metadata_store.utils import sanitize_uri

        return f"DuckDBEngine(database={sanitize_uri(self.database)})"

    @classmethod
    def config_model(cls) -> type[DuckDBMetadataStoreConfig]:
        return DuckDBMetadataStoreConfig
