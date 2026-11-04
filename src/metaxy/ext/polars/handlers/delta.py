"""Delta Lake storage implemented with delta-rs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field

from metaxy._decorators import public
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.io_handler import IOHandler
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.table_ref import DeltaTableIdentifier
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.types import FeatureKey
from metaxy.utils import collect_to_polars


@public
class DeltaMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for DeltaMetadataStore.

    Example:
        ```toml title="metaxy.toml"
        [stores.dev]
        type = "metaxy.ext.polars.handlers.delta.DeltaMetadataStore"

        [stores.dev.config]
        root_path = "s3://my-bucket/metaxy"
        layout = "nested"

        [stores.dev.config.storage_options]
        AWS_REGION = "us-west-2"
        ```
    """

    root_path: str | Path = Field(
        description="Base directory or URI where feature tables are stored.",
    )
    storage_options: dict[str, Any] | None = Field(
        default=None,
        description="Storage backend options passed to delta-rs.",
    )
    layout: Literal["flat", "nested"] = Field(
        default="nested",
        description="Directory layout for feature tables ('nested' or 'flat').",
    )
    delta_write_options: dict[str, Any] | None = Field(
        default=None,
        description="Options passed to [`deltalake.write_deltalake`][deltalake.write_deltalake].",
    )


class DeltaStorageConfig(StorageConfig):
    """Storage configuration for Delta Lake tables under `location`.

    Attributes:
        layout: `"nested"` stores each feature in a directory per key part, `"flat"` in one directory per feature.
        storage_options: Storage backend options passed to delta-rs.
    """

    layout: Literal["flat", "nested"] = "nested"
    storage_options: dict[str, Any] = Field(default_factory=dict)

    def resolve(self, key: FeatureKey) -> DeltaTableIdentifier:
        table_path = "/".join(part for part in key.parts if part) if self.layout == "nested" else key.table_name
        return DeltaTableIdentifier(uri=f"{self.location}/{table_path}.delta", storage_options=self.storage_options)


class DeltaHandler(IOHandler[None, DeltaTableIdentifier]):
    """Reads and writes Delta Lake tables with delta-rs and Polars."""

    def __init__(self, *, write_options: dict[str, Any] | None = None) -> None:
        self.write_options = {"mode": "append", "schema_mode": "merge", **(write_options or {})}
        """Options passed to [`deltalake.write_deltalake`][deltalake.write_deltalake]."""

    def can_handle(self, storage_config: StorageConfig) -> bool:
        return isinstance(storage_config, DeltaStorageConfig)

    def read(
        self,
        conn: None,  # noqa: ARG002
        table_id: DeltaTableIdentifier,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        if not self.has_feature(None, table_id):
            return None
        lazy = nw.from_native(pl.scan_delta(str(table_id.uri), storage_options=table_id.storage_options or None))
        if filters:
            lazy = lazy.filter(*filters)
        if columns is not None:
            lazy = lazy.select(columns)
        return lazy

    def write(
        self,
        conn: None,  # noqa: ARG002
        table_id: DeltaTableIdentifier,
        df: Frame,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        import deltalake

        # delta-rs does not support Polars Enum columns
        frame = collect_to_polars(df).with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))
        options: dict[str, Any] = {"storage_options": table_id.storage_options or None, **self.write_options}
        deltalake.write_deltalake(
            str(table_id.uri),
            frame.to_arrow(),
            mode=options.pop("mode"),
            **options,
        )

    def has_feature(
        self,
        conn: None,  # noqa: ARG002
        table_id: DeltaTableIdentifier,
    ) -> bool:
        # `deltalake.DeltaTable.is_deltatable()` sometimes hangs when called from several threads,
        # so existence is checked by opening the table instead.
        from deltalake.exceptions import TableNotFoundError

        try:
            self._open_table(table_id, without_files=True)
        except TableNotFoundError:
            return False
        return True

    def drop(
        self,
        conn: None,  # noqa: ARG002
        table_id: DeltaTableIdentifier,
    ) -> None:
        if self.has_feature(None, table_id):
            self._open_table(table_id, without_files=True).delete()

    def delete(
        self,
        conn: None,
        table_id: DeltaTableIdentifier,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,  # noqa: ARG002
    ) -> None:
        """Hard-delete rows with the native Delta `DELETE` operation.

        Note:
            Ibis (`ibis-framework`) converts the Narwhals filters to SQL. It is included in the `delta` extras.
        """
        lazy = self.read(conn, table_id)
        if lazy is None:
            return
        if not filters:
            self._open_table(table_id).delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        predicate = narwhals_expr_to_sql_predicate(filters, lazy.collect_schema(), dialect="datafusion")
        self._open_table(table_id).delete(predicate=predicate)

    def get_store_metadata(self, table_id: DeltaTableIdentifier) -> dict[str, Any]:
        return {"uri": str(table_id.uri)}

    def _open_table(self, table_id: DeltaTableIdentifier, *, without_files: bool = False) -> Any:
        import deltalake

        return deltalake.DeltaTable(
            str(table_id.uri), storage_options=table_id.storage_options or None, without_files=without_files
        )


@public
class DeltaMetadataStore(MetadataStore):
    """
    Delta Lake metadata store backed by [delta-rs](https://github.com/delta-io/delta-rs).

    It stores feature metadata in Delta Lake tables located under ``root_path``.
    It uses the Polars versioning engine for provenance calculations.

    Example:

        ```py
        from metaxy.ext.polars.handlers.delta import DeltaMetadataStore

        store = DeltaMetadataStore(
            root_path="s3://my-bucket/metaxy",
            storage_options={"AWS_REGION": "us-west-2"},
        )
        ```
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        root_path: str | Path | None = None,
        *,
        storage_options: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        layout: Literal["flat", "nested"] = "nested",
        delta_write_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if root_path is None:
            raise ValueError("root_path is required")

        from metaxy.ext.polars.engine import PolarsComputeEngine

        if layout not in ("flat", "nested"):
            raise ValueError(f"Invalid layout: {layout}. Must be 'flat' or 'nested'.")

        root_str = str(root_path)
        if is_local_path(root_str):
            root_str = str(Path(root_str.removeprefix("file://").removeprefix("local://")).expanduser().resolve())
        else:
            root_str = root_str.rstrip("/")

        super().__init__(
            engine=PolarsComputeEngine(handlers=[DeltaHandler(write_options=delta_write_options)]),
            storage=DeltaStorageConfig(location=root_str, layout=layout, storage_options=storage_options or {}),
            fallback_stores=fallback_stores,
            versioning_engine="polars",
            **kwargs,
        )

    @property
    def _delta_storage(self) -> DeltaStorageConfig:
        assert isinstance(self._storage, DeltaStorageConfig)
        return self._storage

    @property
    def _root_uri(self) -> str:
        return self._delta_storage.location

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        """Return the URI of the Delta table that holds `feature_key`."""
        return str(self._delta_storage.resolve(feature_key).uri)

    def display(self) -> str:
        """Return human-readable representation of the store."""
        return f"DeltaMetadataStore(path={self._root_uri})"

    @classmethod
    def config_model(cls) -> type[DeltaMetadataStoreConfig]:
        return DeltaMetadataStoreConfig
