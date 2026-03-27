"""Delta Lake storage handler using delta-rs and Polars."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, overload

import deltalake
import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from packaging.version import Version

from metaxy._utils import collect_to_polars
from metaxy.metadata_store.polars.storage_config import PolarsStorageConfig
from metaxy.metadata_store.storage_config import StorageConfig
from metaxy.metadata_store.storage_handler import StorageHandler
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.types import FeatureKey


class DeltaHandler(StorageHandler[None]):
    """Reads and writes Delta Lake tables via delta-rs and Polars.

    Tables are stored under ``root_uri`` using either a nested
    (``a/b/c.delta``) or flat (``a__b__c.delta``) directory layout.
    """

    def __init__(
        self,
        root_uri: str,
        *,
        storage_options: dict[str, Any] | None = None,
        layout: str = "nested",
        delta_write_options: dict[str, Any] | None = None,
    ) -> None:
        root_str = str(root_uri)
        self._storage_options: dict[str, Any] = storage_options or {}
        self._layout = layout
        self._delta_write_options: dict[str, Any] = delta_write_options or {}

        if is_local_path(root_str):
            cleaned = root_str.removeprefix("file://").removeprefix("local://")
            self._root_uri = str(Path(cleaned).expanduser().resolve())
        else:
            self._root_uri = root_str.rstrip("/")

    # -- StorageHandler interface --------------------------------------------------

    def can_read(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        return isinstance(storage_config, PolarsStorageConfig)

    def can_write(self, storage_config: StorageConfig, key: FeatureKey) -> bool:
        return isinstance(storage_config, PolarsStorageConfig)

    def read(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        if not self._table_exists(key):
            return None

        lf = pl.scan_delta(
            self._feature_uri(key),
            storage_options=self._storage_options or None,
        )

        from metaxy.config import MetaxyConfig

        if MetaxyConfig.get().enable_map_datatype:
            lf = self._read_map_columns(lf, key)

        nw_lazy = nw.from_native(lf)
        if filters:
            nw_lazy = nw_lazy.filter(*filters)
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)
        return nw_lazy

    def write(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        table_uri = self._feature_uri(key)
        write_opts = self._default_write_options.copy()
        mode = write_opts.pop("mode", "append")
        storage_options = write_opts.pop("storage_options", None)

        from metaxy.config import MetaxyConfig

        if MetaxyConfig.get().enable_map_datatype:
            self._write_with_map_columns(df, table_uri, mode, storage_options, write_opts)
            return

        can_sink = (
            df.implementation == nw.Implementation.POLARS
            and isinstance(df, nw.LazyFrame)
            and Version(pl.__version__) >= Version("1.37.0")
        )

        if can_sink:
            lf_native = df.to_native()
            assert isinstance(lf_native, pl.LazyFrame)
            self._cast_enum_to_string(lf_native).sink_delta(
                table_uri,
                mode=mode,
                storage_options=storage_options,
                delta_write_options=write_opts or None,
            )
        else:
            self._cast_enum_to_string(collect_to_polars(df)).write_delta(
                table_uri,
                mode=mode,
                storage_options=storage_options,
                delta_write_options=write_opts or None,
            )

    def has_feature(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> bool:
        return self._table_exists(key)

    def drop(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> None:
        if not self._table_exists(key):
            return
        deltalake.DeltaTable(
            self._feature_uri(key),
            storage_options=self._storage_options or None,
            without_files=True,
        ).delete()

    def delete(
        self,
        conn: None,
        storage_config: StorageConfig,
        key: FeatureKey,
        filters: Sequence[nw.Expr] | None,
        *,
        with_feature_history: bool,
    ) -> None:
        if not self._table_exists(key):
            return

        delta_table = deltalake.DeltaTable(
            self._feature_uri(key),
            storage_options=self._storage_options or None,
        )

        if not filters:
            delta_table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        lf = self.read(None, storage_config, key)
        if lf is None:
            return
        delta_table.delete(
            predicate=narwhals_expr_to_sql_predicate(
                filters,
                lf.collect_schema(),
                dialect="datafusion",
            )
        )

    def get_store_metadata(
        self,
        storage_config: StorageConfig,
        key: FeatureKey,
    ) -> dict[str, Any]:
        return {"uri": self._feature_uri(key)}

    # -- Internal helpers ----------------------------------------------------------

    def _feature_uri(self, key: FeatureKey) -> str:
        if self._layout == "nested":
            table_path = "/".join(part for part in key.parts if part)
        else:
            table_path = key.table_name
        return f"{self._root_uri}/{table_path}.delta"

    def _table_exists(self, key: FeatureKey) -> bool:
        # DeltaTable.is_deltatable() can hang in multi-threaded settings,
        # so existence is checked by catching the constructor exception.
        from deltalake.exceptions import TableNotFoundError as DeltaTableNotFoundError

        try:
            deltalake.DeltaTable(
                self._feature_uri(key),
                storage_options=self._storage_options or None,
                without_files=True,
            )
        except DeltaTableNotFoundError:
            return False
        return True

    @cached_property
    def _default_write_options(self) -> dict[str, Any]:
        write_kwargs: dict[str, Any] = {
            "mode": "append",
            "schema_mode": "merge",
            "storage_options": self._storage_options or None,
        }
        write_kwargs.update(self._delta_write_options)
        return write_kwargs

    @overload
    def _cast_enum_to_string(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def _cast_enum_to_string(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def _cast_enum_to_string(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Cast Enum columns to String to avoid delta-rs Utf8View incompatibility."""
        return frame.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

    def _write_with_map_columns(
        self,
        df: Frame,
        table_uri: str,
        mode: Literal["error", "append", "overwrite", "ignore"],
        storage_options: dict[str, str] | None,
        write_opts: dict[str, Any],
    ) -> None:
        """Collect to Arrow and convert Struct *_by_field columns to native MapArray before writing."""
        from metaxy.models.constants import METAXY_DATA_VERSION_BY_FIELD, METAXY_PROVENANCE_BY_FIELD
        from metaxy.versioning._arrow_map import convert_extension_maps_to_native, convert_structs_to_maps

        df_native = self._cast_enum_to_string(collect_to_polars(df))
        df_native = convert_structs_to_maps(
            df_native, columns=[METAXY_PROVENANCE_BY_FIELD, METAXY_DATA_VERSION_BY_FIELD]
        )
        arrow_table = convert_extension_maps_to_native(df_native.to_arrow())
        deltalake.write_deltalake(
            table_uri,
            arrow_table,
            mode=mode,
            storage_options=storage_options,
            **(write_opts or {}),
        )

    def _read_map_columns(self, lf: pl.LazyFrame, key: FeatureKey) -> pl.LazyFrame:
        """Convert native Delta Map columns back to polars_map.Map on read."""
        from deltalake.schema import MapType

        try:
            delta_table = deltalake.DeltaTable(
                self._feature_uri(key),
                storage_options=self._storage_options or None,
                without_files=True,
            )
        except Exception:
            return lf

        map_columns = [field.name for field in delta_table.schema().fields if isinstance(field.type, MapType)]
        if map_columns:
            from metaxy.versioning._arrow_map import convert_maps_to_polars_map

            return convert_maps_to_polars_map(lf, columns=map_columns)
        return lf
