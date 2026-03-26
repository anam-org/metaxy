"""Delta Lake metadata store implemented with delta-rs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Literal, overload

import deltalake
import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from packaging.version import Version

from metaxy._decorators import public
from metaxy._utils import collect_to_polars
from metaxy.ext.delta.config import DeltaMetadataStoreConfig
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


@public
class DeltaMetadataStore(MetadataStore):
    """
    Delta Lake metadata store backed by [delta-rs](https://github.com/delta-io/delta-rs).

    It stores feature metadata in Delta Lake tables located under ``root_path``.
    It uses the Polars versioning engine for provenance calculations.

    !!! tip
        If Polars 1.37 or greater is installed, lazy Polars frames are sinked via
        `LazyFrame.sink_delta`, avoiding unnecessary materialization.

    Example:

        ```py
        from metaxy.ext.metadata_stores.delta import DeltaMetadataStore

        store = DeltaMetadataStore(
            root_path="s3://my-bucket/metaxy",
            storage_options={"AWS_REGION": "us-west-2"},
        )
        ```
    """

    _should_warn_auto_create_tables = False
    versioning_engine_cls = PolarsVersioningEngine

    storage_options: dict[str, Any]
    layout: str
    delta_write_options: dict[str, Any]
    _root_uri: str
    _is_remote: bool

    def __init__(
        self,
        root_path: str | Path | None = None,
        *,
        storage_options: dict[str, Any] | None = None,  # noqa: ARG002
        fallback_stores: list[MetadataStore] | None = None,  # noqa: ARG002
        layout: Literal["flat", "nested"] = "nested",  # noqa: ARG002
        delta_write_options: dict[str, Any] | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        pass  # __new__ already initialized via MetadataStore.__init__

    def __new__(
        cls,
        root_path: str | Path | None = None,
        *,
        storage_options: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        layout: Literal["flat", "nested"] = "nested",
        delta_write_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> DeltaMetadataStore:
        if root_path is None:
            raise ValueError("root_path is required")

        from metaxy.metadata_store.compute_engine import PolarsComputeEngine

        if layout not in ("flat", "nested"):
            raise ValueError(f"Invalid layout: {layout}. Must be 'flat' or 'nested'.")

        instance = MetadataStore.__new__(cls)
        instance.storage_options = storage_options or {}
        instance.layout = layout
        instance.delta_write_options = delta_write_options or {}

        root_str = str(root_path)
        instance._is_remote = not is_local_path(root_str)

        if instance._is_remote:
            instance._root_uri = root_str.rstrip("/")
        else:
            if root_str.startswith("file://"):
                root_str = root_str[7:]
            elif root_str.startswith("local://"):
                root_str = root_str[8:]
            instance._root_uri = str(Path(root_str).expanduser().resolve())

        MetadataStore.__init__(
            instance,
            engine=PolarsComputeEngine(),
            fallback_stores=fallback_stores,
            versioning_engine="polars",
            **kwargs,
        )
        return instance

    # ===== MetadataStore abstract methods =====

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        return self._table_exists(feature)

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        return HashAlgorithm.XXHASH32

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        with self._create_polars_versioning_engine(plan) as engine:
            yield engine

    def _open(self, mode: AccessMode) -> None:  # noqa: ARG002
        pass

    def _close(self) -> None:
        pass

    @cached_property
    def default_delta_write_options(self) -> dict[str, Any]:
        """Default write options for Delta Lake operations.

        Merges base defaults with user-provided delta_write_options.
        """
        write_kwargs: dict[str, Any] = {
            "mode": "append",
            "schema_mode": "merge",
            "storage_options": self.storage_options or None,
        }
        write_kwargs.update(self.delta_write_options)
        return write_kwargs

    # ===== Internal helpers =====

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        if self.layout == "nested":
            table_path = "/".join(part for part in feature_key.parts if part)
        else:
            table_path = feature_key.table_name
        return f"{self._root_uri}/{table_path}.delta"

    def _table_exists(self, feature: CoercibleToFeatureKey) -> bool:
        # DeltaTable.is_deltatable() can hang in multi-threaded settings.
        from deltalake.exceptions import TableNotFoundError as DeltaTableNotFoundError

        try:
            _ = self._open_delta_table(feature, without_files=True)
        except DeltaTableNotFoundError:
            return False
        return True

    @overload
    def _cast_enum_to_string(self, frame: pl.DataFrame) -> pl.DataFrame: ...

    @overload
    def _cast_enum_to_string(self, frame: pl.LazyFrame) -> pl.LazyFrame: ...

    def _cast_enum_to_string(self, frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        return frame.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

    def _open_delta_table(self, feature: CoercibleToFeatureKey, *, without_files: bool = False) -> deltalake.DeltaTable:
        feature_key = self._resolve_feature_key(feature)
        return deltalake.DeltaTable(
            self._feature_uri(feature_key),
            storage_options=self.storage_options or None,
            without_files=without_files,
        )

    # ===== Storage operations =====

    def _write_feature(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        table_uri = self._feature_uri(feature_key)
        write_opts = self.default_delta_write_options.copy()
        mode = write_opts.pop("mode", "append")
        storage_options = write_opts.pop("storage_options", None)

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
            df_native = collect_to_polars(df)

            self._cast_enum_to_string(df_native).write_delta(
                table_uri,
                mode=mode,
                storage_options=storage_options,
                delta_write_options=write_opts or None,
            )

    def _drop_feature(self, feature_key: FeatureKey) -> None:
        if not self._table_exists(feature_key):
            return
        self._open_delta_table(feature_key, without_files=True).delete()

    def _delete_feature(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None = None,
        *,
        with_feature_history: bool,
    ) -> None:
        if not self._table_exists(feature_key):
            return

        delta_table = self._open_delta_table(feature_key)

        if not filters:
            delta_table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        schema = self.read_feature_schema_from_store(feature_key)
        delta_table.delete(
            predicate=narwhals_expr_to_sql_predicate(
                filters,
                schema,
                dialect="datafusion",
            )
        )

    def _read_feature(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        self._check_open()

        if not self._table_exists(feature):
            return None

        feature_key = self._resolve_feature_key(feature)
        nw_lazy = nw.from_native(
            pl.scan_delta(
                self._feature_uri(feature_key),
                storage_options=self.storage_options or None,
            )
        )

        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def display(self) -> str:
        return f"DeltaMetadataStore(path={self._root_uri})"

    def _get_store_metadata_impl(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        return {"uri": self._feature_uri(self._resolve_feature_key(feature_key))}

    @classmethod
    def config_model(cls) -> type[DeltaMetadataStoreConfig]:
        return DeltaMetadataStoreConfig
