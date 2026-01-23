from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import narwhals as nw
import pyarrow as pa
from ray.data import Datasource
from ray.data.block import BlockMetadata
from ray.data.datasource import ReadTask

import metaxy as mx

if TYPE_CHECKING:
    pass


class MetaxyDatasource(Datasource):
    """A Ray Data Datasource for reading from a Metaxy metadata store.

    This datasource reads metadata entries from a Metaxy metadata store as Ray Data blocks,
    associated with a specific feature key.

    !!! example

        ```python
        import metaxy as mx
        import ray

        cfg = mx.init_metaxy()

        ds = ray.data.read_datasource(
            MetaxyDatasource(
                feature="my/feature",
                store=cfg.get_store(),
                config=cfg,
            )
        )
        ```

    !!! example "with filters and column selection"

        ```python
        import narwhals as nw

        ds = ray.data.read_datasource(
            MetaxyDatasource(
                feature="my/feature",
                store=cfg.get_store(),
                config=cfg,
                filters=[nw.col("value") > 10],
                columns=["sample_uid", "value"],
            )
        )
        ```

    Args:
        feature: Feature to read metadata for.
        store: Metadata store to read from.
        config: Metaxy configuration. Will be auto-discovered by the worker if not provided.

            !!! warning
                Ensure the Ray environment is set up properly when not passing `config` explicitly.
                This can be achieved by setting `METAXY_CONFIG` and other `METAXY_` environment variables.
                The best practice is to pass `config` explicitly to avoid any surprises.

        filters: Sequence of Narwhals filter expressions to apply.
        columns: Subset of columns to include. Metaxy's system columns are always included.
        allow_fallback: If `True`, check fallback stores on main store miss.
        current_only: If `True`, only return rows with current feature version.
        feature_version: Explicit feature version to filter by (mutually exclusive with `current_only=True`).
        latest_only: Whether to deduplicate samples within `id_columns` groups ordered by `metaxy_created_at`.
        include_soft_deleted: If `True`, include soft-deleted rows in the result.
    """

    def __init__(
        self,
        feature: mx.CoercibleToFeatureKey,
        store: mx.MetadataStore,
        config: mx.MetaxyConfig | None = None,
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        allow_fallback: bool = True,
        current_only: bool = True,
        latest_only: bool = True,
        include_soft_deleted: bool = False,
    ):
        self.config = mx.init_metaxy(config)

        self.store = store
        self.feature_version = feature_version
        self.filters = list(filters) if filters else None
        self.columns = list(columns) if columns else None
        self.allow_fallback = allow_fallback
        self.current_only = current_only
        self.latest_only = latest_only
        self.include_soft_deleted = include_soft_deleted

        self._feature_key = mx.coerce_to_feature_key(feature)

    def _read_metadata_lazy(self) -> nw.LazyFrame:
        """Create a lazy frame for reading metadata with all configured options."""
        return self.store.read_metadata(
            self._feature_key,
            feature_version=self.feature_version,
            filters=self.filters,
            columns=self.columns,
            allow_fallback=self.allow_fallback,
            current_only=self.current_only,
            latest_only=self.latest_only,
            include_soft_deleted=self.include_soft_deleted,
        )

    def _get_row_count(self) -> int:
        """Get the row count by executing a lightweight count query."""
        with self.store.open("read"):
            lf = self._read_metadata_lazy()
            return lf.select(nw.len()).collect().item()

    def get_read_tasks(self, parallelism: int, per_task_row_limit: int | None = None) -> list[ReadTask]:
        """Return read tasks for the feature metadata.

        Args:
            parallelism: Requested parallelism level (currently ignored, returns single task).
            per_task_row_limit: Maximum rows per returned block. If set, the data will be
                split into multiple blocks of at most this size.

        Returns:
            List containing a single ReadTask that may return multiple blocks.
        """
        # TODO: implement parallelism support via collect_batches to split data into multiple tasks

        num_rows = self._get_row_count()

        # Capture state for the worker closure. These locals avoid serializing the entire
        # datasource instance when the closure is pickled for Ray workers.
        store, config, feature_key = self.store, self.config, self._feature_key
        feature_version, filters, columns = self.feature_version, self.filters, self.columns
        allow_fallback, current_only = self.allow_fallback, self.current_only
        latest_only, include_soft_deleted = self.latest_only, self.include_soft_deleted
        row_limit = per_task_row_limit

        def read_fn() -> list[pa.Table]:
            mx.init_metaxy(config)
            with store.open("read"):
                lf = store.read_metadata(
                    feature_key,
                    feature_version=feature_version,
                    filters=filters,
                    columns=columns,
                    allow_fallback=allow_fallback,
                    current_only=current_only,
                    latest_only=latest_only,
                    include_soft_deleted=include_soft_deleted,
                )
                table = lf.collect(backend="pyarrow").to_arrow()
                batches = table.to_batches(max_chunksize=row_limit)
                return [pa.Table.from_batches([b]) for b in batches]

        metadata = BlockMetadata(
            num_rows=num_rows,
            size_bytes=None,
            input_files=None,
            exec_stats=None,
        )

        return [ReadTask(read_fn, metadata)]

    def estimate_inmemory_data_size(self) -> int | None:
        """Return an estimate of in-memory data size, or None if unknown."""
        return None
