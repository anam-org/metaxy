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
from pydantic import Field
from typing_extensions import Self

from metaxy._decorators import public
from metaxy._utils import collect_to_polars
from metaxy.metadata_store.base import MetadataStore, MetadataStoreConfig
from metaxy.metadata_store.types import AccessMode
from metaxy.metadata_store.utils import is_local_path
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey
from metaxy.versioning.polars import PolarsVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class DeltaMetadataStoreConfig(MetadataStoreConfig):
    """Configuration for DeltaMetadataStore.

    Example:
        ```python
        config = DeltaMetadataStoreConfig(
            root_path="s3://my-bucket/metaxy",
            storage_options={"AWS_REGION": "us-west-2"},
            layout="nested",
        )

        store = DeltaMetadataStore.from_config(config)
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
        description="Options passed to deltalake.write_deltalake().",
    )


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
        from metaxy.metadata_store.delta import DeltaMetadataStore

        store = DeltaMetadataStore(
            root_path="s3://my-bucket/metaxy",
            storage_options={"AWS_REGION": "us-west-2"},
        )
        ```
    """

    _should_warn_auto_create_tables = False
    versioning_engine_cls = PolarsVersioningEngine

    def __init__(
        self,
        root_path: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        layout: Literal["flat", "nested"] = "nested",
        delta_write_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Delta Lake metadata store.

        Args:
            root_path: Base directory or URI where feature tables are stored.
                Supports local paths (`/path/to/dir`), `s3://` URLs, and other object store URIs.
            storage_options: Storage backend options passed to delta-rs.
                Example: `{"AWS_REGION": "us-west-2", "AWS_ACCESS_KEY_ID": "...", ...}`
                See https://delta-io.github.io/delta-rs/ for details on supported options.
            fallback_stores: Ordered list of read-only fallback stores.
            layout: Directory layout for feature tables. Options:

                - `"nested"`: Feature tables stored in nested directories `{part1}/{part2}.delta`

                - `"flat"`: Feature tables stored as `{part1}__{part2}.delta`

            delta_write_options: Additional options passed to deltalake.write_deltalake() - see https://delta-io.github.io/delta-rs/upgrade-guides/guide-1.0.0/#write_deltalake-api.
                Overrides default {"schema_mode": "merge"}. Example: {"max_workers": 4}
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][metaxy.metadata_store.base.MetadataStore].
        """
        self.storage_options = storage_options or {}
        if layout not in ("flat", "nested"):
            raise ValueError(f"Invalid layout: {layout}. Must be 'flat' or 'nested'.")
        self.layout = layout
        self.delta_write_options = delta_write_options or {}

        root_str = str(root_path)
        self._is_remote = not is_local_path(root_str)

        if self._is_remote:
            # Remote path (S3, Azure, GCS, etc.)
            self._root_uri = root_str.rstrip("/")
        else:
            # Local path (including file:// and local:// URLs)
            if root_str.startswith("file://"):
                # Strip file:// prefix
                root_str = root_str[7:]
            elif root_str.startswith("local://"):
                # Strip local:// prefix
                root_str = root_str[8:]
            local_path = Path(root_str).expanduser().resolve()
            self._root_uri = str(local_path)

        super().__init__(
            fallback_stores=fallback_stores,
            versioning_engine="polars",
            **kwargs,
        )

    # ===== MetadataStore abstract methods =====

    def _has_feature_impl(self, feature: CoercibleToFeatureKey) -> bool:
        """Check if feature exists in Delta store.

        Args:
            feature: Feature to check

        Returns:
            True if feature exists, False otherwise
        """
        return self._table_exists(feature)

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH64 by default to match other non-SQL stores."""
        return HashAlgorithm.XXHASH64

    @contextmanager
    def _create_versioning_engine(self, plan: FeaturePlan) -> Iterator[PolarsVersioningEngine]:
        """Create Polars versioning engine for Delta store."""
        with self._create_polars_versioning_engine(plan) as engine:
            yield engine

    @contextmanager
    def open(self, mode: AccessMode = "read") -> Iterator[Self]:  # noqa: ARG002
        """Open the Delta Lake store.

        Delta-rs opens connections lazily per operation, so no connection state management needed.

        Args:
            mode: Access mode for this connection session (accepted for consistency but not used).

        Yields:
            Self: The store instance with connection open
        """
        # Increment context depth to support nested contexts
        self._context_depth += 1

        try:
            # Only perform actual open on first entry
            if self._context_depth == 1:
                # Mark store as open and validate
                # Note: Delta auto-creates tables on first write, no need to pre-create them
                self._is_open = True
                self._validate_after_open()

            yield self
        finally:
            # Decrement context depth
            self._context_depth -= 1

            # Only perform actual close on last exit
            if self._context_depth == 0:
                self._is_open = False

    @cached_property
    def default_delta_write_options(self) -> dict[str, Any]:
        """Default write options for Delta Lake operations.

        Merges base defaults with user-provided delta_write_options.
        Base defaults: mode="append", schema_mode="merge", storage_options.
        """
        write_kwargs: dict[str, Any] = {
            "mode": "append",
            "schema_mode": "merge",  # Allow schema evolution
            "storage_options": self.storage_options or None,
        }
        # Override with custom options from constructor
        write_kwargs.update(self.delta_write_options)
        return write_kwargs

    # ===== Internal helpers =====

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        """Return the URI/path used by deltalake for this feature."""
        if self.layout == "nested":
            # Nested layout: store in directories like "part1/part2/part3"
            # Filter out empty parts to avoid creating absolute paths that would
            # cause os.path.join to discard the root_uri
            table_path = "/".join(part for part in feature_key.parts if part)
        else:
            # Flat layout: store in directories like "part1__part2__part3"
            # table_name already handles this correctly via __join
            table_path = feature_key.table_name
        return f"{self._root_uri}/{table_path}.delta"

    def _table_exists(self, feature: CoercibleToFeatureKey) -> bool:
        """Check whether the feature exists as a Delta table.

        Works for both local and remote (object store) paths.
        """
        # for weird reasons deltalake.DeltaTable.is_deltatable() sometimes hangs in multi-threading settings
        # but a deltalake.DeltaTable can be constructed just fine
        # so we are relying on DeltaTableNotFoundError to check for existence
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
        """Cast Enum columns to String to avoid delta-rs Utf8View incompatibility."""
        return frame.with_columns(pl.selectors.by_dtype(pl.Enum).cast(pl.Utf8))

    def _open_delta_table(self, feature: CoercibleToFeatureKey, *, without_files: bool = False) -> deltalake.DeltaTable:
        feature_key = self._resolve_feature_key(feature)
        table_uri = self._feature_uri(feature_key)
        return deltalake.DeltaTable(
            table_uri,
            storage_options=self.storage_options or None,
            without_files=without_files,
        )

    # ===== Storage operations =====

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
        **kwargs: Any,
    ) -> None:
        """Append metadata to the Delta table for a feature.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata
            **kwargs: Backend-specific parameters that are passed to `write_delta` or `sink_delta`.

        !!! tip
            If Polars 1.37 or greater is installed, lazy Polars frames are sinked via
            `LazyFrame.sink_delta`, avoiding unnecessary materialization.
        """
        table_uri = self._feature_uri(feature_key)

        # Prepare write parameters
        write_opts = self.default_delta_write_options.copy()
        mode = write_opts.pop("mode", "append")
        storage_options = write_opts.pop("storage_options", None)

        # Check if we can use sink_delta (Polars >= 1.37, native Polars LazyFrame)
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

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Delta table for the specified feature using soft delete.

        Uses Delta's delete operation which marks rows as deleted in the transaction log
        rather than physically removing files.
        """
        # Check if table exists first
        if not self._table_exists(feature_key):
            return

        # Load the Delta table
        delta_table = self._open_delta_table(feature_key, without_files=True)

        # Use Delta's delete operation - soft delete all rows
        # This marks rows as deleted in transaction log without physically removing files
        delta_table.delete()

    def _delete_metadata_impl(
        self,
        feature_key: FeatureKey,
        filters: Sequence[nw.Expr] | None = None,
        *,
        current_only: bool,
    ) -> None:
        """Hard-delete rows from a Delta table using the native DELETE operation.

        Note:
            This implementation relies on Ibis (ibis-framework) to generate SQL from Narwhals expressions.
            The `ibis` package is included in the `delta` extras: `pip install metaxy[delta]`.
        """
        if not self._table_exists(feature_key):
            return

        # Load Delta table
        delta_table = self._open_delta_table(feature_key)

        # Convert Narwhals filter expressions to SQL predicate using Ibis
        if not filters:
            delta_table.delete()
            return

        from metaxy.metadata_store.utils import narwhals_expr_to_sql_predicate

        schema = self.read_feature_schema_from_store(feature_key)
        predicate = narwhals_expr_to_sql_predicate(
            filters,
            schema,
            dialect="postgres",
        )

        # Use Delta's native DELETE operation
        delta_table.delete(predicate=predicate)

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata stored in Delta for a single feature using lazy evaluation.

        Args:
            feature: Feature to read metadata for
            filters: List of Narwhals filter expressions
            columns: Subset of columns to return
            **kwargs: Backend-specific parameters (currently unused)
        """
        self._check_open()

        if not self._table_exists(feature):
            return None

        feature_key = self._resolve_feature_key(feature)
        table_uri = self._feature_uri(feature_key)

        # Use scan_delta for lazy evaluation
        lf = pl.scan_delta(
            table_uri,
            storage_options=self.storage_options or None,
        )

        # Convert to Narwhals
        nw_lazy = nw.from_native(lf)

        # Apply filters (unpack list, skip if empty)
        if filters:
            nw_lazy = nw_lazy.filter(*filters)

        # Apply column selection
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self._root_uri}"]
        details.append(f"layout={self.layout}")
        return f"DeltaMetadataStore({', '.join(details)})"

    def _get_store_metadata_impl(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        return {"uri": self._feature_uri(self._resolve_feature_key(feature_key))}

    @classmethod
    def config_model(cls) -> type[DeltaMetadataStoreConfig]:
        return DeltaMetadataStoreConfig
