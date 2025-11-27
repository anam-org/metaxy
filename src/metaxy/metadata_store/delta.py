"""Delta Lake metadata store implemented with delta-rs."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import deltalake
import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from pydantic import Field
from typing_extensions import Self

from metaxy._utils import switch_implementation_to_polars
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


class DeltaMetadataStore(MetadataStore):
    """
    Delta Lake metadata store backed by [delta-rs](https://github.com/delta-io/delta-rs).

    It stores feature metadata in Delta Lake tables located under ``root_path``.
    It uses the Polars versioning engine for provenance calculations.

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
            versioning_engine_cls=PolarsVersioningEngine,
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
        feature_key = self._resolve_feature_key(feature)
        return self._table_exists(self._feature_uri(feature_key))

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH64 by default to match other non-SQL stores."""
        return HashAlgorithm.XXHASH64

    @contextmanager
    def _create_versioning_engine(
        self, plan: FeaturePlan
    ) -> Iterator[PolarsVersioningEngine]:
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

    def _table_exists(self, table_uri: str) -> bool:
        """Check whether the provided URI already contains a Delta table.

        Works for both local and remote (object store) paths.
        """
        # for weird reasons deltalake.DeltaTable.is_deltatable() sometimes hangs in multi-threading settings
        # but a deltalake.DeltaTable can be constructed just fine
        # so we are relying on DeltaTableNotFoundError to check for existence
        from deltalake.exceptions import TableNotFoundError as DeltaTableNotFoundError

        try:
            _ = deltalake.DeltaTable(
                table_uri, storage_options=self.storage_options, without_files=True
            )
        except DeltaTableNotFoundError:
            return False
        return True

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
            df: DataFrame with metadata (already validated)
            **kwargs: Backend-specific parameters (currently unused)
        """
        table_uri = self._feature_uri(feature_key)

        # Delta Lake auto-creates tables on first write, no need to check existence
        # Convert to Polars and collect lazy frames
        df_polars = switch_implementation_to_polars(df)

        # Collect lazy frames, keep eager frames as-is
        if isinstance(df_polars, nw.LazyFrame):
            df_native = df_polars.collect().to_native()
        else:
            df_native = df_polars.to_native()

        assert isinstance(df_native, pl.DataFrame)

        # Prepare write parameters for Polars write_delta
        # Extract mode and storage_options as top-level parameters
        write_opts = self.default_delta_write_options.copy()
        mode = write_opts.pop("mode", "append")
        storage_options = write_opts.pop("storage_options", None)

        # Write using Polars DataFrame.write_delta
        df_native.write_delta(
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
        table_uri = self._feature_uri(feature_key)

        # Check if table exists first
        if not self._table_exists(table_uri):
            return

        # Load the Delta table
        delta_table = deltalake.DeltaTable(
            table_uri,
            storage_options=self.storage_options or None,
            without_files=True,  # Don't track files for this operation
        )

        # Use Delta's delete operation - soft delete all rows
        # This marks rows as deleted in transaction log without physically removing files
        delta_table.delete()

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

        feature_key = self._resolve_feature_key(feature)
        table_uri = self._feature_uri(feature_key)
        if not self._table_exists(table_uri):
            return None

        # Use scan_delta for lazy evaluation
        lf = pl.scan_delta(
            table_uri,
            storage_options=self.storage_options or None,
        )

        # Convert to Narwhals
        nw_lazy = nw.from_native(lf)

        # Apply filters
        if filters is not None:
            nw_lazy = nw_lazy.filter(filters)

        # Apply column selection
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    # ========== Error Tracking Implementation ==========

    def _error_table_uri(self, feature_key: FeatureKey) -> str:
        """Return the URI/path for error table for this feature.

        Args:
            feature_key: Feature key to get error table URI for

        Returns:
            URI string for the error table
        """
        # Get base error table name from base class helper
        error_table_name = self._get_error_table_name(feature_key)

        # Apply same layout logic as regular tables
        if self.layout == "nested":
            # For nested layout, use the feature's path structure
            # but replace the last part with error table name
            parts = list(feature_key.parts[:-1]) if len(feature_key.parts) > 1 else []
            parts.append(error_table_name)
            table_path = "/".join(part for part in parts if part)
        else:
            # For flat layout, error table name already has proper format
            table_path = error_table_name

        return f"{self._root_uri}/{table_path}.delta"

    def write_errors_to_store(
        self,
        feature_key: FeatureKey,
        errors_df: Frame,
    ) -> None:
        """Write error records to Delta error table.

        Args:
            feature_key: Feature key to write errors for
            errors_df: Narwhals DataFrame with error records
        """
        error_table_uri = self._error_table_uri(feature_key)

        # Convert to Polars
        df_polars = switch_implementation_to_polars(errors_df)

        # Collect if lazy
        if isinstance(df_polars, nw.LazyFrame):
            df_native = df_polars.collect().to_native()
        else:
            df_native = df_polars.to_native()

        assert isinstance(df_native, pl.DataFrame)

        # Prepare write parameters
        write_opts = self.default_delta_write_options.copy()
        mode = write_opts.pop("mode", "append")
        storage_options = write_opts.pop("storage_options", None)

        # Write using Polars (Delta auto-creates table on first write)
        df_native.write_delta(
            error_table_uri,
            mode=mode,
            storage_options=storage_options,
            delta_write_options=write_opts or None,
        )

    def read_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read error records from Delta error table.

        Args:
            feature_key: Feature key to read errors for
            filters: Optional Narwhals filter expressions to apply

        Returns:
            Narwhals LazyFrame with error records, or None if error table doesn't exist
        """
        error_table_uri = self._error_table_uri(feature_key)

        # Check if error table exists
        if not self._table_exists(error_table_uri):
            return None

        # Use scan_delta for lazy evaluation
        lf = pl.scan_delta(
            error_table_uri,
            storage_options=self.storage_options or None,
        )

        # Convert to Narwhals
        nw_lazy = nw.from_native(lf)

        # Apply filters
        if filters is not None:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        return nw_lazy

    def clear_errors_from_store(
        self,
        feature_key: FeatureKey,
        *,
        sample_uids: Sequence[dict[str, Any]] | None = None,
        feature_version: str | None = None,
    ) -> None:
        """Clear error records from Delta error table.

        Args:
            feature_key: Feature key to clear errors for
            sample_uids: Optional list of sample ID dicts to clear
            feature_version: Optional feature version to clear
        """
        error_table_uri = self._error_table_uri(feature_key)

        # Check if error table exists
        if not self._table_exists(error_table_uri):
            return  # No-op if table doesn't exist

        # Load Delta table
        delta_table = deltalake.DeltaTable(
            error_table_uri,
            storage_options=self.storage_options or None,
            without_files=True,
        )

        # If no filters, delete all errors (soft delete)
        if sample_uids is None and feature_version is None:
            delta_table.delete()
            return

        # Build predicate for selective deletion
        predicates = []

        # Filter by feature_version if provided
        if feature_version is not None:
            from metaxy.models.constants import METAXY_FEATURE_VERSION

            predicates.append(f"{METAXY_FEATURE_VERSION} = '{feature_version}'")

        # Filter by sample_uids if provided
        if sample_uids is not None and len(sample_uids) > 0:
            # Get id_columns from feature spec
            feature_spec = self._resolve_feature_plan(feature_key).feature
            id_cols = list(feature_spec.id_columns)

            # Build SQL-like predicate for sample_uids
            # For each sample, create an AND condition, then OR them together
            sample_predicates = []
            for uid_dict in sample_uids:
                and_conditions = []
                for col_name in id_cols:
                    col_value = uid_dict[col_name]
                    # Handle different types appropriately for SQL predicate
                    if isinstance(col_value, str):
                        and_conditions.append(f"{col_name} = '{col_value}'")
                    elif isinstance(col_value, (int, float)):
                        and_conditions.append(f"{col_name} = {col_value}")
                    else:
                        # For other types, convert to string
                        and_conditions.append(f"{col_name} = '{col_value}'")

                # Combine with AND
                if len(and_conditions) == 1:
                    sample_pred = and_conditions[0]
                else:
                    sample_pred = " AND ".join(f"({cond})" for cond in and_conditions)

                sample_predicates.append(f"({sample_pred})")

            # Combine all sample predicates with OR
            if len(sample_predicates) == 1:
                samples_pred = sample_predicates[0]
            else:
                samples_pred = " OR ".join(sample_predicates)

            predicates.append(f"({samples_pred})")

        # Execute delete with predicate
        if predicates:
            # Combine all predicates with AND
            if len(predicates) == 1:
                final_predicate = predicates[0]
            else:
                final_predicate = " AND ".join(f"({pred})" for pred in predicates)

            # Delete matching rows
            delta_table.delete(predicate=final_predicate)

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self._root_uri}"]
        details.append(f"layout={self.layout}")
        return f"DeltaMetadataStore({', '.join(details)})"

    def get_store_metadata(self, feature_key: CoercibleToFeatureKey) -> dict[str, Any]:
        return {"path": self._feature_uri(self._resolve_feature_key(feature_key))}

    @classmethod
    def config_model(cls) -> type[DeltaMetadataStoreConfig]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return DeltaMetadataStoreConfig
