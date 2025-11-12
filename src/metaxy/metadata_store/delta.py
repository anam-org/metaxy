"""Delta Lake metadata store implemented with delta-rs."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.exceptions import TableNotFoundError
from metaxy.metadata_store.system import (
    EVENTS_KEY as MIGRATION_EVENTS_KEY,
    FEATURE_VERSIONS_KEY,
    FEATURE_VERSIONS_SCHEMA,
)
from metaxy.metadata_store.system.events import EVENTS_SCHEMA as MIGRATION_EVENTS_SCHEMA
from metaxy.metadata_store.types import AccessMode
from metaxy.models.feature import BaseFeature
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.provenance import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


class DeltaMetadataStore(MetadataStore):
    """
    Delta Lake metadata store backed by [delta-rs](https://github.com/delta-io/delta-rs).

    Stores each feature's metadata in a dedicated Delta table located under ``root_path``.
    Uses Polars/Narwhals components for metadata operations and relies on delta-rs for persistence.

    Example:
        ```py
        store = DeltaMetadataStore(
            "/data/metaxy/metadata",
            storage_options={"AWS_REGION": "us-west-2"},
        )

        with store:
            with store.allow_cross_project_writes():
                store.write_metadata(MyFeature, metadata_df)
        ```
    """

    _should_warn_auto_create_tables = False
    _auto_collect_lazy_frames = False  # Handle lazy frames for streaming writes

    def __init__(
        self,
        root_path: str | Path,
        *,
        storage_options: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Delta Lake metadata store.

        Args:
            root_path: Base directory or URI where feature tables are stored.
                Supports local paths (/path/to/dir), s3:// URLs, and other object store URIs.
            storage_options: Storage backend options passed to delta-rs.
                Example: {"AWS_REGION": "us-west-2", "AWS_ACCESS_KEY_ID": "...", ...}
                For S3: AWS_* environment variables
                For Azure: AZURE_* environment variables
                For GCS: GOOGLE_* environment variables
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][].
        """
        self.storage_options = storage_options or {}
        self._display_root = str(root_path)

        root_str = str(root_path)
        self._is_remote = "://" in root_str and not root_str.startswith("file://")

        if self._is_remote:
            # Remote path (S3, Azure, GCS, etc.)
            self._local_root_path = None
            self._root_uri = root_str.rstrip("/")
        else:
            # Local path (including file:// URLs)
            if root_str.startswith("file://"):
                # Strip file:// prefix
                root_str = root_str[7:]
            local_path = Path(root_str).expanduser().resolve()
            self._local_root_path = local_path
            self._root_uri = str(local_path)

        self.root_path = self._local_root_path
        super().__init__(fallback_stores=fallback_stores, **kwargs)

    # ===== MetadataStore abstract methods =====

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH64 by default to match other non-SQL stores."""
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """DeltaLake store relies on Polars components for provenance calculations."""
        return False

    def native_implementation(self) -> nw.Implementation:
        """Get native implementation for Delta store."""
        return nw.Implementation.POLARS

    @contextmanager
    def _create_provenance_tracker(
        self, plan: FeaturePlan
    ) -> Iterator[ProvenanceTracker]:
        """Create Polars provenance tracker for Delta store.

        Args:
            plan: Feature plan for the feature we're tracking provenance for

        Yields:
            PolarsProvenanceTracker instance
        """
        from metaxy.provenance.polars import PolarsProvenanceTracker

        tracker = PolarsProvenanceTracker(plan=plan)

        try:
            yield tracker
        finally:
            # No cleanup needed for Polars tracker
            pass

    @contextmanager
    def open(self, mode: AccessMode = AccessMode.READ) -> Iterator[Self]:
        """Open the Delta Lake store.

        Args:
            mode: Access mode for this connection session.

        Yields:
            Self: The store instance with connection open
        """
        # Increment context depth to support nested contexts
        self._context_depth += 1

        try:
            # Only perform actual open on first entry
            if self._context_depth == 1:
                # Auto-create system tables if enabled (warning is handled in base class)
                if self.auto_create_tables:
                    self._create_system_tables()

                # Mark store as open and validate
                self._is_open = True
                self._validate_after_open()

            yield self
        finally:
            # Decrement context depth
            self._context_depth -= 1

            # Only perform actual close on last exit
            if self._context_depth == 0:
                # delta-rs is used in one-shot write/read calls, so nothing to close
                self._is_open = False

    # ===== Internal helpers =====

    def _create_system_tables(self) -> None:
        """Create system tables if they don't exist.

        Delta auto-creates tables on first write, so we just need to write
        empty dataframes with the proper schema.
        """
        import deltalake  # pyright: ignore[reportMissingImports]

        # Create feature_versions table if it doesn't exist
        feature_versions_uri = self._feature_uri(FEATURE_VERSIONS_KEY)
        if not deltalake.DeltaTable.is_deltatable(
            feature_versions_uri, storage_options=self.storage_options or None
        ):
            empty_df = pl.DataFrame(schema=FEATURE_VERSIONS_SCHEMA)
            deltalake.write_deltalake(
                feature_versions_uri,
                empty_df,
                mode="append",
                storage_options=self.storage_options or None,
            )

        # Create migration_events table if it doesn't exist
        migration_events_uri = self._feature_uri(MIGRATION_EVENTS_KEY)
        if not deltalake.DeltaTable.is_deltatable(
            migration_events_uri, storage_options=self.storage_options or None
        ):
            empty_df = pl.DataFrame(schema=MIGRATION_EVENTS_SCHEMA)
            deltalake.write_deltalake(
                migration_events_uri,
                empty_df,
                mode="append",
                storage_options=self.storage_options or None,
            )

    def _table_name_to_feature_key(self, table_name: str) -> FeatureKey:
        """Convert table name back to feature key.

        Args:
            table_name: Table name (directory name) to parse

        Returns:
            FeatureKey constructed from table name parts
        """
        # Table names are created by joining parts with "__"
        parts = table_name.split("__")
        return FeatureKey(parts)

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        """Return the URI/path used by deltalake for this feature."""
        table_name = feature_key.table_name
        return os.path.join(self._root_uri, table_name)

    def _feature_local_path(self, feature_key: FeatureKey) -> Path | None:
        """Return filesystem path when operating on local roots."""
        if self._local_root_path is None:
            return None
        return self._local_root_path / feature_key.table_name

    def _table_exists(self, table_uri: str) -> bool:
        """Check whether the provided URI already contains a Delta table.

        Works for both local and remote (object store) paths.
        """
        import deltalake  # pyright: ignore[reportMissingImports]

        return deltalake.DeltaTable.is_deltatable(
            table_uri,
            storage_options=self.storage_options or None,
        )

    # ===== Storage operations =====

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: Frame,
    ) -> None:
        """Append metadata to the Delta table for a feature.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated)

        Raises:
            TableNotFoundError: If table doesn't exist and auto_create_tables is False
        """
        import deltalake  # pyright: ignore[reportMissingImports]

        from metaxy._utils import collect_to_polars

        # Convert Narwhals DataFrame to Polars for delta-rs
        df_polars = collect_to_polars(df)

        table_uri = self._feature_uri(feature_key)
        table_exists = self._table_exists(table_uri)

        # Check if table exists
        if not table_exists and not self.auto_create_tables:
            raise TableNotFoundError(
                f"Delta table does not exist for feature {feature_key.to_string()} at {table_uri}. "
                f"Enable auto_create_tables=True to automatically create tables."
            )

        # Delta automatically creates parent directories, no need to do it manually

        # Use deltalake.write_deltalake for all writes
        deltalake.write_deltalake(
            table_uri,
            df_polars,
            mode="append",
            schema_mode="merge",
            storage_options=self.storage_options or None,
        )

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Delta table for the specified feature using soft delete.

        Uses Delta's delete operation which marks rows as deleted in the transaction log
        rather than physically removing files.
        """
        import deltalake  # pyright: ignore[reportMissingImports]

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
        feature: FeatureKey | type[BaseFeature],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata stored in Delta for a single feature using lazy evaluation."""
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        table_uri = self._feature_uri(feature_key)
        if not self._table_exists(table_uri):
            return None

        # Use pl.scan_delta for lazy reading
        # Build column list for projection
        cols_to_read = None
        if columns is not None:
            cols_to_read = list(columns)
            # Ensure system columns are included for filtering
            if (
                feature_version is not None
                and "metaxy_feature_version" not in cols_to_read
            ):
                cols_to_read.append("metaxy_feature_version")

        # Use scan_delta for lazy evaluation
        lf = pl.scan_delta(
            table_uri,
            storage_options=self.storage_options or None,
        )

        # Apply column selection early if specified (pushdown)
        if cols_to_read is not None:
            lf = lf.select(cols_to_read)

        # Convert to Narwhals
        nw_lazy = nw.from_native(lf)

        # Apply filters
        if feature_version is not None:
            nw_lazy = nw_lazy.filter(
                nw.col("metaxy_feature_version") == feature_version
            )

        if filters is not None:
            for expr in filters:
                nw_lazy = nw_lazy.filter(expr)

        # Apply final column selection if needed (after filtering)
        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def _list_features_local(self) -> list[FeatureKey]:
        """List all features that have Delta tables in this store.

        Returns:
            List of FeatureKey objects (excluding system tables)
        """
        if self._local_root_path is not None:
            return self._list_features_from_local_root()
        return self._list_features_from_object_store()

    def _list_features_from_local_root(self) -> list[FeatureKey]:
        if self._local_root_path is None or not self._local_root_path.exists():
            return []

        feature_keys: list[FeatureKey] = []
        for child in self._local_root_path.iterdir():
            if child.is_dir() and (child / "_delta_log").exists():
                try:
                    feature_key = self._table_name_to_feature_key(child.name)
                    if not self._is_system_table(feature_key):
                        feature_keys.append(feature_key)
                except ValueError as exc:
                    warnings.warn(
                        f"Could not parse Delta table name '{child.name}' as FeatureKey: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
        return sorted(feature_keys)

    def _list_features_from_object_store(self) -> list[FeatureKey]:
        """List features from object store (S3, Azure, GCS, etc.).

        Note: For remote stores, feature discovery relies on the system tables.
        Use `metaxy graph push` to register features in the metadata store.

        Returns:
            Empty list - remote stores require explicit feature registration
        """
        warnings.warn(
            f"Feature discovery not supported for remote Delta stores ({self._root_uri}). "
            "Features must be registered via 'metaxy graph push' or accessed directly by key. "
            "Use system tables (feature_versions) to query registered features.",
            UserWarning,
            stacklevel=2,
        )
        return []

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self._display_root}"]
        if self.storage_options:
            details.append("storage_options=***")
        return f"DeltaMetadataStore({', '.join(details)})"
