"""Delta Lake metadata store implemented with delta-rs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import narwhals as nw
import polars as pl

from metaxy.data_versioning.calculators.base import ProvenanceByFieldCalculator
from metaxy.data_versioning.diff.base import MetadataDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.metadata_store.base import MetadataStore
from metaxy.metadata_store.exceptions import TableNotFoundError
from metaxy.metadata_store.system_tables import (
    FEATURE_VERSIONS_KEY,
    FEATURE_VERSIONS_SCHEMA,
    MIGRATION_EVENTS_KEY,
    MIGRATION_EVENTS_SCHEMA,
)
from metaxy.models.feature import BaseFeature
from metaxy.models.types import FeatureKey


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
            storage_options: Optional storage backend options passed to delta-rs.
                Example: {"AWS_REGION": "us-west-2"}
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][].
        """
        self.root_path = Path(root_path)
        self.storage_options = storage_options or {}
        super().__init__(fallback_stores=fallback_stores, **kwargs)

    # ===== MetadataStore abstract methods =====

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Use XXHASH64 by default to match other non-SQL stores."""
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """DeltaLake store relies on Polars components for provenance calculations."""
        return False

    def _create_native_components(
        self,
    ) -> tuple[
        UpstreamJoiner,
        ProvenanceByFieldCalculator,
        MetadataDiffResolver,
    ]:
        """Delta Lake store does not provide native SQL execution."""
        raise NotImplementedError(
            "DeltaMetadataStore does not support native field provenance calculations"
        )

    def open(self) -> None:
        """Ensure root directory exists and create system tables if needed."""
        self.root_path.mkdir(parents=True, exist_ok=True)

        # Auto-create system tables if enabled (warning is handled in base class)
        if self.auto_create_tables:
            self._create_system_tables()

    def close(self) -> None:
        """No persistent resources to release."""
        # delta-rs is used in one-shot write/read calls, so nothing to close.
        pass

    # ===== Internal helpers =====

    def _create_system_tables(self) -> None:
        """Create system tables if they don't exist.

        Creates empty system tables with proper schemas:
        - metaxy-system__feature_versions: Tracks feature versions and graph snapshots
        - metaxy-system__migration_events: Tracks migration execution events

        This method is idempotent - safe to call multiple times.
        """
        # Check and create feature_versions table
        feature_versions_path = self._feature_path(FEATURE_VERSIONS_KEY)
        if not self._table_exists(feature_versions_path):
            empty_df = pl.DataFrame(schema=FEATURE_VERSIONS_SCHEMA)
            self._write_metadata_impl(FEATURE_VERSIONS_KEY, empty_df)

        # Check and create migration_events table
        migration_events_path = self._feature_path(MIGRATION_EVENTS_KEY)
        if not self._table_exists(migration_events_path):
            empty_df = pl.DataFrame(schema=MIGRATION_EVENTS_SCHEMA)
            self._write_metadata_impl(MIGRATION_EVENTS_KEY, empty_df)

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

    def _feature_path(self, feature_key: FeatureKey) -> Path:
        """Get the filesystem path for a feature's Delta table."""
        table_name = feature_key.table_name
        return self.root_path / table_name

    def _table_exists(self, table_path: Path) -> bool:
        """Check if a Delta table exists at the given path."""
        return (table_path / "_delta_log").exists()

    # ===== Storage operations =====

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """Append metadata to the Delta table for a feature.

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated)

        Raises:
            TableNotFoundError: If table doesn't exist and auto_create_tables is False
        """
        import deltalake  # pyright: ignore[reportMissingImports]

        table_path = self._feature_path(feature_key)
        table_exists = self._table_exists(table_path)

        # Check if table exists
        if not table_exists and not self.auto_create_tables:
            raise TableNotFoundError(
                f"Delta table does not exist for feature {feature_key.to_string()} at {table_path}. "
                f"Enable auto_create_tables=True to automatically create tables."
            )

        # Create parent directory if needed
        table_path.parent.mkdir(parents=True, exist_ok=True)

        arrow_table = df.to_arrow()

        try:
            deltalake.write_deltalake(
                str(table_path),
                arrow_table,
                mode="append",
                schema_mode="merge",
                storage_options=self.storage_options if self.storage_options else None,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to write metadata for feature {feature_key.to_string()}: {e}"
            ) from e

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Delta table for the specified feature."""
        import shutil

        table_path = self._feature_path(feature_key)
        if table_path.exists():
            shutil.rmtree(table_path, ignore_errors=True)

    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata stored in Delta for a single feature."""
        import deltalake  # pyright: ignore[reportMissingImports]

        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        table_path = self._feature_path(feature_key)
        if not self._table_exists(table_path):
            return None

        try:
            delta_table = deltalake.DeltaTable(
                str(table_path),
                storage_options=self.storage_options if self.storage_options else None,
            )

            # Use column projection for efficiency if columns are specified
            if columns is not None:
                # Need to ensure system columns are included for filtering
                cols_to_read = set(columns)
                if feature_version is not None:
                    cols_to_read.add("metaxy_feature_version")
                arrow_table = delta_table.to_pyarrow_table(columns=list(cols_to_read))
            else:
                arrow_table = delta_table.to_pyarrow_table()

            df = cast(pl.DataFrame, pl.from_arrow(arrow_table))
            lf = df.lazy()
            nw_lazy = nw.from_native(lf)

            if feature_version is not None:
                nw_lazy = nw_lazy.filter(
                    nw.col("metaxy_feature_version") == feature_version
                )

            if filters is not None:
                for expr in filters:
                    nw_lazy = nw_lazy.filter(expr)

            if columns is not None:
                nw_lazy = nw_lazy.select(columns)

            return nw_lazy
        except Exception as e:
            raise RuntimeError(
                f"Failed to read metadata for feature {feature_key}: {e}"
            ) from e

    def _list_features_local(self) -> list[FeatureKey]:
        """List all features that have Delta tables in this store.

        Returns:
            List of FeatureKey objects (excluding system tables)
        """
        if not self.root_path.exists():
            return []

        feature_keys: list[FeatureKey] = []
        for child in self.root_path.iterdir():
            if child.is_dir() and (child / "_delta_log").exists():
                try:
                    # Parse table name back to FeatureKey
                    feature_key = self._table_name_to_feature_key(child.name)

                    # Skip system tables
                    if not self._is_system_table(feature_key):
                        feature_keys.append(feature_key)
                except Exception as e:
                    # Log warning but continue - corrupted table names shouldn't break listing
                    import warnings

                    warnings.warn(
                        f"Could not parse Delta table name '{child.name}' as FeatureKey: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
        return sorted(feature_keys)

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self.root_path}"]
        if self.storage_options:
            details.append("storage_options=***")
        if self._is_open:
            try:
                num_features = len(self._list_features_local())
                details.append(f"features={num_features}")
            except Exception:
                # If listing fails, just skip the feature count
                pass
        return f"DeltaMetadataStore({', '.join(details)})"
