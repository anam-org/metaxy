"""Delta Lake metadata store implemented with delta-rs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from urllib.request import url2pathname
import warnings

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

if TYPE_CHECKING:
    from obstore.store import ObjectStore


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
        object_store_kwargs: dict[str, Any] | None = None,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Delta Lake metadata store.

        Args:
            root_path: Base directory or URI where feature tables are stored.
            storage_options: Optional storage backend options passed to delta-rs.
                Example: {"AWS_REGION": "us-west-2"}
            object_store_kwargs: Keyword arguments forwarded to ``obstore.store.from_url``
                when ``root_path`` is remote. Useful for passing ``config`` or
                ``client_options`` dictionaries directly to obstore.
            fallback_stores: Ordered list of read-only fallback stores.
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][].
        """
        self.storage_options = storage_options or {}
        self._object_store_kwargs = dict(object_store_kwargs or {})
        self._display_root = str(root_path)
        self._local_root_path: Path | None
        self._root_uri: str
        self._is_remote = False
        self._object_store: ObjectStore | None = None

        if isinstance(root_path, Path):
            local_path = root_path.expanduser()
            self._local_root_path = local_path
            self._root_uri = str(local_path)
        else:
            root_str = str(root_path)
            if "://" not in root_str:
                local_path = Path(root_str).expanduser()
                self._local_root_path = local_path
                self._root_uri = str(local_path)
            else:
                parsed = urlparse(root_str)
                if parsed.scheme.lower() == "file":
                    local_path_str = url2pathname(parsed.path)
                    if parsed.netloc and parsed.netloc not in {"", "localhost"}:
                        local_path_str = f"//{parsed.netloc}{local_path_str}"
                    local_path = Path(local_path_str).expanduser()
                    self._local_root_path = local_path
                    self._root_uri = str(local_path)
                else:
                    self._local_root_path = None
                    sanitized = root_str.rstrip("/")
                    self._root_uri = sanitized
                    self._is_remote = True

        self.root_path = self._local_root_path
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
        if self._local_root_path is not None:
            self._local_root_path.mkdir(parents=True, exist_ok=True)

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
        feature_versions_uri = self._feature_uri(FEATURE_VERSIONS_KEY)
        if not self._table_exists(feature_versions_uri):
            empty_df = pl.DataFrame(schema=FEATURE_VERSIONS_SCHEMA)
            self._write_metadata_impl(FEATURE_VERSIONS_KEY, empty_df)

        # Check and create migration_events table
        migration_events_uri = self._feature_uri(MIGRATION_EVENTS_KEY)
        if not self._table_exists(migration_events_uri):
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

    def _feature_relative_path(self, feature_key: FeatureKey) -> str:
        """Relative directory used for the feature table."""
        return feature_key.table_name

    def _feature_uri(self, feature_key: FeatureKey) -> str:
        """Return the URI/path used by deltalake for this feature."""
        relative = self._feature_relative_path(feature_key)
        if self._local_root_path is not None:
            return str(self._local_root_path / relative)
        return self._join_remote_uri(relative)

    def _feature_local_path(self, feature_key: FeatureKey) -> Path | None:
        """Return filesystem path when operating on local roots."""
        if self._local_root_path is None:
            return None
        return self._local_root_path / self._feature_relative_path(feature_key)

    def _join_remote_uri(self, relative: str) -> str:
        base = self._root_uri.rstrip("/")
        if not relative:
            return base
        return f"{base}/{relative}"

    def _storage_options_payload(self) -> dict[str, Any] | None:
        return self.storage_options or None

    def _is_delta_table(self, table_uri: str) -> bool:
        import deltalake  # pyright: ignore[reportMissingImports]

        return deltalake.DeltaTable.is_deltatable(
            table_uri,
            storage_options=self._storage_options_payload(),
        )

    def _table_exists(self, table_uri: str) -> bool:
        """Check whether the provided URI already contains a Delta table."""
        return self._is_delta_table(table_uri)

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

        table_uri = self._feature_uri(feature_key)
        table_exists = self._table_exists(table_uri)

        # Check if table exists
        if not table_exists and not self.auto_create_tables:
            raise TableNotFoundError(
                f"Delta table does not exist for feature {feature_key.to_string()} at {table_uri}. "
                f"Enable auto_create_tables=True to automatically create tables."
            )

        # Create parent directory if needed
        local_table_path = self._feature_local_path(feature_key)
        if local_table_path is not None:
            local_table_path.mkdir(parents=True, exist_ok=True)

        arrow_table = df.to_arrow()

        try:
            deltalake.write_deltalake(
                table_uri,
                arrow_table,
                mode="append",
                schema_mode="merge",
                storage_options=self._storage_options_payload(),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to write metadata for feature {feature_key.to_string()}: {e}"
            ) from e

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Delta table for the specified feature."""
        import shutil

        local_table_path = self._feature_local_path(feature_key)
        if local_table_path is not None:
            if local_table_path.exists():
                shutil.rmtree(local_table_path, ignore_errors=True)
            return

        self._delete_remote_prefix(feature_key)

    def _delete_remote_prefix(self, feature_key: FeatureKey) -> None:
        """Delete all objects that belong to a feature when stored remotely."""
        store = self._get_object_store()
        prefix = self._feature_relative_path(feature_key).rstrip("/")
        if prefix:
            prefix = f"{prefix}/"
        stream = store.list(prefix=prefix)
        for chunk in stream:
            paths = [obj["path"] for obj in chunk]
            if paths:
                store.delete(paths)

    def _get_object_store(self) -> ObjectStore:
        """Instantiate (lazily) an obstore client for remote roots."""
        if not self._is_remote:
            msg = "Object store access requested for a non-remote DeltaMetadataStore"
            raise RuntimeError(msg)
        if self._object_store is None:
            try:
                from obstore import store as obstore_store
            except ImportError as exc:  # pragma: no cover - handled at runtime
                raise RuntimeError(
                    "obstore is required for remote DeltaMetadataStore paths. "
                    "Install metaxy[delta] to include the dependency."
                ) from exc

            self._object_store = obstore_store.from_url(
                self._root_uri,
                **self._object_store_kwargs,
            )

        return self._object_store

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
        table_uri = self._feature_uri(feature_key)
        if not self._table_exists(table_uri):
            return None

        try:
            delta_table = deltalake.DeltaTable(
                table_uri,
                storage_options=self._storage_options_payload(),
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
                except Exception as e:
                    warnings.warn(
                        f"Could not parse Delta table name '{child.name}' as FeatureKey: {e}",
                        UserWarning,
                        stacklevel=2,
                    )
        return sorted(feature_keys)

    def _list_features_from_object_store(self) -> list[FeatureKey]:
        store = self._get_object_store()
        try:
            result = store.list_with_delimiter()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to list features under {self._root_uri}: {exc}"
            ) from exc

        prefixes = getattr(result, "common_prefixes", None) or []
        feature_keys: list[FeatureKey] = []
        for prefix in prefixes:
            table_name = prefix.rstrip("/")
            if not table_name:
                continue
            try:
                feature_key = self._table_name_to_feature_key(table_name)
            except Exception as exc:
                warnings.warn(
                    f"Could not parse Delta table name '{table_name}' as FeatureKey: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue

            if self._is_system_table(feature_key):
                continue

            table_uri = self._join_remote_uri(table_name)
            if self._table_exists(table_uri):
                feature_keys.append(feature_key)

        return sorted(feature_keys)

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self._display_root}"]
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
