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
        """Ensure root directory exists."""
        self.root_path.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """No persistent resources to release."""
        # delta-rs is used in one-shot write/read calls, so nothing to close.
        return None

    # ===== Internal helpers =====

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
        """Append metadata to the Delta table for a feature."""
        from deltalake import write_deltalake

        table_path = self._feature_path(feature_key)
        table_path.parent.mkdir(parents=True, exist_ok=True)

        arrow_table = df.to_arrow()

        write_deltalake(
            str(table_path),
            arrow_table,
            mode="append",
            schema_mode="merge",
            storage_options=self.storage_options or None,
        )

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
        self._check_open()

        feature_key = self._resolve_feature_key(feature)
        table_path = self._feature_path(feature_key)
        if not self._table_exists(table_path):
            return None

        from deltalake import DeltaTable

        delta_table = DeltaTable(
            str(table_path),
            storage_options=self.storage_options or None,
        )
        arrow_table = delta_table.to_pyarrow_table()
        df = cast(pl.DataFrame, pl.from_arrow(arrow_table))
        lf = df.lazy()
        nw_lazy = nw.from_native(lf)

        if feature_version is not None:
            nw_lazy = nw_lazy.filter(nw.col("feature_version") == feature_version)

        if filters is not None:
            for expr in filters:
                nw_lazy = nw_lazy.filter(expr)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def _list_features_local(self) -> list[FeatureKey]:
        """List all features that have Delta tables in this store."""
        if not self.root_path.exists():
            return []

        feature_keys: list[FeatureKey] = []
        for child in self.root_path.iterdir():
            if child.is_dir() and (child / "_delta_log").exists():
                feature_keys.append(FeatureKey(child.name.split("__")))
        return sorted(feature_keys)

    def display(self) -> str:
        """Return human-readable representation of the store."""
        details = [f"path={self.root_path}"]
        if self.storage_options:
            details.append("storage_options=***")
        if self._is_open:
            details.append(f"features={len(self._list_features_local())}")
        return f"DeltaMetadataStore({', '.join(details)})"
