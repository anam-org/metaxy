"""LanceDB metadata store implementation."""

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


class LanceDBMetadataStore(MetadataStore):
    """
    LanceDB-backed metadata store.

    Stores each feature in its own Lance table inside a LanceDB database located at ``database_path``.
    """

    _should_warn_auto_create_tables = False

    def __init__(
        self,
        database_path: str | Path,
        *,
        fallback_stores: list[MetadataStore] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize LanceDB metadata store.

        Args:
            database_path: Directory containing LanceDB tables.
            fallback_stores: Optional read-only fallback stores.
            **kwargs: Forwarded to [metaxy.metadata_store.base.MetadataStore][].
        """
        self.database_path = Path(database_path)
        self._conn: Any | None = None
        super().__init__(fallback_stores=fallback_stores, **kwargs)

    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Default hash algorithm."""
        return HashAlgorithm.XXHASH64

    def _supports_native_components(self) -> bool:
        """LanceDB store uses Polars components."""
        return False

    def _create_native_components(
        self,
    ) -> tuple[
        UpstreamJoiner,
        ProvenanceByFieldCalculator,
        MetadataDiffResolver,
    ]:
        """Not supported - LanceDB relies on Polars fallback components."""
        raise NotImplementedError(
            "LanceDBMetadataStore does not support native field provenance calculations"
        )

    def open(self) -> None:
        """Open LanceDB connection."""
        self.database_path.mkdir(parents=True, exist_ok=True)
        import lancedb

        self._conn = lancedb.connect(str(self.database_path))

    def close(self) -> None:
        """Close LanceDB connection."""
        self._conn = None

    # Helpers -----------------------------------------------------------------

    def _table_name(self, feature_key: FeatureKey) -> str:
        return feature_key.table_name

    def _table_exists(self, table_name: str) -> bool:
        assert self._conn is not None, "Store must be open"
        return table_name in self._conn.table_names()  # type: ignore[attr-defined]

    def _get_table(self, table_name: str):
        assert self._conn is not None, "Store must be open"
        return self._conn.open_table(table_name)  # type: ignore[attr-defined]

    # Storage ------------------------------------------------------------------

    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """Append metadata to Lance table."""
        assert self._conn is not None, "Store must be open"
        table_name = self._table_name(feature_key)

        arrow_table = df.to_arrow()

        if self._table_exists(table_name):
            table = self._get_table(table_name)
            table.add(arrow_table)  # type: ignore[attr-defined]
        else:
            self._conn.create_table(table_name, data=arrow_table)  # type: ignore[attr-defined]

    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop Lance table for feature."""
        assert self._conn is not None, "Store must be open"
        table_name = self._table_name(feature_key)
        if self._table_exists(table_name):
            self._conn.drop_table(table_name)  # type: ignore[attr-defined]

    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata from Lance table."""
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        table_name = self._table_name(feature_key)
        if not self._table_exists(table_name):
            return None

        table = self._get_table(table_name)
        arrow_table = table.to_arrow()  # type: ignore[attr-defined]
        df = cast(pl.DataFrame, pl.from_arrow(arrow_table))
        nw_lazy = nw.from_native(df.lazy())

        if feature_version is not None:
            nw_lazy = nw_lazy.filter(nw.col("feature_version") == feature_version)

        if filters is not None:
            for expr in filters:
                nw_lazy = nw_lazy.filter(expr)

        if columns is not None:
            nw_lazy = nw_lazy.select(columns)

        return nw_lazy

    def _list_features_local(self) -> list[FeatureKey]:
        """List Lance tables stored locally."""
        if self._conn is None:
            return []
        names = self._conn.table_names()  # type: ignore[attr-defined]
        return sorted(FeatureKey(name.split("__")) for name in names)

    # Display ------------------------------------------------------------------

    def display(self) -> str:
        """Human-readable representation."""
        details = [f"path={self.database_path}"]
        if self._is_open:
            details.append(f"features={len(self._list_features_local())}")
        return f"LanceDBMetadataStore({', '.join(details)})"
