"""Minimal in-memory metadata store implementation."""

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import narwhals as nw
import polars as pl
from narwhals.typing import Frame
from typing_extensions import Self

from metaxy.metadata_store.base import MetadataStore
from metaxy.models.feature import BaseFeature
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey
from metaxy.provenance.polars import PolarsProvenanceTracker
from metaxy.provenance.tracker import ProvenanceTracker
from metaxy.provenance.types import HashAlgorithm


class SimpleInMemoryMetadataStore(MetadataStore):
    """
    Simple in-memory metadata store using Polars DataFrames.

    Features:
    - Dict storage: {table_name: pl.DataFrame}
    - Uses PolarsProvenanceTracker
    - No persistence
    """

    def __init__(
        self,
        hash_algo: HashAlgorithm = HashAlgorithm.XXHASH64,
        hash_length: int = 16,
        auto_create_tables: bool = True,
    ):
        super().__init__(hash_algo, hash_length, auto_create_tables)
        self._storage: dict[str, pl.DataFrame] = {}

    def supports_native_tracker(self) -> bool:
        return False

    def create_tracker(self, plan: FeaturePlan) -> ProvenanceTracker:
        return PolarsProvenanceTracker(plan)

    def read_metadata_impl(
        self,
        feature: type[BaseFeature] | FeatureKey,
        filters: Sequence[nw.Expr] | None = None,
    ) -> Frame | None:
        if isinstance(feature, FeatureKey):
            feature_key = feature
        else:
            feature_key = feature.spec().key

        table_name = feature_key.table_name

        if table_name not in self._storage:
            return None

        # Get Polars DataFrame and convert to Narwhals LazyFrame
        df = self._storage[table_name]
        nw_lazy = nw.from_native(df.lazy())

        # Apply filters
        if filters:
            for filter_expr in filters:
                nw_lazy = nw_lazy.filter(filter_expr)

        return nw_lazy

    def write_metadata(self, feature: type[BaseFeature] | FeatureKey, data: Frame) -> None:
        if isinstance(feature, FeatureKey):
            feature_key = feature
        else:
            feature_key = feature.spec().key

        table_name = feature_key.table_name

        # Convert to Narwhals if needed
        nw_data = nw.from_native(data) if not isinstance(data, nw.DataFrame | nw.LazyFrame) else data  # type: ignore[arg-type]

        # Convert to Polars DataFrame
        if nw_data.implementation == nw.Implementation.POLARS:
            polars_df = nw_data.to_polars()
            if hasattr(polars_df, 'collect'):
                polars_df = polars_df.collect()
        else:
            # Convert via native interface
            polars_df = pl.from_arrow(nw_data.to_arrow())  # type: ignore[arg-type]

        # Append or create
        if table_name in self._storage:
            # Ensure schema compatibility
            existing_df = self._storage[table_name]

            # Add missing columns to both DataFrames
            for col_name in polars_df.columns:
                if col_name not in existing_df.columns:
                    col_dtype = polars_df.schema[col_name]
                    existing_df = existing_df.with_columns(
                        pl.lit(None).cast(col_dtype).alias(col_name)
                    )

            for col_name in existing_df.columns:
                if col_name not in polars_df.columns:
                    col_dtype = existing_df.schema[col_name]
                    polars_df = polars_df.with_columns(
                        pl.lit(None).cast(col_dtype).alias(col_name)
                    )

            # Ensure consistent column order
            all_columns = sorted(set(existing_df.columns) | set(polars_df.columns))
            existing_df = existing_df.select(all_columns)
            polars_df = polars_df.select(all_columns)

            # Concatenate
            self._storage[table_name] = pl.concat([existing_df, polars_df])
        else:
            if not self.auto_create_tables:
                raise ValueError(
                    f"Table '{table_name}' does not exist and auto_create_tables is False"
                )
            self._storage[table_name] = polars_df

    @contextmanager
    def open(self) -> Any:
        """Open the in-memory store (no-op)."""
        yield self

    def close(self) -> None:
        """Close the store (no-op)."""
        pass
