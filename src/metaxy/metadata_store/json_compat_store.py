"""JSON-compatible metadata store for databases without native STRUCT support.

This module provides a base class for stores that serialize struct columns to JSON.
Works with any SQL database via Ibis (PostgreSQL, SQLite, MySQL, etc.).

Uses Narwhals wrapper around Polars operations for struct handling. While Narwhals
provides a unified API, struct operations like json_encode/json_decode are Polars-specific.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.typing import Frame

from metaxy.metadata_store.ibis import IbisMetadataStore
from metaxy.models.types import CoercibleToFeatureKey, FeatureKey

if TYPE_CHECKING:
    pass


class JsonCompatStore(IbisMetadataStore, ABC):
    """Ibis metadata store for databases without native STRUCT support.

    Uses JSON/TEXT columns for struct storage with Polars-based versioning.
    Automatically serializes structs → JSON on write, deserializes JSON → structs on read.

    This base class is designed for SQL databases that lack native STRUCT types:
    - PostgreSQL (uses JSONB)
    - SQLite (uses TEXT)
    - MySQL (uses JSON)

    All data transformations use pure Narwhals expressions for backend agnosticism.

    Subclasses only need to:
    - Implement database-specific connection setup
    - Override hash functions if needed
    - Handle database-specific features (extensions, schemas, etc.)
    """

    def __init__(self, **kwargs: Any):
        """Initialize JSON-compatible store.

        Forces Polars versioning engine for struct field access support.
        """
        # Force Polars engine (needed for struct operations)
        kwargs.setdefault("versioning_engine", "polars")
        super().__init__(**kwargs)

        # Override versioning_engine_cls to use Polars (IbisMetadataStore sets it to Ibis)
        # This is required because we need struct field access support
        from metaxy.versioning.polars import PolarsVersioningEngine

        self.versioning_engine_cls = PolarsVersioningEngine

    @staticmethod
    def _serialize_struct_column(df: Frame, column_name: str) -> Frame:
        """Convert a struct column to JSON string using Polars operations.

        Args:
            df: Narwhals DataFrame containing the struct column
            column_name: Name of the struct column to serialize

        Returns:
            Narwhals DataFrame with the struct column replaced by JSON strings
        """
        import polars as pl

        if column_name not in df.columns:
            return df

        # Get schema via Narwhals
        schema = df.collect_schema()
        dtype = schema[column_name]

        # Check if it's a Struct type using Narwhals dtype
        if hasattr(dtype, "__class__") and "Struct" in dtype.__class__.__name__:
            # Convert to native Polars for struct operations (json_encode is Polars-specific)
            native_df = df.to_native()

            # Handle both Polars DataFrame and LazyFrame
            if isinstance(native_df, pl.LazyFrame):
                native_df = native_df.collect()

            # Use Polars struct.json_encode() operation
            native_df = native_df.with_columns(
                pl.col(column_name).struct.json_encode().alias(column_name)
            )

            # Return as Narwhals Frame
            return nw.from_native(native_df, eager_only=True)

        # Not a struct column, return unchanged
        return df

    @staticmethod
    def _serialize_struct_columns(df: Frame) -> Frame:
        """Convert all struct columns to JSON strings using Narwhals.

        Serializes both metaxy_provenance_by_field and metaxy_data_version_by_field
        columns, including renamed parent columns like "metaxy_provenance_by_field__parent".

        Args:
            df: Narwhals DataFrame potentially containing struct columns

        Returns:
            Narwhals DataFrame with all struct columns replaced by JSON strings
        """
        from metaxy.models.constants import (
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE_BY_FIELD,
        )

        # Find all provenance columns (main + renamed parent columns after join)
        provenance_cols = [
            col for col in df.columns if col.startswith(METAXY_PROVENANCE_BY_FIELD)
        ]

        # Also find data_version_by_field columns
        data_version_cols = [
            col for col in df.columns if col.startswith(METAXY_DATA_VERSION_BY_FIELD)
        ]

        # Serialize all struct columns using Narwhals
        for col_name in provenance_cols + data_version_cols:
            df = JsonCompatStore._serialize_struct_column(df, col_name)

        return df

    @staticmethod
    def _deserialize_provenance_column(df: Frame) -> Frame:
        """Restore struct provenance columns from JSON text using Polars operations.

        Uses Polars `str.json_decode()` which infers schema automatically.

        Processes all columns starting with METAXY_PROVENANCE_BY_FIELD prefix,
        including renamed parent columns like "metaxy_provenance_by_field__parent",
        and METAXY_DATA_VERSION_BY_FIELD columns.

        **Type Handling:**
        - `Struct`: Already decoded, skip
        - `String/Utf8`: Decode using `str.json_decode()` (most common path)
        - `Object`: Cast to String then decode (handles JSONB returned as Python dicts)
        - Other types: Keep as-is (should not occur in normal usage)

        This ensures "structs at the edges" - reads always return proper Struct columns,
        not String or Object columns.

        Args:
            df: Narwhals DataFrame with JSON string columns

        Returns:
            Narwhals DataFrame with JSON columns decoded to Struct types
        """
        import polars as pl

        from metaxy.models.constants import (
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE_BY_FIELD,
        )

        # Find all provenance columns (main + renamed parent columns after join)
        provenance_cols = [
            col for col in df.columns if col.startswith(METAXY_PROVENANCE_BY_FIELD)
        ]

        # Also find data_version_by_field columns
        data_version_cols = [
            col for col in df.columns if col.startswith(METAXY_DATA_VERSION_BY_FIELD)
        ]

        # Combine both types of struct columns
        struct_cols = provenance_cols + data_version_cols

        if not struct_cols:
            return df

        # Convert to native Polars for struct operations (json_decode is Polars-specific)
        native_df = df.to_native()

        # Handle both Polars DataFrame and LazyFrame
        if isinstance(native_df, pl.LazyFrame):
            native_df = native_df.collect()

        # Get schema from native Polars DataFrame
        polars_schema = native_df.schema

        # Deserialize each struct column using Polars operations
        for col_name in struct_cols:
            dtype = polars_schema[col_name]

            # Skip if already a proper Struct
            if isinstance(dtype, pl.Struct):
                continue

            # For String dtype, use Polars Series json_decode (eager inference)
            if dtype == pl.String or dtype == pl.Utf8:
                try:
                    # Use Series.str.json_decode() which infers schema eagerly
                    # This is more reliable than lazy expression-based decoding
                    decoded_series = native_df.get_column(col_name).str.json_decode()
                    native_df = native_df.with_columns(decoded_series.alias(col_name))
                except Exception:
                    # If decoding fails, keep as string
                    pass
            # For Object dtype (e.g., JSONB returned as dicts), convert to struct
            elif dtype == pl.Object:
                try:
                    import json

                    # Get the column containing Python dicts
                    series = native_df.get_column(col_name)

                    # Convert Python dicts to JSON strings using map_elements
                    # This handles cases where Ibis returns JSONB as Python dicts
                    def dict_to_json_str(obj: object) -> str:
                        """Convert Python dict to JSON string."""
                        if isinstance(obj, dict):
                            return json.dumps(obj)
                        return str(obj)

                    # Convert dicts to JSON strings, then decode to structs
                    string_series = series.map_elements(
                        dict_to_json_str, return_dtype=pl.String
                    )
                    decoded_series = string_series.str.json_decode()
                    native_df = native_df.with_columns(decoded_series.alias(col_name))
                except Exception:
                    # If conversion fails, keep as object
                    # This preserves data but logs a warning
                    import warnings

                    warnings.warn(
                        f"Column '{col_name}' has Object dtype but could not be converted to Struct. "
                        f"Data will be returned as Object type instead of Struct.",
                        UserWarning,
                        stacklevel=4,
                    )

        # Return as Narwhals Frame
        return nw.from_native(native_df, eager_only=True)

    @staticmethod
    def _ensure_utc_created_at(df: Frame) -> Frame:
        """Normalize metaxy_created_at columns to UTC-aware timestamps using Narwhals.

        Args:
            df: Narwhals DataFrame

        Returns:
            Narwhals DataFrame with UTC-aware datetime columns
        """
        from metaxy.models.constants import METAXY_CREATED_AT

        # Get schema via Narwhals
        schema = df.collect_schema()

        updates: list[nw.Expr] = []
        for col_name, dtype in schema.items():
            # Check if it's a datetime column without timezone
            # For Narwhals datetime types, check time_zone attribute if present
            is_datetime_without_tz = False
            if (
                col_name.startswith(METAXY_CREATED_AT)
                and hasattr(dtype, "__class__")
                and "Datetime" in dtype.__class__.__name__
            ):
                # Check if time_zone attribute exists and is None
                time_zone = getattr(
                    dtype, "time_zone", "UTC"
                )  # Default to UTC if not present
                is_datetime_without_tz = time_zone is None

            if is_datetime_without_tz:
                # Use Narwhals dt API to replace timezone
                updates.append(
                    nw.col(col_name).dt.replace_time_zone("UTC").alias(col_name)
                )

        if not updates:
            return df

        return df.with_columns(updates)

    def read_metadata_in_store(
        self,
        feature: CoercibleToFeatureKey,
        *,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> nw.LazyFrame[Any] | None:
        """Read metadata, converting JSON strings to structs using Narwhals.

        Process:
        1. Read from parent (Ibis LazyFrame with JSON as strings)
        2. **Eagerly collect** to Polars DataFrame (required for JSON decoding)
        3. Deserialize JSON → structs using Polars operations
        4. Return as Narwhals LazyFrame (Polars-backed)

        **Why Eager Collection?**

        JSON deserialization requires eager evaluation because:
        - `Series.str.json_decode()` must inspect actual data to infer struct schema
        - Lazy operations cannot determine struct field types without executing
        - This is a necessary trade-off for "structs at the edges" guarantee

        After deserialization, we return a LazyFrame to allow downstream lazy operations
        (filters, projections, joins) before final collection.

        Args:
            feature: Feature to read metadata for
            filters: Optional filter expressions
            columns: Optional column selection
            **kwargs: Additional arguments

        Returns:
            Narwhals LazyFrame with struct columns, or None if no data
        """
        import polars as pl

        # Read from parent (returns Ibis LazyFrame with JSON as strings)
        lazy_frame = super().read_metadata_in_store(
            feature, filters=filters, columns=columns, **kwargs
        )

        if lazy_frame is None:
            return lazy_frame

        # Collect and convert to Polars immediately (Ibis → PyArrow → Polars)
        # This is needed because struct operations require Polars
        native_frame = lazy_frame.collect().to_native()

        # Convert PyArrow/Ibis to Polars DataFrame
        if isinstance(native_frame, pl.DataFrame):
            polars_df = native_frame
        else:
            # Convert from PyArrow Table to Polars
            polars_df = pl.from_arrow(native_frame)  # type: ignore[arg-type]

        assert isinstance(polars_df, pl.DataFrame), (
            "Expected Polars DataFrame after conversion"
        )

        # Wrap in Narwhals for deserialization
        narwhals_df = nw.from_native(polars_df, eager_only=True)

        # Deserialize JSON → structs using Polars operations (returns Polars-backed Frame)
        narwhals_df = self._deserialize_provenance_column(narwhals_df)
        narwhals_df = self._ensure_utc_created_at(narwhals_df)

        # Get native Polars DataFrame and return as Narwhals LazyFrame
        polars_df = narwhals_df.to_native()  # type: ignore[assignment]
        assert isinstance(polars_df, pl.DataFrame), (
            "Expected Polars DataFrame after deserialization"
        )

        # Return as Narwhals LazyFrame backed by Polars
        return nw.from_native(polars_df.lazy(), eager_only=False)

    def write_metadata_to_store(
        self,
        feature_key: FeatureKey,
        df: nw.DataFrame[Any] | nw.LazyFrame[Any],
        **kwargs: Any,
    ) -> None:
        """Write metadata, serializing structs to JSON using Narwhals.

        Process:
        1. Serialize structs → JSON using JsonCompatMixin
        2. Convert to native frame for Ibis insertion
        3. Create table if needed (respects auto_create_tables setting)
        4. Insert data

        Args:
            feature_key: Feature to write metadata for
            df: Narwhals DataFrame/LazyFrame to write
            **kwargs: Additional arguments
        """
        from metaxy.metadata_store.exceptions import TableNotFoundError
        from metaxy.models.constants import (
            METAXY_DATA_VERSION_BY_FIELD,
            METAXY_PROVENANCE_BY_FIELD,
        )

        table_name = self.get_table_name(feature_key)
        table_exists = table_name in self.conn.list_tables()

        # Serialize struct columns to JSON using pure Narwhals operations
        df = self._serialize_struct_column(df, METAXY_PROVENANCE_BY_FIELD)
        df = self._serialize_struct_column(df, METAXY_DATA_VERSION_BY_FIELD)

        # Collect to eager DataFrame if needed, then convert to native for Ibis insertion
        # (Ibis doesn't accept Narwhals frames directly)
        if isinstance(df, nw.LazyFrame):
            eager_df = df.collect()  # type: ignore[attr-defined]
        else:
            eager_df = df
        native_frame = eager_df.to_native()  # type: ignore[attr-defined]

        if not table_exists:
            if not self.auto_create_tables:
                raise TableNotFoundError(
                    f"Table '{table_name}' does not exist. "
                    "Set auto_create_tables=True or create tables manually."
                )

            # Auto-create table
            self.conn.create_table(table_name, obj=native_frame, overwrite=False)
            return

        # Insert data
        self.conn.insert(table_name, obj=native_frame)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
