"""Hash truncation utilities for Metaxy.

This module provides utilities for globally truncating hash outputs to reduce
storage requirements and improve readability. Hash truncation is configured
through the global MetaxyConfig.
"""

from typing import Any

import narwhals as nw
import polars as pl

# Minimum allowed truncation length
MIN_TRUNCATION_LENGTH = 8


def truncate_hash(hash_str: str) -> str:
    """Truncate a hash string using the global truncation setting.

    Uses the global hash truncation setting from MetaxyConfig.
    If the global setting is None, returns the full hash.

    Args:
        hash_str: The hash string to truncate

    Returns:
        Truncated hash string

    Examples:
        ```py
        # With global config set to truncation_length=12:
        truncate_hash("a" * 64)
        # 'aaaaaaaaaaaa'

        # With no truncation setting:
        truncate_hash("abc123")
        # 'abc123'
        ```
    """
    # Get length from global setting
    length = get_hash_truncation_length()

    # No truncation if length is None
    if length is None:
        return hash_str

    # If hash is already shorter than truncation length, return as-is
    if len(hash_str) <= length:
        return hash_str

    # Truncate to specified length
    return hash_str[:length]


def get_hash_truncation_length() -> int:
    """Get the current global hash truncation length from MetaxyConfig.

    Returns:
        Current truncation length, or 64 if no truncation is configured

    Example:
        ```py
        # With MetaxyConfig.hash_truncation_length = 16
        get_hash_truncation_length()
        ```
        16
    """
    from metaxy.config import MetaxyConfig

    config = MetaxyConfig.get()
    return config.hash_truncation_length or 64


def ensure_hash_compatibility(hash1: str, hash2: str) -> bool:
    """Check if two hashes are compatible considering truncation.

    Two hashes are compatible if:
    - They are exactly equal, OR
    - One is a truncated version of the other

    This is useful for comparing hashes that may have been truncated
    at different lengths.

    Args:
        hash1: First hash to compare
        hash2: Second hash to compare

    Returns:
        True if hashes are compatible, False otherwise

    Examples:
        ```py
        ensure_hash_compatibility("abc123", "abc123")
        # True

        ensure_hash_compatibility("abc123456789", "abc12345")
        # True  # Second is truncation of first

        ensure_hash_compatibility("abc123", "def456")
        # False  # Different hashes
        ```
    """
    if hash1 == hash2:
        return True

    # Check if one is a prefix of the other (truncation)
    shorter, longer = sorted([hash1, hash2], key=len)
    return longer.startswith(shorter)


@nw.narwhalify
def truncate_string_column(
    df: nw.DataFrame[Any], column_name: str
) -> nw.DataFrame[Any]:
    """Truncate hash values in a DataFrame column.

    Uses the global hash truncation setting from MetaxyConfig.
    If no truncation is configured, returns the DataFrame unchanged.

    Args:
        df: DataFrame containing the hash column
        column_name: Name of the column containing hash strings

    Returns:
        DataFrame with truncated hash values in the specified column

    Example:
        ```py
        # With global config set to truncation_length=12:
        df = nw.from_native(pd.DataFrame({"hash": ["a" * 64, "b" * 64]}))
        result = truncate_string_column(df, "hash")
        # result["hash"] contains ["aaaaaaaaaaaa", "bbbbbbbbbbbb"]
        ```
    """
    length = get_hash_truncation_length()

    # No truncation if length is None
    if length is None:
        return df

    # Apply truncation to the specified column
    return df.with_columns(nw.col(column_name).str.slice(0, length).alias(column_name))


def _truncate_struct_column_polars(
    df: Any, struct_column: str, length: int
) -> Any:
    """Helper that truncates struct fields on Polars DataFrame/LazyFrame."""
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        return df

    schema = df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema
    if struct_column not in schema:
        return df

    dtype = schema[struct_column]
    if not isinstance(dtype, pl.Struct):
        return df

    field_names = [field.name for field in dtype.fields]
    if not field_names:
        return df

    truncated_struct = pl.struct(
        [
            pl.col(struct_column)
            .struct.field(field_name)
            .str.slice(0, length)
            .alias(field_name)
            for field_name in field_names
        ]
    ).alias(struct_column)

    return df.with_columns(truncated_struct)


def truncate_struct_column(df: Any, struct_column: str) -> Any:
    """Truncate hash values within a struct column.

    Uses the global hash truncation setting from MetaxyConfig.
    Truncates all string values within the struct that appear to be hashes.

    Args:
        df: DataFrame containing the struct column (Polars or Narwhals)
        struct_column: Name of the struct column containing hash values

    Returns:
        DataFrame with truncated hash values within the struct

    Example:
        ```py
        # With global config set to truncation_length=12:
        df = pl.DataFrame({
            "metaxy_provenance_by_field": [{"field1": "a" * 64, "field2": "b" * 64}]
            })
        result = truncate_struct_column(df, "metaxy_provenance_by_field")
        # result["metaxy_provenance_by_field"] contains [{"field1": "aaaaaaaaaaaa", "field2": "bbbbbbbbbbbb"}]
        ```
    """
    length = get_hash_truncation_length()

    # No truncation if length is None
    if length is None:
        return df

    if isinstance(df, nw.LazyFrame):
        native = df.to_native()
        truncated = _truncate_struct_column_polars(native, struct_column, length)
        return nw.from_native(truncated, eager_only=False)

    if isinstance(df, nw.DataFrame):
        native = df.to_native()
        truncated = _truncate_struct_column_polars(native, struct_column, length)
        return nw.from_native(truncated, eager_only=True)

    return _truncate_struct_column_polars(df, struct_column, length)
