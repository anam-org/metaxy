"""Hash truncation utilities for Metaxy.

This module provides utilities for globally truncating hash outputs to reduce
storage requirements and improve readability. Hash truncation is configured
through the global MetaxyConfig.
"""

from typing import Any, TypeVar, overload

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
    return longer.startswith(shorter)  # ty: ignore[possibly-missing-attribute]


@nw.narwhalify
def truncate_string_column(df: nw.DataFrame[Any], column_name: str) -> nw.DataFrame[Any]:
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
        import narwhals as nw
        import polars as pl

        # With global config set to truncation_length=12:
        df = nw.from_native(pl.DataFrame({"hash": ["a" * 64, "b" * 64]}))
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


PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


@overload
def truncate_struct_column(df: pl.DataFrame, struct_column: str) -> pl.DataFrame: ...


@overload
def truncate_struct_column(df: pl.LazyFrame, struct_column: str) -> pl.LazyFrame: ...


def truncate_struct_column(df: pl.DataFrame | pl.LazyFrame, struct_column: str) -> pl.DataFrame | pl.LazyFrame:
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
        df = pl.DataFrame({"metaxy_provenance_by_field": [{"field1": "a" * 64, "field2": "b" * 64}]})
        result = truncate_struct_column(df, "metaxy_provenance_by_field")
        # result["metaxy_provenance_by_field"] contains [{"field1": "aaaaaaaaaaaa", "field2": "bbbbbbbbbbbb"}]
        ```
    """
    length = get_hash_truncation_length()

    # No truncation if length is None
    if length is None:
        return df

    import polars as pl

    # Only handle Polars DataFrames and LazyFrames (structs are Polars-only)
    if not isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        raise TypeError(f"truncate_struct_column only supports Polars DataFrame/LazyFrame, got {type(df)}")

    # For LazyFrame, we need to collect once to get field names
    if isinstance(df, pl.LazyFrame):
        temp_df = df.limit(1).collect()
    else:
        temp_df = df

    # Get field names from the struct column
    if temp_df.height == 0:
        return df

    struct_val = temp_df[struct_column][0]
    if struct_val is None:
        return df

    field_names = list(struct_val.keys())

    # Create expressions to extract and truncate each field
    field_exprs = []
    for field_name in field_names:
        field_exprs.append(pl.col(struct_column).struct.field(field_name).str.slice(0, length).alias(field_name))

    # Extract and truncate fields as separate columns
    df_with_fields = df.with_columns(field_exprs)

    # Recreate the struct from truncated fields
    struct_expr = pl.struct([pl.col(fn) for fn in field_names])
    result = df_with_fields.with_columns(struct_expr.alias(struct_column))

    # Drop temporary columns
    result = result.drop(field_names)
    return result
