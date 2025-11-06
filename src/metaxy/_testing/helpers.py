"""Test data helpers for creating properly formatted test DataFrames."""

import hashlib

import polars as pl


def add_metaxy_provenance_column(
    df: pl.DataFrame,
    hash_algo: str = "md5",
    hash_length: int = 16,
) -> pl.DataFrame:
    """Add metaxy_provenance column computed from metaxy_provenance_by_field.

    This helper ensures test data includes the mandatory metaxy_provenance column,
    which must contain a HASH (not raw concatenation) of all field hashes.

    Args:
        df: DataFrame with metaxy_provenance_by_field column
        hash_algo: Hash algorithm to use (md5, sha256, xxhash64)
        hash_length: Length to truncate hash to

    Returns:
        DataFrame with metaxy_provenance column added

    Example:
        ```python
        df = add_metaxy_provenance_column(
            pl.DataFrame({
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "h1", "audio": "h1"},
                    {"frames": "h2", "audio": "h2"},
                    {"frames": "h3", "audio": "h3"},
                ],
            })
        )
        ```
    """
    # Extract struct fields and sort them
    provenance_by_field = df["metaxy_provenance_by_field"]

    # Get field names from first row
    if len(df) == 0:
        # Empty dataframe - add empty column
        return df.with_columns(pl.lit("").alias("metaxy_provenance"))

    first_row = provenance_by_field[0]
    if first_row is None:
        # Null values - add null column
        return df.with_columns(pl.lit(None).alias("metaxy_provenance"))

    field_names = sorted(first_row.keys())

    # Step 1: Concatenate field values with separator
    concat_expr = pl.concat_str(
        [provenance_by_field.struct.field(fn) for fn in field_names],
        separator="|",
    )

    # Step 2: Hash the concatenation (this is critical - must be a hash, not concat)
    if hash_algo == "md5":
        hash_expr = concat_expr.map_elements(
            lambda x: hashlib.md5(x.encode()).hexdigest()[:hash_length] if x else None,
            return_dtype=pl.Utf8,
        )
    elif hash_algo == "sha256":
        hash_expr = concat_expr.map_elements(
            lambda x: (
                hashlib.sha256(x.encode()).hexdigest()[:hash_length] if x else None
            ),
            return_dtype=pl.Utf8,
        )
    else:
        # For other algos, just use md5 as default in tests
        hash_expr = concat_expr.map_elements(
            lambda x: hashlib.md5(x.encode()).hexdigest()[:hash_length] if x else None,
            return_dtype=pl.Utf8,
        )

    return df.with_columns(hash_expr.alias("metaxy_provenance"))
