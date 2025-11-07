"""Polars implementation of field provenance calculator."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar, cast, overload

import narwhals as nw
import polars as pl
import polars_hash as plh

from metaxy.data_versioning.calculators.base import (
    PROVENANCE_BY_FIELD_COL,
    ProvenanceByFieldCalculator,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.utils.hashing import truncate_struct_column

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


# Map HashAlgorithm enum to polars-hash functions
_HASH_FUNCTION_MAP: dict[HashAlgorithm, Callable[[pl.Expr], pl.Expr]] = {
    HashAlgorithm.XXHASH64: lambda expr: expr.nchash.xxhash64(),  # pyright: ignore[reportAttributeAccessIssue]
    HashAlgorithm.XXHASH32: lambda expr: expr.nchash.xxhash32(),  # pyright: ignore[reportAttributeAccessIssue]
    HashAlgorithm.WYHASH: lambda expr: expr.nchash.wyhash(),  # pyright: ignore[reportAttributeAccessIssue]
    HashAlgorithm.SHA256: lambda expr: expr.chash.sha2_256(),  # pyright: ignore[reportAttributeAccessIssue]
    HashAlgorithm.MD5: lambda expr: expr.nchash.md5(),  # pyright: ignore[reportAttributeAccessIssue]
}


PolarsFrameT = TypeVar("PolarsFrameT", pl.DataFrame, pl.LazyFrame)


@overload
def calculate_provenance_by_field_polars(
    joined_upstream_df: pl.DataFrame,
    feature_spec: "FeatureSpec",
    feature_plan: "FeaturePlan",
    upstream_column_mapping: dict[str, str],
    hash_algorithm: HashAlgorithm,
    hash_truncation_length: int | None = None,
) -> pl.DataFrame: ...


@overload
def calculate_provenance_by_field_polars(
    joined_upstream_df: pl.LazyFrame,
    feature_spec: "FeatureSpec",
    feature_plan: "FeaturePlan",
    upstream_column_mapping: dict[str, str],
    hash_algorithm: HashAlgorithm,
    hash_truncation_length: int | None = None,
) -> pl.LazyFrame: ...


def calculate_provenance_by_field_polars(
    joined_upstream_df: pl.DataFrame | pl.LazyFrame,
    feature_spec: "FeatureSpec",
    feature_plan: "FeaturePlan",
    upstream_column_mapping: dict[str, str],
    hash_algorithm: HashAlgorithm,
    hash_truncation_length: int | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Calculate metaxy_provenance_by_field for a Polars DataFrame.

    This is a standalone function that can be used for testing or direct calculation
    without going through the Narwhals interface.

    Args:
        joined_upstream_df: Polars DataFrame or LazyFrame with upstream data joined
        feature_spec: Feature specification
        feature_plan: Feature plan with field dependencies
        upstream_column_mapping: Maps upstream feature key -> provenance column name
        hash_algorithm: Hash algorithm to use (default: XXHASH64)
        hash_truncation_length: Optional length to truncate hashes to

    Returns:
        Polars frame of the same type as joined_upstream_df with metaxy_provenance_by_field column added

    Example:
        ```python
        from metaxy.data_versioning.calculators.polars import calculate_provenance_by_field_polars
        from metaxy.data_versioning.hash_algorithms import HashAlgorithm

        result = calculate_provenance_by_field_polars(
            joined_df,
            feature_spec,
            feature_plan,
            upstream_column_mapping={"parent": "metaxy_provenance_by_field"},
            hash_algorithm=HashAlgorithm.SHA256,
            hash_truncation_length=16,
        )
        ```
    """
    if hash_algorithm not in _HASH_FUNCTION_MAP:
        raise ValueError(
            f"Hash algorithm {hash_algorithm} not supported. "
            f"Supported: {list(_HASH_FUNCTION_MAP.keys())}"
        )

    hash_fn = _HASH_FUNCTION_MAP[hash_algorithm]

    # Build hash expressions for each field
    field_exprs = {}

    for field in feature_spec.fields:
        field_key_str = field.key.to_struct_key()

        field_deps = feature_plan.field_dependencies.get(field.key, {})

        # Build hash components
        components = [
            pl.lit(field_key_str),
            pl.lit(str(field.code_version)),
        ]

        # Add upstream provenance values in deterministic order
        for upstream_feature_key in sorted(field_deps.keys()):
            upstream_fields = field_deps[upstream_feature_key]
            upstream_key_str = upstream_feature_key.to_string()

            provenance_col_name = upstream_column_mapping.get(
                upstream_key_str, PROVENANCE_BY_FIELD_COL
            )

            for upstream_field in sorted(upstream_fields):
                upstream_field_str = upstream_field.to_struct_key()

                components.append(pl.lit(f"{upstream_key_str}/{upstream_field_str}"))
                components.append(
                    pl.col(provenance_col_name).struct.field(upstream_field_str)
                )

        # Concatenate and hash
        concat_expr = plh.concat_str(*components, separator="|")
        hashed = hash_fn(concat_expr).cast(pl.Utf8)

        # Apply truncation if specified
        if hash_truncation_length is not None:
            hashed = hashed.str.slice(0, hash_truncation_length)

        field_exprs[field_key_str] = hashed

    # Create provenance struct
    provenance_expr = pl.struct(**field_exprs)  # type: ignore[call-overload]

    return joined_upstream_df.with_columns(
        provenance_expr.alias(PROVENANCE_BY_FIELD_COL)
    )


class PolarsProvenanceByFieldCalculator(ProvenanceByFieldCalculator):
    """Calculates metaxy_provenance_by_field values using polars-hash.

    Accepts Narwhals LazyFrames and converts internally to Polars for hashing.
    Supports all hash functions available in polars-hash plugin.
    Default is xxHash64 for cross-database compatibility.
    """

    @property
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """All algorithms supported by polars-hash."""
        return list(_HASH_FUNCTION_MAP.keys())

    @property
    def default_algorithm(self) -> HashAlgorithm:
        """xxHash64 - fast and cross-database compatible."""
        return HashAlgorithm.XXHASH64

    def calculate_provenance_by_field(
        self,
        joined_upstream: nw.LazyFrame[Any],
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> nw.LazyFrame[Any]:
        """Calculate metaxy_provenance_by_field using polars-hash.

        Args:
            joined_upstream: Narwhals LazyFrame with upstream data joined
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_column_mapping: Maps upstream key -> column name
            hash_algorithm: Hash to use (default: xxHash64)

        Returns:
            Narwhals LazyFrame with metaxy_provenance_by_field column added

        Warning:
            joined_upstream DataFrames not backed by Polars **will be materialized to memory** as part of the calculation process.
        """
        algo = hash_algorithm or self.default_algorithm

        # Convert Narwhals to Polars DataFrame
        if joined_upstream.implementation.is_polars():
            pl_df = cast(pl.DataFrame | pl.LazyFrame, joined_upstream.to_native())
        else:
            # we must collect a non-polars DF
            pl_df = joined_upstream.collect().to_polars()

        # Use standalone function for calculation (without truncation here)
        result_pl = calculate_provenance_by_field_polars(
            pl_df,
            feature_spec,
            feature_plan,
            upstream_column_mapping,
            hash_algorithm=algo,
            hash_truncation_length=None,  # Apply truncation separately
        )

        # Convert back to Narwhals LazyFrame and apply truncation
        result_nw_lazy = nw.from_native(result_pl.lazy(), eager_only=False)

        # Use truncate_struct_column to apply hash truncation if configured
        # This needs to be eager for the truncation function
        return truncate_struct_column(result_nw_lazy, PROVENANCE_BY_FIELD_COL)
