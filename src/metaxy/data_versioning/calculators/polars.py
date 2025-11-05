"""Polars implementation of field provenance calculator."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl
import polars_hash as plh

from metaxy.data_versioning.calculators.base import (
    FIELD_NAME_PREFIX,
    ProvenanceByFieldCalculator,
)
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.utils.hashing import truncate_struct_column

if TYPE_CHECKING:
    from metaxy.models.feature_spec import BaseFeatureSpec
    from metaxy.models.plan import FeaturePlan


class PolarsProvenanceByFieldCalculator(ProvenanceByFieldCalculator):
    """Calculates provenance_by_field values using polars-hash.

    Accepts Narwhals LazyFrames and converts internally to Polars for hashing.
    Supports all hash functions available in polars-hash plugin.
    Default is xxHash64 for cross-database compatibility.
    """

    # Map HashAlgorithm enum to polars-hash functions
    _HASH_FUNCTION_MAP: dict[HashAlgorithm, Callable[[pl.Expr], pl.Expr]] = {
        HashAlgorithm.XXHASH64: lambda expr: expr.nchash.xxhash64(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.XXHASH32: lambda expr: expr.nchash.xxhash32(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.WYHASH: lambda expr: expr.nchash.wyhash(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.SHA256: lambda expr: expr.chash.sha2_256(),  # pyright: ignore[reportAttributeAccessIssue]
        HashAlgorithm.MD5: lambda expr: expr.nchash.md5(),  # pyright: ignore[reportAttributeAccessIssue]
    }

    @property
    def supported_algorithms(self) -> list[HashAlgorithm]:
        """All algorithms supported by polars-hash."""
        return list(self._HASH_FUNCTION_MAP.keys())

    @property
    def default_algorithm(self) -> HashAlgorithm:
        """xxHash64 - fast and cross-database compatible."""
        return HashAlgorithm.XXHASH64

    def compute_struct_hash(
        self,
        lazy_frame: nw.LazyFrame[Any],
        struct_column: str,
        output_column: str,
        hash_algorithm: HashAlgorithm | None = None,
    ) -> nw.LazyFrame[Any]:
        """Compute a single hash from a struct column containing field hashes.

        Args:
            lazy_frame: Narwhals LazyFrame containing the struct column
            struct_column: Name of the struct column containing field hashes
            output_column: Name for the output hash column
            hash_algorithm: Hash algorithm to use. If None, uses self.default_algorithm.

        Returns:
            Narwhals LazyFrame with the output hash column added
        """
        algo = hash_algorithm or self.default_algorithm

        if algo not in self.supported_algorithms:
            raise ValueError(
                f"Hash algorithm {algo} not supported by PolarsProvenanceByFieldCalculator. "
                f"Supported: {self.supported_algorithms}"
            )

        # Convert to Polars
        pl_df = lazy_frame.collect().to_polars()

        # Get hash function
        hash_fn = self._HASH_FUNCTION_MAP[algo]

        # Build concatenation expression from struct fields
        # Get field names from the first row's struct
        if len(pl_df) > 0 and struct_column in pl_df.columns:
            struct_value = pl_df[struct_column][0]
            if struct_value is not None and isinstance(struct_value, dict):
                field_names = sorted(struct_value.keys())
            else:
                # Empty or null struct - add empty hash
                pl_df = pl_df.with_columns(pl.lit("").alias(output_column))
                return nw.from_native(pl_df.lazy(), eager_only=False)
        else:
            # No rows or column - return as-is
            return lazy_frame

        # Build concatenation components
        components = []
        for field_name in field_names:
            components.append(pl.lit(field_name))
            components.append(pl.col(struct_column).struct.field(field_name))

        # Concatenate and hash
        concat_expr = plh.concat_str(*components, separator="|")
        hash_expr = hash_fn(concat_expr).cast(pl.Utf8).alias(output_column)

        # Add hash column
        result_pl = pl_df.lazy().with_columns(hash_expr)

        # Apply truncation if configured
        from metaxy.utils.hashing import get_hash_truncation_length

        truncation_length = get_hash_truncation_length()
        if truncation_length is not None:
            result_pl = result_pl.with_columns(
                pl.col(output_column)
                .str.slice(0, truncation_length)
                .alias(output_column)
            )

        # Convert back to Narwhals
        return nw.from_native(result_pl, eager_only=False)

    def calculate_provenance_by_field(
        self,
        joined_upstream: nw.LazyFrame[Any],
        feature_spec: "BaseFeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> nw.LazyFrame[Any]:
        """Calculate provenance_by_field using polars-hash.

        Args:
            joined_upstream: Narwhals LazyFrame with upstream data joined
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_column_mapping: Maps upstream key -> column name
            hash_algorithm: Hash to use (default: xxHash64)

        Returns:
            Narwhals LazyFrame with provenance_by_field column added
        """
        algo = hash_algorithm or self.default_algorithm

        if algo not in self.supported_algorithms:
            raise ValueError(
                f"Hash algorithm {algo} not supported by PolarsProvenanceByFieldCalculator. "
                f"Supported: {self.supported_algorithms}"
            )

        # Convert Narwhals LazyFrame to Polars LazyFrame
        # Must collect first (LazyFrame doesn't have to_polars, only DataFrame does)
        pl_lazy = joined_upstream.collect().to_polars().lazy()

        hash_fn = self._HASH_FUNCTION_MAP[algo]

        # Build hash expressions for each field
        field_exprs = {}

        for field in feature_spec.fields:
            # Use database-safe field names to avoid reserved keywords
            field_key_str = FIELD_NAME_PREFIX + field.key.to_string().replace("/", "_")

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
                    upstream_key_str, "provenance_by_field"
                )

                # Get actual field names from the struct to handle ClickHouse renamed fields
                # First collect a sample to inspect the struct schema
                sample = pl_lazy.select(pl.col(provenance_col_name)).limit(1).collect()
                if len(sample) > 0 and provenance_col_name in sample.columns:
                    struct_value = sample[provenance_col_name][0]
                    if struct_value is not None and isinstance(struct_value, dict):
                        actual_field_names = sorted(list(struct_value.keys()))
                    else:
                        actual_field_names = []
                else:
                    actual_field_names = []

                for i, upstream_field in enumerate(sorted(upstream_fields)):
                    # Use the same prefix for consistency
                    upstream_field_str = (
                        FIELD_NAME_PREFIX + upstream_field.to_string().replace("/", "_")
                    )

                    components.append(
                        pl.lit(f"{upstream_key_str}/{upstream_field_str}")
                    )

                    # Use actual field name if available (handles ClickHouse renaming to f0, f1, etc)
                    if actual_field_names and i < len(actual_field_names):
                        field_to_access = actual_field_names[i]
                    else:
                        field_to_access = upstream_field_str

                    components.append(
                        pl.col(provenance_col_name).struct.field(field_to_access)
                    )

            # Concatenate and hash
            concat_expr = plh.concat_str(*components, separator="|")
            hashed = hash_fn(concat_expr).cast(pl.Utf8)

            field_exprs[field_key_str] = hashed

        # Create provenance_by_field struct
        provenance_expr = pl.struct(**field_exprs)  # type: ignore[call-overload]

        result_pl = pl_lazy.with_columns(provenance_expr.alias("provenance_by_field"))

        # Convert back to Narwhals LazyFrame and apply truncation
        result_nw = nw.from_native(result_pl, eager_only=False)

        # Use truncate_struct_column to apply hash truncation if configured
        # This needs to be eager for the truncation function
        result_eager = result_nw.collect()
        result_truncated = truncate_struct_column(result_eager, "provenance_by_field")

        # Convert back to lazy
        result_lazy = nw.from_native(
            result_truncated.to_native().lazy(), eager_only=False
        )

        # Compute metaxy_provenance using the new method
        result_with_metaxy = self.compute_struct_hash(
            result_lazy,
            struct_column="provenance_by_field",
            output_column="metaxy_provenance",
            hash_algorithm=algo,
        )

        return result_with_metaxy
