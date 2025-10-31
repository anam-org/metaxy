"""Polars implementation of data version calculator."""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import narwhals as nw
import polars as pl
import polars_hash as plh

from metaxy.data_versioning.calculators.base import DataVersionCalculator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.utils.hashing import truncate_struct_column

if TYPE_CHECKING:
    from metaxy.models.feature_spec import BaseFeatureSpec, IDColumns
    from metaxy.models.plan import FeaturePlan


class PolarsDataVersionCalculator(DataVersionCalculator):
    """Calculates data versions using polars-hash.

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

    def calculate_data_versions(
        self,
        joined_upstream: nw.LazyFrame[Any],
        feature_spec: "BaseFeatureSpec[IDColumns]",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> nw.LazyFrame[Any]:
        """Calculate data_version using polars-hash.

        Args:
            joined_upstream: Narwhals LazyFrame with upstream data joined
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_column_mapping: Maps upstream key -> column name
            hash_algorithm: Hash to use (default: xxHash64)

        Returns:
            Narwhals LazyFrame with data_version column added
        """
        algo = hash_algorithm or self.default_algorithm

        if algo not in self.supported_algorithms:
            raise ValueError(
                f"Hash algorithm {algo} not supported by PolarsDataVersionCalculator. "
                f"Supported: {self.supported_algorithms}"
            )

        # Convert Narwhals LazyFrame to Polars LazyFrame
        # Must collect first (LazyFrame doesn't have to_polars, only DataFrame does)
        pl_lazy = joined_upstream.collect().to_polars().lazy()

        hash_fn = self._HASH_FUNCTION_MAP[algo]

        # Build hash expressions for each field
        field_exprs = {}

        for field in feature_spec.fields:
            field_key_str = (
                field.key.to_string()
                if hasattr(field.key, "to_string")
                else "_".join(field.key)
            )

            field_deps = feature_plan.field_dependencies.get(field.key, {})

            # Build hash components
            components = [
                pl.lit(field_key_str),
                pl.lit(str(field.code_version)),
            ]

            # Add upstream data versions in deterministic order
            for upstream_feature_key in sorted(field_deps.keys()):
                upstream_fields = field_deps[upstream_feature_key]
                upstream_key_str = (
                    upstream_feature_key.to_string()
                    if hasattr(upstream_feature_key, "to_string")
                    else "_".join(upstream_feature_key)
                )

                data_version_col_name = upstream_column_mapping.get(
                    upstream_key_str, "data_version"
                )

                for upstream_field in sorted(upstream_fields):
                    upstream_field_str = (
                        upstream_field.to_string()
                        if hasattr(upstream_field, "to_string")
                        else "_".join(upstream_field)
                    )

                    components.append(
                        pl.lit(f"{upstream_key_str}/{upstream_field_str}")
                    )
                    components.append(
                        pl.col(data_version_col_name).struct.field(upstream_field_str)
                    )

            # Concatenate and hash
            concat_expr = plh.concat_str(*components, separator="|")
            hashed = hash_fn(concat_expr).cast(pl.Utf8)

            field_exprs[field_key_str] = hashed

        # Create data_version struct
        data_version_expr = pl.struct(**field_exprs)  # type: ignore[call-overload]

        result_pl = pl_lazy.with_columns(data_version_expr.alias("data_version"))

        # Convert back to Narwhals LazyFrame and apply truncation
        result_nw = nw.from_native(result_pl, eager_only=False)

        # Use truncate_struct_column to apply hash truncation if configured
        # This needs to be eager for the truncation function
        result_eager = result_nw.collect()
        result_truncated = truncate_struct_column(result_eager, "data_version")

        # Convert back to lazy
        return nw.from_native(result_truncated.to_native().lazy(), eager_only=False)
