"""Polars implementation of data version calculator."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import polars as pl
import polars_hash as plh

from metaxy.data_versioning.calculators.base import DataVersionCalculator
from metaxy.data_versioning.hash_algorithms import HashAlgorithm

if TYPE_CHECKING:
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.plan import FeaturePlan


class PolarsDataVersionCalculator(DataVersionCalculator[pl.LazyFrame]):
    """Calculates data versions using polars-hash.

    Type Parameters:
        TRef = pl.LazyFrame

    Supports all hash functions available in polars-hash plugin.
    Default is xxHash64 for cross-database compatibility.
    """

    # Map HashAlgorithm enum to polars-hash functions
    _HASH_FUNCTION_MAP: dict[HashAlgorithm, Callable[[pl.Expr], pl.Expr]] = {
        HashAlgorithm.XXHASH64: lambda expr: expr.nchash.xxhash64(),  # type: ignore[attr-defined]
        HashAlgorithm.XXHASH32: lambda expr: expr.nchash.xxhash32(),  # type: ignore[attr-defined]
        HashAlgorithm.WYHASH: lambda expr: expr.nchash.wyhash(),  # type: ignore[attr-defined]
        HashAlgorithm.SHA256: lambda expr: expr.chash.sha2_256(),  # type: ignore[attr-defined]
        HashAlgorithm.MD5: lambda expr: expr.nchash.md5(),  # type: ignore[attr-defined]
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
        joined_upstream: pl.LazyFrame,
        feature_spec: "FeatureSpec",
        feature_plan: "FeaturePlan",
        upstream_column_mapping: dict[str, str],
        hash_algorithm: HashAlgorithm | None = None,
    ) -> pl.LazyFrame:
        """Calculate data_version using polars-hash.

        Args:
            joined_upstream: LazyFrame with upstream data joined
            feature_spec: Feature specification
            feature_plan: Feature plan
            upstream_column_mapping: Maps upstream key -> column name
            hash_algorithm: Hash to use (default: xxHash64)

        Returns:
            LazyFrame with data_version column added
        """
        algo = hash_algorithm or self.default_algorithm

        if algo not in self.supported_algorithms:
            raise ValueError(
                f"Hash algorithm {algo} not supported by PolarsDataVersionCalculator. "
                f"Supported: {self.supported_algorithms}"
            )

        hash_fn = self._HASH_FUNCTION_MAP[algo]

        # Build hash expressions for each container
        container_exprs = {}

        for container in feature_spec.containers:
            container_key_str = (
                container.key.to_string()
                if hasattr(container.key, "to_string")
                else "_".join(container.key)
            )

            container_deps = feature_plan.container_dependencies.get(container.key, {})

            # Build hash components
            components = [
                pl.lit(container_key_str),
                pl.lit(str(container.code_version)),
            ]

            # Add upstream data versions in deterministic order
            for upstream_feature_key in sorted(container_deps.keys()):
                upstream_containers = container_deps[upstream_feature_key]
                upstream_key_str = (
                    upstream_feature_key.to_string()
                    if hasattr(upstream_feature_key, "to_string")
                    else "_".join(upstream_feature_key)
                )

                data_version_col_name = upstream_column_mapping.get(
                    upstream_key_str, "data_version"
                )

                for upstream_container in sorted(upstream_containers):
                    upstream_container_str = (
                        upstream_container.to_string()
                        if hasattr(upstream_container, "to_string")
                        else "_".join(upstream_container)
                    )

                    components.append(
                        pl.lit(f"{upstream_key_str}/{upstream_container_str}")
                    )
                    components.append(
                        pl.col(data_version_col_name).struct.field(
                            upstream_container_str
                        )
                    )

            # Concatenate and hash
            concat_expr = plh.concat_str(*components, separator="|")
            hashed = hash_fn(concat_expr).cast(pl.Utf8)
            container_exprs[container_key_str] = hashed

        # Create data_version struct
        data_version_expr = pl.struct(**container_exprs)  # type: ignore[call-overload]

        return joined_upstream.with_columns(data_version_expr.alias("data_version"))
