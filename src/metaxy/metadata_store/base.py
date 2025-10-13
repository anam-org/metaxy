"""Abstract base class for metadata storage backends."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import polars as pl

from metaxy.metadata_store.exceptions import DependencyError, FeatureNotFoundError
from metaxy.models.container import ContainerDep, SpecialContainerDep
from metaxy.models.feature import Feature, FeatureRegistry, metaxy_registry
from metaxy.models.plan import FeaturePlan, FQContainerKey
from metaxy.models.types import ContainerKey, FeatureKey

if TYPE_CHECKING:
    pass


class MetadataStore(ABC):
    """
    Abstract base class for metadata storage backends.

    Supports:
    - Immutable metadata storage (append-only)
    - Composable fallback store chains (for branch deployments)
    - Automatic data version calculation
    - Backend-specific computation optimizations

    Backends can override `_compute_data_versions_native()` to use
    native operations (SQL, UDFs) instead of Polars for performance.
    """

    def __init__(
        self,
        *,
        fallback_stores: list["MetadataStore"] | None = None,
        registry: FeatureRegistry = metaxy_registry,
    ):
        """
        Initialize metadata store with optional fallback chain.

        Args:
            fallback_stores: Ordered list of read-only fallback stores.
                Checked in order when metadata is not found locally.
                Useful for branch deployments reading from production.
        """
        self.fallback_stores = fallback_stores or []
        self.registry = registry

    # ========== Helper Methods ==========

    def _resolve_feature_key(self, feature: FeatureKey | type[Feature]) -> FeatureKey:
        """Resolve a Feature class or FeatureKey to FeatureKey."""
        if isinstance(feature, FeatureKey):
            return feature
        else:
            return feature.spec.key

    def _resolve_feature_spec(self, feature: FeatureKey | type[Feature]):
        """Resolve to FeatureSpec for accessing containers and deps."""
        if isinstance(feature, FeatureKey):
            return self.registry.feature_specs_by_key[feature]
        else:
            return feature.spec

    def _resolve_feature_plan(self, feature: FeatureKey | type[Feature]) -> FeaturePlan:
        """Resolve to FeaturePlan for dependency resolution."""
        return self.registry.get_feature_plan(self._resolve_feature_key(feature))

    # ========== Core CRUD Operations ==========

    @abstractmethod
    def write_metadata(
        self,
        feature: FeatureKey | type[Feature],
        df: pl.DataFrame,
    ) -> None:
        """
        Write metadata for a feature (immutable, append-only).

        Args:
            feature: Feature to write metadata for
            df: DataFrame containing metadata. Must have 'data_version' column
                of type pl.Struct with fields matching feature's containers.

        Raises:
            MetadataSchemaError: If DataFrame schema is invalid

        Note: Always writes to current store, never to fallback stores.
        """
        pass

    @abstractmethod
    def _read_metadata_local(
        self,
        feature: FeatureKey | type[Feature],
        *,
        filters: pl.Expr | None = None,
        columns: list[str] | None = None,
    ) -> pl.DataFrame | None:
        """
        Read metadata from THIS store only (no fallback).

        Args:
            feature: Feature to read metadata for
            filters: Polars expression for filtering rows
            columns: Subset of columns to return

        Returns:
            DataFrame with metadata, or None if feature not found locally
        """
        pass

    def read_metadata(
        self,
        feature: FeatureKey | type[Feature],
        *,
        filters: pl.Expr | None = None,
        columns: list[str] | None = None,
        allow_fallback: bool = True,
    ) -> pl.DataFrame:
        """
        Read metadata with optional fallback to upstream stores.

        Args:
            feature: Feature to read metadata for
            filters: Polars expression for filtering rows
            columns: Subset of columns to return
            allow_fallback: If True, check fallback stores on local miss

        Returns:
            DataFrame with metadata

        Raises:
            FeatureNotFoundError: If feature not found in any store
        """
        # Try local first
        df = self._read_metadata_local(feature, filters=filters, columns=columns)

        if df is not None:
            return df

        # Try fallback stores
        if allow_fallback:
            for store in self.fallback_stores:
                try:
                    # Use full read_metadata to handle nested fallback chains
                    return store.read_metadata(
                        feature,
                        filters=filters,
                        columns=columns,
                        allow_fallback=True,  # Allow fallback stores to check their fallbacks
                    )
                except FeatureNotFoundError:
                    # Try next fallback store
                    continue

        # Not found anywhere
        feature_key = self._resolve_feature_key(feature)
        raise FeatureNotFoundError(
            f"Feature {feature_key.to_string()} not found in store"
            + (" or fallback stores" if allow_fallback else "")
        )

    # ========== Feature Existence ==========

    def has_feature(
        self,
        feature: FeatureKey | type[Feature],
        *,
        check_fallback: bool = False,
    ) -> bool:
        """
        Check if feature exists in store.

        Args:
            feature: Feature to check
            check_fallback: If True, also check fallback stores

        Returns:
            True if feature exists, False otherwise
        """
        # Check local
        if self._read_metadata_local(feature) is not None:
            return True

        # Check fallback stores
        if check_fallback:
            for store in self.fallback_stores:
                if store.has_feature(feature, check_fallback=True):
                    return True

        return False

    def list_features(self, *, include_fallback: bool = False) -> list[FeatureKey]:
        """
        List all features in store.

        Args:
            include_fallback: If True, include features from fallback stores

        Returns:
            List of FeatureKey objects
        """
        features = self._list_features_local()

        if include_fallback:
            for store in self.fallback_stores:
                features.extend(store.list_features(include_fallback=True))

        # Deduplicate
        seen = set()
        unique_features = []
        for feature in features:
            key_str = feature.to_string()
            if key_str not in seen:
                seen.add(key_str)
                unique_features.append(feature)

        return unique_features

    @abstractmethod
    def _list_features_local(self) -> list[FeatureKey]:
        """List features in THIS store only."""
        pass

    # ========== Dependency Resolution ==========

    def read_upstream_metadata(
        self,
        feature: FeatureKey | type[Feature],
        container: ContainerKey | None = None,
        *,
        allow_fallback: bool = True,
    ) -> dict[str, pl.DataFrame]:
        """
        Read all upstream dependencies for a feature/container.

        Args:
            feature: Feature whose dependencies to load
            container: Specific container (if None, loads all deps for feature)
            allow_fallback: Whether to check fallback stores

        Returns:
            Dict mapping upstream feature keys (as strings) to metadata DataFrames.
            Each DataFrame has a 'data_version' column (pl.Struct).

        Raises:
            DependencyError: If required upstream feature is missing
        """
        plan = self._resolve_feature_plan(feature)

        # Get all upstream features we need
        upstream_features = set()

        if container is None:
            # All containers' dependencies
            for cont in plan.feature.containers:
                upstream_features.update(
                    self._get_container_dependencies(plan, cont.key)
                )
        else:
            # Specific container's dependencies
            upstream_features.update(self._get_container_dependencies(plan, container))

        # Load metadata for each upstream feature
        upstream_metadata = {}
        for upstream_fq_key in upstream_features:
            upstream_feature_key = upstream_fq_key.feature

            try:
                df = self.read_metadata(
                    upstream_feature_key, allow_fallback=allow_fallback
                )
                # Use string key for dict
                upstream_metadata[upstream_feature_key.to_string()] = df
            except FeatureNotFoundError as e:
                raise DependencyError(
                    f"Missing upstream feature {upstream_feature_key.to_string()} "
                    f"required by {plan.feature.key.to_string()}"
                ) from e

        return upstream_metadata

    def _get_container_dependencies(
        self, plan: FeaturePlan, container_key: ContainerKey
    ) -> set[FQContainerKey]:
        """Get all upstream container dependencies for a given container."""
        container = plan.feature.containers_by_key[container_key]
        upstream = set()

        if container.deps == SpecialContainerDep.ALL:
            # All upstream features and containers
            upstream.update(plan.all_parent_containers_by_key.keys())
        elif isinstance(container.deps, list):
            for dep in container.deps:
                if isinstance(dep, ContainerDep):
                    if dep.containers == SpecialContainerDep.ALL:
                        # All containers of this feature
                        upstream_feature = plan.parent_features_by_key[dep.feature_key]
                        for upstream_container in upstream_feature.containers:
                            upstream.add(
                                FQContainerKey(
                                    feature=dep.feature_key,
                                    container=upstream_container.key,
                                )
                            )
                    elif isinstance(dep.containers, list):
                        # Specific containers
                        for container_key in dep.containers:
                            upstream.add(
                                FQContainerKey(
                                    feature=dep.feature_key, container=container_key
                                )
                            )

        return upstream

    # ========== Data Version Calculation ==========

    def calculate_and_write_data_versions(
        self,
        feature: FeatureKey | type[Feature],
        sample_df: pl.DataFrame,
        *,
        allow_upstream_fallback: bool = True,
    ) -> pl.DataFrame:
        """
        Calculate data versions and write metadata.

        Automatically chooses computation strategy:
        - Native: If backend supports it AND all upstream is local
        - Polars: Otherwise (universal fallback)

        Args:
            feature: Feature to process
            sample_df: Input DataFrame with samples (without data_version column)
            allow_upstream_fallback: Load upstream from fallback stores if needed

        Returns:
            DataFrame with calculated data_version column added

        Raises:
            DependencyError: If upstream dependencies are missing
        """
        # Load upstream metadata
        upstream_metadata = self.read_upstream_metadata(
            feature, allow_fallback=allow_upstream_fallback
        )

        # Check if we can use native computation
        can_use_native = self._can_compute_native()

        if can_use_native:
            # Check if all upstream is in our store (not fallback)
            plan = self._resolve_feature_plan(feature)
            all_local = True

            for upstream_feature_spec in plan.deps or []:
                if not self.has_feature(
                    upstream_feature_spec.key, check_fallback=False
                ):
                    all_local = False
                    break

            if all_local:
                # Use native computation
                result_df = self._compute_data_versions_native(
                    feature=feature,
                    sample_df=sample_df,
                    upstream_metadata=upstream_metadata,
                )
            else:
                # Fall back to Polars (upstream in fallback stores)
                result_df = self._compute_data_versions_polars(
                    feature=feature,
                    sample_df=sample_df,
                    upstream_metadata=upstream_metadata,
                )
        else:
            # Backend doesn't support native, use Polars
            result_df = self._compute_data_versions_polars(
                feature=feature,
                sample_df=sample_df,
                upstream_metadata=upstream_metadata,
            )

        # Write result
        self.write_metadata(feature, result_df)

        return result_df

    def _can_compute_native(self) -> bool:
        """
        Check if this backend supports native data version computation.

        Override in subclasses that support native computation (SQL, UDFs).
        Default: False (use Polars)
        """
        return False

    def _compute_data_versions_native(
        self,
        feature: FeatureKey | type[Feature],
        sample_df: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """
        Compute data versions using backend-native operations.

        Override in subclasses that support native computation.
        This method is only called when _can_compute_native() returns True
        AND all upstream metadata is local.

        Args:
            feature: Feature being processed
            sample_df: Input samples
            upstream_metadata: Dict of upstream feature -> metadata DataFrame

        Returns:
            DataFrame with data_version column added
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support native computation"
        )

    def _compute_data_versions_polars(
        self,
        feature: FeatureKey | type[Feature],
        sample_df: pl.DataFrame,
        upstream_metadata: dict[str, pl.DataFrame],
    ) -> pl.DataFrame:
        """
        Compute data versions using Polars (universal fallback).

        This method uses the polars-hash plugin and works with any backend.

        Args:
            feature: Feature being processed
            sample_df: Input samples
            upstream_metadata: Dict of upstream feature -> metadata DataFrame

        Returns:
            DataFrame with data_version column added
        """
        feature_spec = self._resolve_feature_spec(feature)

        # Join upstream metadata with sample_df
        # We need upstream data_version columns for calculation
        working_df = sample_df
        for upstream_key, upstream_df in upstream_metadata.items():
            # Rename data_version to avoid conflicts
            upstream_renamed = upstream_df.select(
                [
                    pl.col("sample_id"),
                    pl.col("data_version").alias(
                        f"__upstream_{upstream_key}__data_version"
                    ),
                ]
            )

            working_df = working_df.join(
                upstream_renamed,
                on="sample_id",
                how="left",
            )

        # For each container, we need to compute its data version
        # by joining with the relevant upstream data and hashing
        container_data_versions = {}

        for container in feature_spec.containers:
            container_key_str = container.key.to_string()
            container_deps = {}

            if container.deps == SpecialContainerDep.ALL:
                # Depend on all upstream features/containers
                for upstream_key in upstream_metadata.keys():
                    data_version_schema = upstream_metadata[upstream_key].schema[
                        "data_version"
                    ]
                    if hasattr(data_version_schema, "fields"):
                        container_keys = [
                            field.name for field in data_version_schema.fields
                        ]
                        container_deps[upstream_key] = container_keys
            elif isinstance(container.deps, list):
                for dep in container.deps:
                    if isinstance(dep, ContainerDep):
                        upstream_key = dep.feature_key.to_string()
                        if dep.containers == SpecialContainerDep.ALL:
                            upstream_df = upstream_metadata.get(upstream_key)
                            if upstream_df is not None:
                                data_version_schema = upstream_df.schema["data_version"]
                                if hasattr(data_version_schema, "fields"):
                                    container_keys = [
                                        field.name
                                        for field in data_version_schema.fields
                                    ]
                                    container_deps[upstream_key] = container_keys
                        else:
                            container_deps[upstream_key] = [
                                ck.to_string() for ck in dep.containers
                            ]

            # Build hash components for this container
            components = [
                pl.lit(container_key_str),
                pl.lit(str(container.code_version)),
            ]

            # Add upstream data versions in deterministic order
            for upstream_feature_key in sorted(container_deps.keys()):
                upstream_container_keys = sorted(container_deps[upstream_feature_key])

                for upstream_container_key in upstream_container_keys:
                    components.append(
                        pl.lit(f"{upstream_feature_key}/{upstream_container_key}")
                    )
                    # Reference the renamed column in working_df
                    components.append(
                        pl.col(
                            f"__upstream_{upstream_feature_key}__data_version"
                        ).struct.field(upstream_container_key)
                    )

            # Hash and store
            import polars_hash as plh

            container_data_versions[container_key_str] = plh.concat_str(
                *components, separator="|"
            ).chash.sha2_256()

        # Combine all container data versions into a single struct
        data_version_expr = pl.struct(**container_data_versions)

        # Apply calculation to working_df
        result_df = working_df.with_columns(data_version_expr.alias("data_version"))

        # Keep only original sample_df columns plus data_version
        original_cols = sample_df.columns
        return result_df.select(original_cols + ["data_version"])
