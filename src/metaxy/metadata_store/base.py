"""Abstract base class for metadata storage backends."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, Generic, TypeGuard, TypeVar

import polars as pl
from typing_extensions import Self

from metaxy.data_versioning.calculators.base import DataVersionCalculator
from metaxy.data_versioning.calculators.polars import PolarsDataVersionCalculator
from metaxy.data_versioning.diff import DiffResult
from metaxy.data_versioning.diff.base import MetadataDiffResolver
from metaxy.data_versioning.diff.polars import PolarsDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.data_versioning.joiners.polars import PolarsJoiner
from metaxy.metadata_store.exceptions import (
    DependencyError,
    FeatureNotFoundError,
    StoreNotOpenError,
)
from metaxy.models.container import ContainerDep, SpecialContainerDep
from metaxy.models.feature import Feature, FeatureRegistry
from metaxy.models.plan import FeaturePlan, FQContainerKey
from metaxy.models.types import ContainerKey, FeatureKey

if TYPE_CHECKING:
    pass

# Type variable for MetadataStore backend
TRef = TypeVar("TRef")  # Reference type (LazyFrame, ibis.Table, etc.)

# System table keys
FEATURE_VERSIONS_KEY = FeatureKey(["__metaxy__", "feature_versions"])
MIGRATION_HISTORY_KEY = FeatureKey(["__metaxy__", "migrations"])

# Context variable for suppressing feature_version warning in migrations
_suppress_feature_version_warning: ContextVar[bool] = ContextVar(
    "_suppress_feature_version_warning", default=False
)


@contextmanager
def allow_feature_version_override() -> Iterator[None]:
    """
    Context manager to suppress warnings when writing metadata with pre-existing feature_version.

    This should only be used in migration code where writing historical feature versions
    is intentional and necessary.

    Example:
        >>> with allow_feature_version_override():
        ...     # DataFrame already has feature_version column from migration
        ...     store.write_metadata(MyFeature, df_with_feature_version)
    """
    token = _suppress_feature_version_warning.set(True)
    try:
        yield
    finally:
        _suppress_feature_version_warning.reset(token)


def _is_using_polars_components(
    components: tuple[UpstreamJoiner, DataVersionCalculator, MetadataDiffResolver],
) -> TypeGuard[tuple[PolarsJoiner, PolarsDataVersionCalculator, PolarsDiffResolver]]:
    """Type guard to check if using Polars components.

    Returns True if all components are Polars-based, allowing type narrowing.
    """
    joiner, calculator, diff_resolver = components
    return (
        isinstance(joiner, PolarsJoiner)
        and isinstance(calculator, PolarsDataVersionCalculator)
        and isinstance(diff_resolver, PolarsDiffResolver)
    )


class MetadataStore(ABC, Generic[TRef]):
    """
    Abstract base class for metadata storage backends.

    Supports:
    - Immutable metadata storage (append-only)
    - Composable fallback store chains (for branch deployments)
    - Automatic data version calculation using three-component architecture
    - Backend-specific computation optimizations

    Type Parameters:
        TRef: Backend-specific reference type (pl.LazyFrame for Polars, ibis.Table for SQL)

    Components:
        Components are created on-demand in resolve_update() based on:
        - User preference (prefer_native flag)
        - Whether all upstream data is local (or needs fallback stores)
        - Store capabilities (whether it supports native components)

        If prefer_native=True and all conditions met: use native (Ibis, DuckDB, etc.)
        Otherwise: use Polars components

        Subclasses declare what native components they support via abstract methods.

    Context Manager:
        Stores must be used as context managers for resource management.
    """

    def __init__(
        self,
        *,
        hash_algorithm: HashAlgorithm | None = None,
        prefer_native: bool = True,
        fallback_stores: list["MetadataStore"] | None = None,
    ):
        """
        Initialize metadata store.

        Args:
            hash_algorithm: Hash algorithm to use for data versioning.
                Default: None (uses default algorithm for this store type)
            prefer_native: If True, prefer native components when possible.
                If False, always use Polars components. Default: True
            fallback_stores: Ordered list of read-only fallback stores.
                Used when upstream features are not in this store.

        Raises:
            ValueError: If fallback stores use different hash algorithms
        """
        # Initialize state early so properties can check it
        self._is_open = False
        self._context_depth = 0
        self._prefer_native = prefer_native

        # Use store's default algorithm if not specified
        if hash_algorithm is None:
            hash_algorithm = self._get_default_hash_algorithm()

        self.hash_algorithm = hash_algorithm
        self.fallback_stores = fallback_stores or []

        # Validation happens in open()

    @abstractmethod
    def _get_default_hash_algorithm(self) -> HashAlgorithm:
        """Get the default hash algorithm for this store type.

        Returns:
            Default hash algorithm
        """
        pass

    @abstractmethod
    def _supports_native_components(self) -> bool:
        """Check if this store can use native (non-Polars) components.

        Returns:
            True if store has backend-specific native components
            False if store only supports Polars components
        """
        pass

    @abstractmethod
    def _create_native_components(
        self,
    ) -> tuple[
        UpstreamJoiner[TRef], DataVersionCalculator[TRef], MetadataDiffResolver[TRef]
    ]:
        """Create native components for this store.

        Only called if _supports_native_components() returns True.

        Returns:
            Tuple of (joiner, calculator, diff_resolver)

        Raises:
            NotImplementedError: If store doesn't support native components
        """
        pass

    @abstractmethod
    def open(self) -> None:
        """Open/initialize the store for operations.

        Called by __enter__. Subclasses implement connection setup here.
        Can be called manually but context manager usage is recommended.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close/cleanup the store.

        Called by __exit__. Subclasses implement connection cleanup here.
        Can be called manually but context manager usage is recommended.
        """
        pass

    def __enter__(self) -> Self:
        """Enter context manager."""
        # Track nesting depth
        self._context_depth += 1

        # Only open on first enter
        if self._context_depth == 1:
            self.open()
            self._is_open = True

            # Validate after opening (when all components are ready)
            self._validate_after_open()

        return self

    def _validate_after_open(self) -> None:
        """Validate configuration after store is opened.

        Called automatically by __enter__ after open().
        Validates hash algorithm compatibility and fallback store consistency.
        """
        # Validate hash algorithm compatibility with components
        self.validate_hash_algorithm(check_fallback_stores=True)

        # Validate fallback stores use the same hash algorithm
        for i, fallback_store in enumerate(self.fallback_stores):
            if fallback_store.hash_algorithm != self.hash_algorithm:
                raise ValueError(
                    f"Fallback store {i} uses hash_algorithm='{fallback_store.hash_algorithm.value}' "
                    f"but this store uses '{self.hash_algorithm.value}'. "
                    f"All stores in a fallback chain must use the same hash algorithm."
                )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager."""
        # Decrement depth
        self._context_depth -= 1

        # Only close when fully exited
        if self._context_depth == 0:
            self._is_open = False
            self.close()

    def _check_open(self) -> None:
        """Check if store is open, raise error if not.

        Raises:
            StoreNotOpenError: If store is not open
        """
        if not self._is_open:
            raise StoreNotOpenError(
                f"{self.__class__.__name__} must be opened before use. "
                "Use it as a context manager: `with store: ...`"
            )

    # ========== Hash Algorithm Validation ==========

    def validate_hash_algorithm(
        self,
        check_fallback_stores: bool = True,
    ) -> None:
        """Validate that hash algorithm is supported by this store's components.

        Public method - can be called to verify hash compatibility.

        Args:
            check_fallback_stores: If True, also validate hash is supported by
                fallback stores (ensures compatibility for future cross-store operations)

        Raises:
            ValueError: If hash algorithm not supported by components or fallback stores
        """
        # Check if this store can support the algorithm
        # Try native components first (if supported), then Polars
        supported_algorithms = []

        if self._supports_native_components():
            try:
                _, calculator, _ = self._create_native_components()
                supported_algorithms = calculator.supported_algorithms
            except Exception:
                # If native components fail, fall back to Polars
                pass

        # If no native support or prefer_native=False, use Polars
        if not supported_algorithms:
            polars_calc = PolarsDataVersionCalculator()
            supported_algorithms = polars_calc.supported_algorithms

        if self.hash_algorithm not in supported_algorithms:
            from metaxy.metadata_store.exceptions import (
                HashAlgorithmNotSupportedError,
            )

            raise HashAlgorithmNotSupportedError(
                f"Hash algorithm {self.hash_algorithm} not supported by {self.__class__.__name__}. "
                f"Supported: {supported_algorithms}"
            )

        # Check fallback stores
        if check_fallback_stores:
            for fallback in self.fallback_stores:
                fallback.validate_hash_algorithm(check_fallback_stores=False)

    # ========== Helper Methods ==========

    def _is_system_table(self, feature_key: FeatureKey) -> bool:
        """Check if feature key is a system table."""
        return len(feature_key) >= 1 and feature_key[0] == "__metaxy__"

    def _resolve_feature_key(self, feature: FeatureKey | type[Feature]) -> FeatureKey:
        """Resolve a Feature class or FeatureKey to FeatureKey."""
        if isinstance(feature, FeatureKey):
            return feature
        else:
            return feature.spec.key

    def _resolve_feature_spec(self, feature: FeatureKey | type[Feature]):
        """Resolve to FeatureSpec for accessing containers and deps."""
        if isinstance(feature, FeatureKey):
            # When given a FeatureKey, get the registry from the active context
            return FeatureRegistry.get_active().feature_specs_by_key[feature]
        else:
            # When given a Feature class, it already has the registry bound to it
            return feature.spec

    def _resolve_feature_plan(self, feature: FeatureKey | type[Feature]) -> FeaturePlan:
        """Resolve to FeaturePlan for dependency resolution."""
        if isinstance(feature, FeatureKey):
            # When given a FeatureKey, get the registry from the active context
            return FeatureRegistry.get_active().get_feature_plan(feature)
        else:
            # When given a Feature class, use its bound registry
            return feature.registry.get_feature_plan(feature.spec.key)

    # ========== Backend Reference Conversion ==========

    @abstractmethod
    def _feature_to_ref(self, feature: FeatureKey | type[Feature]) -> TRef:
        """Convert feature to backend-specific reference.

        Args:
            feature: Feature to convert

        Returns:
            Backend-specific reference (LazyFrame for Polars, table name for SQL, etc.)

        Example:
            - InMemoryMetadataStore: Returns pl.LazyFrame from stored DataFrame
            - SQL-based stores: Return table name string
        """
        pass

    @abstractmethod
    def _sample_to_ref(self, sample_df: pl.DataFrame) -> TRef:
        """Convert sample DataFrame to backend-specific reference.

        Args:
            sample_df: Input sample DataFrame

        Returns:
            Backend-specific reference

        Example:
            - InMemoryMetadataStore: Returns sample_df.lazy()
            - SQL-based stores: Write to temp table, return table name
        """
        pass

    @abstractmethod
    def _result_to_dataframe(self, result: TRef) -> pl.DataFrame:
        """Convert backend-specific result to DataFrame.

        Args:
            result: Backend-specific result from resolver

        Returns:
            DataFrame ready to be written to metadata store

        Example:
            - InMemoryMetadataStore: Calls result.collect() on LazyFrame
            - SQL-based stores: Execute SQL query, return result as DataFrame
        """
        pass

    # ========== Core CRUD Operations ==========

    @abstractmethod
    def _write_metadata_impl(
        self,
        feature_key: FeatureKey,
        df: pl.DataFrame,
    ) -> None:
        """
        Internal write implementation (backend-specific).

        Args:
            feature_key: Feature key to write to
            df: DataFrame with metadata (already validated)

        Note: Subclasses implement this for their storage backend.
        """
        pass

    def write_metadata(
        self,
        feature: FeatureKey | type[Feature],
        df: pl.DataFrame,
    ) -> None:
        """
        Write metadata for a feature (immutable, append-only).

        Automatically adds 'feature_version' column from current code state,
        unless the DataFrame already contains one (useful for migrations).

        Args:
            feature: Feature to write metadata for
            df: DataFrame containing metadata. Must have 'data_version' column
                of type pl.Struct with fields matching feature's containers.
                May optionally contain 'feature_version' column (for migrations).

        Raises:
            MetadataSchemaError: If DataFrame schema is invalid
            StoreNotOpenError: If store is not open

        Note:
            - Always writes to current store, never to fallback stores.
            - If df already contains 'feature_version' column, it will be used
              as-is (no replacement). This allows migrations to write historical
              versions. A warning is issued unless suppressed via context manager.
        """
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        is_system_table = self._is_system_table(feature_key)

        # For system tables, write directly without feature_version tracking
        if is_system_table:
            self._validate_schema_system_table(df)
            self._write_metadata_impl(feature_key, df)
            return

        # For regular features: add feature_version, validate, and write
        # Check if feature_version already exists in DataFrame
        if "feature_version" in df.columns:
            # DataFrame already has feature_version - use it as-is
            # This is intended for migrations writing historical versions
            # Issue a warning unless we're in a suppression context
            if not _suppress_feature_version_warning.get():
                import warnings

                warnings.warn(
                    f"Writing metadata for {feature_key.to_string()} with existing "
                    f"feature_version column. This is intended for migrations only. "
                    f"Normal code should let write_metadata() add the current version automatically.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Get current feature version from code and add it
            if isinstance(feature, type) and issubclass(feature, Feature):
                current_feature_version = feature.feature_version()  # type: ignore[attr-defined]
            else:
                from metaxy.models.feature import FeatureRegistry

                registry = FeatureRegistry.get_active()
                feature_cls = registry.features_by_key[feature_key]
                current_feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]

            df = df.with_columns(
                pl.lit(current_feature_version).alias("feature_version")
            )

        # Validate schema
        self._validate_schema(df)

        # Write metadata
        self._write_metadata_impl(feature_key, df)

    @abstractmethod
    def _drop_feature_metadata_impl(self, feature_key: FeatureKey) -> None:
        """Drop/delete all metadata for a feature.

        Backend-specific implementation for dropping feature metadata.

        Args:
            feature_key: The feature key to drop metadata for
        """
        pass

    def drop_feature_metadata(self, feature: FeatureKey | type[Feature]) -> None:
        """Drop all metadata for a feature.

        This removes all stored metadata for the specified feature from the store.
        Useful for cleanup in tests or when re-computing feature metadata from scratch.

        Args:
            feature: Feature class or key to drop metadata for

        Example:
            >>> store.drop_feature_metadata(MyFeature)
            >>> assert not store.has_feature(MyFeature)
        """
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        self._drop_feature_metadata_impl(feature_key)

    def _validate_schema(self, df: pl.DataFrame) -> None:
        """
        Validate that DataFrame has required schema.

        Args:
            df: DataFrame to validate

        Raises:
            MetadataSchemaError: If schema is invalid
        """
        from metaxy.metadata_store.exceptions import MetadataSchemaError

        # Check for data_version column
        if "data_version" not in df.columns:
            raise MetadataSchemaError("DataFrame must have 'data_version' column")

        # Check that data_version is a struct
        data_version_type = df.schema["data_version"]
        if not isinstance(data_version_type, pl.Struct):
            raise MetadataSchemaError(
                f"'data_version' column must be pl.Struct, got {data_version_type}"
            )

    def _validate_schema_system_table(self, df: pl.DataFrame) -> None:
        """Validate schema for system tables (minimal validation)."""
        # System tables don't need data_version column
        pass

    def record_feature_version(
        self,
        feature: FeatureKey | type[Feature],
        *,
        snapshot_id: str | None = None,
    ) -> None:
        """Record feature version materialization in system table.

        This should be called explicitly (e.g., in CI) after materializing a feature.
        It enables migration detection by tracking which feature versions exist.

        This is NOT called automatically by write_metadata() for performance reasons.
        Users should call this explicitly when ready to "publish" a feature version.

        Args:
            feature: Feature that was materialized
            snapshot_id: Optional snapshot ID representing the entire feature graph state
                (typically generated by record_all_feature_versions)

        Example:
            >>> # Materialize feature
            >>> store.write_metadata(MyFeature, metadata_df)
            >>>
            >>> # Record the version (call this in CI after successful materialization)
            >>> store.record_feature_version(MyFeature)

            >>> # Or with snapshot_id
            >>> store.record_feature_version(MyFeature, snapshot_id="snapshot_20250113_103000")
        """
        feature_key = self._resolve_feature_key(feature)

        # Get feature class and version
        if isinstance(feature, type) and issubclass(feature, Feature):
            feature_cls = feature
            feature_version = feature.feature_version()  # type: ignore[attr-defined]
        else:
            from metaxy.models.feature import FeatureRegistry

            registry = FeatureRegistry.get_active()
            feature_cls = registry.features_by_key[feature_key]
            feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]

        # Get container names
        containers = [c.key.to_string() for c in feature_cls.spec.containers]  # type: ignore[attr-defined]

        # Check if already recorded (idempotent)
        try:
            existing = self._read_metadata_local(
                FEATURE_VERSIONS_KEY,
                filters=(
                    (pl.col("feature_key") == feature_key.to_string())
                    & (pl.col("feature_version") == feature_version)
                ),
            )
            if existing is not None and len(existing) > 0:
                # Already recorded
                return
        except Exception:
            # Table doesn't exist yet or other error - proceed with recording
            pass

        # Record new version
        version_record = pl.DataFrame(
            {
                "feature_key": [feature_key.to_string()],
                "feature_version": [feature_version],
                "recorded_at": [datetime.now()],
                "containers": [containers],
                "snapshot_id": [snapshot_id],  # Optional graph snapshot ID
            }
        )

        # Write to system table (bypass feature_version tracking to avoid recursion)
        self._write_metadata_impl(FEATURE_VERSIONS_KEY, version_record)

    def record_all_feature_versions(self) -> str:
        """Record all features in registry with a graph snapshot ID.

        This should be called during CD (Continuous Deployment) to record what
        feature versions are being deployed. Typically invoked via `metaxy push`.

        Records all features with the same snapshot_id, representing a consistent
        state of the entire feature graph based on code definitions.

        The snapshot_id is a deterministic hash of all feature_version hashes
        in the registry, making it idempotent - calling multiple times with the
        same feature definitions produces the same snapshot_id.

        Returns:
            The generated snapshot_id (deterministic hash)

        Example:
            >>> # During CD, record what feature versions are being deployed
            >>> snapshot_id = store.record_all_feature_versions()
            >>> print(f"Graph snapshot: {snapshot_id}")
            a3f8b2c1
            >>>
            >>> # Later, in application code, users write metadata
            >>> store.write_metadata(FeatureA, data_a)
            >>> store.write_metadata(FeatureB, data_b)
            >>> store.write_metadata(FeatureC, data_c)
        """
        import hashlib

        from metaxy.models.feature import FeatureRegistry

        registry = FeatureRegistry.get_active()

        # Generate deterministic snapshot_id (no timestamp!)
        # Hash all feature versions together
        hasher = hashlib.sha256()
        for feature_key in sorted(
            registry.features_by_key.keys(), key=lambda k: k.to_string()
        ):
            feature_cls = registry.features_by_key[feature_key]
            feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]
            # Include both key and version in hash
            hasher.update(f"{feature_key.to_string()}:{feature_version}".encode())

        snapshot_id = hasher.hexdigest()[:8]  # 8 chars like git

        # Read existing feature versions once
        try:
            existing_versions = self._read_metadata_local(FEATURE_VERSIONS_KEY)
        except Exception:
            # Table doesn't exist yet
            existing_versions = None

        # Build set of already recorded (feature_key, feature_version) pairs
        already_recorded = set()
        if existing_versions is not None:
            for row in existing_versions.iter_rows(named=True):
                already_recorded.add((row["feature_key"], row["feature_version"]))

        # Build bulk DataFrame for all features
        from metaxy.models.feature import FeatureRegistry

        registry = FeatureRegistry.get_active()

        records = []
        for feature_key in sorted(
            registry.features_by_key.keys(), key=lambda k: k.to_string()
        ):
            feature_cls = registry.features_by_key[feature_key]
            feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]
            containers = [c.key.to_string() for c in feature_cls.spec.containers]  # type: ignore[attr-defined]

            # Skip if already recorded (idempotent)
            if (feature_key.to_string(), feature_version) in already_recorded:
                continue

            records.append(
                {
                    "feature_key": feature_key.to_string(),
                    "feature_version": feature_version,
                    "recorded_at": datetime.now(),
                    "containers": containers,
                    "snapshot_id": snapshot_id,
                }
            )

        # Bulk write all new records at once
        if records:
            version_records = pl.DataFrame(records)
            self._write_metadata_impl(FEATURE_VERSIONS_KEY, version_records)

        return snapshot_id

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
        current_only: bool = True,
    ) -> pl.DataFrame:
        """
        Read metadata with optional fallback to upstream stores.

        Args:
            feature: Feature to read metadata for
            filters: Polars expression for filtering rows
            columns: Subset of columns to return
            allow_fallback: If True, check fallback stores on local miss
            current_only: If True, only return rows with current feature_version
                (default: True for safety)

        Returns:
            DataFrame with metadata

        Raises:
            FeatureNotFoundError: If feature not found in any store
        """
        feature_key = self._resolve_feature_key(feature)
        is_system_table = self._is_system_table(feature_key)

        # Build combined filters
        combined_filters = filters

        # Add feature_version filter for regular features (not system tables)
        # Note: We'll apply this filter after reading, to handle cases where
        # the column doesn't exist in the data yet
        current_feature_version = None
        if current_only and not is_system_table:
            # Get current feature_version
            if isinstance(feature, type) and issubclass(feature, Feature):
                current_feature_version = feature.feature_version()  # type: ignore[attr-defined]
            else:
                from metaxy.models.feature import FeatureRegistry

                registry = FeatureRegistry.get_active()
                feature_cls = registry.features_by_key[feature_key]
                current_feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]

        # Try local first
        df = self._read_metadata_local(
            feature, filters=combined_filters, columns=columns
        )

        if df is not None:
            # Apply feature_version filter if needed (after reading to handle missing column)
            if current_feature_version is not None and "feature_version" in df.columns:
                df = df.filter(pl.col("feature_version") == current_feature_version)
                # If filtering removed all rows, treat as not found
                if len(df) == 0:
                    df = None

        if df is not None:
            return df

        # Try fallback stores
        if allow_fallback:
            for store in self.fallback_stores:
                try:
                    # Use full read_metadata to handle nested fallback chains
                    return store.read_metadata(
                        feature,
                        filters=filters,  # Pass original filters, not combined
                        columns=columns,
                        allow_fallback=True,
                        current_only=current_only,  # Pass through current_only
                    )
                except FeatureNotFoundError:
                    # Try next fallback store
                    continue

        # Not found anywhere
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

        Raises:
            StoreNotOpenError: If store is not open
        """
        self._check_open()

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
        current_only: bool = True,
    ) -> dict[str, pl.DataFrame]:
        """
        Read all upstream dependencies for a feature/container.

        Args:
            feature: Feature whose dependencies to load
            container: Specific container (if None, loads all deps for feature)
            allow_fallback: Whether to check fallback stores
            current_only: If True, only read current feature_version for upstream

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
        # Use the feature's registry to look up upstream feature classes
        if isinstance(feature, FeatureKey):
            from metaxy.models.feature import FeatureRegistry

            registry = FeatureRegistry.get_active()
        else:
            registry = feature.registry

        upstream_metadata = {}
        for upstream_fq_key in upstream_features:
            upstream_feature_key = upstream_fq_key.feature

            try:
                # Look up the Feature class from the registry and pass it to read_metadata
                # This way we use the bound registry instead of relying on active context
                upstream_feature_cls = registry.features_by_key[upstream_feature_key]
                df = self.read_metadata(
                    upstream_feature_cls,
                    allow_fallback=allow_fallback,
                    current_only=current_only,  # Pass through current_only
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

    # ========== Data Versioning API ==========

    def resolve_update(
        self,
        feature: type[Feature],
        **kwargs,
    ) -> DiffResult:
        """Resolve what needs updating for a feature.

        Primary user-facing method. Automatically chooses optimal strategy:
        1. All upstream local → Use native components (stay in DB)
        2. Some upstream in fallback stores → Use Polars components (pull to memory)

        Returns DiffResult with three DataFrames:
        - added: New samples not in current metadata
        - changed: Existing samples with different data_versions
        - removed: Samples in current but not in upstream

        Users can then process only added/changed and call write_metadata().

        Args:
            feature: Feature class to resolve updates for
            **kwargs: Backend-specific parameters (reserved for future use)

        Returns:
            DiffResult[pl.DataFrame] with added, changed, removed
            Each DataFrame has columns: [sample_id, data_version]
        """

        # Check where upstream data lives
        upstream_location = self._check_upstream_location(feature)

        if upstream_location == "all_local":
            # All upstream in this store - use native components
            return self._resolve_update_native(feature)
        else:
            # Some upstream in fallback stores - use Polars components
            return self._resolve_update_polars(feature)

    def _check_upstream_location(self, feature: type[Feature]) -> str:
        """Check if all upstream is in this store or in fallback stores.

        Returns:
            "all_local" if all upstream features are in this store
            "has_fallback" if any upstream is in fallback stores
        """
        plan = feature.registry.get_feature_plan(feature.spec.key)

        if not plan.deps:
            return "all_local"  # No dependencies

        for upstream_spec in plan.deps:
            if not self.has_feature(upstream_spec.key, check_fallback=False):
                return "has_fallback"  # At least one upstream is in fallback

        return "all_local"

    def _resolve_update_native(
        self,
        feature: type[Feature],
    ) -> DiffResult:
        """Resolve using components (all data in this store).

        Creates components on-demand based on:
        - User preference (prefer_native flag)
        - Store capabilities (supports native components)
        - Algorithm support (hash algorithm must be supported)

        Logic:
        - If prefer_native=True AND store supports native AND algorithm supported → use native
        - Otherwise → use Polars
        """
        import warnings

        plan = feature.registry.get_feature_plan(feature.spec.key)

        # Create components based on preference and capabilities
        if self._prefer_native and self._supports_native_components():
            # Use native components (Ibis, DuckDB, etc.)
            components = self._create_native_components()
        else:
            # Fallback to Polars components
            if self._prefer_native:
                # User requested native but we're using Polars - log warning
                warnings.warn(
                    f"{self.__class__.__name__}: prefer_native=True but using Polars components. "
                    f"Store does not support native components.",
                    UserWarning,
                    stacklevel=2,
                )
            components = (
                PolarsJoiner(),
                PolarsDataVersionCalculator(),
                PolarsDiffResolver(),
            )

        # Use type guard to narrow component types
        if _is_using_polars_components(components):
            # Type system knows these are Polars components
            joiner, calculator, diff_resolver = components

            # Load upstream as LazyFrames
            upstream_refs: dict[str, pl.LazyFrame] = {}
            for upstream_spec in plan.deps or []:
                upstream_key_str = (
                    upstream_spec.key.to_string()
                    if hasattr(upstream_spec.key, "to_string")
                    else "_".join(upstream_spec.key)
                )
                upstream_df = self._read_metadata_local(upstream_spec.key)
                if upstream_df is not None:
                    upstream_refs[upstream_key_str] = upstream_df.lazy()

            # Join upstream
            joined, mapping = feature.join_upstream_metadata(
                joiner=joiner,
                upstream_refs=upstream_refs,
            )

            # Calculate data_versions
            target_versions = calculator.calculate_data_versions(
                joined_upstream=joined,
                feature_spec=feature.spec,
                feature_plan=plan,
                upstream_column_mapping=mapping,
                hash_algorithm=self.hash_algorithm,
            )

            # Diff with current
            current_df = self._read_metadata_local(feature)
            current_ref = current_df.lazy() if current_df is not None else None

            return feature.resolve_data_version_diff(
                diff_resolver=diff_resolver,
                target_versions=target_versions,
                current_metadata=current_ref,
            )
        else:
            # Native components - keep data in native format
            joiner, calculator, diff_resolver = components

            # Load upstream as native refs
            upstream_refs_native: dict[str, TRef] = {}
            for upstream_spec in plan.deps or []:
                upstream_key_str = (
                    upstream_spec.key.to_string()
                    if hasattr(upstream_spec.key, "to_string")
                    else "_".join(upstream_spec.key)
                )
                upstream_df = self._read_metadata_local(upstream_spec.key)
                if upstream_df is not None:
                    upstream_refs_native[upstream_key_str] = self._dataframe_to_ref(
                        upstream_df
                    )

            # Join upstream
            joined, mapping = feature.join_upstream_metadata(
                joiner=joiner,  # type: ignore[arg-type]
                upstream_refs=upstream_refs_native,  # type: ignore[arg-type]
            )

            # Calculate data_versions
            target_versions = calculator.calculate_data_versions(
                joined_upstream=joined,  # type: ignore[arg-type]
                feature_spec=feature.spec,
                feature_plan=plan,
                upstream_column_mapping=mapping,
                hash_algorithm=self.hash_algorithm,
            )

            # Diff with current
            current_df = self._read_metadata_local(feature)
            current_ref = (
                self._dataframe_to_ref(current_df) if current_df is not None else None
            )

            return feature.resolve_data_version_diff(
                diff_resolver=diff_resolver,
                target_versions=target_versions,  # type: ignore[arg-type]
                current_metadata=current_ref,  # type: ignore[arg-type]
            )

    def _resolve_update_polars(
        self,
        feature: type[Feature],
    ) -> DiffResult:
        """Resolve using Polars components (cross-store scenario).

        Pulls data from all stores to Polars, performs all operations in memory.
        Uses Polars components (not self.components which might be SQL-based).
        """
        from metaxy.data_versioning.calculators.polars import (
            PolarsDataVersionCalculator,
        )
        from metaxy.data_versioning.diff import PolarsDiffResolver
        from metaxy.data_versioning.joiners.polars import PolarsJoiner

        # Load upstream from all sources (this store + fallbacks) as DataFrames
        upstream_metadata = self.read_upstream_metadata(feature, allow_fallback=True)

        # Convert to LazyFrames
        upstream_refs: dict[str, pl.LazyFrame] = {
            upstream_key: upstream_df.lazy()
            for upstream_key, upstream_df in upstream_metadata.items()
        }

        # Create Polars components
        polars_joiner = PolarsJoiner()
        polars_calculator = PolarsDataVersionCalculator()
        polars_diff = PolarsDiffResolver()

        # Step 1: Join upstream
        plan = feature.registry.get_feature_plan(feature.spec.key)
        joined, mapping = feature.join_upstream_metadata(
            joiner=polars_joiner,
            upstream_refs=upstream_refs,
        )

        # Step 2: Calculate data_versions
        target_versions = polars_calculator.calculate_data_versions(
            joined_upstream=joined,
            feature_spec=feature.spec,
            feature_plan=plan,
            upstream_column_mapping=mapping,
            hash_algorithm=self.hash_algorithm,
        )

        # Select only sample_id and data_version for diff
        # The calculator returns the full joined DataFrame with upstream columns,
        # but diff resolver only needs these two columns
        target_versions = target_versions.select(["sample_id", "data_version"])

        # Step 3: Diff with current
        current_df = self._read_metadata_local(feature)
        current_lazy = current_df.lazy() if current_df is not None else None

        # Diff resolver returns DataFrames directly (already materialized with pl.collect_all)
        return feature.resolve_data_version_diff(
            diff_resolver=polars_diff,
            target_versions=target_versions,
            current_metadata=current_lazy,
        )

    @abstractmethod
    def _dataframe_to_ref(self, df: pl.DataFrame) -> TRef:
        """Convert DataFrame to backend-specific reference.

        For InMemoryMetadataStore: Returns df.lazy()
        For SQL stores: Might upload to temp table and return table reference

        Args:
            df: Polars DataFrame

        Returns:
            Backend-specific reference
        """
        pass
