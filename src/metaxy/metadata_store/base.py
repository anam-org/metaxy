"""Abstract base class for metadata storage backends."""

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from typing import TYPE_CHECKING, TypeGuard, overload

import narwhals as nw
import polars as pl
from typing_extensions import Self

from metaxy.data_versioning.calculators.base import DataVersionCalculator
from metaxy.data_versioning.calculators.polars import PolarsDataVersionCalculator
from metaxy.data_versioning.diff import DiffResult, LazyDiffResult
from metaxy.data_versioning.diff.base import MetadataDiffResolver
from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
from metaxy.data_versioning.hash_algorithms import HashAlgorithm
from metaxy.data_versioning.joiners.base import UpstreamJoiner
from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner
from metaxy.metadata_store.exceptions import (
    DependencyError,
    FeatureNotFoundError,
    StoreNotOpenError,
)
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.field import FieldDep, SpecialFieldDep
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import FeatureKey, FieldKey

if TYPE_CHECKING:
    pass

# Removed TRef - all stores now use Narwhals LazyFrames universally

# System namespace constant (use prefix without __ to avoid conflicts with separator)
SYSTEM_NAMESPACE = "metaxy-system"

# Metaxy-managed column names (to distinguish from user-defined columns)
METAXY_FEATURE_VERSION_COL = "metaxy_feature_version"
METAXY_SNAPSHOT_ID_COL = "metaxy_snapshot_id"
METAXY_DATA_VERSION_COL = "metaxy_data_version"

# System table keys
FEATURE_VERSIONS_KEY = FeatureKey([SYSTEM_NAMESPACE, "feature_versions"])
MIGRATION_HISTORY_KEY = FeatureKey([SYSTEM_NAMESPACE, "migrations"])

# Common Polars schemas for system tables
# TODO: Migrate to use METAXY_*_COL constants instead of plain names
FEATURE_VERSIONS_SCHEMA = {
    "feature_key": pl.String,
    "feature_version": pl.String,  # TODO: Use METAXY_FEATURE_VERSION_COL
    "recorded_at": pl.Datetime("us"),
    "feature_spec": pl.String,
    "feature_class_path": pl.String,
    "snapshot_id": pl.String,  # TODO: Use METAXY_SNAPSHOT_ID_COL
}

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
) -> TypeGuard[
    tuple[NarwhalsJoiner, PolarsDataVersionCalculator, NarwhalsDiffResolver]
]:
    """Type guard to check if using Narwhals components.

    Returns True if all components are Narwhals-based, allowing type narrowing.
    """
    joiner, calculator, diff_resolver = components
    return (
        isinstance(joiner, NarwhalsJoiner)
        and isinstance(calculator, PolarsDataVersionCalculator)
        and isinstance(diff_resolver, NarwhalsDiffResolver)
    )


class MetadataStore(ABC):
    """
    Abstract base class for metadata storage backends.

    Supports:
    - Immutable metadata storage (append-only)
    - Composable fallback store chains (for branch deployments)
    - Automatic data version calculation using three-component architecture
    - Backend-specific computation optimizations

    All stores use Narwhals LazyFrames as their universal interface,
    regardless of the underlying backend (Polars, Ibis/SQL, etc.).

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
    ) -> tuple[UpstreamJoiner, DataVersionCalculator, MetadataDiffResolver]:
        """Create native components for this store.

        Only called if _supports_native_components() returns True.

        Returns:
            Tuple of (joiner, calculator, diff_resolver) with appropriate types
            for this store's backend (Narwhals-compatible)

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
        return len(feature_key) >= 1 and feature_key[0] == SYSTEM_NAMESPACE

    def _resolve_feature_key(self, feature: FeatureKey | type[Feature]) -> FeatureKey:
        """Resolve a Feature class or FeatureKey to FeatureKey."""
        if isinstance(feature, FeatureKey):
            return feature
        else:
            return feature.spec.key

    def _resolve_feature_plan(self, feature: FeatureKey | type[Feature]) -> FeaturePlan:
        """Resolve to FeaturePlan for dependency resolution."""
        if isinstance(feature, FeatureKey):
            # When given a FeatureKey, get the graph from the active context
            return FeatureGraph.get_active().get_feature_plan(feature)
        else:
            # When given a Feature class, use its bound graph
            return feature.graph.get_feature_plan(feature.spec.key)

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
        df: nw.DataFrame | pl.DataFrame,
    ) -> None:
        """
        Write metadata for a feature (immutable, append-only).

        Automatically adds 'feature_version' column from current code state,
        unless the DataFrame already contains one (useful for migrations).

        Args:
            feature: Feature to write metadata for
            df: Narwhals DataFrame or Polars DataFrame containing metadata.
                Must have 'data_version' column of type Struct with fields matching feature's fields.
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

        # Convert Narwhals to Polars if needed
        if isinstance(df, nw.DataFrame):
            df = df.to_polars()
        # nw.DataFrame also matches as DataFrame in some contexts, ensure it's Polars
        if not isinstance(df, pl.DataFrame):
            # Must be some other type - shouldn't happen but handle defensively
            if hasattr(df, "to_polars"):
                df = df.to_polars()
            elif hasattr(df, "to_pandas"):
                df = pl.from_pandas(df.to_pandas())
            else:
                raise TypeError(f"Cannot convert {type(df)} to Polars DataFrame")

        # For system tables, write directly without feature_version tracking
        if is_system_table:
            self._validate_schema_system_table(df)
            self._write_metadata_impl(feature_key, df)
            return

        # For regular features: add feature_version and snapshot_id, validate, and write
        # Check if feature_version and snapshot_id already exist in DataFrame
        if "feature_version" in df.columns and "snapshot_id" in df.columns:
            # DataFrame already has feature_version and snapshot_id - use as-is
            # This is intended for migrations writing historical versions
            # Issue a warning unless we're in a suppression context
            if not _suppress_feature_version_warning.get():
                import warnings

                warnings.warn(
                    f"Writing metadata for {feature_key.to_string()} with existing "
                    f"feature_version and snapshot_id columns. This is intended for migrations only. "
                    f"Normal code should let write_metadata() add the current versions automatically.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Get current feature version and snapshot_id from code and add them
            if isinstance(feature, type) and issubclass(feature, Feature):
                current_feature_version = feature.feature_version()  # type: ignore[attr-defined]
            else:
                from metaxy.models.feature import FeatureGraph

                graph = FeatureGraph.get_active()
                feature_cls = graph.features_by_key[feature_key]
                current_feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]

            # Get snapshot_id from active graph
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
            current_snapshot_id = graph.snapshot_id

            df = df.with_columns(
                [
                    pl.lit(current_feature_version).alias("feature_version"),
                    pl.lit(current_snapshot_id).alias("snapshot_id"),
                ]
            )

        # Validate schema
        self._validate_schema(df)

        # Write metadata
        self._write_metadata_impl(feature_key, df)

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

        # Check for feature_version column
        if "feature_version" not in df.columns:
            raise MetadataSchemaError("DataFrame must have 'feature_version' column")

        # Check for snapshot_id column
        if "snapshot_id" not in df.columns:
            raise MetadataSchemaError("DataFrame must have 'snapshot_id' column")

    def _validate_schema_system_table(self, df: pl.DataFrame) -> None:
        """Validate schema for system tables (minimal validation)."""
        # System tables don't need data_version column
        pass

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

    def record_feature_graph_snapshot(self) -> str:
        """Record all features in graph with a graph snapshot ID.

        This should be called during CD (Continuous Deployment) to record what
        feature versions are being deployed. Typically invoked via `metaxy push`.

        Records all features in the graph with the same snapshot_id, representing
        a consistent state of the entire feature graph based on code definitions.

        The snapshot_id is a deterministic hash of all feature_version hashes
        in the graph, making it idempotent - calling multiple times with the
        same feature definitions produces the same snapshot_id.

        Returns:
            The generated snapshot_id (deterministic hash)

        Example:
            >>> # During CD, record what feature versions are being deployed
            >>> snapshot_id = store.record_feature_graph_snapshot()
            >>> print(f"Graph snapshot: {snapshot_id}")
            a3f8b2c1
        """
        # This is the same as serialize_feature_graph - just an alias
        return self.serialize_feature_graph()

    def serialize_feature_graph(self) -> str:
        """Record all features in graph with a graph snapshot ID.

        This should be called during CD (Continuous Deployment) to record what
        feature versions are being deployed. Typically invoked via `metaxy push`.

        Records all features with the same snapshot_id, representing a consistent
        state of the entire feature graph based on code definitions.

        The snapshot_id is a deterministic hash of all feature_version hashes
        in the graph, making it idempotent - calling multiple times with the
        same feature definitions produces the same snapshot_id.

        Returns:
            The generated snapshot_id (deterministic hash)

        Example:
            >>> # During CD, record what feature versions are being deployed
            >>> snapshot_id = store.serialize_feature_graph()
            >>> print(f"Graph snapshot: {snapshot_id}")
            a3f8b2c1
            >>>
            >>> # Later, in application code, users write metadata
            >>> store.write_metadata(FeatureA, data_a)
            >>> store.write_metadata(FeatureB, data_b)
            >>> store.write_metadata(FeatureC, data_c)
        """

        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()

        # Generate deterministic snapshot_id from graph
        # This uses the same logic as FeatureGraph.snapshot_id property
        snapshot_id = graph.snapshot_id

        # Read existing feature versions once
        try:
            existing_versions_lazy = self._read_metadata_native(FEATURE_VERSIONS_KEY)
            # Materialize to Polars for iteration
            existing_versions = (
                existing_versions_lazy.collect().to_polars()
                if existing_versions_lazy is not None
                else None
            )
        except Exception:
            # Table doesn't exist yet
            existing_versions = None

        # Build set of already recorded (feature_key, feature_version) pairs
        already_recorded = set()
        if existing_versions is not None:
            for row in existing_versions.iter_rows(named=True):
                already_recorded.add((row["feature_key"], row["feature_version"]))

        # Build bulk DataFrame for all features
        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()

        # Check if this exact snapshot already exists
        snapshot_already_exists = False
        if existing_versions is not None:
            snapshot_already_exists = (
                existing_versions.filter(pl.col("snapshot_id") == snapshot_id).height
                > 0
            )

        # If snapshot already exists, we're done (idempotent)
        if snapshot_already_exists:
            return snapshot_id

        records = []
        for feature_key in sorted(
            graph.features_by_key.keys(), key=lambda k: k.to_string()
        ):
            feature_cls = graph.features_by_key[feature_key]
            feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]

            # Serialize complete FeatureSpec
            feature_spec_json = json.dumps(feature_cls.spec.model_dump(mode="json"))  # type: ignore[attr-defined]

            # Get class import path
            class_path = f"{feature_cls.__module__}.{feature_cls.__name__}"

            # Always record all features for this snapshot (don't skip based on feature_version alone)
            # Each snapshot must be complete to support migration detection
            records.append(
                {
                    "feature_key": feature_key.to_string(),
                    "feature_version": feature_version,
                    "recorded_at": datetime.now(),
                    "feature_spec": feature_spec_json,  # Complete FeatureSpec as JSON
                    "feature_class_path": class_path,  # Import path for strict reconstruction
                    "snapshot_id": snapshot_id,
                }
            )

        # Bulk write all new records at once
        if records:
            version_records = pl.DataFrame(
                records,
                schema=FEATURE_VERSIONS_SCHEMA,
            )
            self._write_metadata_impl(FEATURE_VERSIONS_KEY, version_records)

        return snapshot_id

    @abstractmethod
    def _read_metadata_native(
        self,
        feature: FeatureKey | type[Feature],
        *,
        feature_version: str | None = None,
        filters: list[nw.Expr] | None = None,
        columns: list[str] | None = None,
    ) -> nw.LazyFrame | None:
        """
        Read metadata from THIS store only (no fallback).

        Args:
            feature: Feature to read metadata for
            feature_version: Filter by specific feature_version (applied natively in store)
            filters: List of Narwhals filter expressions (backend-agnostic)
                Works with any backend (Polars, Ibis/SQL, Pandas, PyArrow)
            columns: Subset of columns to return

        Returns:
            Narwhals LazyFrame with metadata, or None if feature not found locally
        """
        pass

    def read_metadata(
        self,
        feature: FeatureKey | type[Feature],
        *,
        feature_version: str | None = None,
        filters: list[nw.Expr] | None = None,
        columns: list[str] | None = None,
        allow_fallback: bool = True,
        current_only: bool = True,
    ) -> nw.LazyFrame:
        """
        Read metadata with optional fallback to upstream stores.

        Args:
            feature: Feature to read metadata for
            feature_version: Explicit feature_version to filter by (mutually exclusive with current_only=True)
            filters: List of Narwhals filter expressions (backend-agnostic, works with any store)
            columns: Subset of columns to return
            allow_fallback: If True, check fallback stores on local miss
            current_only: If True, only return rows with current feature_version
                (default: True for safety)

        Returns:
            Narwhals LazyFrame with metadata

        Raises:
            FeatureNotFoundError: If feature not found in any store
            ValueError: If both feature_version and current_only=True are provided
        """
        feature_key = self._resolve_feature_key(feature)
        is_system_table = self._is_system_table(feature_key)

        # Validate mutually exclusive parameters
        if feature_version is not None and current_only:
            raise ValueError(
                "Cannot specify both feature_version and current_only=True. "
                "Use current_only=False with feature_version parameter."
            )

        # Determine which feature_version to use
        feature_version_filter = feature_version
        if current_only and not is_system_table:
            # Get current feature_version
            if isinstance(feature, type) and issubclass(feature, Feature):
                feature_version_filter = feature.feature_version()  # type: ignore[attr-defined]
            else:
                from metaxy.models.feature import FeatureGraph

                graph = FeatureGraph.get_active()
                # Only try to get from graph if feature_key exists in graph
                # This allows reading system tables or external features not in current graph
                if feature_key in graph.features_by_key:
                    feature_cls = graph.features_by_key[feature_key]
                    feature_version_filter = feature_cls.feature_version()  # type: ignore[attr-defined]
                else:
                    # Feature not in graph - skip feature_version filtering
                    feature_version_filter = None

        # Try local first
        lazy_frame = self._read_metadata_native(
            feature,
            feature_version=feature_version_filter,
            filters=filters,
            columns=columns,
        )

        if lazy_frame is not None:
            return lazy_frame

        # Try fallback stores
        if allow_fallback:
            for store in self.fallback_stores:
                try:
                    # Use full read_metadata to handle nested fallback chains
                    return store.read_metadata(
                        feature,
                        feature_version=feature_version,
                        filters=filters,
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
        if self._read_metadata_native(feature) is not None:
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
        field: FieldKey | None = None,
        *,
        allow_fallback: bool = True,
        current_only: bool = True,
    ) -> dict[str, nw.LazyFrame]:
        """
        Read all upstream dependencies for a feature/field.

        Args:
            feature: Feature whose dependencies to load
            field: Specific field (if None, loads all deps for feature)
            allow_fallback: Whether to check fallback stores
            current_only: If True, only read current feature_version for upstream

        Returns:
            Dict mapping upstream feature keys (as strings) to Narwhals LazyFrames.
            Each LazyFrame has a 'data_version' column (Struct).

        Raises:
            DependencyError: If required upstream feature is missing
        """
        plan = self._resolve_feature_plan(feature)

        # Get all upstream features we need
        upstream_features = set()

        if field is None:
            # All fields' dependencies
            for cont in plan.feature.fields:
                upstream_features.update(self._get_field_dependencies(plan, cont.key))
        else:
            # Specific field's dependencies
            upstream_features.update(self._get_field_dependencies(plan, field))

        # Load metadata for each upstream feature
        # Use the feature's graph to look up upstream feature classes
        if isinstance(feature, FeatureKey):
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
        else:
            graph = feature.graph

        upstream_metadata = {}
        for upstream_fq_key in upstream_features:
            upstream_feature_key = upstream_fq_key.feature

            try:
                # Look up the Feature class from the graph and pass it to read_metadata
                # This way we use the bound graph instead of relying on active context
                upstream_feature_cls = graph.features_by_key[upstream_feature_key]
                lazy_frame = self.read_metadata(
                    upstream_feature_cls,
                    allow_fallback=allow_fallback,
                    current_only=current_only,  # Pass through current_only
                )
                # Use string key for dict
                upstream_metadata[upstream_feature_key.to_string()] = lazy_frame
            except FeatureNotFoundError as e:
                raise DependencyError(
                    f"Missing upstream feature {upstream_feature_key.to_string()} "
                    f"required by {plan.feature.key.to_string()}"
                ) from e

        return upstream_metadata

    def _get_field_dependencies(
        self, plan: FeaturePlan, field_key: FieldKey
    ) -> set[FQFieldKey]:
        """Get all upstream field dependencies for a given field."""
        field = plan.feature.fields_by_key[field_key]
        upstream = set()

        if field.deps == SpecialFieldDep.ALL:
            # All upstream features and fields
            upstream.update(plan.all_parent_fields_by_key.keys())
        elif isinstance(field.deps, list):
            for dep in field.deps:
                if isinstance(dep, FieldDep):
                    if dep.fields == SpecialFieldDep.ALL:
                        # All fields of this feature
                        upstream_feature = plan.parent_features_by_key[dep.feature_key]
                        for upstream_field in upstream_feature.fields:
                            upstream.add(
                                FQFieldKey(
                                    feature=dep.feature_key,
                                    field=upstream_field.key,
                                )
                            )
                    elif isinstance(dep.fields, list):
                        # Specific fields
                        for field_key in dep.fields:
                            upstream.add(
                                FQFieldKey(feature=dep.feature_key, field=field_key)
                            )

        return upstream

    # ========== Data Version Calculation ==========

    # ========== Data Versioning API ==========

    @overload
    def resolve_update(
        self,
        feature: type[Feature],
        *,
        lazy: bool = False,
        **kwargs,
    ) -> DiffResult: ...

    @overload
    def resolve_update(
        self,
        feature: type[Feature],
        *,
        lazy: bool = True,
        **kwargs,
    ) -> LazyDiffResult: ...

    def resolve_update(
        self,
        feature: type[Feature],
        *,
        lazy: bool = False,
        **kwargs,
    ) -> DiffResult | LazyDiffResult:
        """Resolve what needs updating for a feature.

        Primary user-facing method. Automatically chooses optimal strategy:
        1. All upstream local → Use native components (stay in DB)
        2. Some upstream in fallback stores → Pull to memory (Polars)

        Args:
            feature: Feature class to resolve updates for
            lazy: If True, return LazyDiffResult with lazy Narwhals LazyFrames.
                If False, return DiffResult with eager Narwhals DataFrames (default).
            **kwargs: Backend-specific parameters (reserved for future use)

        Returns:
            DiffResult (eager, default) or LazyDiffResult (lazy) with:
            - added: New samples not in current metadata
            - changed: Existing samples with different data_versions
            - removed: Samples in current but not in upstream

            Each frame has columns: [sample_id, data_version, ...user columns...]

        Note:
            Users can then process only added/changed and call write_metadata().
        """

        # Check where upstream data lives
        upstream_location = self._check_upstream_location(feature)

        if upstream_location == "all_local":
            # All upstream in this store - use native components
            return self._resolve_update_native(feature, lazy=lazy)
        else:
            # Some upstream in fallback stores - use Polars components
            return self._resolve_update_polars(feature, lazy=lazy)

    def _check_upstream_location(self, feature: type[Feature]) -> str:
        """Check if all upstream is in this store or in fallback stores.

        Returns:
            "all_local" if all upstream features are in this store
            "has_fallback" if any upstream is in fallback stores
        """
        plan = feature.graph.get_feature_plan(feature.spec.key)

        if not plan.deps:
            return "all_local"  # No dependencies

        for upstream_spec in plan.deps:
            if not self.has_feature(upstream_spec.key, check_fallback=False):
                return "has_fallback"  # At least one upstream is in fallback

        return "all_local"

    def _resolve_update_native(
        self,
        feature: type[Feature],
        *,
        lazy: bool = False,
    ) -> DiffResult | LazyDiffResult:
        """Resolve using components (all data in this store).

        Creates components on-demand based on:
        - User preference (prefer_native flag)
        - Store capabilities (supports native components)
        - Algorithm support (hash algorithm must be supported)

        Logic:
        - If prefer_native=True AND store supports native AND algorithm supported → use native
        - Otherwise → use Polars
        """
        plan = feature.graph.get_feature_plan(feature.spec.key)

        # With Narwhals, we have a unified path - use Narwhals components everywhere
        from metaxy.data_versioning.calculators.polars import (
            PolarsDataVersionCalculator,
        )
        from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
        from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner

        narwhals_joiner = NarwhalsJoiner()
        polars_calculator = (
            PolarsDataVersionCalculator()
        )  # Hash application needs Polars
        narwhals_diff = NarwhalsDiffResolver()

        # Load upstream as Narwhals LazyFrames
        upstream_refs: dict[str, nw.LazyFrame] = {}
        for upstream_spec in plan.deps or []:
            upstream_key_str = (
                upstream_spec.key.to_string()
                if hasattr(upstream_spec.key, "to_string")
                else "_".join(upstream_spec.key)
            )
            upstream_lazy = self._read_metadata_native(upstream_spec.key)
            if upstream_lazy is not None:
                upstream_refs[upstream_key_str] = upstream_lazy

        # Join upstream using Narwhals
        joined, mapping = feature.join_upstream_metadata(
            joiner=narwhals_joiner,
            upstream_refs=upstream_refs,
        )

        # Calculate data_versions using Polars calculator (accepts Narwhals LazyFrame)
        target_versions_nw = polars_calculator.calculate_data_versions(
            joined_upstream=joined,
            feature_spec=feature.spec,
            feature_plan=plan,
            upstream_column_mapping=mapping,
            hash_algorithm=self.hash_algorithm,
        )

        # Diff with current (filtered by feature_version at database level)
        current_lazy_nw = self._read_metadata_native(
            feature, feature_version=feature.feature_version()
        )

        # Convert current to Polars to match target (both need same backend for comparison)
        if current_lazy_nw is not None:
            # to_native() returns underlying type without materializing
            current_pl = current_lazy_nw.to_native()
            if isinstance(current_pl, pl.DataFrame):
                current_pl = current_pl.lazy()
            elif not isinstance(current_pl, pl.LazyFrame):
                # Ibis table - convert to Polars
                current_pl = current_pl.to_polars().lazy()
            # Wrap back in Narwhals
            current_lazy_nw = nw.from_native(current_pl)

        return feature.resolve_data_version_diff(
            diff_resolver=narwhals_diff,
            target_versions=target_versions_nw,
            current_metadata=current_lazy_nw,
            lazy=lazy,
        )

    def _resolve_update_polars(
        self,
        feature: type[Feature],
        *,
        lazy: bool = False,
    ) -> DiffResult | LazyDiffResult:
        """Resolve using Polars components (cross-store scenario).

        Pulls data from all stores to Polars, performs all operations in memory.
        Uses Polars components (not self.components which might be SQL-based).
        """
        from metaxy.data_versioning.calculators.polars import (
            PolarsDataVersionCalculator,
        )
        from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
        from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner

        # Load upstream from all sources (this store + fallbacks) as Narwhals LazyFrames
        upstream_refs = self.read_upstream_metadata(feature, allow_fallback=True)

        # Create Narwhals components (work with any backend)
        narwhals_joiner = NarwhalsJoiner()
        polars_calculator = (
            PolarsDataVersionCalculator()
        )  # Still need this for hash calculation
        narwhals_diff = NarwhalsDiffResolver()

        # Step 1: Join upstream using Narwhals
        plan = feature.graph.get_feature_plan(feature.spec.key)
        joined, mapping = feature.join_upstream_metadata(
            joiner=narwhals_joiner,
            upstream_refs=upstream_refs,
        )

        # Step 2: Calculate data_versions
        # to_native() returns underlying type without materializing
        joined_native = joined.to_native()
        if isinstance(joined_native, pl.LazyFrame):
            joined_pl = joined_native
        elif isinstance(joined_native, pl.DataFrame):
            joined_pl = joined_native.lazy()
        else:
            # Ibis table - convert to Polars
            joined_pl = joined_native.to_polars()
            if isinstance(joined_pl, pl.DataFrame):
                joined_pl = joined_pl.lazy()

        # Wrap in Narwhals before passing to calculator
        joined_nw = nw.from_native(joined_pl, eager_only=False)

        target_versions_nw = polars_calculator.calculate_data_versions(
            joined_upstream=joined_nw,
            feature_spec=feature.spec,
            feature_plan=plan,
            upstream_column_mapping=mapping,
            hash_algorithm=self.hash_algorithm,
        )

        # Select only sample_id and data_version for diff
        # The calculator returns the full joined DataFrame with upstream columns,
        # but diff resolver only needs these two columns
        target_versions_nw = target_versions_nw.select(["sample_id", "data_version"])

        # Step 3: Diff with current (filtered by feature_version at database level)
        current_lazy = self._read_metadata_native(
            feature, feature_version=feature.feature_version()
        )

        # Diff resolver returns Narwhals frames (lazy or eager based on flag)
        return feature.resolve_data_version_diff(
            diff_resolver=narwhals_diff,
            target_versions=target_versions_nw,
            current_metadata=current_lazy,
            lazy=lazy,
        )
