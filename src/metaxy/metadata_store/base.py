"""Abstract base class for metadata storage backends."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeGuard, overload

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
from metaxy.metadata_store.system_tables import (
    FEATURE_VERSIONS_KEY,
    FEATURE_VERSIONS_SCHEMA,
    SYSTEM_NAMESPACE,
    _suppress_feature_version_warning,
    allow_feature_version_override,
)
from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import IDColumns
from metaxy.models.field import FieldDep, SpecialFieldDep
from metaxy.models.plan import FeaturePlan, FQFieldKey
from metaxy.models.types import FeatureKey, FieldKey, SnapshotPushResult

if TYPE_CHECKING:
    pass

# Removed TRef - all stores now use Narwhals LazyFrames universally


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
    - Append-only metadata storage patterns

    - Composable fallback store chains for development and testing purposes

    - Backend-specific computation optimizations

    Context Manager:
        Stores must be used as context managers for resource management.
    """

    # Subclasses can override this to disable auto_create_tables warning
    # Set to False for stores where table creation is not applicable (e.g., InMemoryMetadataStore)
    _should_warn_auto_create_tables: bool = True

    def __init__(
        self,
        *,
        hash_algorithm: HashAlgorithm | None = None,
        hash_truncation_length: int | None = None,
        prefer_native: bool = True,
        fallback_stores: list[MetadataStore] | None = None,
        auto_create_tables: bool | None = None,
    ):
        """
        Initialize metadata store.

        Args:
            hash_algorithm: Hash algorithm to use for data versioning.
                Default: None (uses default algorithm for this store type)
            hash_truncation_length: Length to truncate hashes to (minimum 8).
                Default: None (uses global setting or no truncation)
            prefer_native: If True, prefer native data version calculations when possible.
                If False, always use Polars components. Default: True
            fallback_stores: Ordered list of read-only fallback stores.
                Used when upstream features are not in this store.
            auto_create_tables: If True, automatically create tables when opening the store.
                If None (default), reads from global MetaxyConfig (which reads from METAXY_AUTO_CREATE_TABLES env var).
                If False, never auto-create tables.
                WARNING: Auto-create is intended for development/testing only. Do not use in production.
                Use proper database migration tools like Alembic for production deployments.
                Default: None (reads from global config, falls back to False for safety)

        Raises:
            ValueError: If fallback stores use different hash algorithms or truncation lengths
        """
        # Initialize state early so properties can check it
        self._is_open = False
        self._context_depth = 0
        self._prefer_native = prefer_native
        self._allow_cross_project_writes = False

        # Resolve auto_create_tables from global config if not explicitly provided
        if auto_create_tables is None:
            from metaxy.config import MetaxyConfig

            self.auto_create_tables = MetaxyConfig.get().auto_create_tables
        else:
            self.auto_create_tables = auto_create_tables

        # Use store's default algorithm if not specified
        if hash_algorithm is None:
            hash_algorithm = self._get_default_hash_algorithm()

        self.hash_algorithm = hash_algorithm

        # Set hash truncation length - use explicit value or get from global setting
        if hash_truncation_length is not None:
            # Validate minimum length
            if hash_truncation_length < 8:
                raise ValueError(
                    f"hash_truncation_length must be at least 8 characters, got {hash_truncation_length}"
                )
            self.hash_truncation_length = hash_truncation_length
        else:
            # Use global setting if available
            from metaxy.utils.hashing import get_hash_truncation_length

            self.hash_truncation_length = get_hash_truncation_length()

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
            True if store has backend-specific native data version calculations
            False if store only supports Polars components
        """
        pass

    @abstractmethod
    def _create_native_components(
        self,
    ) -> tuple[UpstreamJoiner, DataVersionCalculator, MetadataDiffResolver]:
        """Create native data version calculations for this store.

        Only called if _supports_native_components() returns True.

        Returns:
            Tuple of (joiner, calculator, diff_resolver) with appropriate types
            for this store's backend (Narwhals-compatible)

        Raises:
            NotImplementedError: If store doesn't support native data version calculations
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
            # Warn if auto_create_tables is enabled (and store wants warnings)
            if self.auto_create_tables and self._should_warn_auto_create_tables:
                import warnings

                warnings.warn(
                    f"AUTO_CREATE_TABLES is enabled for {self.display()} - "
                    "do not use in production! "
                    "Use proper database migration tools like Alembic for production deployments.",
                    UserWarning,
                    stacklevel=3,  # stacklevel=3 to point to user's 'with store:' line
                )

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

        # Validate fallback stores use the same hash truncation length
        for i, fallback_store in enumerate(self.fallback_stores):
            if fallback_store.hash_truncation_length != self.hash_truncation_length:
                raise ValueError(
                    f"Fallback store {i} uses hash_truncation_length="
                    f"'{fallback_store.hash_truncation_length}' "
                    f"but this store uses '{self.hash_truncation_length}'. "
                    f"All stores in a fallback chain must use the same hash truncation length."
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
        # Try native data version calculations first (if supported), then Polars
        supported_algorithms = []

        if self._supports_native_components():
            try:
                _, calculator, _ = self._create_native_components()
                supported_algorithms = calculator.supported_algorithms
            except Exception:
                # If native data version calculations fail, fall back to Polars
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

    def _resolve_feature_key(
        self, feature: FeatureKey | type[BaseFeature[IDColumns]]
    ) -> FeatureKey:
        """Resolve a Feature class or FeatureKey to FeatureKey."""
        if isinstance(feature, FeatureKey):
            return feature
        else:
            return feature.spec().key

    def _resolve_feature_plan(
        self, feature: FeatureKey | type[BaseFeature[IDColumns]]
    ) -> FeaturePlan:
        """Resolve to FeaturePlan for dependency resolution."""
        if isinstance(feature, FeatureKey):
            # When given a FeatureKey, get the graph from the active context
            return FeatureGraph.get_active().get_feature_plan(feature)
        else:
            # When given a Feature class, use its bound graph
            return feature.graph.get_feature_plan(feature.spec().key)

    # ========== Core CRUD Operations ==========

    @contextmanager
    def allow_cross_project_writes(self) -> Iterator[None]:
        """Context manager to temporarily allow cross-project writes.

        This is an escape hatch for legitimate cross-project operations like migrations,
        where metadata needs to be written to features from different projects.

        Example:
            ```py
            # During migration, allow writing to features from different projects
            with store.allow_cross_project_writes():
                store.write_metadata(feature_from_project_a, metadata_a)
                store.write_metadata(feature_from_project_b, metadata_b)
            ```

        Yields:
            None: The context manager temporarily disables project validation
        """
        previous_value = self._allow_cross_project_writes
        try:
            self._allow_cross_project_writes = True
            yield
        finally:
            self._allow_cross_project_writes = previous_value

    def _validate_project_write(
        self, feature: FeatureKey | type[BaseFeature[IDColumns]]
    ) -> None:
        """Validate that writing to a feature matches the expected project from config.

        Args:
            feature: Feature to validate project for

        Raises:
            ValueError: If feature's project doesn't match the global config project
        """
        # Skip validation if cross-project writes are allowed
        if self._allow_cross_project_writes:
            return

        # Get the expected project from global config
        from metaxy.config import MetaxyConfig

        config = MetaxyConfig.get()
        expected_project = config.project

        # Use existing method to resolve to FeatureKey
        feature_key = self._resolve_feature_key(feature)

        # Get the Feature class from the graph
        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()
        if feature_key not in graph.features_by_key:
            # Feature not in graph - can't validate, skip
            return

        feature_cls = graph.features_by_key[feature_key]
        feature_project = feature_cls.project  # type: ignore[attr-defined]

        # Validate the project matches
        if feature_project != expected_project:
            raise ValueError(
                f"Cannot write to feature {feature_key.to_string()} from project '{feature_project}' "
                f"when the global configuration expects project '{expected_project}'. "
                f"Use store.allow_cross_project_writes() context manager for legitimate "
                f"cross-project operations like migrations."
            )

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
        feature: FeatureKey | type[BaseFeature[IDColumns]],
        df: nw.DataFrame[Any] | pl.DataFrame,
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
            ValueError: If writing to a feature from a different project than expected

        Note:
            - Always writes to current store, never to fallback stores.
            - If df already contains 'feature_version' column, it will be used
              as-is (no replacement). This allows migrations to write historical
              versions. A warning is issued unless suppressed via context manager.
            - Project validation is performed unless disabled via allow_cross_project_writes()
        """
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        is_system_table = self._is_system_table(feature_key)

        # Validate project for non-system tables
        if not is_system_table:
            self._validate_project_write(feature)

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

        # For regular features: add feature_version and snapshot_version, validate, and write
        # Check if feature_version and snapshot_version already exist in DataFrame
        if "feature_version" in df.columns and "snapshot_version" in df.columns:
            # DataFrame already has feature_version and snapshot_version - use as-is
            # This is intended for migrations writing historical versions
            # Issue a warning unless we're in a suppression context
            if not _suppress_feature_version_warning.get():
                import warnings

                warnings.warn(
                    f"Writing metadata for {feature_key.to_string()} with existing "
                    f"feature_version and snapshot_version columns. This is intended for migrations only. "
                    f"Normal code should let write_metadata() add the current versions automatically.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            # Get current feature version and snapshot_version from code and add them
            if isinstance(feature, type) and issubclass(feature, BaseFeature):
                current_feature_version = feature.feature_version()  # type: ignore[attr-defined]
            else:
                from metaxy.models.feature import FeatureGraph

                graph = FeatureGraph.get_active()
                feature_cls = graph.features_by_key[feature_key]
                current_feature_version = feature_cls.feature_version()  # type: ignore[attr-defined]

            # Get snapshot_version from active graph
            from metaxy.models.feature import FeatureGraph

            graph = FeatureGraph.get_active()
            current_snapshot_version = graph.snapshot_version

            df = df.with_columns(
                [
                    pl.lit(current_feature_version).alias("feature_version"),
                    pl.lit(current_snapshot_version).alias("snapshot_version"),
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

        # Check for snapshot_version column
        if "snapshot_version" not in df.columns:
            raise MetadataSchemaError("DataFrame must have 'snapshot_version' column")

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

    def drop_feature_metadata(
        self, feature: FeatureKey | type[BaseFeature[IDColumns]]
    ) -> None:
        """Drop all metadata for a feature.

        This removes all stored metadata for the specified feature from the store.
        Useful for cleanup in tests or when re-computing feature metadata from scratch.

        Warning:
            This operation is irreversible and will **permanently delete all metadata** for the specified feature.

        Args:
            feature: Feature class or key to drop metadata for

        Example:
            ```py
            store.drop_feature_metadata(MyFeature)
            assert not store.has_feature(MyFeature)
            ```
        """
        self._check_open()
        feature_key = self._resolve_feature_key(feature)
        self._drop_feature_metadata_impl(feature_key)

    def record_feature_graph_snapshot(self) -> SnapshotPushResult:
        """Record all features in graph with a graph snapshot version.

        This should be called during CD (Continuous Deployment) to record what
        feature versions are being deployed. Typically invoked via `metaxy graph push`.

        Records all features in the graph with the same snapshot_version, representing
        a consistent state of the entire feature graph based on code definitions.

        The snapshot_version is a deterministic hash of all feature_version hashes
        in the graph, making it idempotent - calling multiple times with the
        same feature definitions produces the same snapshot_version.

        This method detects three scenarios:
        1. New snapshot (computational changes): No existing rows with this snapshot_version
        2. Metadata-only changes: Snapshot exists but some features have different feature_spec_version
        3. No changes: Snapshot exists with identical feature_spec_versions for all features

        Returns: SnapshotPushResult
        """

        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()

        # Use to_snapshot() to get the snapshot dict
        snapshot_dict = graph.to_snapshot()

        # Generate deterministic snapshot_version from graph
        snapshot_version = graph.snapshot_version

        # Read existing feature versions once
        try:
            existing_versions_lazy = self.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            # Materialize to Polars for iteration
            existing_versions = (
                existing_versions_lazy.collect().to_polars()
                if existing_versions_lazy is not None
                else None
            )
        except Exception:
            # Table doesn't exist yet
            existing_versions = None

        # Get project from any feature in the graph (all should have the same project)
        # Default to empty string if no features in graph
        if graph.features_by_key:
            # Get first feature's project
            first_feature = next(iter(graph.features_by_key.values()))
            project_name = first_feature.project  # type: ignore[attr-defined]
        else:
            project_name = ""

        # Check if this exact snapshot already exists for this project
        snapshot_already_exists = False
        existing_spec_versions: dict[str, str] = {}

        if existing_versions is not None:
            # Check if project column exists (it may not in old tables)
            if "project" in existing_versions.columns:
                snapshot_rows = existing_versions.filter(
                    (pl.col("snapshot_version") == snapshot_version)
                    & (pl.col("project") == project_name)
                )
            else:
                # Old table without project column - just check snapshot_version
                snapshot_rows = existing_versions.filter(
                    pl.col("snapshot_version") == snapshot_version
                )
            snapshot_already_exists = snapshot_rows.height > 0

            if snapshot_already_exists:
                # Check if feature_spec_version column exists (backward compatibility)
                # Old records (before issue #77) won't have this column
                has_spec_version = "feature_spec_version" in snapshot_rows.columns

                if has_spec_version:
                    # Build dict of existing feature_key -> feature_spec_version
                    for row in snapshot_rows.iter_rows(named=True):
                        existing_spec_versions[row["feature_key"]] = row[
                            "feature_spec_version"
                        ]
                # If no spec_version column, existing_spec_versions remains empty
                # This means we'll treat it as "no metadata changes" (conservative approach)

        # Scenario 1: New snapshot (no existing rows)
        if not snapshot_already_exists:
            # Build records from snapshot_dict
            records = []
            for feature_key_str in sorted(snapshot_dict.keys()):
                feature_data = snapshot_dict[feature_key_str]

                # Serialize complete BaseFeatureSpec
                feature_spec_json = json.dumps(feature_data["feature_spec"])

                # Always record all features for this snapshot (don't skip based on feature_version alone)
                # Each snapshot must be complete to support migration detection
                records.append(
                    {
                        "project": project_name,
                        "feature_key": feature_key_str,
                        "feature_version": feature_data["feature_version"],
                        "feature_spec_version": feature_data["feature_spec_version"],
                        "feature_tracking_version": feature_data[
                            "feature_tracking_version"
                        ],
                        "recorded_at": datetime.now(timezone.utc),
                        "feature_spec": feature_spec_json,
                        "feature_class_path": feature_data["feature_class_path"],
                        "snapshot_version": snapshot_version,
                    }
                )

            # Bulk write all new records at once
            if records:
                version_records = pl.DataFrame(
                    records,
                    schema=FEATURE_VERSIONS_SCHEMA,
                )
                self._write_metadata_impl(FEATURE_VERSIONS_KEY, version_records)

            return SnapshotPushResult(
                snapshot_version=snapshot_version,
                already_recorded=False,
                metadata_changed=False,
                features_with_spec_changes=[],
            )

        # Scenario 2 & 3: Snapshot exists - check for metadata changes
        features_with_spec_changes = []

        for feature_key_str, feature_data in snapshot_dict.items():
            current_spec_version = feature_data["feature_spec_version"]
            existing_spec_version = existing_spec_versions.get(feature_key_str)

            if existing_spec_version != current_spec_version:
                features_with_spec_changes.append(feature_key_str)

        # If metadata changed, append new rows for affected features
        if features_with_spec_changes:
            records = []
            for feature_key_str in features_with_spec_changes:
                feature_data = snapshot_dict[feature_key_str]

                # Serialize complete BaseFeatureSpec
                feature_spec_json = json.dumps(feature_data["feature_spec"])

                records.append(
                    {
                        "project": project_name,
                        "feature_key": feature_key_str,
                        "feature_version": feature_data["feature_version"],
                        "feature_spec_version": feature_data["feature_spec_version"],
                        "feature_tracking_version": feature_data[
                            "feature_tracking_version"
                        ],
                        "recorded_at": datetime.now(timezone.utc),
                        "feature_spec": feature_spec_json,
                        "feature_class_path": feature_data["feature_class_path"],
                        "snapshot_version": snapshot_version,
                    }
                )

            # Bulk write updated records (append-only)
            if records:
                version_records = pl.DataFrame(
                    records,
                    schema=FEATURE_VERSIONS_SCHEMA,
                )
                self._write_metadata_impl(FEATURE_VERSIONS_KEY, version_records)

            return SnapshotPushResult(
                snapshot_version=snapshot_version,
                already_recorded=True,
                metadata_changed=True,
                features_with_spec_changes=features_with_spec_changes,
            )

        # Scenario 3: No changes at all
        return SnapshotPushResult(
            snapshot_version=snapshot_version,
            already_recorded=True,
            metadata_changed=False,
            features_with_spec_changes=[],
        )

    @abstractmethod
    def read_metadata_in_store(
        self,
        feature: FeatureKey | type[BaseFeature[IDColumns]],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
    ) -> nw.LazyFrame[Any] | None:
        """
        Read metadata from THIS store only (no fallback).

        Args:
            feature: Feature to read metadata for
            feature_version: Filter by specific feature_version (applied natively in store)
            filters: List of Narwhals filter expressions for this specific feature.
            columns: Subset of columns to return

        Returns:
            Narwhals LazyFrame with metadata, or None if feature not found in the store
        """
        pass

    def read_metadata(
        self,
        feature: FeatureKey | type[BaseFeature[IDColumns]],
        *,
        feature_version: str | None = None,
        filters: Sequence[nw.Expr] | None = None,
        columns: Sequence[str] | None = None,
        allow_fallback: bool = True,
        current_only: bool = True,
    ) -> nw.LazyFrame[Any]:
        """
        Read metadata with optional fallback to upstream stores.

        Args:
            feature: Feature to read metadata for
            feature_version: Explicit feature_version to filter by (mutually exclusive with current_only=True)
            filters: Sequence of Narwhals filter expressions to apply to this feature.
                Example: [nw.col("x") > 10, nw.col("y") < 5]
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
            if isinstance(feature, type) and issubclass(feature, BaseFeature):
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

        # Try local first with filters
        lazy_frame = self.read_metadata_in_store(
            feature,
            feature_version=feature_version_filter,
            filters=filters,  # Pass filters directly
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
                        filters=filters,  # Pass through filters directly
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
        feature: FeatureKey | type[BaseFeature[IDColumns]],
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
        if self.read_metadata_in_store(feature) is not None:
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

    @abstractmethod
    def display(self) -> str:
        """Return a human-readable display string for this store.

        Used in warnings, logs, and CLI output to identify the store.

        Returns:
            Display string (e.g., "DuckDBMetadataStore(database=/path/to/db.duckdb)")
        """
        pass

    def read_graph_snapshots(self, project: str | None = None) -> pl.DataFrame:
        """Read recorded graph snapshots from the feature_versions system table.

        Args:
            project: Project name to filter by. If None, returns snapshots from all projects.

        Returns a DataFrame with columns:
        - snapshot_version: Unique identifier for each graph snapshot
        - recorded_at: Timestamp when the snapshot was recorded
        - feature_count: Number of features in this snapshot

        Returns:
            Polars DataFrame with snapshot information, sorted by recorded_at descending

        Raises:
            StoreNotOpenError: If store is not open

        Example:
            ```py
            with store:
                # Get snapshots for a specific project
                snapshots = store.read_graph_snapshots(project="my_project")
                latest_snapshot = snapshots["snapshot_version"][0]
                print(f"Latest snapshot: {latest_snapshot}")

                # Get snapshots across all projects
                all_snapshots = store.read_graph_snapshots()
            ```
        """
        self._check_open()

        # Build filters based on project parameter
        filters = None
        if project is not None:
            import narwhals as nw

            filters = [nw.col("project") == project]

        versions_lazy = self.read_metadata_in_store(
            FEATURE_VERSIONS_KEY, filters=filters
        )
        if versions_lazy is None:
            # No snapshots recorded yet
            return pl.DataFrame(
                schema={
                    "snapshot_version": pl.String,
                    "recorded_at": pl.Datetime("us"),
                    "feature_count": pl.UInt32,
                }
            )

        versions_df = versions_lazy.collect().to_polars()

        # Group by snapshot_version and get earliest recorded_at and count
        snapshots = (
            versions_df.group_by("snapshot_version")
            .agg(
                [
                    pl.col("recorded_at").min().alias("recorded_at"),
                    pl.col("feature_key").count().alias("feature_count"),
                ]
            )
            .sort("recorded_at", descending=True)
        )

        return snapshots

    def read_features(
        self,
        *,
        current: bool = True,
        snapshot_version: str | None = None,
        project: str | None = None,
    ) -> pl.DataFrame:
        """Read feature version information from the feature_versions system table.

        Args:
            current: If True, only return features from the current code snapshot.
                     If False, must provide snapshot_version.
            snapshot_version: Specific snapshot version to filter by. Required if current=False.
            project: Project name to filter by. Defaults to None.

        Returns:
            Polars DataFrame with columns from FEATURE_VERSIONS_SCHEMA:
            - feature_key: Feature identifier
            - feature_version: Version hash of the feature
            - recorded_at: When this version was recorded
            - feature_spec: JSON serialized feature specification
            - feature_class_path: Python import path to the feature class
            - snapshot_version: Graph snapshot this feature belongs to

        Raises:
            StoreNotOpenError: If store is not open
            ValueError: If current=False but no snapshot_version provided

        Examples:
            ```py
            # Get features from current code
            with store:
                features = store.read_features(current=True)
                print(f"Current graph has {len(features)} features")
            ```

            ```py
            # Get features from a specific snapshot
            with store:
                features = store.read_features(current=False, snapshot_version="abc123")
                for row in features.iter_rows(named=True):
                    print(f"{row['feature_key']}: {row['feature_version']}")
            ```
        """
        self._check_open()

        if not current and snapshot_version is None:
            raise ValueError("Must provide snapshot_version when current=False")

        if current:
            # Get current snapshot from active graph
            graph = FeatureGraph.get_active()
            snapshot_version = graph.snapshot_version

        filters = [nw.col("snapshot_version") == snapshot_version]
        if project is not None:
            filters.append(nw.col("project") == project)

        versions_lazy = self.read_metadata_in_store(
            FEATURE_VERSIONS_KEY, filters=filters
        )
        if versions_lazy is None:
            # No features recorded yet
            return pl.DataFrame(schema=FEATURE_VERSIONS_SCHEMA)

        # Filter by snapshot_version
        versions_df = versions_lazy.collect().to_polars()

        return versions_df

    def copy_metadata(
        self,
        from_store: MetadataStore,
        features: list[FeatureKey | type[BaseFeature[IDColumns]]] | None = None,
        *,
        from_snapshot: str | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        incremental: bool = True,
    ) -> dict[str, int]:
        """Copy metadata from another store with fine-grained filtering.

        This is a reusable method that can be called programmatically or from CLI/migrations.
        Copies metadata for specified features, preserving the original snapshot_version.

        Args:
            from_store: Source metadata store to copy from (must be opened)
            features: List of features to copy. Can be:
                - None: copies all features from source store
                - List of FeatureKey or Feature classes: copies specified features
            from_snapshot: Snapshot version to filter source data by. If None, uses latest snapshot
                from source store. Only rows with this snapshot_version will be copied.
                The snapshot_version is preserved in the destination store.
            filters: Dict mapping feature keys (as strings) to sequences of Narwhals filter expressions.
                These filters are applied when reading from the source store.
                Example: {"feature/key": [nw.col("x") > 10], "other/feature": [...]}
            incremental: If True (default), filter out rows that already exist in the destination
                store by performing an anti-join on sample_uid for the same snapshot_version.

                The implementation uses an anti-join: source LEFT ANTI JOIN destination ON sample_uid
                filtered by snapshot_version.

                Disabling incremental (incremental=False) may improve performance when:
                - You know the destination is empty or has no overlap with source
                - The destination store uses deduplication

                When incremental=False, it's the user's responsibility to avoid duplicates or
                configure deduplication at the storage layer.

        Returns:
            Dict with statistics: {"features_copied": int, "rows_copied": int}

        Raises:
            ValueError: If from_store or self (destination) is not open
            FeatureNotFoundError: If a specified feature doesn't exist in source store

        Examples:
            ```py
            # Simple: copy all features from latest snapshot
            stats = dest_store.copy_metadata(from_store=source_store)
            ```

            ```py
            # Copy specific features from a specific snapshot
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=[FeatureKey(["my_feature"])],
                from_snapshot="abc123",
            )
            ```

            ```py
            # Copy with filters
            stats = dest_store.copy_metadata(
                from_store=source_store,
                filters={"my/feature": [nw.col("sample_uid").is_in(["s1", "s2"])]},
            )
            ```

            ```py
            # Copy specific features with filters
            stats = dest_store.copy_metadata(
                from_store=source_store,
                features=[
                    FeatureKey(["feature_a"]),
                    FeatureKey(["feature_b"]),
                ],
                filters={
                    "feature_a": [nw.col("field_a") > 10, nw.col("sample_uid").is_in(["s1", "s2"])],
                    "feature_b": [nw.col("field_b") < 30],
                },
            )
            ```
        """
        import logging

        logger = logging.getLogger(__name__)

        # Validate destination store is open
        if not self._is_open:
            raise ValueError("Destination store must be opened (use context manager)")

        # Automatically handle source store context manager
        should_close_source = not from_store._is_open
        if should_close_source:
            from_store.__enter__()

        try:
            return self._copy_metadata_impl(
                from_store=from_store,
                features=features,
                from_snapshot=from_snapshot,
                filters=filters,
                incremental=incremental,
                logger=logger,
            )
        finally:
            if should_close_source:
                from_store.__exit__(None, None, None)

    def _copy_metadata_impl(
        self,
        from_store: MetadataStore,
        features: list[FeatureKey | type[BaseFeature[IDColumns]]] | None,
        from_snapshot: str | None,
        filters: Mapping[str, Sequence[nw.Expr]] | None,
        incremental: bool,
        logger,
    ) -> dict[str, int]:
        """Internal implementation of copy_metadata."""
        # Determine which features to copy
        features_to_copy: list[FeatureKey]
        if features is None:
            # Copy all features from source
            features_to_copy = from_store.list_features(include_fallback=False)
            logger.info(
                f"Copying all features from source: {len(features_to_copy)} features"
            )
        else:
            # Convert all to FeatureKey
            features_to_copy = []
            for item in features:
                if isinstance(item, FeatureKey):
                    features_to_copy.append(item)
                else:
                    # Must be Feature class
                    features_to_copy.append(item.spec().key)
            logger.info(f"Copying {len(features_to_copy)} specified features")

        # Determine from_snapshot
        if from_snapshot is None:
            # Get latest snapshot from source store
            try:
                versions_lazy = from_store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
                if versions_lazy is None:
                    # No feature_versions table yet - if no features to copy, that's okay
                    if len(features_to_copy) == 0:
                        logger.info(
                            "No features to copy and no snapshots in source store"
                        )
                        from_snapshot = None  # Will be set later if needed
                    else:
                        raise ValueError(
                            "Source store has no feature_versions table. Cannot determine snapshot."
                        )
                elif versions_lazy is not None:
                    versions_df = versions_lazy.collect().to_polars()
                    if versions_df.height == 0:
                        # Empty versions table - if no features to copy, that's okay
                        if len(features_to_copy) == 0:
                            logger.info(
                                "No features to copy and no snapshots in source store"
                            )
                            from_snapshot = None
                        else:
                            raise ValueError(
                                "Source store feature_versions table is empty. No snapshots found."
                            )
                    else:
                        # Get most recent snapshot_version by recorded_at
                        from_snapshot = (
                            versions_df.sort("recorded_at", descending=True)
                            .select("snapshot_version")
                            .head(1)["snapshot_version"][0]
                        )
                        logger.info(
                            f"Using latest snapshot from source: {from_snapshot}"
                        )
            except Exception as e:
                # If we have no features to copy, continue gracefully
                if len(features_to_copy) == 0:
                    logger.info(f"No features to copy: {e}")
                    from_snapshot = None
                else:
                    raise ValueError(
                        f"Could not determine latest snapshot from source store: {e}"
                    )
        else:
            logger.info(f"Using specified from_snapshot: {from_snapshot}")

        # Copy metadata for each feature
        total_rows = 0
        features_copied = 0

        with allow_feature_version_override():
            for feature_key in features_to_copy:
                try:
                    # Read metadata from source, filtering by from_snapshot
                    # Use current_only=False to avoid filtering by feature_version
                    source_lazy = from_store.read_metadata(
                        feature_key,
                        allow_fallback=False,
                        current_only=False,
                    )

                    # Filter by from_snapshot
                    import narwhals as nw

                    source_filtered = source_lazy.filter(
                        nw.col("snapshot_version") == from_snapshot
                    )

                    # Apply filters for this feature (if any)
                    if filters:
                        feature_key_str = feature_key.to_string()
                        if feature_key_str in filters:
                            for filter_expr in filters[feature_key_str]:
                                source_filtered = source_filtered.filter(filter_expr)

                    # Apply incremental filtering if enabled
                    if incremental:
                        try:
                            # Read existing sample_uids from destination for the same snapshot
                            # This is much cheaper than comparing data_version structs
                            dest_lazy = self.read_metadata(
                                feature_key,
                                allow_fallback=False,
                                current_only=False,
                            )
                            # Filter destination to same snapshot_version
                            dest_for_snapshot = dest_lazy.filter(
                                nw.col("snapshot_version") == from_snapshot
                            )

                            # Materialize destination sample_uids to avoid cross-backend join issues
                            # When copying between different stores (e.g., different DuckDB files),
                            # Ibis can't join tables from different backends
                            dest_sample_uids = (
                                dest_for_snapshot.select("sample_uid")
                                .collect()
                                .to_polars()
                            )

                            # Convert to Polars LazyFrame and wrap in Narwhals
                            dest_sample_uids_lazy = nw.from_native(
                                dest_sample_uids.lazy(), eager_only=False
                            )

                            # Collect source to Polars for anti-join
                            source_df = source_filtered.collect().to_polars()
                            source_lazy = nw.from_native(
                                source_df.lazy(), eager_only=False
                            )

                            # Anti-join: keep only source rows with sample_uid not in destination
                            source_filtered = source_lazy.join(
                                dest_sample_uids_lazy,
                                on="sample_uid",
                                how="anti",
                            )

                            # Collect after filtering
                            source_df = source_filtered.collect().to_polars()

                            logger.info(
                                f"Incremental: copying only new sample_uids for {feature_key.to_string()}"
                            )
                        except FeatureNotFoundError:
                            # Feature doesn't exist in destination yet - copy all rows
                            logger.debug(
                                f"Feature {feature_key.to_string()} not in destination, copying all rows"
                            )
                            source_df = source_filtered.collect().to_polars()
                        except Exception as e:
                            # If incremental check fails, log warning but continue with full copy
                            logger.warning(
                                f"Incremental check failed for {feature_key.to_string()}: {e}. Copying all rows."
                            )
                            source_df = source_filtered.collect().to_polars()
                    else:
                        # Non-incremental: collect all filtered rows
                        source_df = source_filtered.collect().to_polars()

                    if source_df.height == 0:
                        logger.warning(
                            f"No rows found for {feature_key.to_string()} with snapshot {from_snapshot}, skipping"
                        )
                        continue

                    # Write to destination (preserving snapshot_version and feature_version)
                    self.write_metadata(feature_key, source_df)

                    features_copied += 1
                    total_rows += source_df.height
                    logger.info(
                        f"Copied {source_df.height} rows for {feature_key.to_string()}"
                    )

                except FeatureNotFoundError:
                    logger.warning(
                        f"Feature {feature_key.to_string()} not found in source store, skipping"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Error copying {feature_key.to_string()}: {e}", exc_info=True
                    )
                    raise

        logger.info(
            f"Copy complete: {features_copied} features, {total_rows} total rows"
        )

        return {"features_copied": features_copied, "rows_copied": total_rows}

    # ========== Dependency Resolution ==========

    def read_upstream_metadata(
        self,
        feature: FeatureKey | type[BaseFeature[IDColumns]],
        field: FieldKey | None = None,
        *,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        allow_fallback: bool = True,
        current_only: bool = True,
    ) -> dict[str, nw.LazyFrame[Any]]:
        """
        Read all upstream dependencies for a feature/field.

        Args:
            feature: Feature whose dependencies to load
            field: Specific field (if None, loads all deps for feature)
            filters: Dict mapping feature keys (as strings) to lists of Narwhals filter expressions.
                Example: {"upstream/feature1": [nw.col("x") > 10], "upstream/feature2": [...]}
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

            # Extract filters for this specific upstream feature
            upstream_filters = None
            if filters:
                upstream_key_str = upstream_feature_key.to_string()
                if upstream_key_str in filters:
                    upstream_filters = filters[upstream_key_str]

            try:
                # Look up the Feature class from the graph and pass it to read_metadata
                # This way we use the bound graph instead of relying on active context
                upstream_feature_cls = graph.features_by_key[upstream_feature_key]
                lazy_frame = self.read_metadata(
                    upstream_feature_cls,
                    filters=upstream_filters,  # Pass extracted filters (Sequence or None)
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
                        upstream_feature = plan.parent_features_by_key[dep.feature]
                        for upstream_field in upstream_feature.fields:
                            upstream.add(
                                FQFieldKey(
                                    feature=dep.feature,
                                    field=upstream_field.key,
                                )
                            )
                    elif isinstance(dep.fields, list):
                        # Specific fields
                        for field_key in dep.fields:
                            upstream.add(
                                FQFieldKey(feature=dep.feature, field=field_key)
                            )

        return upstream

    # ========== Data Version Calculation ==========

    # ========== Data Versioning API ==========

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature[IDColumns]],
        *,
        samples: nw.DataFrame[Any] | nw.LazyFrame[Any] | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: Literal[False] = False,
        **kwargs: Any,
    ) -> DiffResult: ...

    @overload
    def resolve_update(
        self,
        feature: type[BaseFeature[IDColumns]],
        *,
        samples: nw.DataFrame[Any] | nw.LazyFrame[Any] | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: Literal[True],
        **kwargs: Any,
    ) -> LazyDiffResult: ...

    def resolve_update(
        self,
        feature: type[BaseFeature[IDColumns]],
        *,
        samples: nw.DataFrame[Any] | nw.LazyFrame[Any] | None = None,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: bool = False,
        **kwargs: Any,
    ) -> DiffResult | LazyDiffResult:
        """Calculate an incremental update for a feature.

        Args:
            feature: Feature class to resolve updates for
            samples: Pre-computed DataFrame with ID columns
                and `"data_version"` column. When provided, `MetadataStore` skips upstream loading, joining,
                and data version calculation.

                **Required for root features** (features with no upstream dependencies).
                Root features don't have upstream to calculate `"data_version"` from, so users
                must provide samples with manually computed `"data_version"` column.

                For non-root features, use this when you
                want to bypass the automatic upstream loading and data version calculation.

                Examples:

                - Loading upstream from custom sources

                - Pre-computing data versions with custom logic

                - Testing specific scenarios

                Setting this parameter during normal operations is not required.

            filters: Dict mapping feature keys (as strings) to lists of Narwhals filter expressions.
                Applied when reading upstream metadata to filter samples at the source.
                Example: {"upstream/feature": [nw.col("x") > 10], ...}
            lazy: If `True`, return [metaxy.data_versioning.diff.LazyDiffResult][] with lazy Narwhals LazyFrames.
                If `False`, return [metaxy.data_versioning.diff.DiffResult][] with eager Narwhals DataFrames.
            **kwargs: Backend-specific parameters

        Raises:
            ValueError: If no `samples` DataFrame has been provided when resolving an update for a root feature.

        Examples:
            ```py
            # Root feature - samples required
            samples = pl.DataFrame({
                "sample_uid": [1, 2, 3],
                "data_version": [{"field": "h1"}, {"field": "h2"}, {"field": "h3"}],
            })
            result = store.resolve_update(RootFeature, samples=nw.from_native(samples))
            ```

            ```py
            # Non-root feature - automatic (normal usage)
            result = store.resolve_update(DownstreamFeature)
            ```

            ```py
            # Non-root feature - with escape hatch (advanced)
            custom_samples = compute_custom_data_versions(...)
            result = store.resolve_update(DownstreamFeature, samples=custom_samples)
            ```

        Note:
            Users can then process only added/changed and call write_metadata().
        """
        import narwhals as nw

        plan = feature.graph.get_feature_plan(feature.spec().key)

        # Escape hatch: if samples provided, use them directly (skip join/calculation)
        if samples is not None:
            import logging

            import polars as pl

            logger = logging.getLogger(__name__)

            # Convert samples to lazy if needed
            samples_lazy = (
                samples
                if isinstance(samples, nw.LazyFrame)
                else nw.from_native(samples.to_native().lazy())
            )

            # Check if samples are Polars-backed (common case for escape hatch)
            samples_native = samples_lazy.to_native()
            is_polars_samples = isinstance(samples_native, (pl.DataFrame, pl.LazyFrame))

            if is_polars_samples and self._supports_native_components():
                # User provided Polars samples but store uses native (SQL) backend
                # Need to materialize current metadata to Polars for compatibility
                logger.warning(
                    f"Feature {feature.spec().key}: samples parameter is Polars-backed but store uses native SQL backend. "
                    f"Materializing current metadata to Polars for diff comparison. "
                    f"For better performance, consider using samples with backend matching the store's backend."
                )
                # Get current metadata and materialize to Polars
                current_lazy_native = self.read_metadata_in_store(
                    feature, feature_version=feature.feature_version()
                )
                if current_lazy_native is not None:
                    # Convert to Polars using Narwhals' built-in method
                    current_lazy = nw.from_native(
                        current_lazy_native.collect().to_polars().lazy()
                    )
                else:
                    current_lazy = None
            else:
                # Same backend or no conversion needed - direct read
                current_lazy = self.read_metadata_in_store(
                    feature, feature_version=feature.feature_version()
                )

            # Use diff resolver to compare samples with current
            from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver

            diff_resolver = NarwhalsDiffResolver()

            lazy_result = diff_resolver.find_changes(
                target_versions=samples_lazy,
                current_metadata=current_lazy,
                id_columns=feature.spec().id_columns,  # Get ID columns from feature spec
            )

            return lazy_result if lazy else lazy_result.collect()

        # Root features without samples: error (samples required)
        if not plan.deps:
            raise ValueError(
                f"Feature {feature.spec().key} has no upstream dependencies (root feature). "
                f"Must provide 'samples' parameter with sample_uid and data_version columns. "
                f"Root features require manual data_version computation."
            )

        # Non-root features without samples: automatic upstream loading
        # Check where upstream data lives
        upstream_location = self._check_upstream_location(feature)

        if upstream_location == "all_local":
            # All upstream in this store - use native data version calculations
            return self._resolve_update_native(feature, filters=filters, lazy=lazy)
        else:
            # Some upstream in fallback stores - use Polars components
            return self._resolve_update_polars(feature, filters=filters, lazy=lazy)

    def _check_upstream_location(self, feature: type[BaseFeature[IDColumns]]) -> str:
        """Check if all upstream is in this store or in fallback stores.

        Returns:
            "all_local" if all upstream features are in this store
            "has_fallback" if any upstream is in fallback stores
        """
        plan = feature.graph.get_feature_plan(feature.spec().key)

        if not plan.deps:
            return "all_local"  # No dependencies

        for upstream_spec in plan.deps:
            if not self.has_feature(upstream_spec.key, check_fallback=False):
                return "has_fallback"  # At least one upstream is in fallback

        return "all_local"

    def _resolve_update_native(
        self,
        feature: type[BaseFeature[IDColumns]],
        *,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: bool = False,
    ) -> DiffResult | LazyDiffResult:
        """Resolve using native data version calculations (all data in this store).

        Uses native data version calculations when available (e.g., IbisDataVersionCalculator for SQL stores)
        to execute operations in the database without pulling data into memory.

        For stores that support native data version calculations (DuckDB, ClickHouse), this method:
        - Executes joins and diffs lazily via Narwhals
        - Computes hashes using native SQL functions (xxHash64, MD5, etc.)
        - Does not materialize data into memory (unless lazy=True)

        For stores without native support, falls back to PolarsDataVersionCalculator.
        """
        import logging

        logger = logging.getLogger(__name__)
        plan = feature.graph.get_feature_plan(feature.spec().key)

        # Root features should be handled in resolve_update() with samples parameter
        # This method should only be called for features with upstream
        if not plan.deps:
            raise RuntimeError(
                f"Internal error: _resolve_update_native called for root feature {feature.spec().key}. "
                f"Root features should be handled in resolve_update() with samples parameter."
            )

        # Create components based on native support
        # Only fallback to Polars if store explicitly doesn't support native data version calculations
        if self._supports_native_components():
            joiner, calculator, diff_resolver = self._create_native_components()
            logger.debug(
                f"Using native calculator for {feature.spec().key}: {calculator.__class__.__name__}"
            )
        else:
            # Store doesn't support native data version calculations - use Polars
            from metaxy.data_versioning.calculators.polars import (
                PolarsDataVersionCalculator,
            )
            from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
            from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner

            joiner = NarwhalsJoiner()
            calculator = PolarsDataVersionCalculator()
            diff_resolver = NarwhalsDiffResolver()
            logger.debug(
                f"Using Polars components for {feature.spec().key} (native not supported)"
            )

        # Load upstream as Narwhals LazyFrames (stays lazy in SQL for native stores)
        upstream_refs: dict[str, nw.LazyFrame[Any]] = {}
        for upstream_spec in plan.deps or []:
            upstream_key_str = (
                upstream_spec.key.to_string()
                if hasattr(upstream_spec.key, "to_string")
                else "_".join(upstream_spec.key)
            )
            # Extract filters for this upstream feature
            upstream_filters = None
            if filters and upstream_key_str in filters:
                upstream_filters = filters[upstream_key_str]

            upstream_lazy = self.read_metadata_in_store(
                upstream_spec.key,
                filters=upstream_filters,  # Apply extracted filters
            )
            if upstream_lazy is not None:
                upstream_refs[upstream_key_str] = upstream_lazy

        # Join upstream using Narwhals (stays lazy)
        joined, mapping = feature.load_input(
            joiner=joiner,
            upstream_refs=upstream_refs,
        )

        # Calculate data_versions using the selected calculator
        # For IbisDataVersionCalculator: executes hash computation in SQL
        # For PolarsDataVersionCalculator: materializes to compute hashes in memory
        target_versions_nw = calculator.calculate_data_versions(
            joined_upstream=joined,
            feature_spec=feature.spec(),
            feature_plan=plan,
            upstream_column_mapping=mapping,
            hash_algorithm=self.hash_algorithm,
        )

        # Diff with current (filtered by feature_version at database level)
        current_lazy_nw = self.read_metadata_in_store(
            feature, feature_version=feature.feature_version()
        )

        return feature.resolve_data_version_diff(
            diff_resolver=diff_resolver,
            target_versions=target_versions_nw,
            current_metadata=current_lazy_nw,
            lazy=lazy,
        )

    def _resolve_update_polars(
        self,
        feature: type[BaseFeature[IDColumns]],
        *,
        filters: Mapping[str, Sequence[nw.Expr]] | None = None,
        lazy: bool = False,
    ) -> DiffResult | LazyDiffResult:
        """Resolve using Polars components (cross-store scenario).

        Pulls data from all stores to Polars, performs all operations in memory.
        Uses Polars components instead of native SQL components because upstream
        data is distributed across multiple stores.

        This method is called when upstream features are in fallback stores,
        requiring materialization to join data from different sources.
        """
        import logging

        from metaxy.data_versioning.calculators.polars import (
            PolarsDataVersionCalculator,
        )
        from metaxy.data_versioning.diff.narwhals import NarwhalsDiffResolver
        from metaxy.data_versioning.joiners.narwhals import NarwhalsJoiner

        logger = logging.getLogger(__name__)

        # Warn if native components are available and preferred but can't be used due to cross-store scenario
        if self._prefer_native and self._supports_native_components():
            logger.warning(
                f"Feature {feature.spec().key} has upstream dependencies in fallback stores. "
                f"Falling back to in-memory Polars processing instead of native SQL execution. "
                f"For better performance, ensure all upstream features are in the same store."
            )

        # Load upstream from all sources (this store + fallbacks) as Narwhals LazyFrames
        upstream_refs = self.read_upstream_metadata(
            feature, filters=filters, allow_fallback=True
        )

        # Create Narwhals components (work with any backend)
        narwhals_joiner = NarwhalsJoiner()
        polars_calculator = (
            PolarsDataVersionCalculator()
        )  # Still need this for hash calculation
        narwhals_diff = NarwhalsDiffResolver()

        # Step 1: Join upstream using Narwhals
        plan = feature.graph.get_feature_plan(feature.spec().key)
        joined, mapping = feature.load_input(
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
            feature_spec=feature.spec(),
            feature_plan=plan,
            upstream_column_mapping=mapping,
            hash_algorithm=self.hash_algorithm,
        )

        # Select only sample_uid and data_version for diff
        # The calculator returns the full joined DataFrame with upstream columns,
        # but diff resolver only needs these two columns
        target_versions_nw = target_versions_nw.select(["sample_uid", "data_version"])

        # Step 3: Diff with current (filtered by feature_version at database level)
        current_lazy = self.read_metadata_in_store(
            feature, feature_version=feature.feature_version()
        )

        # Diff resolver returns Narwhals frames (lazy or eager based on flag)
        return feature.resolve_data_version_diff(
            diff_resolver=narwhals_diff,
            target_versions=target_versions_nw,
            current_metadata=current_lazy,
            lazy=lazy,
        )
