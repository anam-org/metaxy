"""Type-safe migration models with Python class paths.

Refactored migration system using:
- Python class paths for polymorphic deserialization via discriminated unions
- Struct-based storage for graph data
- Event-based status tracking
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
from pydantic import AliasChoices, TypeAdapter
from pydantic import Field as PydanticField
from pydantic.types import AwareDatetime

if TYPE_CHECKING:
    from metaxy.graph.diff.diff_models import GraphDiff
    from metaxy.metadata_store.base import MetadataStore


class OperationConfig(pydantic.BaseModel):
    """Configuration for a migration operation.

    The structure directly matches the YAML - no nested 'config' field.
    All operation-specific fields are defined directly on the operation class.

    Required fields:
    - type: Full Python class path to operation class (e.g., "metaxy.migrations.ops.DataVersionReconciliation")

    Optional fields:
    - features: List of feature keys this operation applies to
      - Required for FullGraphMigration
      - Optional for DiffMigration (features determined from graph diff)
    - All other fields are operation-specific and defined by the operation class

    Example (FullGraphMigration):
        {
            "type": "anam_data_utils.migrations.PostgreSQLBackfill",
            "features": ["raw_video", "scene"],
            "postgresql_url": "postgresql://...",  # Direct field, no nesting
            "batch_size": 1000
        }

    Example (DiffMigration):
        {
            "type": "metaxy.migrations.ops.DataVersionReconciliation",
        }

    Note:
        The 'type' field is stored as a string and only imported when the operation
        needs to be instantiated via the Migration.operations property. This allows
        reading migration configurations even if the operation classes have been
        renamed or don't exist yet.
    """

    model_config = pydantic.ConfigDict(extra="allow")

    type: str  # Python class path as string - imported lazily when needed
    features: list[str] = pydantic.Field(default_factory=list)


class Migration(pydantic.BaseModel, ABC):
    """Abstract base class for all migrations.

    Subclasses must define:
    - migration_type: Literal field with class path for discriminated union deserialization
    - execute(): Migration logic
    - get_affected_features(): Return list of affected feature keys

    The migration_type field is used as a discriminator for automatic polymorphic deserialization.

    All migrations form a chain via parent IDs (like git commits):
    - parent: ID of parent migration ("initial" for first migration)
    """

    model_config = pydantic.ConfigDict(extra="forbid")

    # Use AliasChoices to accept both "id" (from generated YAML) and "migration_id" (from tests/manual YAML)
    migration_id: str = PydanticField(validation_alias=AliasChoices("id", "migration_id"), serialization_alias="id")
    parent: str  # Parent migration ID or "initial"
    created_at: AwareDatetime

    @abstractmethod
    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> "MigrationResult":
        """Execute the migration.

        Args:
            store: Metadata store to operate on
            project: Project name for event tracking
            dry_run: If True, only validate without executing

        Returns:
            MigrationResult with execution details

        Raises:
            Exception: If migration fails
        """
        pass

    @abstractmethod
    def get_affected_features(self, store: "MetadataStore", project: str | None) -> list[str]:
        """Get list of affected feature keys in topological order.

        Args:
            store: Metadata store for computing affected features
            project: Project name for filtering snapshots

        Returns:
            List of feature key strings
        """
        pass

    def get_status_info(self, store: "MetadataStore", project: str | None) -> "MigrationStatusInfo":
        """Get comprehensive status information for this migration.

        This is a convenience method that combines information from:
        - The migration YAML (expected features)
        - The database events (completed/failed features)

        Args:
            store: Metadata store for querying events
            project: Project name for filtering events

        Returns:
            MigrationStatusInfo with all status details
        """
        from metaxy.metadata_store.system import SystemTableStorage

        storage = SystemTableStorage(store)

        # Get expected features from YAML (source of truth)
        expected_features = self.get_affected_features(store, project)
        expected_set = set(expected_features)

        # Get actual status from database
        summary = storage.get_migration_summary(self.migration_id, project, expected_features)

        # Filter completed/failed features to only include those in current YAML
        # This handles the case where YAML was modified to remove features
        completed_features = [fk for fk in summary["completed_features"] if fk in expected_set]
        failed_features = {fk: msg for fk, msg in summary["failed_features"].items() if fk in expected_set}

        # Compute pending features
        completed_set = set(completed_features)
        failed_set = set(failed_features.keys())
        pending_features = [fk for fk in expected_features if fk not in completed_set and fk not in failed_set]

        return MigrationStatusInfo(
            migration_id=self.migration_id,
            status=summary["status"],
            expected_features=expected_features,
            completed_features=completed_features,
            failed_features=failed_features,
            pending_features=pending_features,
        )

    @property
    def operations(self) -> list[Any]:
        """Get operations for this migration.

        Dynamically instantiates operations from the ops field (list of dicts with "type" field).
        If the migration doesn't have an ops field, returns empty list.

        Returns:
            List of operation instances

        Raises:
            ValueError: If operation dict is missing "type" field or class cannot be loaded
        """
        import importlib

        # Check if this migration has an ops field (using getattr to avoid type errors)
        ops = getattr(self, "ops", None)
        if ops is None:
            return []

        operations = []
        for op_dict in ops:
            # Validate structure has required fields
            op_config = OperationConfig.model_validate(op_dict)

            # Import the operation class from the string path
            # op_config.type is now a str (e.g., "anam_data_utils.migrations.postgresql_to_metaxy.RootFeatureBackfill")
            module_path, class_name = op_config.type.rsplit(".", 1)

            try:
                module = importlib.import_module(module_path)
                op_cls = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Failed to import operation class '{op_config.type}': {e}") from e

            # Pass the entire dict to the operation class (which inherits from BaseSettings)
            # BaseSettings will extract the fields it needs and read from env vars
            operation = op_cls.model_validate(op_dict)
            operations.append(operation)

        return operations


class DiffMigration(Migration):
    """Migration based on graph diff between two snapshots.

    Migrations form a chain via parent IDs (like git commits):
    - migration_id: Unique identifier for this migration
    - parent: ID of parent migration ("initial" for first migration)
    - from_snapshot_version: Source snapshot version
    - to_snapshot_version: Target snapshot version
    - ops: List of operation dicts with "type" field

    The parent chain ensures migrations are applied in correct order.
    Multiple heads (two migrations with no children) is an error.

    All other information is computed on-demand:
    - affected_features: Computed from GraphDiff when accessed
    - operations: Instantiated from ops
    - description: Auto-generated from affected features count

    The graph diff is computed on-demand when needed using GraphDiffer.

    Examples:
        First migration:
            DiffMigration(
                migration_id="20250113_120000",
                parent="initial",
                from_snapshot_version="abc123...",
                to_snapshot_version="def456...",
                created_at=datetime.now(timezone.utc),
            )

        Subsequent migration:
            DiffMigration(
                migration_id="20250113_130000",
                parent="20250113_120000",
                from_snapshot_version="def456...",
                to_snapshot_version="ghi789...",
                created_at=datetime.now(timezone.utc),
            )
    """

    # Discriminator field for polymorphic deserialization
    migration_type: Literal["metaxy.migrations.models.DiffMigration"] = "metaxy.migrations.models.DiffMigration"

    # Stored fields - persisted to YAML in git
    from_snapshot_version: str
    to_snapshot_version: str
    ops: list[dict[str, Any]]  # Required - must explicitly specify operations

    # Private attribute for caching computed graph diff
    _graph_diff_cache: "GraphDiff | None" = pydantic.PrivateAttr(default=None)

    def _get_graph_diff(self, store: "MetadataStore", project: str | None) -> "GraphDiff":
        """Get or compute graph diff (cached).

        Args:
            store: Metadata store containing snapshots
            project: Project name for filtering snapshots

        Returns:
            GraphDiff between snapshots
        """
        if self._graph_diff_cache is None:
            self._graph_diff_cache = self.compute_graph_diff(store, project)
        return self._graph_diff_cache

    def get_affected_features(self, store: "MetadataStore", project: str | None) -> list[str]:
        """Get affected features in topological order (computed on-demand).

        Args:
            store: Metadata store containing snapshots (required for computation)
            project: Project name for filtering snapshots

        Returns:
            List of feature key strings in topological order
        """
        from metaxy.models.feature import FeatureGraph

        graph_diff = self._get_graph_diff(store, project)

        # Get changed feature keys (root changes)
        changed_keys = [node.feature_key for node in graph_diff.changed_nodes]

        # Also include added nodes
        for node in graph_diff.added_nodes:
            changed_keys.append(node.feature_key)

        # Get the active graph
        active_graph = FeatureGraph.get_active()

        # Get all downstream features (features that depend on changed features)
        downstream_keys = active_graph.get_downstream_features(changed_keys)

        # Combine changed and downstream
        all_affected_keys = changed_keys + downstream_keys

        # Sort topologically
        sorted_keys = active_graph.topological_sort_features(all_affected_keys)

        return [key.to_string() for key in sorted_keys]

    def compute_graph_diff(self, store: "MetadataStore", project: str | None) -> "GraphDiff":
        """Compute GraphDiff on-demand from snapshot versions.

        Args:
            store: Metadata store containing snapshots
            project: Project name for filtering snapshots

        Returns:
            GraphDiff between from_snapshot_version and to_snapshot_version

        Raises:
            ValueError: If snapshots cannot be loaded
        """
        from metaxy.graph.diff.differ import GraphDiffer
        from metaxy.models.feature import FeatureGraph

        differ = GraphDiffer()

        # Load from_snapshot data from store
        from_snapshot_data = differ.load_snapshot_data(store, self.from_snapshot_version)

        # Try to load to_snapshot from store, if it doesn't exist use active graph
        try:
            to_snapshot_data = differ.load_snapshot_data(store, self.to_snapshot_version)
        except ValueError:
            # Snapshot not recorded yet, use active graph
            active_graph = FeatureGraph.get_active()
            if active_graph.snapshot_version != self.to_snapshot_version:
                raise ValueError(
                    f"to_snapshot {self.to_snapshot_version} not found in store "
                    f"and doesn't match active graph ({active_graph.snapshot_version})"
                )
            to_snapshot_data = active_graph.to_snapshot()

        # Compute diff
        return differ.diff(
            from_snapshot_data,
            to_snapshot_data,
            self.from_snapshot_version,
            self.to_snapshot_version,
        )

    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> "MigrationResult":
        """Execute diff-based migration.

        Delegates to MigrationExecutor for execution logic.

        Args:
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        from metaxy.metadata_store.system import SystemTableStorage
        from metaxy.migrations.executor import MigrationExecutor

        storage = SystemTableStorage(store)
        executor = MigrationExecutor(storage)
        return executor._execute_diff_migration(self, store, project, dry_run)


class FullGraphMigration(Migration):
    """Migration that operates within a single snapshot or across snapshots.

    Used for operations that don't involve graph structure changes,
    such as backfills or custom transformations on existing features.

    Each operation specifies which features it applies to, and Metaxy
    handles topological sorting and per-feature execution.
    """

    # Discriminator field for polymorphic deserialization
    migration_type: Literal["metaxy.migrations.models.FullGraphMigration"] = (
        "metaxy.migrations.models.FullGraphMigration"
    )

    snapshot_version: str
    from_snapshot_version: str | None = None  # Optional for cross-snapshot operations
    ops: list[dict[str, Any]]  # List of OperationConfig dicts

    def get_affected_features(self, store: "MetadataStore", project: str | None) -> list[str]:
        """Get all affected features from all operations (deduplicated).

        Args:
            store: Metadata store (not used for FullGraphMigration)
            project: Project name (not used for FullGraphMigration)

        Returns:
            List of unique feature key strings (sorted)
        """
        all_features = set()
        for op_dict in self.ops:
            op_config = OperationConfig.model_validate(op_dict)
            all_features.update(op_config.features)
        return sorted(all_features)  # Return sorted for consistency

    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> "MigrationResult":
        """Execute full graph migration with multiple operations.

        Delegates to MigrationExecutor for execution logic.

        Args:
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        from metaxy.metadata_store.system import SystemTableStorage
        from metaxy.migrations.executor import MigrationExecutor

        storage = SystemTableStorage(store)
        executor = MigrationExecutor(storage)
        return executor._execute_full_graph_migration(self, store, project, dry_run)


class MigrationStatusInfo(pydantic.BaseModel):
    """Status information for a migration computed from events and YAML definition."""

    model_config = pydantic.ConfigDict(extra="forbid")

    migration_id: str
    status: Any  # MigrationStatus enum
    expected_features: list[str]  # All features from YAML
    completed_features: list[str]  # Features completed successfully
    failed_features: dict[str, str]  # feature_key -> error_message
    pending_features: list[str]  # Features not yet started

    @property
    def features_remaining(self) -> int:
        """Number of features still needing processing (pending + failed)."""
        return len(self.pending_features) + len(self.failed_features)

    @property
    def features_total(self) -> int:
        """Total number of features in migration."""
        return len(self.expected_features)


class MigrationResult(pydantic.BaseModel):
    """Result of executing a migration."""

    model_config = pydantic.ConfigDict(extra="forbid")

    migration_id: str
    status: str  # "completed", "failed", "skipped"
    features_completed: int
    features_failed: int
    features_skipped: int  # Features skipped due to failed dependencies
    affected_features: list[str]
    errors: dict[str, str]  # feature_key -> error message
    rows_affected: int
    duration_seconds: float
    timestamp: AwareDatetime

    def summary(self) -> str:
        """Human-readable summary of migration result.

        Returns:
            Multi-line summary string
        """
        lines = [
            f"Migration: {self.migration_id}",
            f"Status: {self.status.upper()}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Duration: {self.duration_seconds:.2f}s",
            f"Features: {self.features_completed} completed, {self.features_failed} failed",
            f"Rows affected: {self.rows_affected}",
        ]

        if self.affected_features:
            lines.append("\nFeatures processed:")
            for feature in self.affected_features:
                lines.append(f"  ✓ {feature}")

        if self.errors:
            lines.append("\nErrors:")
            for feature, error in self.errors.items():
                lines.append(f"  ✗ {feature}: {error}")

        return "\n".join(lines)


# Discriminated union for automatic polymorphic deserialization
# Use Annotated with Field discriminator for type-safe deserialization
MigrationAdapter = TypeAdapter(
    Annotated[
        DiffMigration | FullGraphMigration,
        PydanticField(discriminator="migration_type"),
    ]
)
