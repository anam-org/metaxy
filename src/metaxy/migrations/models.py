"""Type-safe migration models with Python class paths.

Refactored migration system using:
- Python class paths for polymorphic deserialization via discriminated unions
- Struct-based storage for graph data
- Event-based status tracking
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Annotated, Any, Literal

import pydantic
from pydantic import Field as PydanticField
from pydantic import TypeAdapter
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
    """

    model_config = pydantic.ConfigDict(extra="allow")

    type: str
    features: list[str] = pydantic.Field(default_factory=list)


class Migration(pydantic.BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Abstract base class for all migrations.

    Subclasses must define:
    - migration_type: Literal field with class path for discriminated union deserialization
    - execute(): Migration logic
    - get_affected_features(): Return list of affected feature keys

    The migration_type field is used as a discriminator for automatic polymorphic deserialization.

    All migrations form a chain via parent IDs (like git commits):
    - parent: ID of parent migration ("initial" for first migration)
    """

    migration_id: str
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
    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
        """Get list of affected feature keys in topological order.

        Args:
            store: Metadata store for computing affected features
            project: Project name for filtering snapshots

        Returns:
            List of feature key strings
        """
        pass

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
        # Check if this migration has an ops field (using getattr to avoid type errors)
        ops = getattr(self, "ops", None)
        if ops is None:
            return []

        operations = []
        for op_dict in ops:
            # Validate structure has required fields
            op_config = OperationConfig.model_validate(op_dict)

            try:
                # Dynamically import and instantiate the operation class
                module_path, class_name = op_config.type.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                op_cls = getattr(module, class_name)

                # Pass the entire dict to the operation class (which inherits from BaseSettings)
                # BaseSettings will extract the fields it needs and read from env vars
                operation = op_cls.model_validate(op_dict)
                operations.append(operation)
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate operation {op_config.type}: {e}"
                ) from e

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
    migration_type: Literal["metaxy.migrations.models.DiffMigration"] = (
        "metaxy.migrations.models.DiffMigration"
    )

    # Stored fields - persisted to YAML in git
    from_snapshot_version: str
    to_snapshot_version: str
    ops: list[dict[str, Any]]  # Required - must explicitly specify operations

    # Private attribute for caching computed graph diff
    _graph_diff_cache: "GraphDiff | None" = pydantic.PrivateAttr(default=None)

    def _get_graph_diff(
        self, store: "MetadataStore", project: str | None
    ) -> "GraphDiff":
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

    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
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

    def compute_graph_diff(
        self, store: "MetadataStore", project: str | None
    ) -> "GraphDiff":
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
        from_snapshot_data = differ.load_snapshot_data(
            store, self.from_snapshot_version
        )

        # Try to load to_snapshot from store, if it doesn't exist use active graph
        try:
            to_snapshot_data = differ.load_snapshot_data(
                store, self.to_snapshot_version
            )
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

        Process:
        1. Execute each operation in the operations list
        2. For each operation:
           - Check if feature already completed (resume support)
           - Execute operation
           - Record event
        3. Return result

        Args:
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        from metaxy.metadata_store.system_tables import SystemTableStorage

        storage = SystemTableStorage(store)
        start_time = datetime.now(timezone.utc)

        if not dry_run:
            # Write started event
            storage.write_event(self.migration_id, "started", project)

        affected_features_list = []
        errors = {}
        rows_affected_total = 0

        # Execute operations (currently only DataVersionReconciliation is supported)
        from metaxy.migrations.ops import DataVersionReconciliation

        # Get affected features (computed on-demand)
        affected_features_to_process = self.get_affected_features(store, project)

        if len(self.operations) == 1 and isinstance(
            self.operations[0], DataVersionReconciliation
        ):
            # DataVersionReconciliation applies to all affected features
            op = self.operations[0]

            for feature_key_str in affected_features_to_process:
                # Check if already completed (resume support)
                if not dry_run and storage.is_feature_completed(
                    self.migration_id, feature_key_str, project
                ):
                    affected_features_list.append(feature_key_str)
                    continue

                # Log feature started
                if not dry_run:
                    storage.write_event(
                        self.migration_id,
                        "feature_started",
                        project,
                        feature_key=feature_key_str,
                    )

                try:
                    # Execute operation for this feature
                    rows_affected = op.execute_for_feature(
                        store,
                        feature_key_str,
                        snapshot_version=self.to_snapshot_version,
                        from_snapshot_version=self.from_snapshot_version,
                        dry_run=dry_run,
                    )

                    # Log feature completed
                    if not dry_run:
                        storage.write_event(
                            self.migration_id,
                            "feature_completed",
                            project,
                            feature_key=feature_key_str,
                            rows_affected=rows_affected,
                        )

                    affected_features_list.append(feature_key_str)
                    rows_affected_total += rows_affected

                except Exception as e:
                    error_msg = str(e)
                    errors[feature_key_str] = error_msg

                    # Log feature failed
                    if not dry_run:
                        storage.write_event(
                            self.migration_id,
                            "feature_completed",
                            project,
                            feature_key=feature_key_str,
                            error_message=error_msg,
                        )

                    continue
        else:
            # Future: Support other operation types here
            raise NotImplementedError(
                "Only DataVersionReconciliation is currently supported"
            )

        # Determine status
        if dry_run:
            status = "skipped"
        elif len(errors) == 0:
            status = "completed"
            if not dry_run:
                storage.write_event(self.migration_id, "completed", project)
        else:
            status = "failed"
            if not dry_run:
                storage.write_event(self.migration_id, "failed", project)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return MigrationResult(
            migration_id=self.migration_id,
            status=status,
            features_completed=len(affected_features_list),
            features_failed=len(errors),
            affected_features=affected_features_list,
            errors=errors,
            rows_affected=rows_affected_total,
            duration_seconds=duration,
            timestamp=start_time,
        )


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

    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
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

        Process:
        1. For each operation in ops:
           a. Parse OperationConfig
           b. Instantiate operation class from type
           c. Sort operation's features topologically using FeatureGraph
           d. For each feature in topological order:
              - Check if already completed (resume support)
              - Execute operation.execute_for_feature()
              - Record progress event
        2. Return combined MigrationResult

        Args:
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        from metaxy.metadata_store.system_tables import SystemTableStorage
        from metaxy.migrations.ops import BaseOperation
        from metaxy.models.feature import FeatureGraph
        from metaxy.models.types import FeatureKey

        storage = SystemTableStorage(store)
        start_time = datetime.now(timezone.utc)

        if not dry_run:
            storage.write_event(self.migration_id, "started", project)

        affected_features_list = []
        errors = {}
        rows_affected_total = 0

        # Get active graph for topological sorting
        graph = FeatureGraph.get_active()

        # Execute each operation
        for op_index, op_dict in enumerate(self.ops):
            # Parse operation config
            op_config = OperationConfig.model_validate(op_dict)

            # Instantiate operation class
            try:
                module_path, class_name = op_config.type.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                op_cls = getattr(module, class_name)

                if not issubclass(op_cls, BaseOperation):
                    raise TypeError(
                        f"{op_config.type} must be a subclass of BaseOperation"
                    )

                # Instantiate operation with full dict (flat structure)
                operation = op_cls.model_validate(op_dict)

            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate operation {op_config.type}: {e}"
                ) from e

            # Sort features topologically
            feature_keys = [FeatureKey(fk.split("/")) for fk in op_config.features]
            sorted_features = graph.topological_sort_features(feature_keys)

            # Execute for each feature in topological order
            for feature_key_obj in sorted_features:
                feature_key_str = feature_key_obj.to_string()

                # Check if already completed (resume support)
                if not dry_run and storage.is_feature_completed(
                    self.migration_id, feature_key_str, project
                ):
                    affected_features_list.append(feature_key_str)
                    continue

                # Log feature started
                if not dry_run:
                    storage.write_event(
                        self.migration_id,
                        "feature_started",
                        project,
                        feature_key=feature_key_str,
                    )

                try:
                    # Execute operation for this feature
                    rows_affected = operation.execute_for_feature(
                        store,
                        feature_key_str,
                        snapshot_version=self.snapshot_version,
                        from_snapshot_version=self.from_snapshot_version,
                        dry_run=dry_run,
                    )

                    # Log feature completed
                    if not dry_run:
                        storage.write_event(
                            self.migration_id,
                            "feature_completed",
                            project,
                            feature_key=feature_key_str,
                            rows_affected=rows_affected,
                        )

                    affected_features_list.append(feature_key_str)
                    rows_affected_total += rows_affected

                except Exception as e:
                    error_msg = str(e)
                    errors[feature_key_str] = error_msg

                    # Log feature failed
                    if not dry_run:
                        storage.write_event(
                            self.migration_id,
                            "feature_failed",
                            project,
                            feature_key=feature_key_str,
                            error_message=error_msg,
                        )

                    continue

        # Determine status
        if dry_run:
            status = "skipped"
        elif len(errors) == 0:
            status = "completed"
            if not dry_run:
                storage.write_event(self.migration_id, "completed", project)
        else:
            status = "failed"
            if not dry_run:
                storage.write_event(self.migration_id, "failed", project)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        return MigrationResult(
            migration_id=self.migration_id,
            status=status,
            features_completed=len(affected_features_list),
            features_failed=len(errors),
            affected_features=affected_features_list,
            errors=errors,
            rows_affected=rows_affected_total,
            duration_seconds=duration,
            timestamp=start_time,
        )


class MigrationResult(pydantic.BaseModel):
    """Result of executing a migration."""

    migration_id: str
    status: str  # "completed", "failed", "skipped"
    features_completed: int
    features_failed: int
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
