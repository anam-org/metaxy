"""Type-safe migration models with Python class paths.

Refactored migration system using:
- Python class paths for polymorphic deserialization
- Struct-based storage for graph data
- Event-based status tracking
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pydantic
from pydantic.types import AwareDatetime

if TYPE_CHECKING:
    from metaxy.graph.diff.diff_models import GraphDiff
    from metaxy.metadata_store.base import MetadataStore


class Migration(pydantic.BaseModel, ABC):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """Abstract base class for all migrations.

    Subclasses must define:
    - migration_type: Class path as Literal for polymorphic deserialization
    - execute(): Migration logic

    The migration_type field is used for storage and deserialization.
    """

    migration_id: str
    created_at: AwareDatetime

    @property
    @abstractmethod
    def migration_type(self) -> str:
        """Get migration type (Python class path).

        Returns:
            Full Python class path (e.g., "metaxy.migrations.models.DiffMigration")
        """
        pass

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

    def to_storage_dict(self) -> dict[str, Any]:
        """Convert to dict for storage.

        Returns:
            Dict with all fields including migration_type
        """
        data = self.model_dump(mode="python")
        data["migration_type"] = self.migration_type
        return data

    @staticmethod
    def from_storage_dict(data: dict[str, Any]) -> "Migration":
        """Deserialize migration from storage dict.

        Args:
            data: Dict with migration_type and other fields

        Returns:
            Migration instance of appropriate subclass

        Raises:
            ValueError: If migration_type is invalid or class not found
        """
        migration_type = data.get("migration_type")
        if not migration_type:
            raise ValueError("Missing migration_type field")

        # Dynamically import the class
        try:
            module_path, class_name = migration_type.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)

            if not issubclass(cls, Migration):
                raise TypeError(
                    f"{migration_type} must be a subclass of Migration, got {cls}"
                )

            return cls.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"Failed to load migration class {migration_type}: {e}"
            ) from e


class DiffMigration(Migration):
    """Migration based on graph diff between two snapshots.

    Migrations form a chain via parent IDs (like git commits):
    - id: Unique identifier for this migration
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

    # Stored fields - persisted to YAML in git
    parent: str  # Parent migration ID or "initial"
    from_snapshot_version: str
    to_snapshot_version: str
    ops: list[dict[str, Any]]  # Required - must explicitly specify operations

    # Private attribute for caching computed graph diff
    _graph_diff_cache: "GraphDiff | None" = pydantic.PrivateAttr(default=None)

    @pydantic.model_validator(mode="before")
    @classmethod
    def deserialize_json_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize JSON strings for ops (from storage).

        Args:
            data: Raw migration data

        Returns:
            Data with deserialized JSON fields
        """
        import json

        data = dict(data)

        # Deserialize ops from JSON string (from storage)
        if isinstance(data.get("ops"), str):
            data["ops"] = json.loads(data["ops"])

        return data

    @property
    def migration_type(self) -> str:
        """Get migration type."""
        return "metaxy.migrations.models.DiffMigration"

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

    @property
    def operations(self) -> list[Any]:
        """Get operations for this migration.

        Instantiates operations from stored ops (list of dicts with "type" field).

        Returns:
            List of operation instances
        """
        operations = []
        for op_dict in self.ops:
            op_type = op_dict.get("type")
            if not op_type:
                raise ValueError(f"Operation dict missing 'type' field: {op_dict}")
            try:
                # Dynamically import and instantiate the operation class
                module_path, class_name = op_type.rsplit(".", 1)
                module = __import__(module_path, fromlist=[class_name])
                op_cls = getattr(module, class_name)
                operations.append(op_cls())
            except Exception as e:
                raise ValueError(
                    f"Failed to instantiate operation {op_type}: {e}"
                ) from e

        return operations

    @property
    def description(self) -> str:
        """Get auto-generated description for migration.

        Returns:
            Human-readable description based on affected features count
        """
        # Note: This accesses affected_features property which needs store access
        # For display purposes, this is called after affected_features is computed
        return self.auto_description

    @property
    def auto_description(self) -> str:
        """Generate automatic description (requires store context).

        Returns:
            Human-readable description based on affected features
        """
        # This is used internally - callers should use get_description(store)
        return "Migration: snapshot reconciliation"

    def get_description(self, store: "MetadataStore", project: str | None) -> str:
        """Get description for migration.

        Args:
            store: Metadata store for computing affected features
            project: Project name for filtering snapshots

        Returns:
            Description string
        """
        affected = self.get_affected_features(store, project)
        num_features = len(affected)
        if num_features == 0:
            return "No features affected"
        elif num_features == 1:
            return f"Migration: {affected[0]}"
        else:
            return f"Migration: {num_features} features affected"

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

        graph_diff = self._get_graph_diff(store, project)

        # Get changed feature keys (root changes)
        changed_keys = {
            node.feature_key.to_string() for node in graph_diff.changed_nodes
        }

        # Also include added nodes (though they typically don't have existing data to migrate)
        for node in graph_diff.added_nodes:
            changed_keys.add(node.feature_key.to_string())

        # Build dependency map from the GraphDiff added/changed nodes
        # We need to compute downstream dependencies to find all affected features
        from metaxy.graph.diff.models import GraphData, GraphNode
        from metaxy.graph.diff.traversal import GraphWalker
        from metaxy.models.feature import FeatureGraph

        # Get the active graph to extract dependencies
        active_graph = FeatureGraph.get_active()

        # Build GraphData from active graph for dependency analysis
        nodes_dict = {}
        for feature_key, feature_cls in active_graph.features_by_key.items():
            plan = active_graph.get_feature_plan(feature_key)

            # Extract dependencies from plan
            dependencies = []
            if plan.deps:
                for dep in plan.deps:
                    dependencies.append(dep.key)

            nodes_dict[feature_key.to_string()] = GraphNode(
                key=feature_key,
                version=feature_cls.feature_version(),
                dependencies=dependencies,
            )

        to_graph_data = GraphData(
            nodes=nodes_dict, snapshot_version=self.to_snapshot_version
        )

        # Build reverse dependency map (feature -> dependents)
        dependents_map: dict[str, set[str]] = {}
        for node in to_graph_data.nodes.values():
            for dep_key in node.dependencies:
                dep_key_str = dep_key.to_string()
                if dep_key_str not in dependents_map:
                    dependents_map[dep_key_str] = set()
                dependents_map[dep_key_str].add(node.key.to_string())

        # Find all features affected (changed + their downstream)
        affected = set(changed_keys)
        queue = list(changed_keys)
        while queue:
            key_str = queue.pop(0)
            if key_str in dependents_map:
                for dependent in dependents_map[key_str]:
                    if dependent not in affected:
                        affected.add(dependent)
                        queue.append(dependent)

        # Get topological order for affected features
        walker = GraphWalker(to_graph_data)
        sorted_nodes = walker.topological_sort(nodes_to_include=affected)

        return [node.key.to_string() for node in sorted_nodes]

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
                        from_snapshot_version=self.from_snapshot_version,
                        to_snapshot_version=self.to_snapshot_version,
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
    """Migration that operates within a single snapshot.

    Used for operations that don't involve graph structure changes,
    such as backfills or custom transformations on existing features.
    """

    snapshot_version: str
    affected_features: list[str] = pydantic.Field(
        default_factory=list
    )  # Features to process
    operations: list[Any] = pydantic.Field(default_factory=list)  # Custom operations
    description: str | None = None
    metadata: dict[str, Any] = pydantic.Field(default_factory=dict)

    @pydantic.model_validator(mode="before")
    @classmethod
    def deserialize_json_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Deserialize JSON strings for operations and metadata (from storage).

        Args:
            data: Raw migration data

        Returns:
            Data with deserialized JSON fields
        """
        import json

        data = dict(data)

        # Deserialize JSON strings (from storage)
        if isinstance(data.get("operations"), str):
            data["operations"] = json.loads(data["operations"])

        if isinstance(data.get("metadata"), str):
            data["metadata"] = json.loads(data["metadata"])

        return data

    @property
    def migration_type(self) -> str:
        """Get migration type."""
        return "metaxy.migrations.models.FullGraphMigration"

    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
        """Get affected features.

        Args:
            store: Metadata store (not used for FullGraphMigration)
            project: Project name (not used for FullGraphMigration)

        Returns:
            List of feature key strings
        """
        return self.affected_features

    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> "MigrationResult":
        """Execute full graph migration.

        Subclasses should implement custom logic here.

        Args:
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult
        """
        # Base implementation: no-op
        return MigrationResult(
            migration_id=self.migration_id,
            status="completed",
            features_completed=0,
            features_failed=0,
            affected_features=[],
            errors={},
            rows_affected=0,
            duration_seconds=0.0,
            timestamp=datetime.now(timezone.utc),
        )


class CustomMigration(Migration):
    """Base class for user-defined custom migrations.

    Users can subclass this to implement completely custom migration logic.

    Example:
        class S3BackfillMigration(CustomMigration):
            s3_bucket: str
            s3_prefix: str

            @property
            def migration_type(self) -> str:
                return "myproject.migrations.S3BackfillMigration"

            def execute(self, store, *, dry_run=False):
                # Custom logic here
                ...
    """

    @property
    def migration_type(self) -> str:
        """Get migration type.

        Subclasses must override this to return their full class path.
        """
        return f"{self.__class__.__module__}.{self.__class__.__name__}"

    def get_affected_features(
        self, store: "MetadataStore", project: str | None
    ) -> list[str]:
        """Get affected features.

        Args:
            store: Metadata store (not used for CustomMigration base class)
            project: Project name (not used for CustomMigration base class)

        Returns:
            Empty list (subclasses should override)
        """
        return []

    def execute(
        self,
        store: "MetadataStore",
        project: str,
        *,
        dry_run: bool = False,
    ) -> "MigrationResult":
        """Execute custom migration.

        Subclasses must override this to implement custom logic.

        Args:
            store: Metadata store
            project: Project name for event tracking
            dry_run: If True, only validate

        Returns:
            MigrationResult

        Raises:
            NotImplementedError: If not overridden by subclass
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement execute() method"
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
