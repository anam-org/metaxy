"""Data models for migration system."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

import pydantic

if TYPE_CHECKING:
    from metaxy.migrations.ops import BaseOperation


def _load_operation_class(class_path: str) -> type:
    """Dynamically import operation class.

    Args:
        class_path: Full class path (e.g., "myproject.migrations.S3Backfill")

    Returns:
        Operation class

    Raises:
        ImportError: If module or class not found
    """
    from metaxy.migrations.ops import BaseOperation

    module_path, class_name = class_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)

    if not issubclass(cls, BaseOperation):
        raise TypeError(f"{class_path} must be a subclass of BaseOperation, got {cls}")

    return cls


def _register_operation_classes(operations_data: list[dict[str, Any]]) -> None:
    """Register all operation classes for Pydantic discriminated union.

    Dynamically imports user-defined operation classes so Pydantic can
    deserialize them based on the 'type' discriminator field.

    Args:
        operations_data: List of operation dicts from YAML
    """
    for op_data in operations_data:
        op_type = op_data.get("type")
        if not op_type:
            continue

        # Try to import the class (registers it with Pydantic)
        try:
            _load_operation_class(op_type)
        except Exception:
            # Will fail later during validation if class doesn't exist
            # This is just best-effort registration
            pass


class Migration(pydantic.BaseModel):
    """Complete migration definition.

    A migration contains a list of operations that are executed in order.
    Each operation can be any subclass of BaseOperation (e.g.,
    DataVersionReconciliation, MetadataBackfill, or user-defined).

    Operations are polymorphic - deserialized based on the 'type' field.

    Migrations form an explicit dependency chain via parent_migration_id.
    This ensures migrations are applied in the correct order without
    relying on implicit timestamp-based sorting.
    """

    version: int  # Migration schema version
    id: str  # Unique ID (format: "migration_YYYYMMDD_HHMMSS")
    parent_migration_id: str | None = (
        None  # Parent migration (None for first migration)
    )
    from_snapshot_id: (
        str  # Feature graph snapshot before migration (current state in store)
    )
    to_snapshot_id: str  # Feature graph snapshot after migration (target state)
    description: str
    created_at: datetime  # When migration was created

    # Optional overrides for moved/renamed feature classes
    feature_class_overrides: dict[str, str] = pydantic.Field(default_factory=dict)

    # Operations as list of dicts - will be validated/loaded dynamically
    # Can't use Union with discriminator because we support user-defined subclasses
    operations: list[dict[str, Any]]

    def get_operations(self) -> list["BaseOperation"]:
        """Parse operations into BaseOperation objects.

        Returns:
            List of parsed operation objects

        Raises:
            ImportError: If operation type cannot be imported
            ValidationError: If operation data is invalid
        """

        parsed_operations = []

        for op_data in self.operations:
            op_type = op_data.get("type")
            if not op_type:
                raise ValueError(f"Operation missing 'type' field: {op_data}")

            # Load the operation class
            op_class = _load_operation_class(op_type)

            # Parse and validate
            op_instance = op_class.model_validate(op_data)  # type: ignore[attr-defined]
            parsed_operations.append(op_instance)

        return parsed_operations

    @staticmethod
    def from_yaml(path: str) -> "Migration":
        """Load migration from YAML file.

        Supports polymorphic operations - dynamically loads operation classes
        based on the 'type' field.

        Args:
            path: Path to YAML migration file

        Returns:
            Migration object

        Raises:
            FileNotFoundError: If file doesn't exist
            pydantic.ValidationError: If YAML structure is invalid
            ImportError: If operation type class cannot be imported
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Register operation classes for polymorphic deserialization
        _register_operation_classes(data.get("operations", []))

        # Pydantic will handle validation and parsing with discriminated unions
        return Migration.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Save migration to YAML file.

        Args:
            path: Path to write YAML file
        """
        import yaml

        # Use Pydantic's model_dump for serialization
        # Use by_alias=True to export "from" instead of "from_"
        data = self.model_dump(mode="python", by_alias=True)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


class MigrationResult(pydantic.BaseModel):
    """Result of executing a migration."""

    migration_id: str
    status: Literal["completed", "failed", "skipped"]
    operations_applied: int
    operations_failed: int
    affected_features: list[str]
    errors: dict[str, str]  # feature_key -> error message
    duration_seconds: float
    timestamp: datetime

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
            f"Operations: {self.operations_applied} applied, {self.operations_failed} failed",
            f"Affected features: {len(self.affected_features)}",
        ]

        if self.affected_features:
            lines.append("\nFeatures updated:")
            for feature in self.affected_features:
                lines.append(f"  ✓ {feature}")

        if self.errors:
            lines.append("\nErrors:")
            for feature, error in self.errors.items():
                lines.append(f"  ✗ {feature}: {error}")

        return "\n".join(lines)
