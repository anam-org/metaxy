"""Data models for migration system."""

from datetime import datetime
from typing import Literal

import pydantic


class FeatureVersionMigration(pydantic.BaseModel):
    """Migration operation for a single feature version change."""

    model_config = pydantic.ConfigDict(populate_by_name=True)

    feature_key: list[str]
    from_: str = pydantic.Field(
        alias="from",  # YAML uses "from", Python uses "from_"
    )
    to: str  # New feature_version
    change_type: str  # Type of change: "code_version", "dependencies", "unknown"
    reason: str  # Human-readable explanation
    type: Literal["feature_version_migration"] = "feature_version_migration"


class Migration(pydantic.BaseModel):
    """Complete migration definition."""

    version: int  # Migration schema version
    id: str  # Unique ID (format: "migration_YYYYMMDD_HHMMSS")
    description: str
    created_at: datetime  # When migration was created
    operations: list[FeatureVersionMigration]

    @staticmethod
    def from_yaml(path: str) -> "Migration":
        """Load migration from YAML file.

        Args:
            path: Path to YAML migration file

        Returns:
            Migration object

        Raises:
            FileNotFoundError: If file doesn't exist
            pydantic.ValidationError: If YAML structure is invalid
        """
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        # Pydantic will handle validation and parsing
        return Migration.model_validate(data)

    def to_yaml(self, path: str) -> None:
        """Save migration to YAML file.

        Args:
            path: Path to write YAML file
        """
        import yaml

        # Use Pydantic's model_dump for serialization
        data = self.model_dump(mode="python")

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
