"""Migration system for metadata version updates."""

from metaxy.migrations.detector import FeatureChange, detect_feature_changes
from metaxy.migrations.executor import MigrationResult, apply_migration
from metaxy.migrations.generator import generate_migration
from metaxy.migrations.models import (
    FeatureVersionMigration,
    Migration,
)

__all__ = [
    "Migration",
    "FeatureVersionMigration",
    "MigrationResult",
    "FeatureChange",
    "detect_feature_changes",
    "generate_migration",
    "apply_migration",
]
