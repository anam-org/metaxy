"""Migration system for metadata version updates."""

from metaxy.migrations.detector import detect_feature_changes
from metaxy.migrations.executor import MigrationStatus, apply_migration
from metaxy.migrations.generator import generate_migration
from metaxy.migrations.models import Migration, MigrationResult
from metaxy.migrations.ops import (
    BaseOperation,
    DataVersionReconciliation,
    MetadataBackfill,
)

__all__ = [
    "Migration",
    "MigrationResult",
    "MigrationStatus",
    "BaseOperation",
    "DataVersionReconciliation",
    "MetadataBackfill",
    "detect_feature_changes",
    "generate_migration",
    "apply_migration",
]
