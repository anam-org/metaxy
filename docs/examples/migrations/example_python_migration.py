"""Example Python migration file.

This file demonstrates how to create a migration in Python instead of YAML.
Python migrations provide more flexibility for complex migration logic.

File naming convention: {timestamp}_{name}.py
Example: 20250101_120000_example_migration.py
"""

from datetime import datetime, timezone

from metaxy.migrations.models import PythonMigration
from metaxy.migrations.ops import DataVersionReconciliation


class ExampleMigration(PythonMigration):
    """Example diff-based migration defined in Python.

    This migration would be equivalent to a YAML migration with the same fields.
    Python migrations allow for:
    - Custom validation logic
    - Programmatic field generation
    - Type-safe definitions with IDE support
    - Complex migration logic by overriding execute()
    """

    # Required fields for DiffMigration
    migration_id: str = "20250101_120000_example_migration"
    created_at: datetime = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    parent: str = "initial"
    from_snapshot_version: str = (
        "d49e39c7ad7523cd9d25e26f9f350b73c66c979abccf2f0caee84e489035ce82"
    )
    to_snapshot_version: str = (
        "155f26076028a45ebcf937aad81704f62c8d84efd204965ce5aa8d46ba8eec9e"
    )

    def build_operations(self) -> list[DataVersionReconciliation]:
        # Return actual operation objects for better IDE support
        return [
            DataVersionReconciliation(),
        ]


# Custom migration with complex logic
class CustomBackfillMigration(PythonMigration):
    """Custom migration with additional business logic.

    This demonstrates how Python migrations can include custom logic
    beyond what's possible with YAML.
    """

    migration_id: str = "20250102_120000_custom_backfill"
    created_at: datetime = datetime(2025, 1, 2, 12, 0, 0, tzinfo=timezone.utc)
    parent: str = "20250101_120000_example_migration"
    from_snapshot_version: str = (
        "155f26076028a45ebcf937aad81704f62c8d84efd204965ce5aa8d46ba8eec9e"
    )
    to_snapshot_version: str = "new_snapshot_version_here"

    def build_operations(self) -> list[DataVersionReconciliation]:
        return [
            DataVersionReconciliation(),
        ]

    def execute(self, store, project, *, dry_run: bool = False):
        """Override execute for custom migration logic.

        This allows you to:
        - Add pre/post migration hooks
        - Implement custom validation
        - Add logging and monitoring
        - Handle special cases
        """
        # Pre-migration validation
        print(f"Starting migration: {self.migration_id}")

        # Call parent implementation
        result = super().execute(store, project, dry_run=dry_run)

        # Post-migration actions
        print(f"Migration completed: {result.status}")

        return result
