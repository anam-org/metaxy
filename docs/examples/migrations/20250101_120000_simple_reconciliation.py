"""Example 1: Simple DiffMigration with DataVersionReconciliation.

This is the simplest form of Python migration, demonstrating how to define
a DiffMigration that reconciles metadata when feature definitions change but
the underlying computation remains the same.

Use Case:
---------
You refactored your feature definitions (e.g., improved type annotations,
reorganized imports, clarified dependencies) but the actual computation logic
didn't change. The metadata (field provenance) needs to be updated to reflect
the new feature versions without recomputing all the data.

When to Use:
------------
- Code refactoring that doesn't change computation results
- Dependency graph updates (more precise field-level dependencies)
- Field schema improvements (renaming, restructuring)
- Type annotation improvements
- Import reorganization

When NOT to Use:
----------------
- Algorithm changes that affect output (re-run pipeline instead)
- Bug fixes that change results (re-run pipeline instead)
- New model versions (re-run pipeline instead)

Migration Anatomy:
------------------
- migration_id: Unique identifier (timestamp format: YYYYMMDD_HHMMSS)
- created_at: Timestamp when migration was created
- parent: ID of parent migration ("initial" for first migration)
- from_snapshot_version: Source snapshot version (before changes)
- to_snapshot_version: Target snapshot version (after changes)
- ops: List of operations (here: DataVersionReconciliation)

Example Usage:
--------------
Place this file in .metaxy/migrations/ directory and run:
    metaxy migrations apply

The migration will:
1. Load the feature graph from to_snapshot_version
2. Compute affected features (root changes + downstream)
3. For each affected feature:
   - Load existing metadata with old feature_version
   - Recalculate field_provenance based on new feature definition
   - Write new rows with updated feature_version and snapshot_version
4. Preserve all user metadata columns (immutable, copy-on-write)
"""

from datetime import datetime, timezone

from metaxy.migrations.models import PythonMigration
from metaxy.migrations.ops import DataVersionReconciliation


class SimpleReconciliationMigration(PythonMigration):
    """Simple reconciliation when feature definitions change.

    This migration demonstrates the most common use case: updating metadata
    after code refactoring that doesn't change computation.

    The affected features are automatically computed from the snapshot diff.
    The DataVersionReconciliation operation handles all the metadata updates.
    """

    # Required: Unique migration ID (must match filename prefix for convention)
    migration_id: str = "20250101_120000_simple_reconciliation"

    # Required: Timestamp when migration was created
    created_at: datetime = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Required: Parent migration ID (use "initial" for first migration)
    parent: str = "initial"

    # Required: Source snapshot version (run `metaxy graph push` in CD to record)
    # This should be the snapshot version BEFORE your code changes
    from_snapshot_version: str = (
        "abc123..."  # Replace with actual snapshot version from your store
    )

    # Required: Target snapshot version (the current code state)
    # This should be the snapshot version AFTER your code changes
    # You can get this by running `metaxy graph push` after making changes
    to_snapshot_version: str = (
        "def456..."  # Replace with actual snapshot version from your store
    )

    # Define operations in Python for IDE/type safety
    # DataVersionReconciliation updates field_provenance for all affected features
    def build_operations(self) -> list[DataVersionReconciliation]:
        return [
            DataVersionReconciliation(),
        ]


# Notes:
# ------
# 1. Migration IDs must be unique across all migration files (YAML and Python)
# 2. The filename should match the migration ID prefix for easy identification
# 3. You can define the migration as a class (shown here) or instantiate it
# 4. All fields can be defined as class attributes (Pydantic supports this)
# 5. The migration file must define exactly ONE Migration subclass
# 6. Affected features are computed automatically from snapshot diff
# 7. Operations apply to all affected features in topological order

# Alternative: Instantiate the migration directly
# -----------------------------------------------
# If you prefer, you can instantiate the migration instead of defining a class:
#
# simple_migration = DiffMigration(
#     migration_id="20250101_120000_simple_reconciliation",
#     created_at=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
#     parent="initial",
#     from_snapshot_version="abc123...",
#     to_snapshot_version="def456...",
#     ops=[
#         {"type": "metaxy.migrations.ops.DataVersionReconciliation"}
#     ],
# )
#
# However, using a class allows you to add custom methods and better type hints.
