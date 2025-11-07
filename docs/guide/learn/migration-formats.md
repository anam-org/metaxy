# Migration Formats

Metaxy migrations live alongside your code in `.metaxy/migrations/` and can be written in either YAML (`.yaml`) or Python (`.py`).
Both formats participate in the same parent chain, so you can freely mix them within the same project.

- **YAML migrations** are best for straightforward `DataVersionReconciliation` workflows where no additional logic or validation is required.
- **Python migrations** provide a full programming surface for conditional logic, helper methods, and custom operations.
  The `PythonMigration` base class lets you define migrations as normal Python classes and return strongly typed operations:

```python
from metaxy.migrations import PythonMigration, DataVersionReconciliation


class Migration_2024_01_15_refactor_features(PythonMigration):
    migration_id = "20240115_120000_refactor_features"
    parent = "20240110_080000_initial"
    from_snapshot_version = "abc123..."
    to_snapshot_version = "def456..."

    def build_operations(self):
        return [
            DataVersionReconciliation(),
        ]
```

When a migration is defined in Python, Metaxy automatically serializes the returned operations so that you can still export the migration to YAML for reviews or documentation via `metaxy migrations export <migration_id> --output review.yaml`.
