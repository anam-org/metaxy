# Migration Formats

Metaxy stores migrations alongside your repository in `.metaxy/migrations/`.
The migration chain can mix both YAML (`.yaml`) and Python (`.py`) files.
Each file contributes a single migration node linked through the `parent` field.
This page explains when to reach for each format, documents the Python migration API, and links to fully worked examples under `docs/examples/migrations/`.

> **Note:** The example files are reference implementations and may intentionally demonstrate advanced techniques (dynamic imports, optional dependencies such as `boto3`/`psycopg`, etc.).
> Treat them as educational references—you can adopt lighter-weight patterns in production.

## Choosing Between YAML and Python

### Use YAML Migrations When

- The standard `DataVersionReconciliation` operation is sufficient.
- Operations are known at migration creation time.
- No custom validation or external dependencies are involved.
- You prefer a simple, declarative artifact for code-review.

### Use Python Migrations When

- You need custom execution logic or helper methods.
- Operations must be generated dynamically at runtime.
- You’re validating configuration (Pydantic validators, connectivity checks).
- External systems (S3, databases, APIs) must be queried inside the migration.
- You’re building reusable operation classes (e.g., subclasses of `MetadataBackfill`).
- You want richer logging, conditional logic, or alternative rollback strategies.

## Python Migration Quick Start

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

Python subclasses can either:

1. Set `ops = [{"type": "..."}]` directly (mirrors YAML), or
2. Override `build_operations()` / legacy `operations()` to return actual operation objects. The base class serializes them back into the `ops` list so the migration can still be exported to YAML (via `metaxy migrations export`).

### File Naming Convention

```
YYYYMMDD_HHMMSS_description.py
```

The migration ID inside the file should match the filename prefix, e.g., `20250101_120000_simple_reconciliation.py`.

## Example Library

The `docs/examples/migrations/` directory contains canonical examples for common scenarios:

| Example                        | File                                       | Highlights                                                                                                                      |
| ------------------------------ | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| 1. Simple reconciliation       | `20250101_120000_simple_reconciliation.py` | Minimal `PythonMigration` returning `DataVersionReconciliation` operations.                                                     |
| 2. S3 backfill                 | `20250101_130000_custom_backfill.py`       | `CustomMigration` loading metadata from S3, filtering, and writing to the store.                                                |
| 3. Dynamic operations          | `20250101_140000_dynamic_operations.py`    | `PythonMigration` that inspects the feature graph/config to build operation lists on the fly.                                   |
| 4. Validation-heavy migration  | `20250101_150000_custom_validation.py`     | Demonstrates Pydantic field/model validators, external dependency checks, and clear error reporting.                            |
| 5. MetadataBackfill operations | `20250101_160000_metadata_backfill.py`     | Shows how to implement reusable operation classes (subclassing `MetadataBackfill`) and invoke them from YAML/Python migrations. |
| Reference file                 | `example_python_migration.py`              | Two miniature migrations showing the basic class structure and custom `execute()` overrides.                                    |

Each example includes docstrings describing the use case, key concepts, and when to prefer the pattern.

## Migration + Operation Hierarchy

```
Migration (abstract base)
├── DiffMigration
│   └── Snapshot-diff driven, supports DataVersionReconciliation and custom ops.
├── CustomMigration
│   └── User-defined `execute()` logic; no automatic affected-features computation.
└── FullGraphMigration
    └── Operates within a single snapshot or on custom metadata workflows.
```

```
BaseOperation
├── DataVersionReconciliation (built-in, no user config)
└── MetadataBackfill (abstract)
    └── User-defined operations (S3 backfills, API ingestion, etc.)
```

## Common Patterns

### Dynamic Operation Lists

```python
def build_operations(self) -> list[dict[str, str]]:
    operations = []
    config = load_config()
    if config.get("enable_validation"):
        operations.append({"type": "myproject.ops.CustomValidation"})
    return operations
```

### External Data Loading

```python
def execute(self, store, project, *, dry_run: bool = False):
    external = load_external_data()
    diff = store.resolve_update(feature_cls, samples=external.select("sample_uid"))
    to_write = external.join(diff.added, on="sample_uid")
    if not dry_run:
        store.write_metadata(feature_cls, to_write)
    return MigrationResult(...)
```

### Validation

```python
@field_validator("url")
@classmethod
def validate_url(cls, value: str) -> str:
    if not value.startswith("https://"):
        raise ValueError("URL must be HTTPS")
    return value


@model_validator(mode="after")
def validate_connectivity(self):
    if not ping(self.url):
        raise ValueError(f"Cannot reach {self.url}")
    return self
```

### Filtering Affected Features

```python
def get_affected_features(self, store, project):
    targets = super().get_affected_features(store, project)
    graph = FeatureGraph.get_active()
    return [key for key in targets if has_upstream_dependencies(graph, key)]
```

## Running and Inspecting Migrations

```bash
# List chain status
metaxy migrations status

# Dry-run a specific migration
metaxy migrations apply --dry-run 20250101_120000_simple_reconciliation

# Apply all pending migrations
metaxy migrations apply
```

Always run with `--dry-run` first when developing new logic, and monitor system-table status via `metaxy migrations status`.

## Troubleshooting Checklist

- **Migration not found:** ensure the file resides in `.metaxy/migrations/`, defines exactly one `Migration` subclass, and that its `migration_id` matches the filename prefix.
- **Validation errors:** Pydantic errors pinpoint invalid fields—verify types, ranges, and connectivity tests.
- **Import errors:** install optional dependencies (`boto3`, `psycopg2`, etc.) or guard them with `try/except ImportError`.
- **Execution failures:** confirm features exist in the active graph, check that upstream metadata is available, and review migration logs (system tables) for per-feature errors.

## Additional Resources

- **Operations reference:** `src/metaxy/migrations/ops.py`
- **Migration models / base classes:** `src/metaxy/migrations/models.py`
- **CLI guide:** `docs/reference/cli.md` (`metaxy migrations ...` commands)
- **Tests:** The suites under `tests/migrations/` provide extensive runnable examples.
