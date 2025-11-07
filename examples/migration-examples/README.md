# Python Migration Examples

This directory hosts runnable samples that exercise the Python migration APIs.
Each file focuses on a single pattern (reconciliation, custom backfills, dynamic operations, validation-first workflows, reusable `MetadataBackfill` operations, etc.).
The examples double as integration tests—see `docs/learn/migration-formats.md` for the full walkthrough, decision matrix, and troubleshooting guide.

## How to Use the Examples

1. **Read the docs:** Start with the [Migration Formats guide](../../docs/learn/migration-formats.md) to understand when to choose YAML vs Python and how each file fits into the bigger picture.
2. **Inspect the code:** Open any file under this directory; each migration is fully documented and can be copied into your project’s `.metaxy/migrations/` folder.
3. **Run the tests:** Execute `uv run pytest tests/migrations/test_python_migration_execution.py` (or the full suite) to see the examples in action.

> **Type checking:** Some examples intentionally import optional dependencies (e.g., `boto3`, `psycopg2`) or use dynamic patterns that may trigger warnings. These are guarded with `pyright` ignores purely for demonstration.
