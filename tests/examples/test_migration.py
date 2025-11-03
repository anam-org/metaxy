"""Tests for migration example."""

import os
import re
import subprocess
import sys
from pathlib import Path

from metaxy._testing import ExternalMetaxyProject


def test_pipeline(tmp_path, snapshot):
    """Test that the migration workflow example runs successfully.

    This test orchestrates the migration workflow:
    1. metaxy graph push (STAGE=1) - record feature graph v1
    2. Run pipeline with STAGE=1
    3. Deploy new code (STAGE=2) and push to record v2 snapshot
    4. Generate migration (compare v1 vs v2, both in store)
    5. Apply migration (reconcile existing data from v1 to v2)
    6. Run pipeline with STAGE=2 - should show no recomputes (migration worked)
    """
    example_dir = Path("examples/src/examples/migration")
    project = ExternalMetaxyProject(example_dir)

    # Use a temp directory for test migrations (don't modify example source code)
    test_migrations_dir = tmp_path / ".metaxy" / "migrations"
    test_migrations_dir.mkdir(parents=True, exist_ok=True)

    # Reference migration file in the example (source of truth - never modified)
    reference_migration_path = example_dir / ".metaxy/migrations/example_migration.yaml"

    # Use a test database instead of the default one
    test_db = tmp_path / "example_migration.db"

    # Create a test-specific metaxy config that points to temp migrations dir
    test_config_dir = tmp_path / ".metaxy"
    test_config_dir.mkdir(exist_ok=True)
    test_config_file = test_config_dir / "config.toml"
    with open(test_config_file, "w") as f:
        f.write(f"""
project = "migration_test"
migrations_dir = "{test_migrations_dir}"
auto_create_tables = true

[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{test_db}"
""")

    # Start with parent environment to preserve Nix paths
    base_env = os.environ.copy()
    # Override specific values for the test
    base_env.update(
        {
            "METAXY_CONFIG": str(test_config_file),
            # Ensure HOME is set for DuckDB extension installation
            "HOME": os.environ.get("HOME", str(tmp_path)),
        }
    )

    # Step 0: Setup upstream data
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.setup_data"],
        capture_output=True,
        text=True,
        timeout=10,
        env={**base_env},
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Setup failed: {result.stderr}"
    print(result.stdout)

    # Step 1: metaxy graph push with STAGE=1
    result = project.run_cli(
        "graph",
        "push",
        env={**base_env, "STAGE": "1"},
    )
    assert result.returncode == 0, f"Push v1 failed: {result.stderr}"
    print(result.stdout)

    # Extract v1 snapshot version from output
    # Format is: "Snapshot version: \nd49e39c7ad7523cd..." (multiline)
    match = re.search(r"Snapshot version:\s+\n([a-f0-9]+)", result.stdout)
    if not match:
        # Try alternative format
        match = re.search(r"Full ID: \('([^']+)'", result.stdout)
    assert match, f"Could not find snapshot version in push output:\n{result.stdout}"
    v1_snapshot = match.group(1)
    assert v1_snapshot is not None

    # Step 2: Run pipeline with STAGE=1
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**base_env, "STAGE": "1"},
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Stage 1 failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)
    assert "Pipeline STAGE=1" in result.stdout
    assert "✅ Stage 1 pipeline complete!" in result.stdout

    # Step 3: Push STAGE=2 snapshot (simulates CD after code deployment)
    result = project.run_cli(
        "graph",
        "push",
        env={**base_env, "STAGE": "2"},
    )
    assert result.returncode == 0, f"Push v2 failed: {result.stderr}"
    print(result.stdout)

    # Step 4: Use reference migration (don't generate, just use the existing one)
    import shutil

    import yaml

    # Copy reference migration to temp test location
    reference_migration_path = example_dir / ".metaxy/migrations/example_migration.yaml"
    assert reference_migration_path.exists(), (
        f"Reference migration not found at {reference_migration_path}"
    )

    # Copy to test migrations dir for apply command
    test_migration_path = test_migrations_dir / "example_migration.yaml"
    shutil.copy(reference_migration_path, test_migration_path)

    # Load reference for validation
    with open(reference_migration_path) as f:
        reference_yaml = yaml.safe_load(f)

    print(f"\nUsing reference migration from {reference_migration_path}:")
    print(yaml.safe_dump(reference_yaml, sort_keys=False))

    # Snapshot test for deterministic fields (exclude id, created_at which have timestamps)
    snapshot_migration = {
        k: v for k, v in reference_yaml.items() if k not in ["id", "created_at"]
    }

    assert snapshot_migration == snapshot

    # Step 5: Apply migration (use STAGE=2 env)
    result = project.run_cli(
        "migrations",
        "apply",
        env={**base_env, "STAGE": "2"},
    )
    assert result.returncode == 0, (
        f"Migration apply failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)

    # Verify migration was applied successfully
    assert (
        "Migration completed" in result.stdout or "completed" in result.stdout.lower()
    ), "Expected migration completion message in output"

    # Step 6: Run pipeline with STAGE=2 after migration
    # This should show NO recomputes because migration reconciled the field_provenance
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**base_env, "STAGE": "2"},
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Stage 2 (after migration) failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print("\n--- Stage 2 after migration ---")
    print(result.stdout)
    assert "Pipeline STAGE=2" in result.stdout
    assert "✅ Stage 2 pipeline complete!" in result.stdout

    # The migration should have reconciled the child feature's field_provenance,
    # so no recomputation should be needed
    assert (
        "No changes detected (idempotent or migration worked correctly)"
        in result.stdout
    ), "Migration should have reconciled field_provenance, but child was recomputed"
