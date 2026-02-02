"""Tests for migration example."""

import os
import subprocess
import sys
from pathlib import Path

from metaxy_testing import ExternalMetaxyProject


def test_pipeline(tmp_path, snapshot):
    """Test that the migration workflow example runs successfully.

    This test orchestrates the migration workflow:
    1. metaxy push (STAGE=1) - record feature graph v1
    2. Run pipeline with STAGE=1
    3. Deploy new code (STAGE=2) and push to record v2 snapshot
    4. Generate migration (compare v1 vs v2, both in store)
    5. Apply migration (reconcile existing data from v1 to v2)
    6. Run pipeline with STAGE=2 - should show no recomputes (migration worked)
    """
    example_dir = Path("examples/example-migration")
    project = ExternalMetaxyProject(example_dir)

    # Use a temp directory for test migrations (don't modify example source code)
    test_migrations_dir = tmp_path / ".metaxy" / "migrations"
    test_migrations_dir.mkdir(parents=True, exist_ok=True)

    # Reference migration file in the example (source of truth - never modified)
    reference_migration_path = example_dir / ".metaxy/migrations/example_migration.yaml"

    # Use a test database instead of the default one
    test_db = tmp_path / "example_migration.db"

    # Start with parent environment to preserve Nix paths
    base_env = os.environ.copy()
    # Override specific values for the test using proper Pydantic environment variables
    base_env.update(
        {
            # Override the root path for the dev store (DeltaMetadataStore uses root_path)
            "METAXY_STORES__DEV__CONFIG__ROOT_PATH": str(test_db),
            # Override the migrations directory
            "METAXY_MIGRATIONS_DIR": str(test_migrations_dir),
            # Ensure HOME is set for DuckDB extension installation
            "HOME": os.environ.get("HOME", str(tmp_path)),
        }
    )

    # Step 0: Setup upstream data
    result = subprocess.run(
        [sys.executable, "-m", f"{project.package_name}.setup_data"],
        capture_output=True,
        text=True,
        timeout=10,
        env={**base_env},
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Setup failed: {result.stderr}"
    print(result.stdout)

    # Step 1: metaxy push with STAGE=1
    result = project.run_cli(
        ["push"],
        env={**base_env, "STAGE": "1"},
    )
    assert result.returncode == 0, f"Push v1 failed: {result.stderr}"

    # After the stream separation changes, stdout contains ONLY the raw snapshot ID hash
    v1_snapshot = result.stdout.strip()
    # v1_snapshot should be the expected hash from the snapshot data
    assert v1_snapshot == "a6b7f865"

    # Step 2: Run pipeline with STAGE=1
    result = subprocess.run(
        [sys.executable, "-m", f"{project.package_name}.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**base_env, "STAGE": "1"},
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Stage 1 failed: {result.stderr}\nstdout: {result.stdout}"
    print(result.stdout)
    assert "Pipeline STAGE=1" in result.stdout
    assert "[OK] Stage 1 pipeline complete!" in result.stdout

    # Step 3: Push STAGE=2 snapshot (simulates CD after code deployment)
    result = project.run_cli(
        ["push"],
        env={**base_env, "STAGE": "2"},
    )
    assert result.returncode == 0, f"Push v2 failed: {result.stderr}"

    # stdout contains ONLY the raw snapshot ID hash
    v2_snapshot = result.stdout.strip()
    print(f"V2 snapshot: {v2_snapshot}")

    # Step 4: Use reference migration (don't generate, just use the existing one)
    import shutil

    import yaml

    # Copy reference migration to temp test location
    reference_migration_path = example_dir / ".metaxy/migrations/example_migration.yaml"
    assert reference_migration_path.exists(), f"Reference migration not found at {reference_migration_path}"

    # Copy to test migrations dir for apply command
    test_migration_path = test_migrations_dir / "example_migration.yaml"
    shutil.copy(reference_migration_path, test_migration_path)

    # Load reference for validation
    with open(reference_migration_path) as f:
        reference_yaml = yaml.safe_load(f)

    print(f"\nUsing reference migration from {reference_migration_path}:")
    print(yaml.safe_dump(reference_yaml, sort_keys=False))

    # Snapshot test for deterministic fields (exclude id, created_at which have timestamps)
    snapshot_migration = {k: v for k, v in reference_yaml.items() if k not in ["id", "created_at"]}

    assert snapshot_migration == snapshot

    # Step 5: Apply migration (use STAGE=2 env)
    result = project.run_cli(
        ["migrations", "apply"],
        env={**base_env, "STAGE": "2"},
    )
    assert result.returncode == 0, f"Migration apply failed: {result.stderr}\nstdout: {result.stdout}"
    print(result.stderr)  # Migration status messages go to stderr now

    # Verify migration was applied successfully (check stderr for status messages)
    assert "Migration completed" in result.stderr or "completed" in result.stderr.lower(), (
        f"Expected migration completion message in stderr, got: {result.stderr}"
    )

    # Step 6: Run pipeline with STAGE=2 after migration
    # This should show NO recomputes because migration reconciled the field_provenance
    result = subprocess.run(
        [sys.executable, "-m", f"{project.package_name}.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env={**base_env, "STAGE": "2"},
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Stage 2 (after migration) failed: {result.stderr}\nstdout: {result.stdout}"
    print("\n--- Stage 2 after migration ---")
    print(result.stdout)
    assert "Pipeline STAGE=2" in result.stdout
    assert "[OK] Stage 2 pipeline complete!" in result.stdout

    # The migration should have reconciled the child feature's field_provenance,
    # so no recomputation should be needed
    assert "No changes detected (idempotent or migration worked correctly)" in result.stdout, (
        "Migration should have reconciled field_provenance, but child was recomputed"
    )
