"""Tests for migration example."""

import os
import subprocess
import sys
from pathlib import Path


def test_pipeline(tmp_path, snapshot):
    """Test that the migration workflow example runs successfully.

    This test orchestrates the migration workflow:
    1. metaxy push (STAGE=1) - record feature graph v1
    2. Run pipeline with STAGE=1
    3. Switch to STAGE=2 and generate migration
    4. Apply migration
    5. metaxy push (STAGE=2) - record feature graph v2
    6. Run pipeline with STAGE=2 - should show no recomputes (migration worked)
    """
    example_dir = Path("examples/src/examples/migration")

    # Set up environment
    env = os.environ.copy()
    test_db = tmp_path / "example_migration.db"
    env["METAXY_STORES__DEV__CONFIG__DATABASE"] = str(test_db)

    # Step 0: Setup upstream data
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.setup_data"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Setup failed: {result.stderr}"
    print(result.stdout)

    # Step 1: metaxy push with STAGE=1
    env["STAGE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.push"],
        capture_output=True,
        text=True,
        timeout=10,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Push v1 failed: {result.stderr}"
    print(result.stdout)

    # Step 2: Run pipeline with STAGE=1
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Stage 1 failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)
    assert "Pipeline STAGE=1" in result.stdout
    assert "✅ Stage 1 pipeline complete!" in result.stdout

    # Step 3: Generate migration (STAGE=2)
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    migration_env = env.copy()
    migration_env["STAGE"] = "2"  # Load v2 features for migration generation
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "metaxy.cli.app",
            "migrations",
            "generate",
            "--migrations-dir",
            str(migrations_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=migration_env,
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Migration generation failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)

    # Check migration file was created
    migration_files = list(migrations_dir.glob("*.yaml"))
    assert len(migration_files) > 0, "No migration file generated"

    # Read and snapshot the migration file (excluding non-deterministic fields)
    import yaml

    with open(migration_files[0]) as f:
        migration_data = yaml.safe_load(f)

    # Remove non-deterministic fields for snapshotting
    snapshot_data = migration_data.copy()
    snapshot_data.pop("id", None)  # Contains timestamp
    snapshot_data.pop("created_at", None)  # Timestamp
    snapshot_data.pop("from_snapshot_id", None)  # Hash
    snapshot_data.pop("to_snapshot_id", None)  # Hash

    # Snapshot the deterministic parts
    assert snapshot_data == snapshot

    # Step 4: Apply migration (use STAGE=2 env)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "metaxy.cli.app",
            "migrations",
            "apply",
            "--migrations-dir",
            str(migrations_dir),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        env=migration_env,  # Use migration_env which has STAGE=2
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Migration apply failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print(result.stdout)

    # Step 5: metaxy push with STAGE=2 (record new feature graph)
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.push"],
        capture_output=True,
        text=True,
        timeout=10,
        env=migration_env,
        cwd=example_dir,
    )
    assert result.returncode == 0, f"Push v2 failed: {result.stderr}"
    print(result.stdout)

    # Step 6: Run pipeline with STAGE=2 after migration
    # This should show NO recomputes because migration reconciled the data_versions
    env["STAGE"] = "2"
    result = subprocess.run(
        [sys.executable, "-m", "examples.migration.pipeline"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        cwd=example_dir,
    )
    assert result.returncode == 0, (
        f"Stage 2 (after migration) failed: {result.stderr}\nstdout: {result.stdout}"
    )
    print("\n--- Stage 2 after migration ---")
    print(result.stdout)
    assert "Pipeline STAGE=2" in result.stdout
    assert "✅ Stage 2 pipeline complete!" in result.stdout

    # The migration should have reconciled the child feature's data_versions,
    # so no recomputation should be needed
    assert (
        "No changes detected (idempotent or migration worked correctly)"
        in result.stdout
    ), "Migration should have reconciled data_versions, but child was recomputed"
