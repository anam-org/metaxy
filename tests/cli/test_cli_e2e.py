"""End-to-end CLI tests with real workflows."""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion


@pytest.fixture
def e2e_project(tmp_path: Path):
    """Create a complete test project with features, config, and migrations."""
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()

    # Create features module (v1)
    features_dir = project_dir / "features"
    features_dir.mkdir()

    (features_dir / "__init__.py").write_text("")

    (features_dir / "video.py").write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class VideoFiles(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "files"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)]
)):
    pass
""")

    (features_dir / "processing.py").write_text("""
from metaxy import (
    Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey,
    FeatureDep, FieldDep
)

class VideoProcessing(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "processing"]),
    deps=[FeatureDep(key=FeatureKey(["video", "files"]))],
    fields=[FieldSpec(
        key=FieldKey(["frames"]),
        code_version=1,
        deps=[FieldDep(
            feature_key=FeatureKey(["video", "files"]),
            fields=[FieldKey(["default"])]
        )]
    )]
)):
    pass
""")

    # Create config file with entrypoints and unique database path
    db_path = project_dir / "metadata.duckdb"
    (project_dir / "metaxy.toml").write_text(f"""
store = "dev"
migrations_dir = "migrations"
entrypoints = ["features.video", "features.processing"]

[stores.dev]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.dev.config]
database = "{db_path}"
""")

    # Add project to path
    sys.path.insert(0, str(project_dir))

    yield project_dir

    # Cleanup
    if str(project_dir) in sys.path:
        sys.path.remove(str(project_dir))


def test_cli_e2e_duckdb_workflow(e2e_project: Path, snapshot: SnapshotAssertion):
    """Test complete CLI workflow with DuckDB store.

    Workflow:
    1. Load features v1
    2. Write initial metadata
    3. Push (record feature versions)
    4. Update feature code (v2)
    5. Generate migration
    6. Apply migration
    7. Verify results
    """
    pytest.importorskip("duckdb")

    # Change to project directory
    import os

    original_cwd = os.getcwd()
    os.chdir(e2e_project)

    # Ensure project is in path for imports
    sys.path.insert(0, str(e2e_project))

    try:
        # Step 1: Load features and write initial data
        # Use config to load entrypoints (this is what the CLI will do)
        from metaxy import FeatureKey
        from metaxy.config import MetaxyConfig

        # Load config which will auto-load entrypoints
        config = MetaxyConfig.load()

        # Debug: check config
        print(f"Config entrypoints: {config.entrypoints}")
        print(f"Config store: {config.store}")

        # Get features from the now-populated global graph
        from metaxy.models.feature import FeatureGraph

        graph = FeatureGraph.get_active()

        # Debug: check what's loaded
        print(f"Loaded features: {list(graph.features_by_key.keys())}")

        VideoFiles = graph.features_by_key[FeatureKey(["video", "files"])]
        VideoProcessing = graph.features_by_key[FeatureKey(["video", "processing"])]

        # Get store and write initial data
        store = config.get_store()

        with store:
            # Write initial metadata
            video_files_data = pl.DataFrame(
                {
                    "sample_id": ["video1.mp4", "video2.mp4"],
                    "path": ["/data/video1.mp4", "/data/video2.mp4"],
                    "data_version": [{"default": "hash1"}, {"default": "hash2"}],
                }
            )
            store.write_metadata(VideoFiles, video_files_data)

            # Write processing metadata
            processing_data = pl.DataFrame(
                {
                    "sample_id": ["video1.mp4", "video2.mp4"],
                    "data_version": [{"frames": "proc1"}, {"frames": "proc2"}],
                }
            )
            store.write_metadata(VideoProcessing, processing_data)

            # Step 2: Push (record feature versions)
            snapshot_id_v1 = store.serialize_feature_graph()

            assert snapshot_id_v1 is not None
            assert len(snapshot_id_v1) == 64  # Full SHA256 hash

        # Store is now closed, data is persisted to DuckDB

        # Step 3: Update feature code (simulate code change)
        (e2e_project / "features" / "processing.py").write_text("""
from metaxy import (
    Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey,
    FeatureDep, FieldDep
)

class VideoProcessing(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "processing"]),
    deps=[FeatureDep(key=FeatureKey(["video", "files"]))],
    fields=[FieldSpec(
        key=FieldKey(["frames"]),
        code_version=2,  # Changed from 1 to 2
        deps=[FieldDep(
            feature_key=FeatureKey(["video", "files"]),
            fields=[FieldKey(["default"])]
        )]
    )]
)):
    pass
""")

        # Step 4: Generate migration using CLI subprocess
        # The subprocess will load the new v2 code and compare with v1 in DB
        result = subprocess.run(
            [sys.executable, "-m", "metaxy.cli.app", "migrations", "generate"],
            cwd=str(e2e_project),
            capture_output=True,
            text=True,
        )

        # Debug output if failed
        if result.returncode != 0 or "No feature changes detected" in result.stdout:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")

        assert result.returncode == 0, f"Generate failed: {result.stderr}"
        assert "No feature changes detected" not in result.stdout, (
            f"No changes detected. Output: {result.stdout}"
        )

        # Verify migration file created (in metaxy/migrations dir from config)
        migrations_dir = e2e_project / config.migrations_dir
        assert migrations_dir.exists(), (
            f"Migrations dir not created. Output: {result.stdout}"
        )

        migration_files = list(migrations_dir.glob("*.yaml"))
        assert len(migration_files) == 1

        migration_file = migration_files[0]

        # Load and verify migration
        from metaxy.migrations import Migration

        migration = Migration.from_yaml(str(migration_file))

        # Should have 1 operation (only processing changed, it has upstream so it's reconcilable)
        ops = migration.get_operations()
        assert len(ops) == 1
        assert ops[0].feature_key == ["video", "processing"]

        # Snapshot the generated migration structure
        migration_snapshot = {
            "version": migration.version,
            "parent_migration_id": migration.parent_migration_id,
            "operations": [
                {
                    "id": op.id,
                    "type": op.type,  # type: ignore[attr-defined]
                    "feature_key": op.feature_key,
                    "reason": op.reason,
                }
                for op in ops
            ],
        }
        assert migration_snapshot == snapshot

        # Step 5: Apply migration using CLI (dry run first)
        # Use migration ID instead of file path
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "metaxy.cli.app",
                "migrations",
                "apply",
                migration.id,  # Migration ID, not file path
                "--dry-run",
            ],
            cwd=str(e2e_project),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Apply dry-run failed: {result.stderr}"
        assert (
            "skipped" in result.stdout.lower() or "completed" in result.stdout.lower()
        )

        # Step 6: Apply migration for real
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "metaxy.cli.app",
                "migrations",
                "apply",
                migration.id,  # Migration ID, not file path
            ],
            cwd=str(e2e_project),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Apply failed: {result.stderr}"

        # Step 7: Verify migration was applied
        # Reload config and graph to get v2 features
        config = MetaxyConfig.load()
        graph = FeatureGraph.get_active()
        VideoProcessingV2 = graph.features_by_key[FeatureKey(["video", "processing"])]

        store = config.get_store()
        with store:
            # Check migration system tables
            from metaxy.migrations.executor import MIGRATIONS_KEY

            migrations = store.read_metadata(MIGRATIONS_KEY, current_only=False)
            assert len(migrations) == 1
            assert migrations["migration_id"][0] == migration.id

            # Verify processing feature was updated
            all_processing = store.read_metadata(
                VideoProcessingV2,
                current_only=False,
            )

            # Should have original + migrated rows
            assert len(all_processing) >= 2

            # Snapshot the final data state
            final_snapshot = {
                "row_count": len(all_processing),
                "sample_ids": sorted(all_processing["sample_id"].unique().to_list()),
                "feature_versions": sorted(
                    all_processing["feature_version"].unique().to_list()
                ),
            }
            assert final_snapshot == snapshot

        # Store is now closed, can run CLI commands without lock conflicts

        # Check migration status via CLI
        status_result = subprocess.run(
            [sys.executable, "-m", "metaxy.cli.app", "migrations", "status"],
            cwd=str(e2e_project),
            capture_output=True,
            text=True,
        )

        assert status_result.returncode == 0, f"Status failed: {status_result.stderr}"

        # Snapshot migration status output (excluding timestamp-based migration_id)
        status_snapshot = {
            "has_migrations": "migration_" in status_result.stdout,
            "migration_completed": "COMPLETED" in status_result.stdout,
            "has_migration_id": migration.id is not None if migration else False,
        }
        assert status_snapshot == snapshot

    finally:
        os.chdir(original_cwd)
        if str(e2e_project) in sys.path:
            sys.path.remove(str(e2e_project))


def test_cli_migration_status_command(e2e_project: Path):
    """Test migrations status command shows correct info."""
    pytest.importorskip("duckdb")

    import os

    original_cwd = os.getcwd()
    os.chdir(e2e_project)
    sys.path.insert(0, str(e2e_project))

    try:
        # First generate a migration (sets up the workflow)
        from metaxy import FeatureKey
        from metaxy.config import MetaxyConfig
        from metaxy.models.feature import FeatureGraph

        # Load config (auto-loads entrypoints)
        config = MetaxyConfig.load()
        graph = FeatureGraph.get_active()

        store = config.get_store()

        with store:
            # Write some data and record versions
            VideoFiles = graph.features_by_key[FeatureKey(["video", "files"])]

            store.write_metadata(
                VideoFiles,
                pl.DataFrame(
                    {
                        "sample_id": ["v1"],
                        "data_version": [{"default": "h1"}],
                    }
                ),
            )
            store.serialize_feature_graph()

        # Update code to trigger a change
        (e2e_project / "features" / "video.py").write_text("""
from metaxy import Feature, FeatureSpec, FeatureKey, FieldSpec, FieldKey

class VideoFiles(Feature, spec=FeatureSpec(
    key=FeatureKey(["video", "files"]),
    deps=None,
    fields=[FieldSpec(key=FieldKey(["default"]), code_version=2)]  # Changed!
)):
    pass
""")

        # Check status (should show no migrations initially)
        result = subprocess.run(
            [sys.executable, "-m", "metaxy.cli.app", "migrations", "status"],
            cwd=str(e2e_project),
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert (
            "No migrations found" in result.stdout
            or "Migration Status:" in result.stdout
        )

    finally:
        os.chdir(original_cwd)
