"""Tests for migrations CLI commands."""

import pytest
from metaxy_testing import TempMetaxyProject


def test_migrations_list_empty(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test migrations list with no migrations."""

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # No migrations created yet
        result = metaxy_project.run_cli(["migrations", "list"], capsys=capsys)

        assert result.returncode == 0
        assert "No migrations found" in result.stderr


def test_migrations_list_single_migration(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test migrations list with a single migration."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Create migrations directory
        migrations_dir = metaxy_project.project_dir / ".metaxy" / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Write a test migration YAML
        migration_yaml = {
            "migration_type": "metaxy.migrations.models.FullGraphMigration",
            "migration_id": "test_migration_001",
            "created_at": datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            "parent": "initial",
            "snapshot_version": "b" * 64,
            "ops": [
                {
                    "type": "metaxy.migrations.ops.DataVersionReconciliation",
                    "features": ["video/files"],
                }
            ],
        }

        yaml_path = migrations_dir / "test_001.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(migration_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli(["migrations", "list"], capsys=capsys)

        assert result.returncode == 0
        # Check for table contents
        assert "test_migration_001" in result.stderr
        assert "2025-01-27 12:00" in result.stderr
        assert "DataVersionReconciliation" in result.stderr
        # Check table headers
        assert "ID" in result.stderr
        assert "Created" in result.stderr
        assert "Operations" in result.stderr


def test_migrations_list_multiple_migrations(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test migrations list with multiple migrations in chain order."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Create migrations directory
        migrations_dir = metaxy_project.project_dir / ".metaxy" / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Write first migration YAML
        migration1_yaml = {
            "migration_type": "metaxy.migrations.models.DiffMigration",
            "migration_id": "migration_001",
            "created_at": datetime(2025, 1, 27, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
            "parent": "initial",
            "from_snapshot_version": "a" * 64,
            "to_snapshot_version": "b" * 64,
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path1 = migrations_dir / "first.yaml"
        with open(yaml_path1, "w") as f:
            yaml.dump(migration1_yaml, f)

        # Write second migration YAML (depends on first)
        migration2_yaml = {
            "migration_type": "metaxy.migrations.models.DiffMigration",
            "migration_id": "migration_002",
            "created_at": datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            "parent": "migration_001",
            "from_snapshot_version": "b" * 64,
            "to_snapshot_version": "c" * 64,
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path2 = migrations_dir / "second.yaml"
        with open(yaml_path2, "w") as f:
            yaml.dump(migration2_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli(["migrations", "list"], capsys=capsys)

        assert result.returncode == 0
        # Check both migrations are listed
        assert "migration_001" in result.stderr
        assert "migration_002" in result.stderr
        assert "2025-01-27 10:00" in result.stderr
        assert "2025-01-27 12:00" in result.stderr
        # Check they appear in chain order (migration_001 before migration_002)
        pos_001 = result.stderr.index("migration_001")
        pos_002 = result.stderr.index("migration_002")
        assert pos_001 < pos_002


def test_migrations_list_multiple_operations(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test migrations list with migration having multiple operations."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Create migrations directory
        migrations_dir = metaxy_project.project_dir / ".metaxy" / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Write migration with multiple operations
        migration_yaml = {
            "migration_type": "metaxy.migrations.models.FullGraphMigration",
            "migration_id": "multi_op_migration",
            "created_at": datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            "parent": "initial",
            "snapshot_version": "b" * 64,
            "ops": [
                {
                    "type": "metaxy.migrations.ops.DataVersionReconciliation",
                    "features": ["video/files"],
                },
                {"type": "myproject.ops.CustomBackfill", "features": ["video/files"]},
            ],
        }

        yaml_path = migrations_dir / "multi_op.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(migration_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli(["migrations", "list"], capsys=capsys)

        assert result.returncode == 0
        assert "multi_op_migration" in result.stderr
        # Check both operation names appear (shortened)
        assert "DataVersionReconciliation" in result.stderr
        assert "CustomBackfill" in result.stderr
        # Check they're comma-separated (may wrap across lines in table)
        # Remove whitespace to handle line wrapping
        output_normalized = "".join(result.stderr.split())
        assert "DataVersionReconciliation,CustomBackfill" in output_normalized


def test_migrations_list_invalid_chain(metaxy_project: TempMetaxyProject, capsys: pytest.CaptureFixture[str]):
    """Test migrations list with invalid migration chain."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy_testing.models import SampleFeatureSpec

        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Create migrations directory
        migrations_dir = metaxy_project.project_dir / ".metaxy" / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Write two migrations that both claim to be head (invalid chain)
        migration1_yaml = {
            "migration_type": "metaxy.migrations.models.DiffMigration",
            "migration_id": "migration_001",
            "created_at": datetime(2025, 1, 27, 10, 0, 0, tzinfo=timezone.utc).isoformat(),
            "parent": "initial",
            "from_snapshot_version": "a" * 64,
            "to_snapshot_version": "b" * 64,
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path1 = migrations_dir / "first.yaml"
        with open(yaml_path1, "w") as f:
            yaml.dump(migration1_yaml, f)

        # Second migration also has parent "initial" (creates two heads)
        migration2_yaml = {
            "migration_type": "metaxy.migrations.models.DiffMigration",
            "migration_id": "migration_002",
            "created_at": datetime(2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc).isoformat(),
            "parent": "initial",
            "from_snapshot_version": "b" * 64,
            "to_snapshot_version": "c" * 64,
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path2 = migrations_dir / "second.yaml"
        with open(yaml_path2, "w") as f:
            yaml.dump(migration2_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli(["migrations", "list"], check=False, capsys=capsys)

        assert result.returncode == 0  # Doesn't exit with error, just prints error
        assert "Invalid migration" in result.stderr
        assert "Multiple migration heads" in result.stderr
