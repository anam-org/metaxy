"""Tests for migrations CLI commands."""

from metaxy._testing import TempMetaxyProject


def test_migrations_list_empty(metaxy_project: TempMetaxyProject):
    """Test migrations list with no migrations."""

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
        result = metaxy_project.run_cli("migrations", "list")

        assert result.returncode == 0
        assert "No migrations found" in result.stderr


def test_migrations_list_single_migration(metaxy_project: TempMetaxyProject):
    """Test migrations list with a single migration."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
            "migration_type": "metaxy.migrations.models.DiffMigration",
            "migration_id": "test_migration_001",
            "created_at": datetime(
                2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc
            ).isoformat(),
            "parent": "initial",
            "from_snapshot_version": "a" * 64,
            "to_snapshot_version": "b" * 64,
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path = migrations_dir / "test_001.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(migration_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli("migrations", "list")

        assert result.returncode == 0
        # Check for table contents
        assert "test_migration_001" in result.stderr
        assert "2025-01-27 12:00" in result.stderr
        assert "DataVersionReconciliation" in result.stderr
        # Check table headers
        assert "ID" in result.stderr
        assert "Created" in result.stderr
        assert "Operations" in result.stderr


def test_migrations_list_multiple_migrations(metaxy_project: TempMetaxyProject):
    """Test migrations list with multiple migrations in chain order."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
            "created_at": datetime(
                2025, 1, 27, 10, 0, 0, tzinfo=timezone.utc
            ).isoformat(),
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
            "created_at": datetime(
                2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc
            ).isoformat(),
            "parent": "migration_001",
            "from_snapshot_version": "b" * 64,
            "to_snapshot_version": "c" * 64,
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path2 = migrations_dir / "second.yaml"
        with open(yaml_path2, "w") as f:
            yaml.dump(migration2_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli("migrations", "list")

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


def test_migrations_list_multiple_operations(metaxy_project: TempMetaxyProject):
    """Test migrations list with migration having multiple operations."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
            "migration_type": "metaxy.migrations.models.DiffMigration",
            "migration_id": "multi_op_migration",
            "created_at": datetime(
                2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc
            ).isoformat(),
            "parent": "initial",
            "from_snapshot_version": "a" * 64,
            "to_snapshot_version": "b" * 64,
            "ops": [
                {"type": "metaxy.migrations.ops.DataVersionReconciliation"},
                {"type": "myproject.ops.CustomBackfill"},
            ],
        }

        yaml_path = migrations_dir / "multi_op.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(migration_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli("migrations", "list")

        assert result.returncode == 0
        assert "multi_op_migration" in result.stderr
        # Check both operation names appear (may be truncated with …)
        # DataVersionReconciliation may be truncated to DataVersionReconciliatio…
        assert "DataVersionReconciliatio" in result.stderr  # Allow truncation
        assert "CustomBackfill" in result.stderr


def test_migrations_list_invalid_chain(metaxy_project: TempMetaxyProject):
    """Test migrations list with invalid migration chain."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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
            "created_at": datetime(
                2025, 1, 27, 10, 0, 0, tzinfo=timezone.utc
            ).isoformat(),
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
            "created_at": datetime(
                2025, 1, 27, 12, 0, 0, tzinfo=timezone.utc
            ).isoformat(),
            "parent": "initial",
            "from_snapshot_version": "b" * 64,
            "to_snapshot_version": "c" * 64,
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path2 = migrations_dir / "second.yaml"
        with open(yaml_path2, "w") as f:
            yaml.dump(migration2_yaml, f)

        # Run migrations list
        result = metaxy_project.run_cli("migrations", "list", check=False)

        assert result.returncode == 0  # Doesn't exit with error, just prints error
        assert "Invalid migration:" in result.stderr
        assert "Multiple migration heads" in result.stderr


def test_migrations_apply_with_error_logging(metaxy_project: TempMetaxyProject):
    """Test that migration errors are logged with full tracebacks."""
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

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

        # Create a migration with invalid snapshot versions (will cause errors)
        migration_yaml = {
            "migration_type": "metaxy.migrations.models.DiffMigration",
            "migration_id": "test_error_migration",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent": "initial",
            "from_snapshot_version": "a" * 64,  # Nonexistent snapshot
            "to_snapshot_version": "b" * 64,  # Nonexistent snapshot
            "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
        }

        yaml_path = migrations_dir / "test_error.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(migration_yaml, f)

        # Apply migration (will fail because snapshots don't exist)
        result = metaxy_project.run_cli("migrations", "apply", "--dry-run", check=False)

        # This particular error causes fatal exit before feature processing
        assert result.returncode == 1
        output = result.stderr

        # Verify that tracebacks are printed (not just error messages)
        assert "Traceback (most recent call last):" in output
        # Verify actual error details are shown (snapshot-related errors)
        assert "snapshot" in output.lower() and "cannot load" in output.lower()
        # Should show file paths and line numbers from traceback
        assert ".py" in output and "line " in output


def test_generated_migration_is_valid(metaxy_project: TempMetaxyProject):
    """Test that generated migrations can be loaded and used with status command.

    Regression test for bug where generated YAML used 'id' but model expected 'migration_id'.
    """

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Generate a full migration
        result = metaxy_project.run_cli(
            "migrations",
            "generate",
            "--op",
            "metaxy.migrations.ops.DataVersionReconciliation",
            "--type",
            "full",
        )
        assert result.returncode == 0

        # Status command should work without validation errors
        result = metaxy_project.run_cli("migrations", "status")
        assert result.returncode == 0
        assert "validation error" not in result.stderr.lower()


def test_migrations_apply_rerun_displays_reprocessing_message(
    metaxy_project: TempMetaxyProject,
):
    """Test that --rerun flag shows 'Reprocessing all N feature(s)' message.

    Regression test for bug where --rerun incorrectly displayed database completion
    state instead of indicating all features will be reprocessed from YAML.
    """

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class VideoFiles(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["video", "files"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Generate a full migration
        result = metaxy_project.run_cli(
            "migrations",
            "generate",
            "--op",
            "metaxy.migrations.ops.DataVersionReconciliation",
            "--type",
            "full",
        )
        assert result.returncode == 0

        # Apply with --rerun and --dry-run to see the message without executing
        result = metaxy_project.run_cli(
            "migrations", "apply", "--rerun", "--dry-run", check=False
        )

        # Should show RERUN MODE banner
        assert "RERUN MODE" in result.stderr
        # Should show "Reprocessing all N feature(s)" message (not "already completed")
        assert "Reprocessing all" in result.stderr
        assert "feature(s)" in result.stderr
        # Should NOT show "already completed" message when in rerun mode
        assert "already completed)" not in result.stderr


def test_migrations_status_uses_yaml_as_source_of_truth(
    metaxy_project: TempMetaxyProject,
):
    """Test that migration status uses YAML features as source of truth.

    Regression test for bug where status displayed historical completed features
    from database even when they were removed from the YAML file.

    Scenario: YAML originally had 3 features, 2 were executed. Then YAML is
    modified to only have 1 feature. Status should show 0/1 completed (not 2/1).
    """
    from datetime import datetime, timezone

    import yaml

    def features():
        from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
        from metaxy._testing.models import SampleFeatureSpec

        class FeatureA(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature_a"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        class FeatureB(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature_b"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

    with metaxy_project.with_features(features):
        # Create migrations directory
        migrations_dir = metaxy_project.project_dir / ".metaxy" / "migrations"
        migrations_dir.mkdir(parents=True, exist_ok=True)

        # Write a FullGraphMigration YAML with both features
        migration_yaml = {
            "migration_type": "metaxy.migrations.models.FullGraphMigration",
            "migration_id": "test_yaml_source_of_truth",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "parent": "initial",
            "snapshot_version": "a" * 64,
            "ops": [
                {
                    "type": "metaxy.migrations.ops.DataVersionReconciliation",
                    "features": ["test/feature_a", "test/feature_b"],
                }
            ],
        }

        yaml_path = migrations_dir / "test_yaml_truth.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(migration_yaml, f)

        # Check initial status - should show 0/2 completed
        result = metaxy_project.run_cli("migrations", "status")
        assert result.returncode == 0
        assert "0/2 completed" in result.stderr

        # Now modify YAML to only have feature_a (remove feature_b)
        migration_yaml["ops"] = [
            {
                "type": "metaxy.migrations.ops.DataVersionReconciliation",
                "features": ["test/feature_a"],
            }
        ]
        with open(yaml_path, "w") as f:
            yaml.dump(migration_yaml, f)

        # Check status again - should show 0/1 completed (YAML is source of truth)
        result = metaxy_project.run_cli("migrations", "status")
        assert result.returncode == 0
        assert "0/1 completed" in result.stderr
        # Should NOT show 0/2 (the old value)
        assert "0/2" not in result.stderr
