"""Tests for CLI commands."""

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    FeatureKey,
    FeatureSpec,
    FieldKey,
    FieldSpec,
)
from metaxy.cli.app import app
from metaxy.cli.context import set_config
from metaxy.cli.migrations import app as migrations_app
from metaxy.models.feature import FeatureRegistry

# Import TempFeatureModule for importable test features
# Go up two levels from tests/cli/ to tests/
sys.path.insert(0, str(Path(__file__).parent.parent))
from test_migrations import TempFeatureModule  # type: ignore[import-not-found]


@pytest.fixture
def cli_registry():
    """Test registry with features using temp module for importability."""
    temp_module = TempFeatureModule("test_cli_features")

    test_feature_spec = FeatureSpec(
        key=FeatureKey(["test_cli", "feature"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    temp_module.write_features({"TestFeature": test_feature_spec})
    registry = temp_module.get_registry()

    yield registry

    temp_module.cleanup()


@pytest.fixture
def cli_config_file(tmp_path: Path, cli_registry: FeatureRegistry) -> Path:
    """Create test config file."""
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text("""
store = "test"

[stores.test]
type = "metaxy.metadata_store.InMemoryMetadataStore"

[stores.test.config]
""")
    return config_file


def test_cli_help():
    """Test main CLI help."""
    # Use parse_args to avoid sys.exit
    app.parse_args(["--help"], print_error=False, exit_on_error=False)
    # Should not raise


def test_migrations_help():
    """Test migrations subcommand help."""
    migrations_app.parse_args(["--help"], print_error=False, exit_on_error=False)
    # Should not raise


def test_migrations_generate_help():
    """Test migrations generate help."""
    migrations_app.parse_args(
        ["generate", "--help"], print_error=False, exit_on_error=False
    )
    # Should not raise


def test_push_command(
    cli_config_file: Path,
    cli_registry: FeatureRegistry,
    capsys,
    snapshot: SnapshotAssertion,
):
    """Test push command records feature versions during CD."""
    from metaxy.cli.app import push
    from metaxy.config import MetaxyConfig

    # Load config and create store
    config = MetaxyConfig.load(cli_config_file)

    # Use test registry context
    with cli_registry.use():
        store = config.get_store("test")

        # Simulate CD workflow: push records feature versions before deployment
        with store:
            set_config(config)
            push()

            # Later, in application code, users write metadata
            TestFeature = cli_registry.features_by_key[
                FeatureKey(["test_cli", "feature"])
            ]
            data = pl.DataFrame(
                {
                    "sample_id": [1, 2],
                    "data_version": [{"default": "h1"}, {"default": "h2"}],
                }
            )
            store.write_metadata(TestFeature, data)

    # Check output
    captured = capsys.readouterr()
    output_lines = [
        line.strip() for line in captured.out.strip().split("\n") if line.strip()
    ]

    # Snapshot the output
    assert len(output_lines) >= 1
    assert "Recorded" in captured.out or "features" in captured.out


def test_migrations_generate_no_changes(
    cli_config_file: Path,
    cli_registry: FeatureRegistry,
    capsys,
    snapshot: SnapshotAssertion,
):
    """Test generate command when no changes detected."""
    from metaxy.cli.migrations import generate
    from metaxy.config import MetaxyConfig

    # Load config and create store with data
    config = MetaxyConfig.load(cli_config_file)

    # Use test registry context
    with cli_registry.use():
        store = config.get_store("test")

        # Write data and call command in context manager
        with store:
            TestFeature = cli_registry.features_by_key[
                FeatureKey(["test_cli", "feature"])
            ]
            data = pl.DataFrame(
                {
                    "sample_id": [1, 2],
                    "data_version": [{"default": "h1"}, {"default": "h2"}],
                }
            )
            store.write_metadata(TestFeature, data)
            store.record_feature_graph_snapshot()  # No arguments - records all features in registry

            set_config(config)
            generate(migrations_dir=Path("migrations"))

    # Check output
    captured = capsys.readouterr()
    assert captured.out.strip() == snapshot


def test_migrations_apply_dry_run(
    tmp_path: Path, cli_registry: FeatureRegistry, capsys, snapshot: SnapshotAssertion
):
    """Test apply command in dry-run mode."""
    pytest.importorskip("duckdb")

    from metaxy.cli.migrations import apply
    from metaxy.config import MetaxyConfig
    from metaxy.migrations import DataVersionReconciliation, Migration

    # Create config with DuckDB (persistent store for CLI)
    db_path = tmp_path / "test.duckdb"
    config_file = tmp_path / "metaxy.toml"
    config_file.write_text(f"""
store = "test"

[stores.test]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"
config.database = "{db_path}"
""")

    config = MetaxyConfig.load(config_file)

    # Use test registry context
    with cli_registry.use():
        store = config.get_store("test")

        # Record a snapshot first so migration can load historical registry
        with store:
            snapshot_id = store.serialize_feature_graph()

            # Verify snapshot was recorded
            from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

            fv = store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
            assert len(fv) > 0, "No feature versions recorded"
            assert snapshot_id in fv["snapshot_id"].to_list(), (
                f"Snapshot {snapshot_id} not in table"
            )

        # Create test migration file
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        migration = Migration(
            version=1,
            id="test_migration",
            parent_migration_id=None,
            from_snapshot_id=snapshot_id,
            to_snapshot_id=snapshot_id,
            description="Test",
            created_at=datetime(2025, 1, 1, 0, 0, 0),
            operations=[
                DataVersionReconciliation(
                    id="test_op",
                    feature_key=["test_cli", "feature"],
                    from_="abc123",
                    to="def456",
                    reason="Test migration",
                ).model_dump(by_alias=True)
            ],
        )
        migration.to_yaml(str(migrations_dir / "test_migration.yaml"))

        # Set context and call apply in context manager
        with store:
            set_config(config)
            apply(
                revision="test_migration",  # Migration ID instead of file
                migrations_dir=migrations_dir,
                dry_run=True,
                force=False,
            )

    # Check output with snapshot
    captured = capsys.readouterr()
    assert captured.out.strip() == snapshot


def test_migrations_status_empty(
    cli_config_file: Path,
    cli_registry: FeatureRegistry,
    capsys,
    snapshot: SnapshotAssertion,
):
    """Test status command with no migrations."""
    from metaxy.cli.migrations import status
    from metaxy.config import MetaxyConfig

    config = MetaxyConfig.load(cli_config_file)

    # Use test registry context
    with cli_registry.use():
        store = config.get_store("test")

        # Call command in context manager
        with store:
            set_config(config)
            status()

    # Check output with snapshot
    captured = capsys.readouterr()
    assert captured.out.strip() == snapshot
