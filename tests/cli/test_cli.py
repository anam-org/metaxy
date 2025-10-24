"""Tests for CLI commands."""

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
from metaxy._testing import TempFeatureModule
from metaxy._utils import collect_to_polars
from metaxy.cli.app import app
from metaxy.cli.context import set_config
from metaxy.cli.migrations import app as migrations_app
from metaxy.models.feature import FeatureGraph


@pytest.fixture
def cli_graph():
    """Test graph with features using temp module for importability."""
    temp_module = TempFeatureModule("test_cli_features")

    test_feature_spec = FeatureSpec(
        key=FeatureKey(["test_cli", "feature"]),
        deps=None,
        fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
    )

    temp_module.write_features({"TestFeature": test_feature_spec})
    graph = temp_module.get_graph()

    yield graph

    temp_module.cleanup()


@pytest.fixture
def cli_config_file(tmp_path: Path, cli_graph: FeatureGraph) -> Path:
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


def test_migrations_generate_no_changes(
    tmp_path: Path,
    cli_graph: FeatureGraph,
    capsys,
    snapshot: SnapshotAssertion,
):
    """Test generate command when no changes detected."""

    from metaxy.cli.migrations import generate
    from metaxy.config import MetaxyConfig

    # Create config with DuckDB store (not InMemory)
    config_file = tmp_path / "metaxy.toml"
    db_path = tmp_path / "test.duckdb"
    config_file.write_text(f"""
store = "test"

[stores.test]
type = "metaxy.metadata_store.duckdb.DuckDBMetadataStore"

[stores.test.config]
database = "{db_path}"
""")

    # Load config and create store with data
    config = MetaxyConfig.load(config_file)

    # Use test graph context
    with cli_graph.use():
        store = config.get_store("test")

        # Write data and call command in context manager
        with store:
            TestFeature = cli_graph.features_by_key[FeatureKey(["test_cli", "feature"])]
            data = pl.DataFrame(
                {
                    "sample_id": [1, 2],
                    "data_version": [{"default": "h1"}, {"default": "h2"}],
                }
            )
            store.write_metadata(TestFeature, data)
            store.record_feature_graph_snapshot()  # No arguments - records all features in graph

            set_config(config)
            generate(migrations_dir=Path("migrations"))

    # Check output
    captured = capsys.readouterr()
    assert captured.out.strip() == snapshot


def test_migrations_apply_dry_run(
    tmp_path: Path, cli_graph: FeatureGraph, capsys, snapshot: SnapshotAssertion
):
    """Test apply command in dry-run mode."""

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

    # Use test graph context
    with cli_graph.use():
        store = config.get_store("test")

        # Record a snapshot first so migration can load historical graph
        with store:
            snapshot_id, _ = store.record_feature_graph_snapshot()

            # Verify snapshot was recorded
            from metaxy.metadata_store.base import FEATURE_VERSIONS_KEY

            fv = collect_to_polars(
                store.read_metadata(FEATURE_VERSIONS_KEY, current_only=False)
            )
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
    cli_graph: FeatureGraph,
    capsys,
    snapshot: SnapshotAssertion,
):
    """Test status command with no migrations."""
    from metaxy.cli.migrations import status
    from metaxy.config import MetaxyConfig

    config = MetaxyConfig.load(cli_config_file)

    # Use test graph context
    with cli_graph.use():
        store = config.get_store("test")

        # Call command in context manager
        with store:
            set_config(config)
            status()

    # Check output with snapshot
    captured = capsys.readouterr()
    assert captured.out.strip() == snapshot
