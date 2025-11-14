"""Integration tests for Python migration execution and end-to-end workflows.

These tests verify that Python migrations (.py files) execute correctly with
actual metadata stores, mixed chains work properly, and custom execute() logic
runs as expected.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import (
    FeatureDep,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
    InMemoryMetadataStore,
    SampleFeatureSpec,
)
from metaxy._testing import TempFeatureModule
from metaxy._utils import collect_to_polars
from metaxy.config import MetaxyConfig
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.migrations import MigrationExecutor
from metaxy.migrations.loader import build_migration_chain, load_migration_from_file
from metaxy.migrations.models import DiffMigration, Migration
from metaxy.models.feature import FeatureGraph


@pytest.fixture(autouse=True)
def setup_default_config():
    """Set up default MetaxyConfig for all tests so features use project='default'."""
    config = MetaxyConfig(project="default", stores={})
    MetaxyConfig.set(config)
    yield
    MetaxyConfig.reset()


def migrate_store_to_graph(
    source_store: InMemoryMetadataStore,
    target_graph: FeatureGraph,
) -> InMemoryMetadataStore:
    """Create new store with target graph context but source store's data.

    This includes system tables (snapshots, migrations, events) so that
    migration detection can find the previous snapshot.
    """
    new_store = InMemoryMetadataStore()
    # Copy all storage including system tables
    new_store._storage = source_store._storage.copy()
    return new_store


@pytest.fixture
def upstream_downstream_v1():
    """Graph with upstream and downstream features."""
    temp_module = TempFeatureModule("test_py_migration_chain_v1")

    upstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_py_migration", "upstream"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="1"),
        ],
    )

    downstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_py_migration", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test_py_migration", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test_py_migration", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
    yield temp_module.graph
    temp_module.cleanup()


@pytest.fixture
def upstream_downstream_v2():
    """Graph with upstream code_version changed."""
    temp_module = TempFeatureModule("test_py_migration_chain_v2")

    upstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_py_migration", "upstream"]),
        fields=[
            FieldSpec(key=FieldKey(["default"]), code_version="2"),  # Changed!
        ],
    )

    downstream_spec = SampleFeatureSpec(
        key=FeatureKey(["test_py_migration", "downstream"]),
        deps=[FeatureDep(feature=FeatureKey(["test_py_migration", "upstream"]))],
        fields=[
            FieldSpec(
                key=FieldKey(["default"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature=FeatureKey(["test_py_migration", "upstream"]),
                        fields=[FieldKey(["default"])],
                    )
                ],
            ),
        ],
    )

    temp_module.write_features(
        {"Upstream": upstream_spec, "Downstream": downstream_spec}
    )
    yield temp_module.graph
    temp_module.cleanup()


def write_python_diff_migration(
    migrations_dir: Path,
    migration_id: str,
    parent: str,
    from_snapshot: str,
    to_snapshot: str,
    custom_code: str = "",
    *,
    python_api: bool = False,
) -> Path:
    """Helper to write a Python DiffMigration file.

    Args:
        migrations_dir: Directory to write migration to
        migration_id: Unique migration ID
        parent: Parent migration ID
        from_snapshot: Source snapshot version
        to_snapshot: Target snapshot version
        custom_code: Optional custom code to add to migration class

    Returns:
        Path to created migration file
    """
    py_path = migrations_dir / f"{migration_id}.py"

    base_class = "PythonMigration" if python_api else "DiffMigration"
    extra_import = ""
    ops_block = ""
    if python_api:
        extra_import = "\nfrom metaxy.migrations.ops import DataVersionReconciliation"
        ops_block = """
    def operations(self):
        return [
            DataVersionReconciliation(),
        ]
"""
    else:
        ops_block = """
    ops: list = [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]
"""

    code = f'''"""Migration {migration_id}."""

from datetime import datetime, timezone
from metaxy.migrations.models import {base_class}{extra_import}

class Migration({base_class}):
    """Migration from {from_snapshot[:8]} to {to_snapshot[:8]}."""
    migration_id: str = "{migration_id}"
    parent: str = "{parent}"
    created_at: datetime = datetime.now(timezone.utc)
    from_snapshot_version: str = "{from_snapshot}"
    to_snapshot_version: str = "{to_snapshot}"
{ops_block}

{custom_code}
'''

    with open(py_path, "w") as f:
        f.write(code)

    return py_path


class TestBasicPythonMigrationExecution:
    """Test executing simple Python migrations."""

    @pytest.mark.parametrize(
        "use_python_api", [False, True], ids=["ops_field", "operations_method"]
    )
    def test_execute_simple_python_diff_migration(
        self,
        tmp_path: Path,
        upstream_downstream_v1: FeatureGraph,
        upstream_downstream_v2: FeatureGraph,
        snapshot: SnapshotAssertion,
        use_python_api: bool,
    ):
        """Test executing a simple Python DiffMigration with DataVersionReconciliation."""
        # Step 1: Setup v1 data
        store_v1 = InMemoryMetadataStore()
        UpstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]
        DownstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "downstream"])
        ]

        with upstream_downstream_v1.use(), store_v1:
            # Write upstream (root feature)
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                        {"default": "h3"},
                    ],
                }
            )
            store_v1.write_metadata(UpstreamV1, upstream_data)

            # Write downstream (derived feature)
            diff = store_v1.resolve_update(DownstreamV1)
            if len(diff.added) > 0:
                store_v1.write_metadata(DownstreamV1, diff.added)

            # Record v1 snapshot
            store_v1.record_feature_graph_snapshot()

        # Step 2: Migrate to v2 graph
        store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
        UpstreamV2 = upstream_downstream_v2.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]

        with upstream_downstream_v2.use(), store_v2:
            # Record v2 snapshot
            store_v2.record_feature_graph_snapshot()

            # Step 3: Create Python migration file
            migrations_dir = tmp_path / "migrations"
            migrations_dir.mkdir()

            write_python_diff_migration(
                migrations_dir=migrations_dir,
                migration_id="20250107_120000",
                parent="initial",
                from_snapshot=upstream_downstream_v1.snapshot_version,
                to_snapshot=upstream_downstream_v2.snapshot_version,
                python_api=use_python_api,
            )

            # Step 4: Load and execute migration
            migration_file = migrations_dir / "20250107_120000.py"
            migration = load_migration_from_file(migration_file)

            assert isinstance(migration, DiffMigration)
            assert migration.migration_id == "20250107_120000"

            # Update upstream manually (root feature cannot be auto-reconciled)
            new_upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "metaxy_provenance_by_field": [
                        {"default": "new_h1"},
                        {"default": "new_h2"},
                        {"default": "new_h3"},
                    ],
                }
            )
            store_v2.write_metadata(UpstreamV2, new_upstream_data)

            # Execute migration
            storage = SystemTableStorage(store_v2)
            executor = MigrationExecutor(storage)
            result = executor.execute(
                migration, store_v2, project="default", dry_run=False
            )

            # Verify result
            assert result.migration_id == "20250107_120000"
            # Upstream will fail (root), downstream will be skipped due to failed dependency
            assert result.status == "failed"
            assert result.features_completed == 0  # Nothing completed
            assert result.features_failed == 1  # Upstream failed
            assert result.features_skipped == 1  # Downstream skipped

            # Verify upstream error
            assert "test_py_migration/upstream" in result.errors
            assert "Root features" in result.errors["test_py_migration/upstream"]

            # Verify downstream was skipped
            assert "test_py_migration/downstream" in result.errors
            assert (
                "Skipped due to failed dependencies"
                in result.errors["test_py_migration/downstream"]
            )

    def test_python_migration_with_custom_method(
        self,
        tmp_path: Path,
        upstream_downstream_v1: FeatureGraph,
        upstream_downstream_v2: FeatureGraph,
    ):
        """Test Python migration with custom helper methods."""
        # Setup v1 data
        store_v1 = InMemoryMetadataStore()
        UpstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]
        DownstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "downstream"])
        ]

        with upstream_downstream_v1.use(), store_v1:
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                    ],
                }
            )
            store_v1.write_metadata(UpstreamV1, upstream_data)

            diff = store_v1.resolve_update(DownstreamV1)
            if len(diff.added) > 0:
                store_v1.write_metadata(DownstreamV1, diff.added)

            store_v1.record_feature_graph_snapshot()

        # Migrate to v2
        store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
        UpstreamV2 = upstream_downstream_v2.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]

        with upstream_downstream_v2.use(), store_v2:
            store_v2.record_feature_graph_snapshot()

            # Create Python migration with custom helper method
            migrations_dir = tmp_path / "migrations"
            migrations_dir.mkdir()

            custom_code = '''
    def get_reconciliation_ops(self) -> list:
        """Helper method to generate operations."""
        return [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]

    def validate_snapshots(self) -> bool:
        """Helper method to validate snapshots."""
        return self.from_snapshot_version != self.to_snapshot_version
'''

            write_python_diff_migration(
                migrations_dir=migrations_dir,
                migration_id="20250107_130000",
                parent="initial",
                from_snapshot=upstream_downstream_v1.snapshot_version,
                to_snapshot=upstream_downstream_v2.snapshot_version,
                custom_code=custom_code,
            )

            # Load migration
            migration_file = migrations_dir / "20250107_130000.py"
            migration = load_migration_from_file(migration_file)

            # Verify custom methods work
            assert hasattr(migration, "get_reconciliation_ops")
            # Type checker doesn't know about dynamically added methods, use getattr
            reconciliation_ops = getattr(migration, "get_reconciliation_ops")()
            assert len(reconciliation_ops) == 1
            assert (
                reconciliation_ops[0]["type"]
                == "metaxy.migrations.ops.DataVersionReconciliation"
            )

            # Verify validation method
            assert hasattr(migration, "validate_snapshots")
            assert getattr(migration, "validate_snapshots")() is True

            # Update upstream
            new_upstream = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"default": "new_h1"},
                        {"default": "new_h2"},
                    ],
                }
            )
            store_v2.write_metadata(UpstreamV2, new_upstream)

            # Execute
            storage = SystemTableStorage(store_v2)
            executor = MigrationExecutor(storage)
            result = executor.execute(
                migration, store_v2, project="default", dry_run=False
            )

            # Verify execution worked
            assert result.status == "failed"  # Upstream will fail (root)
            assert result.features_completed == 0  # Nothing completed
            assert (
                result.features_skipped == 1
            )  # Downstream skipped due to upstream failure


class TestCustomExecuteLogic:
    """Test Python migrations with custom execute() implementations."""

    def test_python_migration_with_custom_execute(
        self,
        tmp_path: Path,
        upstream_downstream_v1: FeatureGraph,
        upstream_downstream_v2: FeatureGraph,
    ):
        """Test Python migration that overrides execute() with custom logic."""
        # Setup v1 data
        store_v1 = InMemoryMetadataStore()
        UpstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]
        DownstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "downstream"])
        ]

        with upstream_downstream_v1.use(), store_v1:
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"default": "h1"}],
                }
            )
            store_v1.write_metadata(UpstreamV1, upstream_data)

            diff = store_v1.resolve_update(DownstreamV1)
            if len(diff.added) > 0:
                store_v1.write_metadata(DownstreamV1, diff.added)

            store_v1.record_feature_graph_snapshot()

        # Migrate to v2
        store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
        UpstreamV2 = upstream_downstream_v2.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]

        with upstream_downstream_v2.use(), store_v2:
            store_v2.record_feature_graph_snapshot()

            # Create Python migration with custom execute
            migrations_dir = tmp_path / "migrations"
            migrations_dir.mkdir()

            # Track that custom logic ran
            marker_file = tmp_path / "custom_logic_ran.txt"

            custom_code = f'''
    def execute(self, store, project, *, dry_run=False):
        """Custom execute with pre/post processing."""
        # Pre-processing: write marker file
        from pathlib import Path
        marker = Path(r"{marker_file}")
        marker.write_text("Custom logic executed")

        # Call parent execute
        result = super().execute(store, project, dry_run=dry_run)

        # Post-processing: modify result description
        # (In real usage, could add custom metrics, notifications, etc.)
        return result
'''

            write_python_diff_migration(
                migrations_dir=migrations_dir,
                migration_id="20250107_140000",
                parent="initial",
                from_snapshot=upstream_downstream_v1.snapshot_version,
                to_snapshot=upstream_downstream_v2.snapshot_version,
                custom_code=custom_code,
            )

            # Update upstream
            new_upstream = pl.DataFrame(
                {
                    "sample_uid": [1],
                    "metaxy_provenance_by_field": [{"default": "new_h1"}],
                }
            )
            store_v2.write_metadata(UpstreamV2, new_upstream)

            # Load and execute
            migration_file = migrations_dir / "20250107_140000.py"
            migration = load_migration_from_file(migration_file)

            storage = SystemTableStorage(store_v2)
            executor = MigrationExecutor(storage)
            result = executor.execute(
                migration, store_v2, project="default", dry_run=False
            )

            # Verify custom logic ran
            assert marker_file.exists()
            assert marker_file.read_text() == "Custom logic executed"

            # Verify standard execution worked
            assert result.status == "failed"  # Upstream will fail
            assert result.features_completed == 0  # Nothing completed
            assert result.features_skipped == 1  # Downstream skipped


class TestMixedChainExecution:
    """Test executing chains with mixed YAML and Python migrations."""

    def test_execute_yaml_then_python_chain(
        self,
        tmp_path: Path,
        upstream_downstream_v1: FeatureGraph,
        upstream_downstream_v2: FeatureGraph,
    ):
        """Test executing a chain with YAML migration followed by Python migration."""
        # Setup v1 data
        store_v1 = InMemoryMetadataStore()
        UpstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]
        DownstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "downstream"])
        ]

        with upstream_downstream_v1.use(), store_v1:
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                    ],
                }
            )
            store_v1.write_metadata(UpstreamV1, upstream_data)

            diff = store_v1.resolve_update(DownstreamV1)
            if len(diff.added) > 0:
                store_v1.write_metadata(DownstreamV1, diff.added)

            store_v1.record_feature_graph_snapshot()

        # Create intermediate graph (v1.5)
        temp_module_v15 = TempFeatureModule("test_py_migration_chain_v15")
        upstream_spec_v15 = SampleFeatureSpec(
            key=FeatureKey(["test_py_migration", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1.5"),
            ],
        )
        downstream_spec_v15 = SampleFeatureSpec(
            key=FeatureKey(["test_py_migration", "downstream"]),
            deps=[FeatureDep(feature=FeatureKey(["test_py_migration", "upstream"]))],
            fields=[
                FieldSpec(
                    key=FieldKey(["default"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["test_py_migration", "upstream"]),
                            fields=[FieldKey(["default"])],
                        )
                    ],
                ),
            ],
        )
        temp_module_v15.write_features(
            {"Upstream": upstream_spec_v15, "Downstream": downstream_spec_v15}
        )
        graph_v15 = temp_module_v15.graph

        # Migrate to v1.5
        store_v15 = migrate_store_to_graph(store_v1, graph_v15)

        # Create migrations directory (outside context managers)
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        with graph_v15.use(), store_v15:
            store_v15.record_feature_graph_snapshot()

            # Migration 1: YAML (v1 -> v1.5)
            import yaml

            yaml_migration = {
                "id": "20250107_150000",
                "parent": "initial",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "from_snapshot_version": upstream_downstream_v1.snapshot_version,
                "to_snapshot_version": graph_v15.snapshot_version,
                "ops": [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
            }
            yaml_path = migrations_dir / "20250107_150000.yaml"
            with open(yaml_path, "w") as f:
                yaml.dump(yaml_migration, f)

        # Now migrate to v2
        store_v2 = migrate_store_to_graph(store_v15, upstream_downstream_v2)

        with upstream_downstream_v2.use(), store_v2:
            store_v2.record_feature_graph_snapshot()

            # Migration 2: Python (v1.5 -> v2)
            write_python_diff_migration(
                migrations_dir=migrations_dir,
                migration_id="20250107_160000",
                parent="20250107_150000",
                from_snapshot=graph_v15.snapshot_version,
                to_snapshot=upstream_downstream_v2.snapshot_version,
            )

            # Build and verify chain
            chain = build_migration_chain(migrations_dir)
            assert len(chain) == 2
            assert chain[0].migration_id == "20250107_150000"
            assert chain[1].migration_id == "20250107_160000"

            # Verify mixed types
            from metaxy.migrations.models import DiffMigration

            assert isinstance(chain[0], DiffMigration)
            assert isinstance(chain[1], DiffMigration)

        temp_module_v15.cleanup()


class TestIdempotencyAndFailureRecovery:
    """Test idempotency and recovery from failures."""

    def test_python_migration_idempotency(
        self,
        tmp_path: Path,
        upstream_downstream_v1: FeatureGraph,
        upstream_downstream_v2: FeatureGraph,
    ):
        """Test that Python migrations can be re-run safely (idempotent)."""
        # Setup v1 data
        store_v1 = InMemoryMetadataStore()
        UpstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]
        DownstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "downstream"])
        ]

        with upstream_downstream_v1.use(), store_v1:
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                    ],
                }
            )
            store_v1.write_metadata(UpstreamV1, upstream_data)

            diff = store_v1.resolve_update(DownstreamV1)
            if len(diff.added) > 0:
                store_v1.write_metadata(DownstreamV1, diff.added)

            store_v1.record_feature_graph_snapshot()

        # Migrate to v2
        store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
        UpstreamV2 = upstream_downstream_v2.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]

        with upstream_downstream_v2.use(), store_v2:
            store_v2.record_feature_graph_snapshot()

            # Create Python migration
            migrations_dir = tmp_path / "migrations"
            migrations_dir.mkdir()

            write_python_diff_migration(
                migrations_dir=migrations_dir,
                migration_id="20250107_170000",
                parent="initial",
                from_snapshot=upstream_downstream_v1.snapshot_version,
                to_snapshot=upstream_downstream_v2.snapshot_version,
            )

            # Update upstream
            new_upstream = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"default": "new_h1"},
                        {"default": "new_h2"},
                    ],
                }
            )
            store_v2.write_metadata(UpstreamV2, new_upstream)

            # Load migration
            migration_file = migrations_dir / "20250107_170000.py"
            migration = load_migration_from_file(migration_file)

            # Execute first time
            storage = SystemTableStorage(store_v2)
            executor = MigrationExecutor(storage)
            result1 = executor.execute(
                migration, store_v2, project="default", dry_run=False
            )

            assert result1.status == "failed"  # Upstream fails
            assert result1.features_completed == 0  # Nothing completes
            assert result1.features_failed == 1  # Upstream fails
            assert result1.features_skipped == 1  # Downstream skipped

            # Execute second time (should be idempotent)
            result2 = executor.execute(
                migration, store_v2, project="default", dry_run=False
            )

            # Should have same results (upstream still fails, downstream still skipped)
            assert result2.status == "failed"
            assert result2.features_completed == 0
            assert result2.features_failed == 1
            assert result2.features_skipped == 1


class TestErrorHandling:
    """Test proper error messages when Python migrations fail."""

    def test_python_migration_validation_errors(self, tmp_path: Path):
        """Test that Pydantic validation catches errors in Python migrations."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create Python migration with invalid field (missing required fields)
        invalid_migration = '''"""Invalid migration."""

from metaxy.migrations.models import DiffMigration

class Migration(DiffMigration):
    """Invalid migration missing required fields."""
    migration_id: str = "20250107_180000"
    # Missing: parent, from_snapshot_version, to_snapshot_version, ops
'''

        migration_file = migrations_dir / "20250107_180000.py"
        with open(migration_file, "w") as f:
            f.write(invalid_migration)

        # Try to load - should raise validation error
        with pytest.raises(Exception) as exc_info:
            load_migration_from_file(migration_file)

        # Verify error mentions validation
        error_str = str(exc_info.value)
        assert "validation" in error_str.lower() or "required" in error_str.lower()

    def test_python_migration_with_custom_fields(self, tmp_path: Path):
        """Test Python migration with custom fields (extending DiffMigration)."""
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        # Create Python migration with custom fields
        custom_migration = '''"""Migration with custom fields."""

from datetime import datetime, timezone
from metaxy.migrations.models import DiffMigration

class Migration(DiffMigration):
    """Migration with custom metadata."""
    migration_id: str = "20250107_190000"
    parent: str = "initial"
    created_at: datetime = datetime.now(timezone.utc)
    from_snapshot_version: str = "snap1"
    to_snapshot_version: str = "snap2"
    ops: list = [{"type": "metaxy.migrations.ops.DataVersionReconciliation"}]

    # Custom fields
    author: str = "test@example.com"
    ticket_id: str = "JIRA-123"
    description: str = "Test migration with custom metadata"
'''

        migration_file = migrations_dir / "20250107_190000.py"
        with open(migration_file, "w") as f:
            f.write(custom_migration)

        # Load migration
        migration = load_migration_from_file(migration_file)

        # Verify custom fields are accessible
        assert isinstance(migration, DiffMigration)
        assert migration.migration_id == "20250107_190000"
        # Custom fields should be accessible via __dict__ or model_dump
        migration_dict = migration.model_dump()
        assert migration_dict.get("author") == "test@example.com"
        assert migration_dict.get("ticket_id") == "JIRA-123"


class TestPythonMigrationTypes:
    """Test custom migration types (not DiffMigration)."""

    def test_custom_migration_execution(self, tmp_path: Path):
        """Test executing a custom migration with completely custom logic."""
        # Create custom migration in Python file
        migrations_dir = tmp_path / "migrations"
        migrations_dir.mkdir()

        marker_file = tmp_path / "custom_migration_executed.txt"

        custom_migration = f'''"""Custom migration with user-defined logic."""

from datetime import datetime, timezone
from metaxy.migrations.models import Migration, MigrationResult
from pathlib import Path

class CustomMigration(Migration):
    """Custom migration for data cleanup."""
    migration_id: str = "20250107_200000"
    parent: str = "initial"
    created_at: datetime = datetime.now(timezone.utc)

    # Custom fields
    cleanup_threshold_days: int = 30

    def get_affected_features(self, store, project):
        """Return empty list for this custom migration."""
        return []

    def execute(self, store, project, *, dry_run=False):
        """Custom cleanup logic."""
        # Write marker to prove execution
        marker = Path(r"{marker_file}")
        marker.write_text(f"Cleanup executed with threshold: {{self.cleanup_threshold_days}} days")

        # Return custom result
        return MigrationResult(
            migration_id=self.migration_id,
            status="completed",
            features_completed=0,
            features_failed=0,
            features_skipped=0,
            affected_features=[],
            errors={{}},
            rows_affected=42,  # Custom value
            duration_seconds=0.1,
            timestamp=datetime.now(timezone.utc),
        )
'''

        migration_file = migrations_dir / "20250107_200000.py"
        with open(migration_file, "w") as f:
            f.write(custom_migration)

        # Load migration
        migration = load_migration_from_file(migration_file)

        # Verify it's a Migration
        assert isinstance(migration, Migration)
        assert migration.migration_id == "20250107_200000"

        # Execute with dummy store
        store = InMemoryMetadataStore()
        with store:
            storage = SystemTableStorage(store)
            executor = MigrationExecutor(storage)
            result = executor.execute(
                migration, store, project="default", dry_run=False
            )

            # Verify custom logic ran
            assert marker_file.exists()
            assert "threshold: 30 days" in marker_file.read_text()

            # Verify custom result
            assert result.status == "completed"
            assert result.rows_affected == 42
            assert result.features_completed == 0


class TestDryRunMode:
    """Test dry-run mode with Python migrations."""

    def test_python_migration_dry_run(
        self,
        tmp_path: Path,
        upstream_downstream_v1: FeatureGraph,
        upstream_downstream_v2: FeatureGraph,
    ):
        """Test that dry_run=True doesn't modify data."""
        # Setup v1 data
        store_v1 = InMemoryMetadataStore()
        UpstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]
        DownstreamV1 = upstream_downstream_v1.features_by_key[
            FeatureKey(["test_py_migration", "downstream"])
        ]

        # Will be populated in first context, used in second
        initial_downstream = pl.DataFrame()

        with upstream_downstream_v1.use(), store_v1:
            upstream_data = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"default": "h1"},
                        {"default": "h2"},
                    ],
                }
            )
            store_v1.write_metadata(UpstreamV1, upstream_data)

            diff = store_v1.resolve_update(DownstreamV1)
            if len(diff.added) > 0:
                store_v1.write_metadata(DownstreamV1, diff.added)

            store_v1.record_feature_graph_snapshot()

            # Get initial downstream data for later comparison
            initial_downstream = collect_to_polars(
                store_v1.read_metadata(DownstreamV1, current_only=False)
            )

        # Migrate to v2
        store_v2 = migrate_store_to_graph(store_v1, upstream_downstream_v2)
        UpstreamV2 = upstream_downstream_v2.features_by_key[
            FeatureKey(["test_py_migration", "upstream"])
        ]
        DownstreamV2 = upstream_downstream_v2.features_by_key[
            FeatureKey(["test_py_migration", "downstream"])
        ]

        with upstream_downstream_v2.use(), store_v2:
            store_v2.record_feature_graph_snapshot()

            # Update upstream
            new_upstream = pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "metaxy_provenance_by_field": [
                        {"default": "new_h1"},
                        {"default": "new_h2"},
                    ],
                }
            )
            store_v2.write_metadata(UpstreamV2, new_upstream)

            # Create and execute with dry_run=True
            migrations_dir = tmp_path / "migrations"
            migrations_dir.mkdir()

            write_python_diff_migration(
                migrations_dir=migrations_dir,
                migration_id="20250107_210000",
                parent="initial",
                from_snapshot=upstream_downstream_v1.snapshot_version,
                to_snapshot=upstream_downstream_v2.snapshot_version,
            )

            migration_file = migrations_dir / "20250107_210000.py"
            migration = load_migration_from_file(migration_file)

            storage = SystemTableStorage(store_v2)
            executor = MigrationExecutor(storage)
            result = executor.execute(
                migration, store_v2, project="default", dry_run=True
            )

            # Verify dry run status
            assert result.status == "skipped"
            # In dry run, no rows are affected and features are counted differently
            # Upstream fails, downstream skipped
            assert result.features_completed == 0
            assert result.features_failed == 1
            assert result.features_skipped == 1

            # Verify data unchanged
            final_downstream = collect_to_polars(
                store_v2.read_metadata(DownstreamV2, current_only=False)
            )

            # Compare data (should be identical)
            assert len(final_downstream) == len(initial_downstream)
            assert set(final_downstream["sample_uid"]) == set(
                initial_downstream["sample_uid"]
            )
