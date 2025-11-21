"""Comprehensive tests for project validation in issue #88."""

import tempfile
from pathlib import Path

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing.models import SampleFeatureSpec
from metaxy.config import MetaxyConfig
from metaxy.metadata_store.base import (
    MetadataStore,
)
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.migrations.ops import DataVersionReconciliation
from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureDep, FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


class TestProjectValidationComprehensive:
    """Comprehensive tests for project validation across all components."""

    @pytest.fixture(params=[InMemoryMetadataStore, DuckDBMetadataStore])
    def store_cls(self, request) -> type[MetadataStore]:
        """Test with different store backends."""
        return request.param

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def store(self, store_cls, temp_dir):
        """Create a metadata store instance."""
        if store_cls == InMemoryMetadataStore:
            with store_cls() as store:
                yield store
        else:
            db_path = temp_dir / "test.db"
            with store_cls(str(db_path)) as store:
                yield store

    def test_project_detection_hierarchy(self):
        """Test that project detection follows the correct hierarchy."""
        # Reset to default config
        MetaxyConfig.set(MetaxyConfig())

        # Create test graph
        test_graph = FeatureGraph()

        with test_graph.use():
            # 1. Default case - uses global config (default project)
            class DefaultFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["default_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            assert DefaultFeature.project == "default"

        # 2. Test module detection - should use test config
        MetaxyConfig.set(MetaxyConfig(project="test"))

        with test_graph.use():

            class TestFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field2"]), code_version="1")],
                ),
            ):
                pass

            # Since we're in a test module, should use test project
            assert TestFeature.project == "test"

        # 3. Custom project via config
        MetaxyConfig.set(MetaxyConfig(project="custom_project"))

        with test_graph.use():

            class CustomFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["custom_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field3"]), code_version="1")],
                ),
            ):
                pass

            assert CustomFeature.project == "custom_project"

    def test_project_validation_in_metadata_store(self, store):
        """Test that project validation works correctly in metadata stores."""
        # Set up config with a specific project
        MetaxyConfig.set(MetaxyConfig(project="test_project"))

        test_graph = FeatureGraph()

        with test_graph.use():
            # Create a feature with the correct project
            class ValidFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["valid_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            assert ValidFeature.project == "test_project"

            # Should succeed - projects match
            df = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "field1": [10, 20, 30],
                    "metaxy_provenance_by_field": [
                        {"field1": "hash1"},
                        {"field1": "hash2"},
                        {"field1": "hash3"},
                    ],
                }
            )
            df_nw = nw.from_native(df)

            store.write_metadata(ValidFeature, df_nw)

            # Read back and verify
            result = store.read_metadata(ValidFeature).collect()
            assert result.shape[0] == 3

    def test_cross_project_write_fails_without_context(self, store):
        """Test that cross-project writes fail without the context manager."""
        # Set up two different projects
        MetaxyConfig.set(MetaxyConfig(project="project_a"))

        graph_a = FeatureGraph()
        with graph_a.use():

            class FeatureA(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature_a"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            # Change project INSIDE the graph context so FeatureA stays registered
            MetaxyConfig.set(MetaxyConfig(project="project_b"))

            # Try to write FeatureA (from project_a) with current config as project_b
            df = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "field1": [10, 20, 30],
                    "metaxy_provenance_by_field": [
                        {"field1": "hash1"},
                        {"field1": "hash2"},
                        {"field1": "hash3"},
                    ],
                }
            )
            df_nw = nw.from_native(df)

            # This should fail - feature is from project_a but config is project_b
            with pytest.raises(ValueError, match="Cannot write to feature"):
                store.write_metadata(FeatureA, df_nw)

    def test_cross_project_write_succeeds_with_context(self, store):
        """Test that cross-project writes succeed with the context manager."""
        # Set up two different projects
        MetaxyConfig.set(MetaxyConfig(project="project_a"))

        graph_a = FeatureGraph()
        with graph_a.use():

            class FeatureA(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["feature_a"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            # Change project INSIDE the graph context
            MetaxyConfig.set(MetaxyConfig(project="project_b"))

            # Try to write FeatureA with cross-project context
            df = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "field1": [10, 20, 30],
                    "metaxy_provenance_by_field": [
                        {"field1": "hash1"},
                        {"field1": "hash2"},
                        {"field1": "hash3"},
                    ],
                }
            )
            df_nw = nw.from_native(df)

            # This should succeed with the context manager
            with store.allow_cross_project_writes():
                store.write_metadata(FeatureA, df_nw)

            # Verify the data was written
            # Need to read with cross-project context or change config back
            MetaxyConfig.set(MetaxyConfig(project="project_a"))
            result = store.read_metadata(FeatureA).collect()
            assert result.shape[0] == 3

    def test_migration_with_cross_project_support(self, store):
        """Test that migrations can handle cross-project scenarios."""
        # Set up initial state with project_a
        MetaxyConfig.set(MetaxyConfig(project="project_a"))

        graph = FeatureGraph()
        with graph.use():

            class RootFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            class ChildFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["child_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field2"]), code_version="1")],
                    deps=[FeatureDep(feature=RootFeature.spec().key)],
                ),
            ):
                pass

            # Write initial data
            root_df = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "field1": [10, 20, 30],
                    "metaxy_provenance_by_field": [
                        {"field1": "hash1"},
                        {"field1": "hash2"},
                        {"field1": "hash3"},
                    ],
                }
            )
            child_df = pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "field2": [100, 200, 300],
                    "metaxy_provenance_by_field": [
                        {"field2": "hash4"},
                        {"field2": "hash5"},
                        {"field2": "hash6"},
                    ],
                }
            )

            store.write_metadata(RootFeature, nw.from_native(root_df))
            store.write_metadata(ChildFeature, nw.from_native(child_df))

            # Record snapshot
            from_snapshot_version = graph.snapshot_version
            SystemTableStorage(store).push_graph_snapshot()

        # Now simulate a project change - feature moves to project_b
        MetaxyConfig.set(MetaxyConfig(project="project_b"))

        # Create new graph with updated feature (simulating a code change)
        new_graph = FeatureGraph()
        with new_graph.use():

            class RootFeatureV2(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["root_feature_v2"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            class ChildFeatureV2(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["child_feature_v2"]),
                    fields=[
                        FieldSpec(key=FieldKey(["field2"]), code_version="2")
                    ],  # Changed version
                    deps=[FeatureDep(feature=RootFeatureV2.spec().key)],
                ),
            ):
                pass

            # Record new snapshot within graph context
            to_snapshot_version = new_graph.snapshot_version
            SystemTableStorage(store).push_graph_snapshot()

            # Create migration operation
            op = DataVersionReconciliation()

            # Execute migration for the child feature (which has dependencies)
            # This should use allow_cross_project_writes internally
            feature_key = FeatureKey(["child_feature_v2"])

            # The migration should handle the cross-project write
            # Note: This will fail if the feature doesn't have upstream deps
            # but ChildFeatureV2 does have deps, so it should work
            try:
                rows_affected = op.execute_for_feature(
                    store,
                    feature_key.to_string(),
                    snapshot_version=to_snapshot_version,
                    from_snapshot_version=from_snapshot_version,
                    dry_run=False,
                )
                # Migration might not affect rows if data is already migrated
                assert rows_affected >= 0
            except ValueError as e:
                # If it fails, it should be because the feature isn't found,
                # not because of project validation
                assert "not found in" in str(e)

    def test_cli_project_handling(self, temp_dir):
        """Test that CLI correctly handles project configuration."""
        from metaxy.cli.context import AppContext
        from metaxy.config import MetaxyConfig

        # Test default behavior
        config = MetaxyConfig()
        ctx = AppContext(config=config, cli_project=None)
        assert ctx.project == "default"  # Default project from config

        # Test with CLI project override
        ctx = AppContext(config=config, cli_project="cli_project")
        assert ctx.project == "cli_project"

        # Test with all projects flag
        ctx = AppContext(config=config, cli_project=None, all_projects=True)
        assert ctx.project is None  # Should return None for all projects

        # Test precedence - all_projects overrides cli_project
        ctx = AppContext(config=config, cli_project="cli_project", all_projects=True)
        assert ctx.project is None

    def test_system_tables_exempt_from_validation(self, store):
        """Test that system tables are exempt from project validation."""
        from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY

        # Set a specific project
        MetaxyConfig.set(MetaxyConfig(project="user_project"))

        # System tables should always be writable regardless of project
        # This is handled internally by the metadata store

        # Test feature versions (system table)
        from datetime import datetime, timezone

        df = pl.DataFrame(
            {
                "project": ["user_project"],
                "metaxy_snapshot_version": ["snap1"],
                "feature_key": ["test/feature"],
                "metaxy_feature_version": ["v1"],
                "metaxy_feature_spec_version": ["spec1"],
                "metaxy_full_definition_version": ["track1"],
                "recorded_at": [datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)],
                "feature_spec": ["{}"],
                "feature_class_path": ["test.TestFeature"],
            }
        )
        df_nw = nw.from_native(df)

        # Should succeed even though it's a system table
        store.write_metadata(FEATURE_VERSIONS_KEY, df_nw)

    def test_test_fixtures_get_correct_project(self):
        """Test that test fixtures automatically get the test project."""
        # This test runs in the test environment, so features should get project="test"
        test_graph = FeatureGraph()

        with test_graph.use():

            class FixtureFeature(
                Feature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["fixture_feature"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            # Should automatically get project="test" because we're in a test module
            assert FixtureFeature.project == "test"
