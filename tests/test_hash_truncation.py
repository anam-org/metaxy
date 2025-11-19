"""Tests for global hash truncation feature."""

import hashlib
from contextvars import copy_context

import polars as pl
import pytest

from metaxy.config import MetaxyConfig
from metaxy.metadata_store.memory import InMemoryMetadataStore
from metaxy.metadata_store.system import SystemTableStorage
from metaxy.models.feature import TestingFeature
from metaxy.models.feature_spec import FieldSpec, SampleFeatureSpec
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.utils.hashing import (
    MIN_TRUNCATION_LENGTH,
    ensure_hash_compatibility,
    get_hash_truncation_length,
    truncate_hash,
    truncate_string_column,
    truncate_struct_column,
)


class TestHashTruncationUtils:
    """Test hash truncation utility functions."""

    def test_truncate_hash_basic(self):
        """Test basic hash truncation."""
        full_hash = "a" * 64

        # Test with different config settings
        config = MetaxyConfig(hash_truncation_length=16)
        with config.use():
            assert truncate_hash(full_hash) == "a" * 16

        config = MetaxyConfig(hash_truncation_length=8)
        with config.use():
            assert truncate_hash(full_hash) == "a" * 8

        config = MetaxyConfig(hash_truncation_length=32)
        with config.use():
            assert truncate_hash(full_hash) == "a" * 32

    def test_truncate_hash_shorter_than_length(self):
        """Test truncation when hash is shorter than truncation length."""
        short_hash = "abc123def789"  # 12 characters

        # No truncation (hash is shorter)
        config = MetaxyConfig(hash_truncation_length=20)
        with config.use():
            assert truncate_hash(short_hash) == "abc123def789"

        # Exact length
        config = MetaxyConfig(hash_truncation_length=12)
        with config.use():
            assert truncate_hash(short_hash) == "abc123def789"

        # Truncated to minimum
        config = MetaxyConfig(hash_truncation_length=8)
        with config.use():
            assert truncate_hash(short_hash) == "abc123de"

    def test_truncate_hash_minimum_length(self):
        """Test minimum truncation length validation in config."""
        full_hash = "a" * 64

        # Test that config validation prevents invalid values
        with pytest.raises(ValueError, match="at least 8 characters"):
            MetaxyConfig(hash_truncation_length=7)
        with pytest.raises(ValueError, match="at least 8 characters"):
            MetaxyConfig(hash_truncation_length=0)

        # Minimum allowed should work
        config = MetaxyConfig(hash_truncation_length=MIN_TRUNCATION_LENGTH)
        with config.use():
            assert truncate_hash(full_hash) == "a" * MIN_TRUNCATION_LENGTH

    def test_truncate_hash_none_length(self):
        """Test no truncation when length is None."""
        full_hash = "a" * 64

        # With no truncation config
        config = MetaxyConfig(hash_truncation_length=None)
        with config.use():
            assert truncate_hash(full_hash) == full_hash

    def test_global_truncation_setting(self):
        """Test global hash truncation via MetaxyConfig."""
        # Initial state - default truncation (64)
        MetaxyConfig.reset()
        assert get_hash_truncation_length() == 64

        # Set truncation length via config
        config = MetaxyConfig(hash_truncation_length=16)
        with config.use():
            assert get_hash_truncation_length() == 16

            # Truncate using global setting
            full_hash = "a" * 64
            assert truncate_hash(full_hash) == "a" * 16

        # Reset to None
        MetaxyConfig.reset()
        assert get_hash_truncation_length() == 64
        full_hash = "a" * 64
        assert truncate_hash(full_hash) == full_hash

    def test_global_truncation_minimum_validation(self):
        """Test validation of global truncation length."""
        with pytest.raises(ValueError, match="at least 8 characters"):
            MetaxyConfig(hash_truncation_length=7)

    def test_ensure_hash_compatibility(self):
        """Test hash compatibility checking."""
        # Exact match
        assert ensure_hash_compatibility("abc123", "abc123") is True

        # Truncated version
        full_hash = "abc123456789"
        truncated = "abc12345"
        assert ensure_hash_compatibility(full_hash, truncated) is True
        assert ensure_hash_compatibility(truncated, full_hash) is True

        # Different hashes
        assert ensure_hash_compatibility("abc123", "def456") is False

    def test_context_isolation(self):
        """Test that hash truncation setting works with MetaxyConfig context."""
        # Set in main context
        original_config = MetaxyConfig(hash_truncation_length=16)
        with original_config.use():
            assert get_hash_truncation_length() == 16

            # Create new context - should inherit setting
            ctx = copy_context()

            def check_in_new_context():
                # Should inherit from parent context
                assert get_hash_truncation_length() == 16
                # Change in this context
                new_config = MetaxyConfig(hash_truncation_length=12)
                with new_config.use():
                    assert get_hash_truncation_length() == 12

            ctx.run(check_in_new_context)

            # Main context should be unchanged
            assert get_hash_truncation_length() == 16


class TestNarwhalsFunctions:
    """Test Narwhals-based hash truncation functions."""

    def test_truncate_string_column(self):
        """Test truncating a string column containing hashes."""
        # Create test data with hash-like strings
        df = pl.DataFrame(
            {"hash_column": ["a" * 64, "b" * 64, "c" * 64], "other_column": [1, 2, 3]}
        )

        # No truncation when config is None
        MetaxyConfig.reset()
        # The function accepts the native DataFrame directly, thanks to @nw.narwhalify
        result_pl = truncate_string_column(df, "hash_column")
        assert result_pl["hash_column"][0] == "a" * 64

        # With truncation
        config = MetaxyConfig(hash_truncation_length=12)
        with config.use():
            result_pl = truncate_string_column(df, "hash_column")

            assert result_pl["hash_column"][0] == "a" * 12
            assert result_pl["hash_column"][1] == "b" * 12
            assert result_pl["hash_column"][2] == "c" * 12

            # Other columns unchanged
            assert result_pl["other_column"].to_list() == [1, 2, 3]

    def test_truncate_struct_column(self):
        """Test truncating hash values within a struct column."""
        # Create test data with struct containing hash values
        df = pl.DataFrame(
            {
                "metaxy_provenance_by_field": [
                    {"field1": "a" * 64, "field2": "b" * 64},
                    {"field1": "c" * 64, "field2": "d" * 64},
                ],
                "sample_uid": [1, 2],
            }
        )

        # No truncation when config is None
        MetaxyConfig.reset()
        result_pl = truncate_struct_column(df, "metaxy_provenance_by_field")
        assert result_pl["metaxy_provenance_by_field"][0]["field1"] == "a" * 64

        # With truncation
        config = MetaxyConfig(hash_truncation_length=16)
        with config.use():
            result_pl = truncate_struct_column(df, "metaxy_provenance_by_field")

            # Check struct values are truncated
            assert result_pl["metaxy_provenance_by_field"][0]["field1"] == "a" * 16
            assert result_pl["metaxy_provenance_by_field"][0]["field2"] == "b" * 16
            assert result_pl["metaxy_provenance_by_field"][1]["field1"] == "c" * 16
            assert result_pl["metaxy_provenance_by_field"][1]["field2"] == "d" * 16

            # Other columns unchanged
            assert result_pl["sample_uid"].to_list() == [1, 2]

    def test_truncate_functions_with_empty_df(self):
        """Test truncate functions with empty DataFrames."""
        # Setup truncation
        config = MetaxyConfig(hash_truncation_length=10)
        with config.use():
            # Empty DataFrame with hash column
            df = pl.DataFrame({"hash_column": pl.Series([], dtype=pl.Utf8)})

            result_pl = truncate_string_column(df, "hash_column")
            assert result_pl.height == 0

            # Empty DataFrame with struct column
            df = pl.DataFrame(
                {
                    "metaxy_provenance_by_field": pl.Series(
                        [], dtype=pl.Struct({"field1": pl.Utf8})
                    )
                }
            )

            result_pl = truncate_struct_column(df, "metaxy_provenance_by_field")
            assert result_pl.height == 0


class TestFeatureVersionTruncation:
    """Test hash truncation in feature versioning."""

    def test_feature_version_truncation(self, graph):
        """Test that feature versions are truncated."""

        # Create feature without truncation
        class TestFeature1(
            TestingFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature1"]),
                fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                # Root feature has no dependencies
            ),
        ):
            pass

        version_full = TestFeature1.feature_version()
        assert len(version_full) == 64  # SHA256 hex digest length

        # Enable truncation
        config = MetaxyConfig(hash_truncation_length=16)
        with config.use():

            class TestFeature2(
                TestingFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature2"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                    # Root feature has no dependencies
                ),
            ):
                pass

            version_truncated = TestFeature2.feature_version()
            assert len(version_truncated) == 16

    def test_snapshot_version_truncation(self, graph):
        """Test that snapshot versions are truncated."""

        class TestFeature(
            TestingFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
            ),
        ):
            pass

        # Get snapshot without truncation
        snapshot_full = graph.snapshot_version
        assert len(snapshot_full) == 64

        # Enable truncation
        config = MetaxyConfig(hash_truncation_length=12)
        with config.use():
            snapshot_truncated = graph.snapshot_version
            assert len(snapshot_truncated) == 12
            # Note: The truncated version is computed fresh with truncated dependencies,
            # not just a truncation of the full version, so they may differ

    def test_field_version_truncation(self, graph):
        """Test that field versions are truncated."""

        class TestFeature(
            TestingFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                    FieldSpec(key=FieldKey(["field2"]), code_version="2"),
                ],
            ),
        ):
            pass

        # Get field versions without truncation
        provenance_full = TestFeature.provenance_by_field()
        assert all(len(v) == 64 for v in provenance_full.values())

        # Enable truncation
        config = MetaxyConfig(hash_truncation_length=20)
        with config.use():
            provenance_truncated = TestFeature.provenance_by_field()
            assert all(len(v) == 20 for v in provenance_truncated.values())


class TestConfigIntegration:
    """Test hash truncation configuration integration."""

    def test_config_sets_global_truncation(self, tmp_path):
        """Test that MetaxyConfig stores truncation length."""
        # Create config with truncation
        config_file = tmp_path / "metaxy.toml"
        config_file.write_text("""
hash_truncation_length = 16

[stores.dev]
type = "metaxy.metadata_store.memory.InMemoryMetadataStore"
""")

        # Load config - should store truncation setting
        config = MetaxyConfig.load(config_file)
        assert config.hash_truncation_length == 16

        # The hash truncation is now retrieved via MetaxyConfig.get()
        assert get_hash_truncation_length() == 16

        # Clean up - reset config
        MetaxyConfig.reset()

    def test_config_validation(self):
        """Test config validation for truncation length."""
        with pytest.raises(ValueError, match="at least 8 characters"):
            MetaxyConfig(hash_truncation_length=7)

        # Valid values should work
        config = MetaxyConfig(hash_truncation_length=8)
        assert config.hash_truncation_length == 8

        config = MetaxyConfig(hash_truncation_length=None)
        assert config.hash_truncation_length is None

    def test_environment_variable(self):
        """Test setting truncation via environment variable."""
        import os

        # Set via environment
        os.environ["METAXY_HASH_TRUNCATION_LENGTH"] = "24"
        config = MetaxyConfig()
        assert config.hash_truncation_length == 24

        # Clean up
        del os.environ["METAXY_HASH_TRUNCATION_LENGTH"]
        MetaxyConfig.reset()


class TestMetadataStoreTruncation:
    """Test hash truncation in metadata stores."""

    def test_store_truncation_property(self):
        """Test that stores have hash_truncation_length property that pulls from config."""
        # Default truncation (64)
        MetaxyConfig.reset()
        with InMemoryMetadataStore() as store:
            assert store.hash_truncation_length == 64

        # With global truncation
        config = MetaxyConfig(hash_truncation_length=16)
        with config.use():
            with InMemoryMetadataStore() as store:
                assert store.hash_truncation_length == 16

        # Different config value
        config = MetaxyConfig(hash_truncation_length=20)
        with config.use():
            with InMemoryMetadataStore() as store:
                assert store.hash_truncation_length == 20

    def test_provenance_truncation(self, graph):
        """Test that field provenances are truncated in stores."""

        # Create feature
        class TestFeature(
            TestingFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
            ),
        ):
            pass

        # Enable truncation (preserve project from test setup)
        config = MetaxyConfig(project="test", hash_truncation_length=16)
        with config.use():
            # Store should use truncated field provenances
            with InMemoryMetadataStore() as store:
                # Write some dummy metadata with provenance_by_field
                metadata = pl.DataFrame(
                    {
                        "sample_uid": ["s1", "s2"],
                        "metaxy_provenance_by_field": [
                            {"field1": "a" * 16},  # Already truncated
                            {"field1": "b" * 16},  # Already truncated
                        ],
                    }
                )
                store.write_metadata(TestFeature, metadata)

                # Read back and verify

                result = store.read_metadata(TestFeature).collect()
                result_pl = result.to_polars()
                assert result_pl.height == 2

                # Field provenance should be truncated
                for row in result_pl.iter_rows(named=True):
                    provenance_by_field = row["metaxy_provenance_by_field"]
                    assert len(provenance_by_field["field1"]) == 16


class TestMigrationCompatibility:
    """Test migration detection with hash truncation."""

    def test_migration_with_truncation(self, graph):
        """Test that migration detection works with truncated hashes."""
        from metaxy.migrations.detector import detect_diff_migration

        # Enable truncation (preserve project from test setup)
        config = MetaxyConfig(project="test", hash_truncation_length=12)
        with config.use():

            class TestFeature(
                TestingFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["test", "feature"]),
                    fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
                ),
            ):
                pass

            # Record snapshot
            with InMemoryMetadataStore() as store:
                result = SystemTableStorage(store).push_graph_snapshot()

                snapshot_v1 = result.snapshot_version

                assert len(snapshot_v1) == 12  # Truncated

                # Modify feature to trigger migration
                graph.remove_feature(FeatureKey(["test", "feature"]))

                class TestFeature(  # noqa: F811
                    TestingFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["test", "feature"]),
                        fields=[
                            FieldSpec(key=FieldKey(["field1"]), code_version="2")
                        ],  # Changed
                    ),
                ):
                    pass

                # Detect migration - should work with truncated versions
                migration = detect_diff_migration(
                    store,
                    project="test",  # Use the same project as in config
                    ops=[{"type": "metaxy.migrations.ops.DataVersionReconciliation"}],
                )
                assert migration is not None
                assert len(migration.from_snapshot_version) == 12
                assert len(migration.to_snapshot_version) == 12

    def test_hash_compatibility_in_migration(self):
        """Test hash compatibility checking in migrations."""
        from metaxy.utils.hashing import ensure_hash_compatibility

        # Full hashes
        hash1 = hashlib.sha256(b"test1").hexdigest()
        hash2 = hashlib.sha256(b"test2").hexdigest()

        # Not compatible when different
        assert not ensure_hash_compatibility(hash1, hash2)

        # Compatible when one is truncation of the other
        truncated = hash1[:16]
        assert ensure_hash_compatibility(hash1, truncated)
        assert ensure_hash_compatibility(truncated, hash1)


class TestEndToEnd:
    """End-to-end tests with hash truncation."""

    def test_full_workflow_with_truncation(self, graph):
        """Test complete workflow with hash truncation enabled."""
        # Enable truncation globally (preserve project from test setup)
        config = MetaxyConfig(project="test", hash_truncation_length=16)
        with config.use():
            # Create features
            class ParentFeature(
                TestingFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["parent"]),
                    fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                ),
            ):
                pass

            from metaxy.models.feature_spec import FeatureDep

            class ChildFeature(
                TestingFeature,
                spec=SampleFeatureSpec(
                    key=FeatureKey(["child"]),
                    fields=[FieldSpec(key=FieldKey(["derived"]), code_version="1")],
                    deps=[FeatureDep(feature=FeatureKey(["parent"]))],
                ),
            ):
                pass

            # Verify truncation
            assert len(ParentFeature.feature_version()) == 16
            assert len(ChildFeature.feature_version()) == 16
            assert len(graph.snapshot_version) == 16

            # Store metadata
            with InMemoryMetadataStore() as store:
                # Record snapshot
                result = SystemTableStorage(store).push_graph_snapshot()

                snapshot_version = result.snapshot_version

                assert len(snapshot_version) == 16

                # Write parent metadata
                parent_data = pl.DataFrame(
                    {
                        "sample_uid": ["s1", "s2"],
                        "metaxy_provenance_by_field": [
                            {"value": "h" * 16},  # Truncated hash
                            {"value": "i" * 16},  # Truncated hash
                        ],
                    }
                )
                store.write_metadata(ParentFeature, parent_data)

                # Verify stored versions are truncated

                result = store.read_metadata(ParentFeature).collect()
                result_pl = result.to_polars()

                for row in result_pl.iter_rows(named=True):
                    assert len(row["metaxy_feature_version"]) == 16
                    assert len(row["metaxy_snapshot_version"]) == 16
                    assert len(row["metaxy_provenance_by_field"]["value"]) == 16
