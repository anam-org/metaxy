"""Tests for cross-project write validation in metadata stores.

Tests the project isolation feature that prevents writing to features from
different projects unless explicitly allowed via allow_cross_project_writes().
"""

from __future__ import annotations

import polars as pl
import pytest
from syrupy.assertion import SnapshotAssertion

from metaxy import Feature, FeatureKey, FeatureSpec, FieldKey, FieldSpec
from metaxy.config import MetaxyConfig
from metaxy.metadata_store import InMemoryMetadataStore
from metaxy.models.feature import FeatureGraph


def test_write_to_same_project_succeeds(snapshot: SnapshotAssertion) -> None:
    """Test that writing to a feature from the same project succeeds."""
    config = MetaxyConfig(project="test_project")
    MetaxyConfig.set(config)

    try:

        class TestFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        # Create metadata
        import narwhals as nw

        metadata = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2, 3],
                    "data_version": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                        {"default": "hash3"},
                    ],
                }
            )
        )

        with InMemoryMetadataStore() as store:
            # Write should succeed (same project)
            store.write_metadata(TestFeature, metadata)

            # Verify write succeeded
            result = store.read_metadata(TestFeature)
            assert result.collect().shape[0] == 3

    finally:
        MetaxyConfig.reset()


def test_write_to_different_project_fails() -> None:
    """Test that writing to a feature from a different project fails."""
    # Create feature in project_a
    config_a = MetaxyConfig(project="project_a")
    MetaxyConfig.set(config_a)

    class FeatureA(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
    ):
        pass

    # Switch to project_b
    config_b = MetaxyConfig(project="project_b")
    MetaxyConfig.set(config_b)

    # Try to write to FeatureA (which belongs to project_a)
    import narwhals as nw

    metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        )
    )

    with InMemoryMetadataStore() as store:
        # Write should fail (different project)
        with pytest.raises(
            ValueError,
            match="Cannot write to feature .* from project 'project_a' when the global configuration expects project 'project_b'",
        ):
            store.write_metadata(FeatureA, metadata)

    MetaxyConfig.reset()


def test_allow_cross_project_writes_context_manager() -> None:
    """Test that allow_cross_project_writes() context manager enables cross-project writes."""
    # Create feature in project_a
    config_a = MetaxyConfig(project="project_a")
    MetaxyConfig.set(config_a)

    class FeatureA(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
    ):
        pass

    # Switch to project_b
    config_b = MetaxyConfig(project="project_b")
    MetaxyConfig.set(config_b)

    import narwhals as nw

    metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        )
    )

    with InMemoryMetadataStore() as store:
        # With context manager, write should succeed
        with store.allow_cross_project_writes():
            store.write_metadata(FeatureA, metadata)

        # Verify write succeeded
        result = store.read_metadata(FeatureA, current_only=False)
        assert result.collect().shape[0] == 3

    MetaxyConfig.reset()


def test_system_tables_exempt_from_project_validation() -> None:
    """Test that system tables (metaxy-system) are exempt from project validation."""
    from metaxy.metadata_store.system_tables import SYSTEM_NAMESPACE

    config = MetaxyConfig(project="test_project")
    MetaxyConfig.set(config)

    # Create a system table feature key
    system_key = FeatureKey([SYSTEM_NAMESPACE, "test_table"])

    import narwhals as nw

    metadata = nw.from_native(
        pl.DataFrame(
            {
                "test_column": [1, 2, 3],
            }
        )
    )

    with InMemoryMetadataStore() as store:
        # System tables should not require project validation
        store.write_metadata(system_key, metadata)

        # Verify write succeeded
        result = store.read_metadata(system_key, current_only=False)
        assert result.collect().shape[0] == 3

    MetaxyConfig.reset()


def test_write_multiple_features_same_project() -> None:
    """Test writing multiple features from the same project."""
    config = MetaxyConfig(project="multi_feature_project")
    MetaxyConfig.set(config)

    try:

        class Feature1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature1"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        class Feature2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature2"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        import narwhals as nw

        metadata1 = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "data_version": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                    ],
                }
            )
        )

        metadata2 = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "data_version": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                    ],
                }
            )
        )

        with InMemoryMetadataStore() as store:
            # Both writes should succeed (same project)
            store.write_metadata(Feature1, metadata1)
            store.write_metadata(Feature2, metadata2)

            # Verify both writes succeeded
            result1 = store.read_metadata(Feature1)
            result2 = store.read_metadata(Feature2)

            assert result1.collect().shape[0] == 2
            assert result2.collect().shape[0] == 2

    finally:
        MetaxyConfig.reset()


def test_cross_project_write_during_migration() -> None:
    """Test that migrations can write to features from different projects.

    This simulates a migration scenario where we need to reconcile metadata
    for features that have moved between projects.
    """
    graph = FeatureGraph()

    # Keep graph active throughout the test
    with graph.use():
        # Create features from two different projects
        config_a = MetaxyConfig(project="project_a")
        MetaxyConfig.set(config_a)

        class FeatureA(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_a"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        config_b = MetaxyConfig(project="project_b")
        MetaxyConfig.set(config_b)

        class FeatureB(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["feature_b"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
            ),
        ):
            pass

        # Now simulate a migration writing to both features
        # Set config to project_a
        MetaxyConfig.set(config_a)

        import narwhals as nw

        metadata_a = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "data_version": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                    ],
                }
            )
        )

        metadata_b = nw.from_native(
            pl.DataFrame(
                {
                    "sample_uid": [1, 2],
                    "data_version": [
                        {"default": "hash1"},
                        {"default": "hash2"},
                    ],
                }
            )
        )

        with InMemoryMetadataStore() as store:
            # Write to FeatureA should succeed (same project)
            store.write_metadata(FeatureA, metadata_a)

            # Write to FeatureB should fail (different project) without context manager
            with pytest.raises(ValueError, match="Cannot write to feature"):
                store.write_metadata(FeatureB, metadata_b)

            # But with allow_cross_project_writes, should succeed
            with store.allow_cross_project_writes():
                # Only write to FeatureB here (FeatureA already written above)
                store.write_metadata(FeatureB, metadata_b)

            # Verify both writes succeeded (inside store context)
            result_a = store.read_metadata(FeatureA, current_only=False)
            result_b = store.read_metadata(FeatureB, current_only=False)

            assert result_a.collect().shape[0] == 2
            assert result_b.collect().shape[0] == 2

    MetaxyConfig.reset()


def test_project_validation_with_feature_key() -> None:
    """Test that project validation works when passing FeatureKey directly."""
    # Create a feature in project_a
    config_a = MetaxyConfig(project="project_a")
    MetaxyConfig.set(config_a)

    # Use the active graph so the feature is registered globally
    class FeatureA(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
    ):
        pass

    # Switch to project_b
    config_b = MetaxyConfig(project="project_b")
    MetaxyConfig.set(config_b)

    import narwhals as nw

    metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "data_version": [
                    {"default": "hash1"},
                    {"default": "hash2"},
                    {"default": "hash3"},
                ],
            }
        )
    )

    with InMemoryMetadataStore() as store:
        # Pass FeatureKey instead of Feature class
        feature_key = FeatureKey(["test", "feature"])

        # Should still validate project
        with pytest.raises(ValueError, match="Cannot write to feature"):
            store.write_metadata(feature_key, metadata)

    MetaxyConfig.reset()


def test_nested_cross_project_writes_context_managers() -> None:
    """Test that nested allow_cross_project_writes() context managers work correctly."""
    config_a = MetaxyConfig(project="project_a")
    MetaxyConfig.set(config_a)

    class FeatureA(
        Feature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "feature"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version=1)],
        ),
    ):
        pass

    # Switch to project_b
    config_b = MetaxyConfig(project="project_b")
    MetaxyConfig.set(config_b)

    import narwhals as nw

    metadata = nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1],
                "data_version": [{"default": "hash1"}],
            }
        )
    )

    with InMemoryMetadataStore() as store:
        # Nested context managers should work
        with store.allow_cross_project_writes():
            # Inner context manager
            with store.allow_cross_project_writes():
                store.write_metadata(FeatureA, metadata)

            # Still within outer context manager
            store.write_metadata(FeatureA, metadata)

        # Outside context manager, should fail again
        with pytest.raises(ValueError, match="Cannot write to feature"):
            store.write_metadata(FeatureA, metadata)

    MetaxyConfig.reset()
