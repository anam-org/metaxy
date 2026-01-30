"""Tests for tags field and feature_schema field in feature_versions system table (GitHub issue #149)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from metaxy_testing.models import SampleFeatureSpec

from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._version import __version__
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY, SystemTableStorage
from metaxy.models.feature import FeatureGraph


def test_push_graph_snapshot_with_default_tags(tmp_path: Path):
    """Test that push_graph_snapshot() automatically adds metaxy_version."""
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Push without explicit tags
            SystemTableStorage(store).push_graph_snapshot()

            # Read and verify tags field
            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            assert versions_df.height == 1
            tags_json = versions_df["tags"][0]
            tags = json.loads(tags_json)

            # Should contain metaxy tag with nested version
            assert "metaxy" in tags
            metaxy_data = json.loads(tags["metaxy"])
            assert metaxy_data["version"] == __version__


def test_push_graph_snapshot_with_custom_tags(tmp_path: Path):
    """Test that custom tags are merged with automatic tags."""
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Push with custom tags
            custom_tags = {
                "environment": "production",
                "version": "1.2.3",
                "deployed_by": "ci-system",
            }
            SystemTableStorage(store).push_graph_snapshot(tags=custom_tags)

            # Read and verify tags field
            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            assert versions_df.height == 1
            tags_json = versions_df["tags"][0]
            tags = json.loads(tags_json)

            # Should contain both automatic and custom tags
            assert "metaxy" in tags
            metaxy_data = json.loads(tags["metaxy"])
            assert metaxy_data["version"] == __version__
            assert tags["environment"] == "production"
            assert tags["version"] == "1.2.3"
            assert tags["deployed_by"] == "ci-system"


def test_push_graph_snapshot_tags_persist_across_pushes(tmp_path: Path):
    """Test that tags are stored with each push."""
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # First push
            SystemTableStorage(store).push_graph_snapshot(tags={"environment": "staging"})

            # Second push (no changes, shouldn't write new rows)
            SystemTableStorage(store).push_graph_snapshot(tags={"environment": "production"})

            # Read and verify - should only have one row (no changes)
            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            # No changes, so no new row
            assert versions_df.height == 1

            # Tags from first push should be preserved
            tags_json = versions_df["tags"][0]
            tags = json.loads(tags_json)
            assert tags["environment"] == "staging"
            # Metaxy version should also be present
            assert "metaxy" in tags
            metaxy_data = json.loads(tags["metaxy"])
            assert "version" in metaxy_data


def test_push_graph_snapshot_tags_updated_with_feature_changes(tmp_path: Path):
    """Test that tags are updated when feature definitions change."""
    graph_v1 = FeatureGraph()

    with graph_v1.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            pass

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # First push
            SystemTableStorage(store).push_graph_snapshot(tags={"environment": "staging", "build": "123"})

            # Change feature (metadata-only change)
            graph_v2 = FeatureGraph()
            with graph_v2.use():

                class TestFeature2(
                    BaseFeature,
                    spec=SampleFeatureSpec(
                        key=FeatureKey(["test", "feature"]),
                        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
                        metadata={"owner": "team_a"},  # Changed
                    ),
                ):
                    pass

                # Second push with different tags
                SystemTableStorage(store).push_graph_snapshot(tags={"environment": "production", "build": "124"})

                # Read and verify - should have two rows
                versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
                assert versions_lazy is not None
                versions_df = versions_lazy.collect().to_polars()

                # Two rows due to feature spec change
                assert versions_df.height == 2

                # Sort by recorded_at to get in order
                versions_df = versions_df.sort("recorded_at")

                # First push tags
                tags_json_1 = versions_df["tags"][0]
                tags_1 = json.loads(tags_json_1)
                assert tags_1["environment"] == "staging"
                assert tags_1["build"] == "123"
                assert "metaxy" in tags_1
                metaxy_data_1 = json.loads(tags_1["metaxy"])
                assert "version" in metaxy_data_1

                # Second push tags
                tags_json_2 = versions_df["tags"][1]
                tags_2 = json.loads(tags_json_2)
                assert tags_2["environment"] == "production"
                assert tags_2["build"] == "124"
                assert "metaxy" in tags_2
                metaxy_data_2 = json.loads(tags_2["metaxy"])
                assert "version" in metaxy_data_2


def test_tags_field_schema():
    """Test that tags field is properly defined in schema."""
    from metaxy.metadata_store.system.models import FEATURE_VERSIONS_SCHEMA

    # Verify tags field exists in schema
    assert "tags" in FEATURE_VERSIONS_SCHEMA
    assert FEATURE_VERSIONS_SCHEMA["tags"] == pl.String


def test_feature_versions_model_has_tags():
    """Test that FeatureVersionsModel Pydantic model has tags field."""
    from datetime import datetime, timezone

    from metaxy.metadata_store.system.models import FeatureVersionsModel

    # Create a model instance with dict tags
    model = FeatureVersionsModel(
        project="test_project",
        feature_key="test/feature",
        metaxy_feature_version="abc123",
        metaxy_definition_version="ghi789",
        recorded_at=datetime.now(timezone.utc),
        feature_spec='{"key": "value"}',
        feature_schema='{"type": "object", "properties": {}}',
        feature_class_path="test.module.TestFeature",
        metaxy_snapshot_version="jkl012",
        tags={"custom_tag": "custom_value"},
    )

    # Verify tags field is a JSON string with metaxy.version injected
    assert isinstance(model.tags, str)
    tags_dict = json.loads(model.tags)
    assert "metaxy" in tags_dict
    assert "custom_tag" in tags_dict
    assert tags_dict["custom_tag"] == "custom_value"
    metaxy_data = json.loads(tags_dict["metaxy"])
    assert metaxy_data["version"] == __version__

    # Convert to Polars and verify - already a JSON string
    df = model.to_polars()
    assert "tags" in df.columns
    # Verify structure in DataFrame
    df_tags = json.loads(df["tags"][0])
    assert "metaxy" in df_tags
    assert "custom_tag" in df_tags

    # Test that explicit empty dict gets metaxy.version injected via validator
    model_empty = FeatureVersionsModel(
        project="test_project",
        feature_key="test/feature",
        metaxy_feature_version="abc123",
        metaxy_definition_version="ghi789",
        recorded_at=datetime.now(timezone.utc),
        feature_spec='{"key": "value"}',
        feature_schema='{"type": "object", "properties": {}}',
        feature_class_path="test.module.TestFeature",
        metaxy_snapshot_version="jkl012",
        tags={},  # Explicit empty dict triggers validator
    )

    # Validator should inject metaxy.version
    assert isinstance(model_empty.tags, str)
    tags_dict_empty = json.loads(model_empty.tags)
    assert "metaxy" in tags_dict_empty
    metaxy_data_empty = json.loads(tags_dict_empty["metaxy"])
    assert metaxy_data_empty["version"] == __version__

    # Verify in DataFrame too
    df_empty = model_empty.to_polars()

    tags_dict = json.loads(df_empty["tags"][0])
    assert "metaxy" in tags_dict
