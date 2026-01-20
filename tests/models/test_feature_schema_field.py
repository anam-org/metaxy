"""Tests for feature_schema field in feature_versions system table."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from pydantic import Field

from metaxy import BaseFeature, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store.delta import DeltaMetadataStore
from metaxy.metadata_store.system import FEATURE_VERSIONS_KEY, SystemTableStorage
from metaxy.models.feature import FeatureGraph


def test_push_graph_snapshot_stores_feature_schema(tmp_path: Path):
    """Test that push_graph_snapshot() stores the Pydantic model schema."""
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Test feature with custom field."""

            custom_field: str = Field(description="A custom field")
            another_field: int = Field(default=42, description="An integer field")

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Push snapshot
            SystemTableStorage(store).push_graph_snapshot()

            # Read and verify feature_schema field
            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            assert versions_df.height == 1

            # Check that feature_schema column exists
            assert "feature_schema" in versions_df.columns

            # Parse and verify the schema
            schema_json = versions_df["feature_schema"][0]
            schema = json.loads(schema_json)

            # Verify it's a valid Pydantic schema
            assert "properties" in schema
            assert "type" in schema
            assert schema["type"] == "object"

            # Check for our custom fields in the schema
            assert "custom_field" in schema["properties"]
            assert "another_field" in schema["properties"]

            # Verify field descriptions are preserved
            assert (
                schema["properties"]["custom_field"]["description"] == "A custom field"
            )
            assert (
                schema["properties"]["another_field"]["description"]
                == "An integer field"
            )

            # Verify field types
            assert schema["properties"]["custom_field"]["type"] == "string"
            assert schema["properties"]["another_field"]["type"] == "integer"

            # Verify default value
            assert schema["properties"]["another_field"]["default"] == 42


def test_feature_schema_differs_between_features(tmp_path: Path):
    """Test that different features have different schemas."""
    graph = FeatureGraph()

    with graph.use():

        class Feature1(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature", "one"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """First feature."""

            field_a: str

        class Feature2(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["feature", "two"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Second feature."""

            field_b: int
            field_c: bool = Field(default=True)

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Push snapshot
            SystemTableStorage(store).push_graph_snapshot()

            # Read and verify
            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            assert versions_df.height == 2

            # Get schemas for both features (using correct key format with slashes)
            feature1_rows = versions_df.filter(pl.col("feature_key") == "feature/one")
            feature2_rows = versions_df.filter(pl.col("feature_key") == "feature/two")

            assert feature1_rows.height == 1
            assert feature2_rows.height == 1

            schema1 = json.loads(feature1_rows["feature_schema"][0])
            schema2 = json.loads(feature2_rows["feature_schema"][0])

            # Verify they're different
            assert schema1 != schema2

            # Feature1 should have field_a
            assert "field_a" in schema1["properties"]
            assert "field_b" not in schema1["properties"]
            assert "field_c" not in schema1["properties"]

            # Feature2 should have field_b and field_c
            assert "field_a" not in schema2["properties"]
            assert "field_b" in schema2["properties"]
            assert "field_c" in schema2["properties"]

            # Verify descriptions
            assert schema1["description"] == "First feature."
            assert schema2["description"] == "Second feature."


def test_feature_schema_included_in_snapshot():
    """Test that to_snapshot() includes feature_schema."""
    graph = FeatureGraph()

    with graph.use():

        class TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Test feature."""

            test_field: str = Field(description="Test field")

        snapshot = graph.to_snapshot()

        # Verify snapshot structure (using correct key format with slashes)
        assert "test/feature" in snapshot
        feature_data = snapshot["test/feature"]

        # Check that feature_schema is included
        assert "feature_schema" in feature_data

        # Verify it's a dict (not a string yet)
        assert isinstance(feature_data["feature_schema"], dict)

        # Check schema content
        schema = feature_data["feature_schema"]
        assert "properties" in schema
        assert "test_field" in schema["properties"]
        assert schema["properties"]["test_field"]["description"] == "Test field"


def test_feature_schema_field_in_polars_schema():
    """Test that feature_schema field is properly defined in Polars schema."""
    from metaxy.metadata_store.system.models import FEATURE_VERSIONS_SCHEMA

    # Verify feature_schema field exists in schema
    assert "feature_schema" in FEATURE_VERSIONS_SCHEMA
    assert FEATURE_VERSIONS_SCHEMA["feature_schema"] == pl.String


def test_feature_versions_model_has_feature_schema():
    """Test that FeatureVersionsModel Pydantic model has feature_schema field."""
    from datetime import datetime, timezone

    from metaxy.metadata_store.system.models import FeatureVersionsModel

    # Create a model instance with feature_schema
    model = FeatureVersionsModel(
        project="test_project",
        feature_key="test/feature",
        metaxy_feature_version="abc123",
        metaxy_feature_spec_version="def456",
        metaxy_full_definition_version="ghi789",
        recorded_at=datetime.now(timezone.utc),
        feature_spec='{"key": "value"}',
        feature_schema='{"type": "object", "properties": {}}',
        feature_class_path="test.module.TestClass",
        metaxy_snapshot_version="snapshot123",
        tags="{}",
    )

    # Verify the field exists and is set correctly
    assert hasattr(model, "feature_schema")
    assert model.feature_schema == '{"type": "object", "properties": {}}'

    # Convert to Polars DataFrame
    df = model.to_polars()
    assert "feature_schema" in df.columns
    assert df["feature_schema"][0] == '{"type": "object", "properties": {}}'


def test_feature_schema_for_feature_without_custom_fields(tmp_path: Path):
    """Test that features without custom fields still have a schema."""
    graph = FeatureGraph()

    with graph.use():

        class SimpleFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["simple", "feature"]),
                fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
            ),
        ):
            """Simple feature without custom fields."""

            pass  # No custom fields, just inherits from Feature

        with DeltaMetadataStore(root_path=tmp_path / "delta_store") as store:
            # Push snapshot
            SystemTableStorage(store).push_graph_snapshot()

            # Read and verify
            versions_lazy = store.read_metadata_in_store(FEATURE_VERSIONS_KEY)
            assert versions_lazy is not None
            versions_df = versions_lazy.collect().to_polars()

            assert versions_df.height == 1

            # Parse the schema
            schema_json = versions_df["feature_schema"][0]
            schema = json.loads(schema_json)

            # Even without custom fields, should have a valid schema
            assert "type" in schema
            assert schema["type"] == "object"
            assert "properties" in schema

            # The schema might be empty for a class with no custom fields
            # This is okay - the important thing is that we have a valid schema structure

            # Check the description
            assert schema["description"] == "Simple feature without custom fields."
