"""Tests for FeatureDefinition model."""

import pytest

from metaxy import FeatureDefinition
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


def test_feature_definition_from_feature_class(test_graph, test_features):
    """Test creating FeatureDefinition from a Feature class."""
    feature_cls = test_features["UpstreamFeatureA"]

    definition = FeatureDefinition.from_feature_class(feature_cls)

    assert definition.spec == feature_cls.spec()
    assert definition.feature_schema == feature_cls.model_json_schema()
    assert definition.feature_class_path.endswith(feature_cls.__name__)
    assert definition.project == feature_cls.metaxy_project()
    assert definition.feature_definition_version  # non-empty hash


def test_feature_definition_project_field():
    """Test project is stored as a field."""
    spec = FeatureSpec(
        key=FeatureKey(["ns", "feat"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["val"]))],
    )

    definition = FeatureDefinition(
        spec=spec,
        feature_schema={"type": "object"},
        feature_class_path="myproject.features.MyFeature",
        project="myproject",
    )

    assert definition.project == "myproject"


def test_feature_definition_convenience_properties():
    """Test key, table_name, id_columns properties."""
    spec = FeatureSpec(
        key=FeatureKey(["video", "frames"]),
        id_columns=("video_id", "frame_idx"),
        fields=[FieldSpec(key=FieldKey(["data"]))],
    )

    definition = FeatureDefinition(
        spec=spec,
        feature_schema={},
        feature_class_path="pkg.mod.Cls",
        project="pkg",
    )

    assert definition.key == FeatureKey(["video", "frames"])
    assert definition.table_name == "video__frames"
    assert definition.id_columns == ("video_id", "frame_idx")


def test_feature_definition_version_determinism():
    """Test that feature_definition_version is deterministic."""
    spec = FeatureSpec(
        key=FeatureKey(["test", "feat"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["value"]))],
    )
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}

    version1 = FeatureDefinition._compute_definition_version(spec, schema)
    version2 = FeatureDefinition._compute_definition_version(spec, schema)

    assert version1 == version2


def test_feature_definition_version_changes_with_spec():
    """Test that version changes when spec changes."""
    spec1 = FeatureSpec(
        key=FeatureKey(["test", "feat"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
    )
    spec2 = FeatureSpec(
        key=FeatureKey(["test", "feat"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["value"]), code_version="2")],
    )
    schema = {"type": "object"}

    version1 = FeatureDefinition._compute_definition_version(spec1, schema)
    version2 = FeatureDefinition._compute_definition_version(spec2, schema)

    assert version1 != version2


def test_feature_definition_version_changes_with_schema():
    """Test that version changes when schema changes."""
    spec = FeatureSpec(
        key=FeatureKey(["test", "feat"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["value"]))],
    )
    schema1 = {"type": "object", "properties": {"id": {"type": "string"}}}
    schema2 = {"type": "object", "properties": {"id": {"type": "integer"}}}

    version1 = FeatureDefinition._compute_definition_version(spec, schema1)
    version2 = FeatureDefinition._compute_definition_version(spec, schema2)

    assert version1 != version2


def test_feature_definition_version_excludes_project():
    """Test that feature_definition_version is independent of project."""
    spec = FeatureSpec(
        key=FeatureKey(["test", "feat"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["value"]))],
    )
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}

    definition1 = FeatureDefinition(
        spec=spec,
        feature_schema=schema,
        feature_class_path="project_a.features.Feature",
        project="project_a",
    )
    definition2 = FeatureDefinition(
        spec=spec,
        feature_schema=schema,
        feature_class_path="project_b.features.Feature",
        project="project_b",
    )

    # Same spec and schema should produce same definition version regardless of project
    assert definition1.feature_definition_version == definition2.feature_definition_version
    # But the projects are different
    assert definition1.project != definition2.project


def test_feature_definition_is_frozen():
    """Test that FeatureDefinition is immutable."""
    spec = FeatureSpec(
        key=FeatureKey(["test", "feat"]),
        id_columns=("id",),
        fields=[FieldSpec(key=FieldKey(["value"]))],
    )

    definition = FeatureDefinition(
        spec=spec,
        feature_schema={},
        feature_class_path="pkg.Cls",
        project="pkg",
    )

    with pytest.raises(Exception):  # Pydantic's ValidationError for frozen models
        definition.feature_class_path = "other.path"
