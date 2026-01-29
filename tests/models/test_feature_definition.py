"""Tests for FeatureDefinition model."""

import pytest

from metaxy import FeatureDefinition
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


def test_feature_definition_from_feature_class(graph):
    """Test creating FeatureDefinition from a Feature class."""
    from metaxy_testing.models import SampleFeatureSpec

    from metaxy import BaseFeature

    # Create a feature class directly for this test
    class TestFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "from_class"]),
            fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    definition = FeatureDefinition.from_feature_class(TestFeature)

    assert definition.spec == TestFeature.spec()
    assert definition.feature_schema == TestFeature.model_json_schema()
    assert definition.feature_class_path is not None and definition.feature_class_path.endswith(TestFeature.__name__)
    assert definition.project == TestFeature.metaxy_project()
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


class TestGetFeatureByKey:
    """Tests for the get_feature_by_key public function."""

    def test_get_feature_by_key_returns_definition(self, graph):
        """Test that get_feature_by_key returns a FeatureDefinition."""
        from metaxy_testing.models import SampleFeatureSpec

        import metaxy as mx

        class TestFeature(
            mx.BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "get_by_key"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        result = mx.get_feature_by_key(FeatureKey(["test", "get_by_key"]))

        assert isinstance(result, FeatureDefinition)
        assert result.spec == TestFeature.spec()
        assert result.feature_class_path is not None and result.feature_class_path.endswith("TestFeature")

    def test_get_feature_by_key_coerces_list(self, graph):
        """Test that get_feature_by_key accepts list of strings."""
        from metaxy_testing.models import SampleFeatureSpec

        import metaxy as mx

        class AnotherFeature(
            mx.BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "coerce_list"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Pass list instead of FeatureKey
        result = mx.get_feature_by_key(["test", "coerce_list"])

        assert isinstance(result, FeatureDefinition)
        assert result.key == FeatureKey(["test", "coerce_list"])

    def test_get_feature_by_key_coerces_string(self, graph):
        """Test that get_feature_by_key accepts slashed string."""
        from metaxy_testing.models import SampleFeatureSpec

        import metaxy as mx

        class StringFeature(
            mx.BaseFeature,
            spec=SampleFeatureSpec(
                key=FeatureKey(["test", "coerce_string"]),
                fields=[FieldSpec(key=FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        # Pass slashed string instead of FeatureKey
        result = mx.get_feature_by_key("test/coerce_string")

        assert isinstance(result, FeatureDefinition)
        assert result.key == FeatureKey(["test", "coerce_string"])

    def test_get_feature_by_key_raises_on_missing(self, graph):
        """Test that get_feature_by_key raises KeyError for missing feature."""
        import metaxy as mx

        with pytest.raises(KeyError, match="No feature with key"):
            mx.get_feature_by_key(FeatureKey(["nonexistent", "feature"]))


def test_external_feature_definition_creation():
    """External definitions can be created without a Feature class."""
    import metaxy as mx

    spec = mx.FeatureSpec(
        key=mx.FeatureKey(["external", "feature"]),
        id_columns=["id"],
        fields=["value"],
    )
    schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["id", "value"],
    }

    definition = mx.FeatureDefinition.external(
        spec=spec,
        feature_schema=schema,
        project="external-project",
    )

    assert definition.spec.key == spec.key
    assert definition.spec.id_columns == spec.id_columns
    assert definition.spec.fields == spec.fields
    assert definition.feature_schema == schema
    assert definition.project == "external-project"
    assert definition.is_external is True


def test_external_feature_definition_class_path_is_none():
    """External definitions have no class path."""
    import metaxy as mx

    spec = mx.FeatureSpec(
        key=mx.FeatureKey(["my", "external"]),
        id_columns=["id"],
    )
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}

    definition = mx.FeatureDefinition.external(
        spec=spec,
        feature_schema=schema,
        project="proj",
    )

    assert definition.feature_class_path is None


def test_is_external_property_true_for_external():
    """External definitions report is_external=True."""
    import metaxy as mx

    spec = mx.FeatureSpec(
        key=mx.FeatureKey(["ext", "feat"]),
        id_columns=["id"],
    )
    schema = {"type": "object", "properties": {"id": {"type": "string"}}}

    definition = mx.FeatureDefinition.external(
        spec=spec,
        feature_schema=schema,
        project="proj",
    )

    assert definition.is_external is True


def test_is_external_property_false_for_regular(graph):
    """Regular definitions report is_external=False."""
    from metaxy_testing.models import SampleFeatureSpec

    import metaxy as mx

    class RegularFeature(
        mx.BaseFeature,
        spec=SampleFeatureSpec(
            key=mx.FeatureKey(["test", "is_external_regular"]),
            fields=[mx.FieldSpec(key=mx.FieldKey(["default"]), code_version="1")],
        ),
    ):
        pass

    definition = mx.FeatureDefinition.from_feature_class(RegularFeature)
    assert definition.is_external is False
