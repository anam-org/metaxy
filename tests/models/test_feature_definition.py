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


class TestExternalProvenanceOverride:
    """Tests for external features with provenance override."""

    def test_external_with_provenance_override_creation(self):
        """External definitions can include provenance override."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "with_provenance"]),
            id_columns=["id"],
            fields=["value", "other"],
        )
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "value": {"type": "integer"},
                "other": {"type": "string"},
            },
        }
        provenance = {"value": "abc123", "other": "def456"}

        definition = mx.FeatureDefinition.external(
            spec=spec,
            feature_schema=schema,
            project="external-project",
            provenance_by_field=provenance,
        )

        assert definition.is_external is True
        assert definition.has_provenance_override is True
        assert definition.provenance_by_field_override == provenance

    def test_external_without_provenance_override_properties(self):
        """External definitions without provenance override report correctly."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "no_provenance"]),
            id_columns=["id"],
        )
        schema = {"type": "object", "properties": {"id": {"type": "string"}}}

        definition = mx.FeatureDefinition.external(
            spec=spec,
            feature_schema=schema,
            project="proj",
        )

        assert definition.is_external is True
        assert definition.has_provenance_override is False

    def test_regular_feature_has_provenance_override_raises(self, graph):
        """Accessing has_provenance_override on non-external features raises."""
        from metaxy_testing.models import SampleFeatureSpec

        import metaxy as mx
        from metaxy.utils.exceptions import MetaxyInvariantViolationError

        class RegularFeatureProvCheck(
            mx.BaseFeature,
            spec=SampleFeatureSpec(
                key=mx.FeatureKey(["test", "no_prov_override"]),
                fields=[mx.FieldSpec(key=mx.FieldKey(["default"]), code_version="1")],
            ),
        ):
            pass

        definition = mx.FeatureDefinition.from_feature_class(RegularFeatureProvCheck)
        with pytest.raises(MetaxyInvariantViolationError, match="not an external feature"):
            _ = definition.has_provenance_override

    def test_provenance_by_field_accepts_coercible_keys(self):
        """provenance_by_field accepts various field key formats."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "coercible"]),
            id_columns=["id"],
            fields=["simple", "nested/field"],
        )
        schema = {"type": "object", "properties": {"id": {}, "simple": {}, "nested/field": {}}}

        # Use different key formats: string, tuple, FieldKey
        # (lists can't be dict keys since they're unhashable)
        definition = mx.FeatureDefinition.external(
            spec=spec,
            feature_schema=schema,
            project="proj",
            provenance_by_field={
                "simple": "hash1",  # string
                ("nested", "field"): "hash2",  # tuple
                mx.FieldKey(["another"]): "hash3",  # FieldKey
            },
        )

        # All should be normalized to string format
        assert definition.provenance_by_field_override == {
            "simple": "hash1",
            "nested/field": "hash2",
            "another": "hash3",
        }


class TestOnVersionMismatch:
    """Tests for on_version_mismatch behavior."""

    def test_on_version_mismatch_default_is_warn(self):
        """Default on_version_mismatch is 'warn'."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "default_warn"]),
            id_columns=["id"],
        )
        definition = mx.FeatureDefinition.external(spec=spec, project="proj")
        assert definition.on_version_mismatch == "warn"

    def test_on_version_mismatch_can_be_set_to_error(self):
        """on_version_mismatch can be set to 'error'."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "error_mode"]),
            id_columns=["id"],
        )
        definition = mx.FeatureDefinition.external(spec=spec, project="proj", on_version_mismatch="error")
        assert definition.on_version_mismatch == "error"

    def test_check_version_mismatch_warns_on_difference(self):
        """check_version_mismatch warns when versions differ and mode is 'warn'."""
        import metaxy as mx

        external_spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "warn_test"]),
            id_columns=["id"],
            fields=["field_a"],
        )
        external_def = mx.FeatureDefinition.external(spec=external_spec, project="proj", on_version_mismatch="warn")

        with pytest.warns(UserWarning, match="Version mismatch"):
            external_def.check_version_mismatch(
                expected_version="expected_v1",
                actual_version="actual_v2",
                expected_version_by_field={"field_a": "field_hash_1"},
                actual_version_by_field={"field_a": "field_hash_2"},
            )

    def test_check_version_mismatch_raises_on_difference_when_error(self):
        """check_version_mismatch raises when versions differ and mode is 'error'."""
        import metaxy as mx

        external_spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "error_test"]),
            id_columns=["id"],
            fields=["field_a"],
        )
        external_def = mx.FeatureDefinition.external(spec=external_spec, project="proj", on_version_mismatch="error")

        with pytest.raises(ValueError, match="Version mismatch"):
            external_def.check_version_mismatch(
                expected_version="expected_v1",
                actual_version="actual_v2",
                expected_version_by_field={"field_a": "field_hash_1"},
                actual_version_by_field={"field_a": "field_hash_2"},
            )

    def test_check_version_mismatch_silent_when_versions_match(self):
        """check_version_mismatch does nothing when versions match."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "match_test"]),
            id_columns=["id"],
            fields=["field_a"],
        )
        external_def = mx.FeatureDefinition.external(spec=spec, project="proj", on_version_mismatch="error")

        # Should not raise or warn when versions match
        external_def.check_version_mismatch(
            expected_version="same_version",
            actual_version="same_version",
            expected_version_by_field={"field_a": "same_hash"},
            actual_version_by_field={"field_a": "same_hash"},
        )

    def test_check_version_mismatch_shows_field_level_details(self):
        """check_version_mismatch includes field-level details in the message."""
        import metaxy as mx

        external_spec = mx.FeatureSpec(
            key=mx.FeatureKey(["external", "field_details"]),
            id_columns=["id"],
            fields=["field_a", "field_b"],
        )
        external_def = mx.FeatureDefinition.external(spec=external_spec, project="proj", on_version_mismatch="warn")

        with pytest.warns(UserWarning, match="field_a.*expected.*got") as record:
            external_def.check_version_mismatch(
                expected_version="expected_v1",
                actual_version="actual_v2",
                expected_version_by_field={"field_a": "hash_a1", "field_b": "hash_b"},
                actual_version_by_field={"field_a": "hash_a2", "field_b": "hash_b"},
            )

        # The warning message should mention field_a (which differs) but not field_b (which matches)
        warning_message = str(record[0].message)
        assert "field_a" in warning_message
        assert "hash_a1" in warning_message
        assert "hash_a2" in warning_message


class TestSourceProperty:
    """Tests for the source property that tracks where a definition came from."""

    def test_source_returns_explicit_source_when_set(self):
        """source property returns explicit _source when set."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["test", "explicit_source"]),
            id_columns=["id"],
        )
        definition = mx.FeatureDefinition.external(
            spec=spec,
            project="proj",
            source="my-custom-source",
        )
        assert definition.source == "my-custom-source"

    def test_source_returns_feature_class_path_when_no_explicit_source(self):
        """source property returns feature_class_path when _source is not set."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["test", "class_path_source"]),
            id_columns=["id"],
            fields=[mx.FieldSpec(key=mx.FieldKey(["default"]))],
        )
        definition = mx.FeatureDefinition(
            spec=spec,
            feature_schema={},
            feature_class_path="mypackage.features.MyFeature",
            project="proj",
        )
        assert definition.source == "mypackage.features.MyFeature"

    def test_source_returns_question_mark_when_nothing_available(self):
        """source property returns '?' when neither _source nor feature_class_path is set."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["test", "no_source"]),
            id_columns=["id"],
        )
        # Create definition directly without using external() to bypass auto-capture
        definition = mx.FeatureDefinition(
            spec=spec,
            feature_schema={},
            feature_class_path=None,
            project="proj",
        )
        assert definition.source == "?"

    def test_external_captures_call_site_when_source_not_provided(self):
        """external() captures call site location when source is not provided."""
        import metaxy as mx

        spec = mx.FeatureSpec(
            key=mx.FeatureKey(["test", "auto_capture"]),
            id_columns=["id"],
        )
        # The source should include this test file and function name
        definition = mx.FeatureDefinition.external(
            spec=spec,
            project="proj",
        )
        # The source should contain the current module and function name
        assert "test_feature_definition" in definition.source
        assert "test_external_captures_call_site_when_source_not_provided" in definition.source

    def test_from_stored_data_accepts_source_parameter(self):
        """from_stored_data() accepts source parameter."""
        import metaxy as mx

        spec_dict = {
            "key": ["test", "stored_source"],
            "id_columns": ["id"],
        }
        schema_dict = {"type": "object", "properties": {"id": {"type": "string"}}}

        definition = mx.FeatureDefinition.from_stored_data(
            feature_spec=spec_dict,
            feature_schema=schema_dict,
            feature_class_path="pkg.Cls",
            project="proj",
            source="DuckDBMetadataStore(database=/path/to/db)",
        )
        assert definition.source == "DuckDBMetadataStore(database=/path/to/db)"
