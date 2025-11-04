from metaxy.models.field import CoersibleToFieldSpecsTypeAdapter, FieldSpec
from metaxy.models.types import FieldKey


def test_default_code_version():
    field = FieldSpec("my_field")

    # this default is EXTREMELY important
    # changing it will affect **all versions on all fields and features**
    assert field.code_version == "__metaxy_initial__"


def test_field_spec_from_string():
    """Test that FieldSpec can be constructed from just a string key."""
    field = FieldSpec("my_field")

    assert isinstance(field.key, FieldKey)
    assert field.key.to_string() == "my_field"
    assert field.code_version == "__metaxy_initial__"


def test_field_spec_from_string_with_code_version():
    """Test FieldSpec construction with explicit code_version."""
    field = FieldSpec("my_field", code_version="2")

    assert field.key.to_string() == "my_field"
    assert field.code_version == "2"


def test_field_spec_adapter_validates_string():
    """Test that CoersibleToFieldSpecsTypeAdapter can validate strings into FieldSpec instances."""
    # Validate from string list
    fields = CoersibleToFieldSpecsTypeAdapter.validate_python(["my_field"])

    assert len(fields) == 1
    assert isinstance(fields[0], FieldSpec)
    assert fields[0].key.to_string() == "my_field"
    assert fields[0].code_version == "__metaxy_initial__"


def test_field_spec_adapter_validates_dict():
    """Test that CoersibleToFieldSpecsTypeAdapter can validate dicts into FieldSpec instances."""
    # Validate from dict list
    fields = CoersibleToFieldSpecsTypeAdapter.validate_python(
        [{"key": "my_field", "code_version": "3"}]
    )

    assert len(fields) == 1
    assert isinstance(fields[0], FieldSpec)
    assert fields[0].key.to_string() == "my_field"
    assert fields[0].code_version == "3"


def test_field_spec_adapter_preserves_field_spec():
    """Test that CoersibleToFieldSpecsTypeAdapter preserves existing FieldSpec instances."""
    original = FieldSpec("my_field", code_version="5")
    validated = CoersibleToFieldSpecsTypeAdapter.validate_python([original])

    assert len(validated) == 1
    assert validated[0] is original
    assert validated[0].key.to_string() == "my_field"
    assert validated[0].code_version == "5"


def test_feature_spec_with_string_fields():
    """Test that FeatureSpec can be initialized with string field keys."""
    from metaxy.models.feature_spec import FeatureSpec

    spec = FeatureSpec(key="test/feature", fields=["field1", "field2", "field3"])

    assert len(spec.fields) == 3
    assert spec.fields[0].key.to_string() == "field1"
    assert spec.fields[1].key.to_string() == "field2"
    assert spec.fields[2].key.to_string() == "field3"

    # All should have default code_version
    for field in spec.fields:
        assert field.code_version == "__metaxy_initial__"


def test_feature_spec_with_mixed_fields():
    """Test that FeatureSpec can mix string fields and FieldSpec objects."""
    from metaxy.models.feature_spec import FeatureSpec

    spec = FeatureSpec(
        key="test/feature",
        fields=[
            "simple_field",
            FieldSpec("complex_field", code_version="2"),
            "another_simple_field",
        ],
    )

    assert len(spec.fields) == 3
    assert spec.fields[0].key.to_string() == "simple_field"
    assert spec.fields[0].code_version == "__metaxy_initial__"

    assert spec.fields[1].key.to_string() == "complex_field"
    assert spec.fields[1].code_version == "2"

    assert spec.fields[2].key.to_string() == "another_simple_field"
    assert spec.fields[2].code_version == "__metaxy_initial__"
