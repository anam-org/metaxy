"""Tests for ID columns validation in BaseFeature.

This module tests that BaseFeature validates that all id_columns specified
in the feature spec are present as fields in the feature class.
"""

import pytest

from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature import BaseFeature
from metaxy.models.feature_spec import FeatureSpec
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


def test_valid_single_id_column():
    """Test that features with valid single id_column pass validation."""

    class MyFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "single_id"]),
            id_columns=["sample_uid"],  # This field exists
        ),
    ):
        sample_uid: str

    # Should not raise
    instance = MyFeature(sample_uid="123")
    assert instance.sample_uid == "123"


def test_valid_multiple_id_columns():
    """Test that features with valid multiple id_columns pass validation."""

    class MyFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "multi_id"]),
            id_columns=["sample_uid", "chunk_id"],  # Both fields exist
        ),
    ):
        sample_uid: str
        chunk_id: int

    # Should not raise
    instance = MyFeature(sample_uid="123", chunk_id=42)
    assert instance.sample_uid == "123"
    assert instance.chunk_id == 42


def test_missing_single_id_column_raises_error():
    """Test that missing id_column raises helpful error."""

    class BadFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "bad_single"]),
            id_columns=["nonexistent"],  # This field doesn't exist
        ),
    ):
        sample_uid: str

    with pytest.raises(
        ValueError,
        match=r"ID columns \{'nonexistent'\} specified in spec are not present in model fields",
    ):
        BadFeature(sample_uid="123")


def test_missing_multiple_id_columns_raises_error():
    """Test that missing multiple id_columns raises helpful error."""
    from pydantic_core import ValidationError as PydanticValidationError

    class BadFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "bad_multi"]),
            id_columns=["sample_uid", "missing_col", "another_missing"],
        ),
    ):
        sample_uid: str

    # Pydantic wraps ValueError in ValidationError, so check the message contains both columns
    # Note: Set ordering is not deterministic, so we just check that both are present
    with pytest.raises(
        PydanticValidationError,
        match=r"(missing_col.*another_missing|another_missing.*missing_col)",
    ):
        BadFeature(sample_uid="123")


def test_error_message_shows_available_fields():
    """Test that error message includes available fields for debugging."""

    class BadFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "show_fields"]),
            id_columns=["nonexistent"],
        ),
    ):
        sample_uid: str
        other_field: int

    with pytest.raises(ValueError, match=r"Available fields:.*sample_uid.*other_field"):
        BadFeature(sample_uid="123", other_field=42)


def test_id_columns_with_inherited_fields():
    """Test that id_columns validation works with inherited fields."""

    # Base feature class with some fields
    class BaseCustomFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "base"]),
            id_columns=["base_id"],
        ),
    ):
        base_id: str

    # Inherited feature class
    class ChildFeature(
        BaseCustomFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "child"]),
            id_columns=["base_id", "child_id"],  # base_id is inherited
        ),
    ):
        child_id: str

    # Should not raise - base_id is inherited
    instance = ChildFeature(base_id="base123", child_id="child456")
    assert instance.base_id == "base123"
    assert instance.child_id == "child456"


def test_id_columns_case_sensitive():
    """Test that id_columns validation is case-sensitive."""

    class BadFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "case_sensitive"]),
            id_columns=["Sample_UID"],  # Wrong case - should be sample_uid
        ),
    ):
        sample_uid: str

    with pytest.raises(
        ValueError,
        match=r"ID columns \{'Sample_UID'\} specified in spec are not present in model fields",
    ):
        BadFeature(sample_uid="123")


def test_single_id_column_with_production_feature_spec():
    """Test that production FeatureSpec works with validation."""

    class ProductionFeature(
        BaseFeature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "production"]),
            id_columns=["uid"],  # Must be explicit with FeatureSpec
        ),
    ):
        uid: str
        some_field: str

    # Should not raise
    instance = ProductionFeature(uid="123", some_field="value")
    assert instance.uid == "123"
    assert instance.some_field == "value"


def test_id_columns_with_optional_fields():
    """Test that id_columns can reference optional fields."""

    class OptionalIDFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "optional_id"]),
            id_columns=["sample_uid"],
        ),
    ):
        sample_uid: str | None  # Optional field

    # Should not raise even with None value
    instance = OptionalIDFeature(sample_uid=None)
    assert instance.sample_uid is None

    # Should work with actual value
    instance2 = OptionalIDFeature(sample_uid="123")
    assert instance2.sample_uid == "123"


def test_id_columns_validation_with_extra_fields():
    """Test that extra fields not in id_columns are allowed."""

    class ExtraFieldsFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "extra"]),
            id_columns=["sample_uid"],  # Only one ID column
        ),
    ):
        sample_uid: str
        extra_field1: str
        extra_field2: int
        extra_field3: float

    # Should not raise - extra fields are fine
    instance = ExtraFieldsFeature(
        sample_uid="123",
        extra_field1="value",
        extra_field2=42,
        extra_field3=3.14,
    )
    assert instance.sample_uid == "123"
    assert instance.extra_field1 == "value"


def test_id_columns_with_field_specs():
    """Test id_columns validation with explicit FieldSpecs."""

    class FieldSpecFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "field_spec"]),
            id_columns=["sample_uid", "chunk_id"],
            fields=[
                FieldSpec(key=FieldKey(["output"]), code_version="1"),
                FieldSpec(key=FieldKey(["metadata"]), code_version="1"),
            ],
        ),
    ):
        sample_uid: str
        chunk_id: int
        output: str
        metadata: dict[str, str]

    # Should not raise
    instance = FieldSpecFeature(
        sample_uid="123",
        chunk_id=42,
        output="result",
        metadata={"key": "value"},
    )
    assert instance.sample_uid == "123"
    assert instance.chunk_id == 42


def test_id_columns_partial_match_raises_error():
    """Test that having only some id_columns raises error."""

    class PartialMatchFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "partial"]),
            id_columns=["sample_uid", "missing_id", "chunk_id"],
        ),
    ):
        sample_uid: str
        chunk_id: int
        # missing_id is not defined

    with pytest.raises(
        ValueError,
        match=r"ID columns \{'missing_id'\} specified in spec are not present in model fields",
    ):
        PartialMatchFeature(sample_uid="123", chunk_id=42)


def test_id_columns_with_tuple_type():
    """Test that id_columns as tuple works with validation."""

    class TupleIDFeature(
        BaseFeature,
        spec=FeatureSpec(
            key=FeatureKey(["test", "tuple_id"]),
            id_columns=("uid", "gid"),  # Tuple instead of list
        ),
    ):
        uid: str
        gid: str

    # Should not raise
    instance = TupleIDFeature(uid="u123", gid="g456")
    assert instance.uid == "u123"
    assert instance.gid == "g456"


def test_id_columns_validation_error_format():
    """Test that validation error message is properly formatted."""

    class BadFormatFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "format"]),
            id_columns=["missing1", "missing2"],
        ),
    ):
        sample_uid: str
        field1: str
        field2: int

    try:
        BadFormatFeature(sample_uid="123", field1="value", field2=42)
        pytest.fail("Expected ValueError to be raised")
    except ValueError as e:
        error_msg = str(e)
        # Check that error message contains key parts
        assert "ID columns" in error_msg
        assert "missing1" in error_msg
        assert "missing2" in error_msg
        assert "not present in model fields" in error_msg
        assert "Available fields" in error_msg
        assert "sample_uid" in error_msg
        assert "field1" in error_msg
        assert "field2" in error_msg


def test_id_columns_order_does_not_matter():
    """Test that order of id_columns doesn't affect validation."""

    # First order
    class Feature1(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "order1"]),
            id_columns=["sample_uid", "chunk_id", "version"],
        ),
    ):
        version: int
        chunk_id: int
        sample_uid: str  # Different order in class definition

    # Should not raise
    instance1 = Feature1(sample_uid="123", chunk_id=42, version=1)
    assert instance1.sample_uid == "123"

    # Second order
    class Feature2(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "order2"]),
            id_columns=["version", "chunk_id", "sample_uid"],  # Different order
        ),
    ):
        sample_uid: str
        chunk_id: int
        version: int

    # Should not raise
    instance2 = Feature2(sample_uid="456", chunk_id=99, version=2)
    assert instance2.sample_uid == "456"


def test_id_columns_with_complex_types():
    """Test that id_columns can have complex types (list, dict, etc)."""

    class ComplexIDFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "complex"]),
            id_columns=["sample_uid", "tags", "metadata"],
        ),
    ):
        sample_uid: str
        tags: list[str]
        metadata: dict[str, int]

    # Should not raise
    instance = ComplexIDFeature(
        sample_uid="123",
        tags=["tag1", "tag2"],
        metadata={"key": 42},
    )
    assert instance.sample_uid == "123"
    assert instance.tags == ["tag1", "tag2"]


def test_base_feature_direct_instantiation_with_valid_columns():
    """Test BaseFeature direct instantiation validates id_columns."""

    class DirectBaseFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "direct"]),
            id_columns=["uid"],
        ),
    ):
        uid: str
        data: str

    # Should not raise
    instance = DirectBaseFeature(uid="123", data="value")
    assert instance.uid == "123"


def test_base_feature_direct_instantiation_with_invalid_columns():
    """Test BaseFeature direct instantiation catches invalid id_columns."""

    class DirectBadFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "direct_bad"]),
            id_columns=["nonexistent"],
        ),
    ):
        uid: str
        data: str

    with pytest.raises(
        ValueError,
        match=r"ID columns \{'nonexistent'\} specified in spec are not present in model fields",
    ):
        DirectBadFeature(uid="123", data="value")
