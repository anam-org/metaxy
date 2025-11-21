"""Tests for FeatureKey and FieldKey types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError
from typing_extensions import assert_type

from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature import Feature
from metaxy.models.feature_spec import FeatureDep
from metaxy.models.types import (
    FeatureKey,
    FieldKey,
    # FieldKeyType,
)

if TYPE_CHECKING:
    pass


# TODO: switch to property testing with hypothesis, need to establish a feature_key_data and field_key_data strategies first
class TestFeatureKey:
    """Tests for FeatureKey class."""

    def test_string_input(self):
        """Test FeatureKey construction with string."""
        key = FeatureKey("a/b/c")
        assert_type(key, FeatureKey)
        assert_type(key.parts, tuple[str, ...])
        assert list(key.parts) == ["a", "b", "c"]
        assert key.to_string() == "a/b/c"

    def test_list_input(self):
        """Test FeatureKey construction with list."""
        key = FeatureKey(["a", "b", "c"])
        assert_type(key, FeatureKey)
        assert_type(key.parts, tuple[str, ...])
        assert list(key.parts) == ["a", "b", "c"]
        assert key.to_string() == "a/b/c"

    def test_tuple_input(self):
        """Test FeatureKey construction with tuple (Sequence support)."""
        key = FeatureKey(("a", "b", "c"))
        assert_type(key, FeatureKey)
        assert_type(key.parts, tuple[str, ...])
        assert list(key.parts) == ["a", "b", "c"]
        assert key.to_string() == "a/b/c"

    def test_feature_key_input(self):
        """Test FeatureKey construction with FeatureKey (copy)."""
        original = FeatureKey(["a", "b", "c"])
        copy = FeatureKey(original)
        assert list(copy.parts) == ["a", "b", "c"]
        assert copy == original
        assert copy is not original  # Different instances

    def test_equality(self):
        """Test equality comparison."""
        key1 = FeatureKey("a/b/c")
        key2 = FeatureKey(["a", "b", "c"])
        assert key1 == key2

        # All should have same hash
        assert hash(key1) == hash(key2)

    def test_inequality(self):
        """Test inequality comparison."""
        key1 = FeatureKey("a/b/c")
        key2 = FeatureKey("x/y/z")
        assert key1 != key2
        assert key1 != "a/b/c"  # Different type

    def test_hash_consistency(self):
        """Test hash consistency for use as dict keys."""
        key1 = FeatureKey("a/b/c")
        key2 = FeatureKey(["a", "b", "c"])
        assert_type(hash(key1), int)
        assert hash(key1) == hash(key2)
        # Can be used in sets and dicts
        my_set = {key1, key2}  # pyright: ignore[reportUnhashable]
        assert_type(my_set, set[FeatureKey])
        assert my_set == {key1}  # pyright: ignore[reportUnhashable]
        my_dict = {key1: 1, key2: 2}  # pyright: ignore[reportUnhashable]
        assert_type(my_dict, dict[FeatureKey, int])
        assert my_dict == {key1: 2}  # pyright: ignore[reportUnhashable]

    def test_table_name_property(self):
        """Test table_name property."""
        key = FeatureKey("a/b/c")
        assert key.table_name == "a__b__c"

    def test_repr(self):
        """Test __repr__ returns string representation."""
        key = FeatureKey("a/b/c")
        assert repr(key) == "a/b/c"

    def test_validation_double_underscore(self):
        """Test validation rejects double underscores."""
        with pytest.raises(ValidationError, match="cannot contain double underscores"):
            FeatureKey("a/b__c")

    def test_validation_forward_slash_in_part(self):
        """Test validation rejects forward slashes in list parts."""
        with pytest.raises(ValidationError, match="cannot contain forward slashes"):
            FeatureKey(["a", "b/c"])

    def test_validation_non_string_parts(self):
        """Test validation rejects non-string parts."""
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            FeatureKey([1, 2, 3])  # pyright: ignore[reportCallIssue,reportArgumentType]

    def test_validation_multiple_args_rejected(self):
        """Test validation rejects multiple positional arguments."""
        with pytest.raises(TypeError, match="takes from 1 to 2 positional arguments"):
            FeatureKey("a", "b", "c")  # pyright: ignore[reportCallIssue]

    def test_empty_string_splits_to_empty_list(self):
        """Test empty string handling."""
        key = FeatureKey("")
        assert list(key.parts) == [""]

    def test_single_part(self):
        """Test single part key."""
        key = FeatureKey("single")
        assert list(key.parts) == ["single"]
        assert key.to_string() == "single"

    def test_backward_compatibility_list_operations(self):
        """Test that FeatureKey still works with list-like operations."""
        key = FeatureKey("a/b/c")
        assert len(key) == 3
        assert key[0] == "a"
        assert key[1] == "b"
        assert key[2] == "c"
        assert "a" in key
        assert list(reversed(key)) == ["c", "b", "a"]

    def test_immutability(self):
        """Test that FeatureKey is immutable (frozen)."""
        key = FeatureKey("a/b/c")
        with pytest.raises(ValidationError, match="frozen"):
            key.parts = ("x", "y", "z")  # pyright: ignore[reportAttributeAccessIssue]

    def test_iteration(self):
        """Test iteration over parts."""
        key = FeatureKey("a/b/c")
        parts = []
        for part in key:
            parts.append(part)
        assert parts == ["a", "b", "c"]

    def test_contains(self):
        """Test contains operator."""
        key = FeatureKey("a/b/c")
        assert "a" in key
        assert "b" in key
        assert "c" in key
        assert "d" not in key


class TestFieldKey:
    """Tests for FieldKey class."""

    def test_string_input(self):
        """Test FieldKey construction with string."""
        key = FieldKey("a/b/c")
        assert list(key.parts) == ["a", "b", "c"]
        assert key.to_string() == "a/b/c"

    def test_list_input(self):
        """Test FieldKey construction with list."""
        key = FieldKey(["a", "b", "c"])
        assert list(key.parts) == ["a", "b", "c"]
        assert key.to_string() == "a/b/c"

    def test_tuple_input(self):
        """Test FieldKey construction with tuple (Sequence support)."""
        key = FieldKey(("a", "b", "c"))
        assert list(key.parts) == ["a", "b", "c"]
        assert key.to_string() == "a/b/c"

    def test_field_key_input(self):
        """Test FieldKey construction with FieldKey (copy)."""
        original = FieldKey(["a", "b", "c"])
        copy = FieldKey(original)
        assert list(copy.parts) == ["a", "b", "c"]
        assert copy == original
        assert copy is not original  # Different instances

    def test_equality(self):
        """Test equality comparison."""
        key1 = FieldKey("a/b/c")
        key2 = FieldKey(["a", "b", "c"])
        assert key1 == key2
        assert key2 == key2

        # All should have same hash
        assert hash(key1) == hash(key2)

    def test_inequality(self):
        """Test inequality comparison."""
        key1 = FieldKey("a/b/c")
        key2 = FieldKey("x/y/z")
        assert key1 != key2
        assert key1 != "a/b/c"  # Different type

    def test_hash_consistency(self):
        """Test hash consistency for use as dict keys."""
        key1 = FieldKey("a/b/c")
        key2 = FieldKey(["a", "b", "c"])
        assert hash(key1) == hash(key2)
        # Can be used in sets and dicts
        assert {key1, key2} == {key1}  # pyright: ignore[reportUnhashable]
        assert {key1: 1, key2: 2} == {key1: 2}  # pyright: ignore[reportUnhashable]

    def test_repr(self):
        """Test __repr__ returns string representation."""
        key = FieldKey("a/b/c")
        assert repr(key) == "a/b/c"

    def test_validation_double_underscore(self):
        """Test validation rejects double underscores."""
        with pytest.raises(ValidationError, match="cannot contain double underscores"):
            FieldKey("a/b__c")

    def test_validation_forward_slash_in_part(self):
        """Test validation rejects forward slashes in list parts."""
        with pytest.raises(ValidationError, match="cannot contain forward slashes"):
            FieldKey(["a", "b/c"])

    def test_validation_non_string_parts(self):
        """Test validation rejects non-string parts."""
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            FieldKey([1, 2, 3])  # pyright: ignore[reportCallIssue,reportArgumentType]

    def test_validation_multiple_args_rejected(self):
        """Test validation rejects multiple positional arguments."""
        with pytest.raises(TypeError, match="takes from 1 to 2 positional arguments"):
            FieldKey("a", "b", "c")  # pyright: ignore[reportCallIssue]

    def test_immutability(self):
        """Test that FieldKey is immutable (frozen)."""
        key = FieldKey("a/b/c")
        with pytest.raises(ValidationError, match="frozen"):
            key.parts = ("x", "y", "z")  # pyright: ignore[reportAttributeAccessIssue]

    def test_backward_compatibility_list_operations(self):
        """Test that FieldKey still works with list-like operations."""
        key = FieldKey("a/b/c")
        assert len(key) == 3
        assert key[0] == "a"
        assert key[1] == "b"
        assert key[2] == "c"
        assert "a" in key
        assert list(reversed(key)) == ["c", "b", "a"]


class TestTypeChecking:
    """Tests for type checking with overloaded constructors."""

    def test_feature_key_type_checking(self):
        """Test FeatureKey type checking with different inputs."""
        # These should all type-check correctly
        key_str = FeatureKey("a/b/c")
        key_list = FeatureKey(["a", "b", "c"])
        key_tuple = FeatureKey(("a", "b", "c"))
        key_copy = FeatureKey(key_str)

        # All should produce FeatureKey instances
        assert isinstance(key_str, FeatureKey)
        assert isinstance(key_list, FeatureKey)
        assert isinstance(key_tuple, FeatureKey)
        assert isinstance(key_copy, FeatureKey)

        # All should be equal
        assert key_str == key_list == key_tuple == key_copy


class TestJSONSerialization:
    """Tests for JSON serialization of FeatureKey and FieldKey."""

    def test_feature_key_json_serialization_from_string(self):
        """Test that FeatureKey from string serializes to list format."""
        key = FeatureKey("a/b/c")
        json_data = key.model_dump()

        # Should be a list, not a dict with "parts" key
        assert isinstance(json_data, list)
        assert json_data == ["a", "b", "c"]

    def test_feature_key_json_serialization_from_list(self):
        """Test that FeatureKey from list serializes to list format."""
        key = FeatureKey(["a", "b", "c"])
        json_data = key.model_dump()

        assert isinstance(json_data, list)
        assert json_data == ["a", "b", "c"]

    def test_field_key_json_serialization_from_string(self):
        """Test that FieldKey from string serializes to list format."""
        key = FieldKey("a/b/c")
        json_data = key.model_dump()

        assert isinstance(json_data, list)
        assert json_data == ["a", "b", "c"]

    def test_field_key_json_serialization_from_list(self):
        """Test that FieldKey from list serializes to list format."""
        key = FieldKey(["a", "b", "c"])
        json_data = key.model_dump()

        assert isinstance(json_data, list)
        assert json_data == ["a", "b", "c"]

    def test_feature_key_json_roundtrip(self):
        """Test that FeatureKey can be serialized and deserialized."""
        import json

        key = FeatureKey("a/b/c")
        json_str = json.dumps(key.model_dump())

        # Should serialize to ["a", "b", "c"]
        assert json_str == '["a", "b", "c"]'

        # Should be able to deserialize
        json_data = json.loads(json_str)
        key_restored = FeatureKey(json_data)

        assert key_restored == key

    def test_field_key_json_roundtrip(self):
        """Test that FieldKey can be serialized and deserialized."""
        import json

        key = FieldKey("x/y/z")
        json_str = json.dumps(key.model_dump())

        # Should serialize to ["x", "y", "z"]
        assert json_str == '["x", "y", "z"]'

        # Should be able to deserialize
        json_data = json.loads(json_str)
        key_restored = FieldKey(json_data)

        assert key_restored == key

    def test_feature_key_in_pydantic_model(self):
        """Test FeatureKey serialization when used as a field in Pydantic model."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            key: FeatureKey

        # Create with list
        model = MyModel(key=FeatureKey(["a", "b", "c"]))
        json_data = model.model_dump()

        # key should be serialized as a list
        assert isinstance(json_data["key"], list)
        assert json_data["key"] == ["a", "b", "c"]

    def test_field_key_in_pydantic_model(self):
        """Test FieldKey serialization when used as a field in Pydantic model."""
        from pydantic import BaseModel

        class MyModel(BaseModel):
            key: FieldKey

        # Create with list
        model = MyModel(key=FieldKey(["x", "y"]))
        json_data = model.model_dump()

        # key should be serialized as a list
        assert isinstance(json_data["key"], list)
        assert json_data["key"] == ["x", "y"]


class TestFeatureSpecIntegration:
    """Tests for FeatureKey and FieldKey usage in SampleFeatureSpec with various formats."""

    def test_feature_spec_with_string_key(self):
        """Test SampleFeatureSpec accepts FeatureKey as string."""
        from metaxy._testing.models import SampleFeatureSpec

        spec = SampleFeatureSpec(key="my/feature")

        assert isinstance(spec.key, FeatureKey)
        assert spec.key.to_string() == "my/feature"
        assert list(spec.key.parts) == ["my", "feature"]

    def test_feature_spec_with_list_key(self):
        """Test SampleFeatureSpec accepts FeatureKey as list."""
        from metaxy._testing.models import SampleFeatureSpec

        spec = SampleFeatureSpec(key=["my", "feature"])

        assert isinstance(spec.key, FeatureKey)
        assert spec.key.to_string() == "my/feature"
        assert list(spec.key.parts) == ["my", "feature"]

    def test_feature_spec_with_feature_key_instance(self):
        """Test SampleFeatureSpec accepts FeatureKey instance."""
        from metaxy._testing.models import SampleFeatureSpec

        key = FeatureKey(["my", "feature"])
        spec = SampleFeatureSpec(key=key)

        assert isinstance(spec.key, FeatureKey)
        assert spec.key.to_string() == "my/feature"
        assert list(spec.key.parts) == ["my", "feature"]

    def test_feature_dep_with_string_key(self):
        """Test FeatureDep accepts key as string."""
        from metaxy.models.feature_spec import FeatureDep

        dep = FeatureDep(feature="upstream/feature")

        assert isinstance(dep.feature, FeatureKey)
        assert dep.feature.to_string() == "upstream/feature"

    def test_feature_dep_with_list_key(self):
        """Test FeatureDep accepts key as list."""
        from metaxy.models.feature_spec import FeatureDep

        dep = FeatureDep(feature=["upstream", "feature"])

        assert isinstance(dep.feature, FeatureKey)
        assert dep.feature.to_string() == "upstream/feature"

    def test_feature_dep_with_feature_key_instance(self):
        """Test FeatureDep accepts FeatureKey instance."""
        from metaxy.models.feature_spec import FeatureDep

        dep = FeatureDep(feature=FeatureKey(["upstream", "feature"]))

        assert isinstance(dep.feature, FeatureKey)
        assert dep.feature.to_string() == "upstream/feature"

    def test_field_spec_with_string_key(self):
        """Test FieldSpec accepts FieldKey as string."""
        from metaxy.models.field import FieldSpec

        field = FieldSpec(key="my/field", code_version="1")

        assert isinstance(field.key, FieldKey)
        assert field.key.to_string() == "my/field"

    def test_field_spec_with_list_key(self):
        """Test FieldSpec accepts FieldKey as list."""
        from metaxy.models.field import FieldSpec

        field = FieldSpec(key=["my", "field"], code_version="1")

        assert isinstance(field.key, FieldKey)
        assert field.key.to_string() == "my/field"

    def test_field_spec_with_field_key_instance(self):
        """Test FieldSpec accepts FieldKey instance."""
        from metaxy.models.field import FieldSpec

        field = FieldSpec(key=FieldKey(["my", "field"]), code_version="1")

        assert isinstance(field.key, FieldKey)
        assert field.key.to_string() == "my/field"

    def test_complete_feature_spec_with_list_keys(self):
        """Test complete SampleFeatureSpec with all keys created using list format."""
        from metaxy._testing.models import SampleFeatureSpec
        from metaxy.models.feature_spec import FeatureDep
        from metaxy.models.field import FieldSpec

        spec = SampleFeatureSpec(
            key=FeatureKey(["my", "complete", "feature"]),
            deps=[
                FeatureDep(feature=FeatureKey(["upstream", "one"])),
                FeatureDep(feature=FeatureKey(["upstream", "two"])),
            ],
            fields=[
                FieldSpec(key=FieldKey(["field", "one"]), code_version="1"),
                FieldSpec(key=FieldKey(["field", "two"]), code_version="1"),
            ],
        )

        # Verify all keys are properly constructed
        assert spec.key.to_string() == "my/complete/feature"
        assert spec.deps is not None
        assert spec.deps[0].feature.to_string() == "upstream/one"
        assert spec.deps[1].feature.to_string() == "upstream/two"
        assert spec.fields[0].key.to_string() == "field/one"
        assert spec.fields[1].key.to_string() == "field/two"

        # Verify JSON serialization
        json_data = spec.model_dump(mode="json")
        assert json_data["key"] == ["my", "complete", "feature"]
        assert json_data["deps"] is not None  # Type narrowing for type checker
        assert json_data["deps"][0]["feature"] == ["upstream", "one"]
        assert json_data["deps"][1]["feature"] == ["upstream", "two"]
        assert json_data["fields"][0]["key"] == ["field", "one"]
        assert json_data["fields"][1]["key"] == ["field", "two"]


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_feature_key_empty_parts_via_list(self):
        """Test FeatureKey with empty list (edge case)."""
        key = FeatureKey([])
        assert list(key.parts) == []
        assert key.to_string() == ""


feature_key = FeatureKey("a/b")
feature_spec = SampleFeatureSpec(key=feature_key, id_columns=("id",))


def test_feature_dep_key_overloads():
    _ = FeatureDep(feature=feature_key)
    # Note: FeatureDep doesn't accept spec instances directly anymore
    # _ = FeatureDep(feature=feature_spec)

    class TestFeature(Feature, spec=feature_spec): ...

    _ = FeatureDep(feature=TestFeature)


class TestCoercibleToFeatureKey:
    """Tests for CoercibleToFeatureKey type and adapter."""

    def test_coercible_in_pydantic_model_with_string(self):
        """Test ValidatedFeatureKey in Pydantic model accepts string."""
        from pydantic import BaseModel

        from metaxy.models.types import ValidatedFeatureKey

        class MyModel(BaseModel):
            key: ValidatedFeatureKey

        model = MyModel(key="a/b/c")  # pyright: ignore[reportArgumentType]
        assert isinstance(model.key, FeatureKey)
        assert model.key.to_string() == "a/b/c"

    def test_coercible_in_pydantic_model_with_list(self):
        """Test ValidatedFeatureKey in Pydantic model accepts list."""
        from pydantic import BaseModel

        from metaxy.models.types import ValidatedFeatureKey

        class MyModel(BaseModel):
            key: ValidatedFeatureKey

        model = MyModel(key=["a", "b", "c"])  # pyright: ignore[reportArgumentType]
        assert isinstance(model.key, FeatureKey)
        assert model.key.to_string() == "a/b/c"

    def test_coercible_in_pydantic_model_with_feature_key(self):
        """Test ValidatedFeatureKey in Pydantic model accepts FeatureKey."""
        from pydantic import BaseModel

        from metaxy.models.types import ValidatedFeatureKey

        class MyModel(BaseModel):
            key: ValidatedFeatureKey

        feature_key = FeatureKey("a/b/c")
        model = MyModel(key=feature_key)
        assert isinstance(model.key, FeatureKey)
        assert model.key.to_string() == "a/b/c"

    def test_coercible_in_pydantic_model_with_feature_class(self):
        """Test ValidatedFeatureKey in Pydantic model accepts Feature class."""
        from pydantic import BaseModel

        from metaxy.models.types import ValidatedFeatureKey

        class MyModel(BaseModel):
            key: ValidatedFeatureKey

        class TestFeature(Feature, spec=feature_spec): ...

        model = MyModel(key=TestFeature)  # pyright: ignore[reportArgumentType]
        assert isinstance(model.key, FeatureKey)
        assert model.key == feature_key

    def test_coercible_adapter_with_string(self):
        """Test ValidatedFeatureKeyAdapter.validate_python with string."""
        from metaxy.models.types import ValidatedFeatureKeyAdapter

        result = ValidatedFeatureKeyAdapter.validate_python("a/b/c")
        assert isinstance(result, FeatureKey)
        assert result.to_string() == "a/b/c"

    def test_coercible_adapter_with_list(self):
        """Test ValidatedFeatureKeyAdapter.validate_python with list."""
        from metaxy.models.types import ValidatedFeatureKeyAdapter

        result = ValidatedFeatureKeyAdapter.validate_python(["a", "b", "c"])
        assert isinstance(result, FeatureKey)
        assert result.to_string() == "a/b/c"

    def test_coercible_adapter_with_feature_key(self):
        """Test ValidatedFeatureKeyAdapter.validate_python with FeatureKey."""
        from metaxy.models.types import ValidatedFeatureKeyAdapter

        feature_key = FeatureKey("a/b/c")
        result = ValidatedFeatureKeyAdapter.validate_python(feature_key)
        assert isinstance(result, FeatureKey)
        assert result == feature_key

    def test_coercible_adapter_with_feature_class(self):
        """Test ValidatedFeatureKeyAdapter.validate_python with Feature class."""
        from metaxy.models.types import ValidatedFeatureKeyAdapter

        class TestFeature(Feature, spec=feature_spec): ...

        result = ValidatedFeatureKeyAdapter.validate_python(TestFeature)
        assert isinstance(result, FeatureKey)
        assert result == feature_key

    def test_coercible_adapter_invalid_input(self):
        """Test ValidatedFeatureKeyAdapter.validate_python with invalid input."""
        from metaxy.models.types import ValidatedFeatureKeyAdapter

        with pytest.raises(ValidationError):
            ValidatedFeatureKeyAdapter.validate_python(123)

    def test_feature_dep_accepts_all_types(self):
        """Test FeatureDep accepts all CoercibleToFeatureKey types."""
        # String
        dep1 = FeatureDep(feature="a/b/c")
        assert dep1.feature.to_string() == "a/b/c"

        # List
        dep2 = FeatureDep(feature=["a", "b", "c"])
        assert dep2.feature.to_string() == "a/b/c"

        # FeatureKey
        dep3 = FeatureDep(feature=FeatureKey("a/b/c"))
        assert dep3.feature.to_string() == "a/b/c"

        # Feature class
        class TestFeature(Feature, spec=feature_spec): ...

        dep4 = FeatureDep(feature=TestFeature)
        assert dep4.feature == feature_key


class TestCoercibleToFieldKey:
    """Tests for CoercibleToFieldKey type and adapter."""

    def test_coercible_in_pydantic_model_with_string(self):
        """Test ValidatedFieldKey in Pydantic model accepts string."""
        from pydantic import BaseModel

        from metaxy.models.types import ValidatedFieldKey

        class MyModel(BaseModel):
            key: ValidatedFieldKey

        model = MyModel(key="a/b/c")  # pyright: ignore[reportArgumentType]
        assert isinstance(model.key, FieldKey)
        assert model.key.to_string() == "a/b/c"

    def test_coercible_in_pydantic_model_with_list(self):
        """Test ValidatedFieldKey in Pydantic model accepts list."""
        from pydantic import BaseModel

        from metaxy.models.types import ValidatedFieldKey

        class MyModel(BaseModel):
            key: ValidatedFieldKey

        model = MyModel(key=["a", "b", "c"])  # pyright: ignore[reportArgumentType]
        assert isinstance(model.key, FieldKey)
        assert model.key.to_string() == "a/b/c"

    def test_coercible_in_pydantic_model_with_field_key(self):
        """Test ValidatedFieldKey in Pydantic model accepts FieldKey."""
        from pydantic import BaseModel

        from metaxy.models.types import ValidatedFieldKey

        class MyModel(BaseModel):
            key: ValidatedFieldKey

        field_key = FieldKey("a/b/c")
        model = MyModel(key=field_key)
        assert isinstance(model.key, FieldKey)
        assert model.key.to_string() == "a/b/c"

    def test_coercible_adapter_with_string(self):
        """Test ValidatedFieldKeyAdapter.validate_python with string."""
        from metaxy.models.types import ValidatedFieldKeyAdapter

        result = ValidatedFieldKeyAdapter.validate_python("a/b/c")
        assert isinstance(result, FieldKey)
        assert result.to_string() == "a/b/c"

    def test_coercible_adapter_with_list(self):
        """Test ValidatedFieldKeyAdapter.validate_python with list."""
        from metaxy.models.types import ValidatedFieldKeyAdapter

        result = ValidatedFieldKeyAdapter.validate_python(["a", "b", "c"])
        assert isinstance(result, FieldKey)
        assert result.to_string() == "a/b/c"

    def test_coercible_adapter_with_field_key(self):
        """Test ValidatedFieldKeyAdapter.validate_python with FieldKey."""
        from metaxy.models.types import ValidatedFieldKeyAdapter

        field_key = FieldKey("a/b/c")
        result = ValidatedFieldKeyAdapter.validate_python(field_key)
        assert isinstance(result, FieldKey)
        assert result == field_key

    def test_coercible_adapter_invalid_input(self):
        """Test ValidatedFieldKeyAdapter.validate_python with invalid input."""
        from metaxy.models.types import ValidatedFieldKeyAdapter

        with pytest.raises(ValidationError):
            ValidatedFieldKeyAdapter.validate_python(123)
