"""Hypothesis-based property tests for FeatureKey, FieldKey and classes that use them."""

from __future__ import annotations

import json
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings
from pydantic import BaseModel, ValidationError

from metaxy.models.feature_spec import FeatureDep, SampleFeatureSpec
from metaxy.models.field import FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureKey, FieldKey

# ============================================================================
# Hypothesis Strategies for Generating Test Data
# ============================================================================


# Strategy for valid key part strings (no "/" or "__")
@st.composite
def valid_key_part(draw: st.DrawFn) -> str:
    """Generate a valid key part that doesn't contain "/" or "__"."""
    # Generate alphanumeric with underscores and hyphens
    part = draw(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"
            ),
            min_size=1,
            max_size=20,
        )
    )
    # Ensure no double underscores
    assume("__" not in part)
    assume("/" not in part)
    return part


# Strategy for generating lists of valid key parts
@st.composite
def key_parts_list(draw: st.DrawFn) -> list[str]:
    """Generate a list of valid key parts."""
    n_parts = draw(st.integers(min_value=1, max_value=5))
    return draw(st.lists(valid_key_part(), min_size=n_parts, max_size=n_parts))


# Strategy for generating unique lists of key parts
@st.composite
def unique_key_parts_lists(
    draw: st.DrawFn, min_size: int = 1, max_size: int = 3
) -> list[list[str]]:
    """Generate a list of unique key parts lists."""
    n_keys = draw(st.integers(min_value=min_size, max_value=max_size))

    # Generate unique keys by adding index to ensure uniqueness
    keys = []
    for i in range(n_keys):
        parts = draw(key_parts_list())
        # Add index to first part to ensure uniqueness
        # Use single underscore to avoid double underscore issues
        if parts:
            parts[0] = f"{parts[0]}-{i}"  # Use hyphen instead of underscore
        keys.append(parts)

    return keys


# Strategy for slashed string format
@st.composite
def key_as_slashed_string(draw: st.DrawFn) -> str:
    """Generate key in slashed string format like 'a/b/c'."""
    parts = draw(key_parts_list())
    return "/".join(parts)


# Strategy for generating coercible FeatureKey data
@st.composite
def coercible_to_feature_key(draw: st.DrawFn) -> Any:
    """Generate data that can be coerced to FeatureKey."""
    parts = draw(key_parts_list())

    # Choose a format randomly
    format_choice = draw(
        st.sampled_from(
            [
                "string",
                "list",
                "tuple",
                "feature_key",
                "dict_with_list",
                "dict_with_tuple",
            ]
        )
    )

    if format_choice == "string":
        return "/".join(parts)
    elif format_choice == "list":
        return parts
    elif format_choice == "tuple":
        return tuple(parts)
    elif format_choice == "feature_key":
        # Create a FeatureKey instance
        return FeatureKey(parts)
    elif format_choice == "dict_with_list":
        # Dict format that might come from JSON deserialization
        return {"parts": parts}
    elif format_choice == "dict_with_tuple":
        # Dict format with tuple
        return {"parts": tuple(parts)}
    else:
        return parts


# Strategy for generating coercible FieldKey data
@st.composite
def coercible_to_field_key(draw: st.DrawFn) -> Any:
    """Generate data that can be coerced to FieldKey."""
    parts = draw(key_parts_list())

    # Choose a format randomly
    format_choice = draw(
        st.sampled_from(
            [
                "string",
                "list",
                "tuple",
                "field_key",
                "dict_with_list",
                "dict_with_tuple",
            ]
        )
    )

    if format_choice == "string":
        return "/".join(parts)
    elif format_choice == "list":
        return parts
    elif format_choice == "tuple":
        return tuple(parts)
    elif format_choice == "field_key":
        # Create a FieldKey instance
        return FieldKey(parts)
    elif format_choice == "dict_with_list":
        # Dict format that might come from JSON deserialization
        return {"parts": parts}
    elif format_choice == "dict_with_tuple":
        # Dict format with tuple
        return {"parts": tuple(parts)}
    else:
        return parts


# Strategy for invalid key data
@st.composite
def invalid_key_data(draw: st.DrawFn) -> Any:
    """Generate data that should fail to coerce to a key."""
    invalid_type = draw(
        st.sampled_from(
            [
                "int",
                "float",
                "bool",
                "none",
                "dict_without_parts",
                "list_with_invalid_types",
                "string_with_double_underscore",
                "list_with_slash",
            ]
        )
    )

    if invalid_type == "int":
        return draw(st.integers())
    elif invalid_type == "float":
        return draw(st.floats())
    elif invalid_type == "bool":
        return draw(st.booleans())
    elif invalid_type == "none":
        return None
    elif invalid_type == "dict_without_parts":
        return {"not_parts": ["a", "b"]}
    elif invalid_type == "list_with_invalid_types":
        return [1, 2, 3]  # Non-string elements
    elif invalid_type == "string_with_double_underscore":
        return "a/b__c/d"
    elif invalid_type == "list_with_slash":
        return ["a", "b/c", "d"]
    else:
        return 42


# ============================================================================
# Property Tests for FeatureKey
# ============================================================================


class TestFeatureKeyProperties:
    """Property tests for FeatureKey."""

    @given(data=coercible_to_feature_key())
    @settings(max_examples=200)
    def test_feature_key_accepts_all_coercible_formats(self, data: Any):
        """Test that FeatureKey accepts all coercible formats."""
        key = FeatureKey(data)
        assert isinstance(key, FeatureKey)
        assert isinstance(key.parts, tuple)
        assert all(isinstance(part, str) for part in key.parts)

    @given(data=coercible_to_feature_key())
    @settings(max_examples=100)
    def test_feature_key_consistency(self, data: Any):
        """Test that FeatureKey produces consistent results regardless of input format."""
        key1 = FeatureKey(data)

        # Create from the parts to get canonical form
        key2 = FeatureKey(list(key1.parts))
        key3 = FeatureKey("/".join(key1.parts))

        # All should be equal
        assert key1 == key2
        assert key1 == key3
        assert hash(key1) == hash(key2) == hash(key3)
        assert key1.to_string() == key2.to_string() == key3.to_string()

    @given(data=invalid_key_data())
    def test_feature_key_rejects_invalid_data(self, data: Any):
        """Test that FeatureKey rejects invalid data."""
        with pytest.raises((ValueError, ValidationError, TypeError)):
            FeatureKey(data)

    @given(parts=key_parts_list())
    @settings(max_examples=100)
    def test_feature_key_serialization_roundtrip(self, parts: list[str]):
        """Test JSON serialization roundtrip."""
        key = FeatureKey(parts)

        # Serialize
        json_data = key.model_dump()
        assert isinstance(json_data, list)
        assert json_data == parts

        # JSON string roundtrip
        json_str = json.dumps(json_data)
        restored_data = json.loads(json_str)
        restored_key = FeatureKey(restored_data)

        assert restored_key == key

    @given(parts=key_parts_list())
    def test_feature_key_immutability(self, parts: list[str]):
        """Test that FeatureKey is immutable."""
        key = FeatureKey(parts)

        with pytest.raises(ValidationError, match="frozen"):
            key.parts = tuple(["x", "y", "z"])  # pyright: ignore[reportAttributeAccessIssue]

    @given(parts1=key_parts_list(), parts2=key_parts_list())
    def test_feature_key_ordering(self, parts1: list[str], parts2: list[str]):
        """Test ordering operations."""
        key1 = FeatureKey(parts1)
        key2 = FeatureKey(parts2)

        # Test comparison operators are consistent
        if key1 < key2:
            assert key2 > key1
            assert key1 <= key2
            assert key2 >= key1
            assert key1 != key2
        elif key1 > key2:
            assert key2 < key1
            assert key1 >= key2
            assert key2 <= key1
            assert key1 != key2
        else:
            assert key1 == key2
            assert key1 <= key2
            assert key1 >= key2


# ============================================================================
# Property Tests for FieldKey
# ============================================================================


class TestFieldKeyProperties:
    """Property tests for FieldKey."""

    @given(data=coercible_to_field_key())
    @settings(max_examples=200)
    def test_field_key_accepts_all_coercible_formats(self, data: Any):
        """Test that FieldKey accepts all coercible formats."""
        key = FieldKey(data)
        assert isinstance(key, FieldKey)
        assert isinstance(key.parts, tuple)
        assert all(isinstance(part, str) for part in key.parts)

    @given(data=coercible_to_field_key())
    @settings(max_examples=100)
    def test_field_key_consistency(self, data: Any):
        """Test that FieldKey produces consistent results regardless of input format."""
        key1 = FieldKey(data)

        # Create from the parts to get canonical form
        key2 = FieldKey(list(key1.parts))
        key3 = FieldKey("/".join(key1.parts))

        # All should be equal
        assert key1 == key2
        assert key1 == key3
        assert hash(key1) == hash(key2) == hash(key3)
        assert key1.to_string() == key2.to_string() == key3.to_string()

    @given(data=invalid_key_data())
    def test_field_key_rejects_invalid_data(self, data: Any):
        """Test that FieldKey rejects invalid data."""
        with pytest.raises((ValueError, ValidationError, TypeError)):
            FieldKey(data)

    @given(parts=key_parts_list())
    @settings(max_examples=100)
    def test_field_key_serialization_roundtrip(self, parts: list[str]):
        """Test JSON serialization roundtrip."""
        key = FieldKey(parts)

        # Serialize
        json_data = key.model_dump()
        assert isinstance(json_data, list)
        assert json_data == parts

        # JSON string roundtrip
        json_str = json.dumps(json_data)
        restored_data = json.loads(json_str)
        restored_key = FieldKey(restored_data)

        assert restored_key == key


# ============================================================================
# Property Tests for SampleFeatureSpec
# ============================================================================


class TestFeatureSpecProperties:
    """Property tests for SampleFeatureSpec accepting coercible keys."""

    @given(key_data=coercible_to_feature_key())
    @settings(max_examples=100)
    def test_feature_spec_accepts_coercible_key(self, key_data: Any):
        """Test that SampleFeatureSpec accepts all coercible key formats."""
        spec = SampleFeatureSpec(key=key_data)

        assert isinstance(spec.key, FeatureKey)
        assert isinstance(spec.key.parts, tuple)

    @given(
        key_data=coercible_to_feature_key(),
        dep_keys=st.lists(coercible_to_feature_key(), max_size=3),
    )
    @settings(max_examples=50)
    def test_feature_spec_with_deps(self, key_data: Any, dep_keys: list[Any]):
        """Test SampleFeatureSpec with dependencies using coercible keys."""
        # Build deps list; omit parameter if empty (don't pass None)
        if dep_keys:
            deps = [FeatureDep(feature=dep_key) for dep_key in dep_keys]
            spec = SampleFeatureSpec(key=key_data, deps=deps)
        else:
            spec = SampleFeatureSpec(key=key_data)

        assert isinstance(spec.key, FeatureKey)
        if dep_keys:
            assert spec.deps is not None
            for dep in spec.deps:
                assert isinstance(dep.feature, FeatureKey)

    @given(
        key_data=coercible_to_feature_key(),
        unique_parts=unique_key_parts_lists(min_size=1, max_size=3),
    )
    @settings(max_examples=50)
    def test_feature_spec_with_fields(
        self, key_data: Any, unique_parts: list[list[str]]
    ):
        """Test SampleFeatureSpec with fields using coercible keys."""
        # Convert unique parts to various formats
        fields = []
        for i, parts in enumerate(unique_parts):
            # Choose different format for each field
            if i % 3 == 0:
                field_key = "/".join(parts)  # String format
            elif i % 3 == 1:
                field_key = parts  # List format
            else:
                field_key = FieldKey(parts)  # FieldKey instance

            fields.append(
                FieldSpec(key=field_key, code_version="1", deps=SpecialFieldDep.ALL)
            )

        spec = SampleFeatureSpec(key=key_data, fields=fields)

        assert isinstance(spec.key, FeatureKey)
        assert len(spec.fields) == len(fields)
        for field in spec.fields:
            assert isinstance(field.key, FieldKey)

    @given(key_data=coercible_to_feature_key())
    @settings(max_examples=50)
    def test_feature_spec_json_serialization(self, key_data: Any):
        """Test SampleFeatureSpec JSON serialization with coercible keys."""
        spec = SampleFeatureSpec(key=key_data)

        # Serialize to JSON
        json_data = spec.model_dump(mode="json")

        # Key should be serialized as a list
        assert isinstance(json_data["key"], list)

        # Should be able to reconstruct
        spec_restored = SampleFeatureSpec(**json_data)
        assert spec_restored.key == spec.key


# ============================================================================
# Property Tests for FeatureDep
# ============================================================================


class TestFeatureDepProperties:
    """Property tests for FeatureDep accepting coercible keys."""

    @given(key_data=coercible_to_feature_key())
    @settings(max_examples=100)
    def test_feature_dep_accepts_coercible_key(self, key_data: Any):
        """Test that FeatureDep accepts all coercible key formats."""
        dep = FeatureDep(feature=key_data)

        assert isinstance(dep.feature, FeatureKey)
        assert isinstance(dep.feature.parts, tuple)

    @given(
        key_data=coercible_to_feature_key(),
        columns=st.one_of(
            st.none(),
            st.just(()),  # Empty tuple means only system columns
            st.lists(
                st.text(
                    alphabet=st.characters(whitelist_categories=("Lu", "Ll")),
                    min_size=1,
                    max_size=10,
                ),
                min_size=1,
                max_size=5,
            ).map(tuple),
        ),
    )
    @settings(max_examples=50)
    def test_feature_dep_with_columns(
        self, key_data: Any, columns: tuple[str, ...] | None
    ):
        """Test FeatureDep with column selection."""
        dep = FeatureDep(feature=key_data, columns=columns)

        assert isinstance(dep.feature, FeatureKey)
        assert dep.columns == columns

    @given(
        key_data=coercible_to_feature_key(),
        rename_pairs=st.lists(
            st.tuples(
                st.text(alphabet="abcdefgh", min_size=1, max_size=5),
                st.text(alphabet="ijklmnop", min_size=1, max_size=5),
            ),
            max_size=3,
        ),
    )
    @settings(max_examples=50)
    def test_feature_dep_with_rename(
        self, key_data: Any, rename_pairs: list[tuple[str, str]]
    ):
        """Test FeatureDep with column renaming."""
        rename = dict(rename_pairs) if rename_pairs else None

        dep = FeatureDep(feature=key_data, rename=rename)

        assert isinstance(dep.feature, FeatureKey)
        assert dep.rename == rename

    @given(key_data=coercible_to_feature_key())
    @settings(max_examples=50)
    def test_feature_dep_json_serialization(self, key_data: Any):
        """Test FeatureDep JSON serialization."""
        dep = FeatureDep(
            feature=key_data, columns=("col1", "col2"), rename={"col1": "new_col1"}
        )

        # Serialize to JSON
        json_data = dep.model_dump(mode="json")

        # Key should be serialized as a list
        assert isinstance(json_data["feature"], list)
        assert json_data["columns"] == ["col1", "col2"]  # Tuple becomes list in JSON
        assert json_data["rename"] == {"col1": "new_col1"}

        # Should be able to reconstruct
        dep_restored = FeatureDep(**json_data)
        assert dep_restored.feature == dep.feature
        assert dep_restored.columns == ("col1", "col2")
        assert dep_restored.rename == {"col1": "new_col1"}


# ============================================================================
# Property Tests for FieldSpec
# ============================================================================


class TestFieldSpecProperties:
    """Property tests for FieldSpec accepting coercible keys."""

    @given(key_data=coercible_to_field_key())
    @settings(max_examples=100)
    def test_field_spec_accepts_coercible_key(self, key_data: Any):
        """Test that FieldSpec accepts all coercible key formats."""
        field = FieldSpec(key=key_data, code_version="1")

        assert isinstance(field.key, FieldKey)
        assert isinstance(field.key.parts, tuple)

    @given(
        key_data=coercible_to_field_key(),
        code_version=st.integers(min_value=1, max_value=100).map(str),
    )
    @settings(max_examples=50)
    def test_field_spec_with_code_version(self, key_data: Any, code_version: str):
        """Test FieldSpec with different code versions."""
        field = FieldSpec(key=key_data, code_version=code_version)

        assert isinstance(field.key, FieldKey)
        assert field.code_version == code_version

    @given(key_data=coercible_to_field_key())
    @settings(max_examples=50)
    def test_field_spec_json_serialization(self, key_data: Any):
        """Test FieldSpec JSON serialization."""
        field = FieldSpec(key=key_data, code_version="7", deps=SpecialFieldDep.ALL)

        # Serialize to JSON
        json_data = field.model_dump(mode="json")

        # Key should be serialized as a list
        assert isinstance(json_data["key"], list)
        assert json_data["code_version"] == "7"

        # Should be able to reconstruct
        field_restored = FieldSpec(**json_data)
        assert field_restored.key == field.key
        assert field_restored.code_version == field.code_version


# ============================================================================
# Integration Tests with Complex Nested Structures
# ============================================================================


class TestComplexIntegration:
    """Integration tests with complex nested structures."""

    @given(
        main_key=coercible_to_feature_key(),
        dep_keys=st.lists(coercible_to_feature_key(), min_size=0, max_size=2),
        unique_field_parts=unique_key_parts_lists(min_size=1, max_size=2),
    )
    @settings(max_examples=30, deadline=5000)
    def test_complete_feature_spec_with_all_coercible_types(
        self, main_key: Any, dep_keys: list[Any], unique_field_parts: list[list[str]]
    ):
        """Test complete SampleFeatureSpec with all components using coercible types."""
        # Build fields with unique keys
        fields = []
        for i, parts in enumerate(unique_field_parts):
            # Vary the format
            if i % 2 == 0:
                field_key = "/".join(parts)
            else:
                field_key = parts

            fields.append(
                FieldSpec(
                    key=field_key,
                    code_version=str(i + 1),
                    deps=SpecialFieldDep.ALL,  # Always use ALL since NONE doesn't exist
                )
            )

        # Build dependencies and create spec (omit deps if empty)
        if dep_keys:
            deps = [
                FeatureDep(
                    feature=dep_key,
                    columns=("col1",) if i % 2 == 0 else None,
                    rename={"col1": f"dep{i}_col1"} if i % 3 == 0 else None,
                )
                for i, dep_key in enumerate(dep_keys)
            ]
            spec = SampleFeatureSpec(key=main_key, deps=deps, fields=fields)
        else:
            spec = SampleFeatureSpec(key=main_key, fields=fields)

        # Verify structure
        assert isinstance(spec.key, FeatureKey)
        if dep_keys:
            assert spec.deps is not None
            for dep in spec.deps:
                assert isinstance(dep.feature, FeatureKey)
        for field in spec.fields:
            assert isinstance(field.key, FieldKey)

        # Test JSON serialization roundtrip
        json_str = spec.model_dump_json()
        json_data = json.loads(json_str)
        spec_restored = SampleFeatureSpec(**json_data)

        assert spec_restored.key == spec.key
        assert len(spec_restored.fields) == len(spec.fields)

    @given(data=st.data())
    @settings(max_examples=20)
    def test_nested_pydantic_model_with_keys(self, data: st.DataObject):
        """Test nested Pydantic models with coercible keys."""

        class MyNestedModel(BaseModel):
            """A nested model that uses FeatureKey and FieldKey."""

            feature_key: FeatureKey
            field_key: FieldKey
            metadata: dict[str, Any]

        class MyContainerModel(BaseModel):
            """A container model with nested models."""

            items: list[MyNestedModel]
            main_feature: SampleFeatureSpec

        # Generate data
        n_items = data.draw(st.integers(min_value=1, max_value=3))
        items = []
        for _ in range(n_items):
            items.append(
                MyNestedModel(
                    feature_key=data.draw(coercible_to_feature_key()),
                    field_key=data.draw(coercible_to_field_key()),
                    metadata={"some": "data"},
                )
            )

        main_feature = SampleFeatureSpec(key=data.draw(coercible_to_feature_key()))

        # Create container
        container = MyContainerModel(items=items, main_feature=main_feature)

        # Verify all keys are properly converted
        for item in container.items:
            assert isinstance(item.feature_key, FeatureKey)
            assert isinstance(item.field_key, FieldKey)
        assert isinstance(container.main_feature.key, FeatureKey)

        # Test JSON serialization
        json_data = container.model_dump(mode="json")

        # All keys should be serialized as lists
        for item in json_data["items"]:
            assert isinstance(item["feature_key"], list)
            assert isinstance(item["field_key"], list)
        assert isinstance(json_data["main_feature"]["key"], list)

        # Roundtrip
        container_restored = MyContainerModel(**json_data)
        assert len(container_restored.items) == len(container.items)


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCasesWithHypothesis:
    """Edge case tests using Hypothesis."""

    @given(n=st.integers(min_value=0, max_value=100))
    def test_very_deep_keys(self, n: int):
        """Test keys with many parts."""
        parts = [f"part{i}" for i in range(n)]

        if n > 0:
            # Test FeatureKey
            feature_key = FeatureKey(parts)
            assert len(feature_key.parts) == n
            assert feature_key.to_string() == "/".join(parts)

            # Test FieldKey
            field_key = FieldKey(parts)
            assert len(field_key.parts) == n
            assert field_key.to_string() == "/".join(parts)
        else:
            # Empty list case
            feature_key = FeatureKey([])
            assert len(feature_key.parts) == 0
            assert feature_key.to_string() == ""

    @given(
        text=st.text(
            alphabet=st.characters(blacklist_categories=("Cc", "Cs")), min_size=1
        )
    )
    def test_unicode_in_keys(self, text: str):
        """Test Unicode characters in keys (excluding "/" and "__")."""
        assume("/" not in text)
        assume("__" not in text)

        # Single part with unicode
        feature_key = FeatureKey([text])
        assert feature_key.parts == (text,)
        assert feature_key.to_string() == text

        # Multiple parts with unicode
        parts = [text, "normal", text + "2"]
        field_key = FieldKey(parts)
        assert list(field_key.parts) == parts
        assert field_key.to_string() == "/".join(parts)

    @given(parts=st.lists(valid_key_part(), min_size=1, max_size=3))
    def test_key_as_dict_key(self, parts: list[str]):
        """Test that keys work properly as dictionary keys."""
        feature_key1 = FeatureKey(parts)
        feature_key2 = FeatureKey("/".join(parts))
        field_key1 = FieldKey(parts)
        field_key2 = FieldKey("/".join(parts))

        # Create dictionary with keys
        d = {
            feature_key1: "value1",
            field_key1: "value2",
        }

        # Should be able to access with equivalent keys
        assert d[feature_key2] == "value1"
        assert d[field_key2] == "value2"

        # Keys should be hashable and comparable
        assert hash(feature_key1) == hash(feature_key2)
        assert hash(field_key1) == hash(field_key2)
