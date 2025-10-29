"""Tests for type coercion in FeatureSpec parameters.

Note: This file uses # type: ignore[arg-type] comments because Pydantic validators
accept more types at runtime than the static type annotations indicate.
The tests verify that the validators correctly coerce types at runtime.
"""

import pytest

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)


def test_feature_key_from_str() -> None:
    """Test FeatureKey accepts single string."""
    spec = FeatureSpec(key="simple", deps=None)  # type: ignore[arg-type]
    assert spec.key == FeatureKey(["simple"])
    assert spec.key.to_string() == "simple"


def test_feature_key_from_list_str() -> None:
    """Test FeatureKey accepts list of strings."""
    spec = FeatureSpec(key=["my", "feature"], deps=None)  # type: ignore[arg-type]
    assert spec.key == FeatureKey(["my", "feature"])
    assert spec.key.to_string() == "my/feature"


def test_feature_key_from_feature_key() -> None:
    """Test FeatureKey accepts FeatureKey (no-op)."""
    key = FeatureKey(["my", "feature"])
    spec = FeatureSpec(key=key, deps=None)
    assert spec.key == key
    assert spec.key is key  # Should be the same object


def test_field_key_from_str() -> None:
    """Test FieldKey accepts single string."""
    field = FieldSpec(key="age", code_version=1)  # type: ignore[arg-type]
    assert field.key == FieldKey(["age"])
    assert field.key.to_string() == "age"


def test_field_key_from_list_str() -> None:
    """Test FieldKey accepts list of strings."""
    field = FieldSpec(key=["user", "age"], code_version=1)  # type: ignore[arg-type]
    assert field.key == FieldKey(["user", "age"])
    assert field.key.to_string() == "user/age"


def test_field_key_from_field_key() -> None:
    """Test FieldKey accepts FieldKey (no-op)."""
    key = FieldKey(["user", "age"])
    field = FieldSpec(key=key, code_version=1)
    assert field.key == key
    assert field.key is key  # Should be the same object


def test_feature_dep_key_from_str() -> None:
    """Test FeatureDep.key accepts single string."""
    dep = FeatureDep(key="parent")  # type: ignore[arg-type]
    assert dep.key == FeatureKey(["parent"])
    assert dep.key.to_string() == "parent"


def test_feature_dep_key_from_list_str() -> None:
    """Test FeatureDep.key accepts list of strings."""
    dep = FeatureDep(key=["my", "parent"])  # type: ignore[arg-type]
    assert dep.key == FeatureKey(["my", "parent"])
    assert dep.key.to_string() == "my/parent"


def test_feature_dep_key_from_feature_key() -> None:
    """Test FeatureDep.key accepts FeatureKey."""
    key = FeatureKey(["my", "parent"])
    dep = FeatureDep(key=key)
    assert dep.key == key


def test_feature_dep_key_from_feature_class() -> None:
    """Test FeatureDep.key accepts Feature class."""
    graph = FeatureGraph()
    with graph.use():

        class ParentFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["examples", "parent"]),
                deps=None,
            ),
        ):
            pass

        # Create FeatureDep using Feature class
        dep = FeatureDep(key=ParentFeature)  # type: ignore[arg-type]
        assert dep.key == FeatureKey(["examples", "parent"])


def test_feature_dep_key_from_feature_spec() -> None:
    """Test FeatureDep.key accepts FeatureSpec."""
    parent_spec = FeatureSpec(
        key=FeatureKey(["examples", "parent"]),
        deps=None,
    )

    # Create FeatureDep using FeatureSpec
    dep = FeatureDep(key=parent_spec)  # type: ignore[arg-type]
    assert dep.key == FeatureKey(["examples", "parent"])


def test_field_dep_feature_key_from_str() -> None:
    """Test FieldDep.feature_key accepts single string."""
    dep = FieldDep(feature_key="parent", fields=[FieldKey(["age"])])  # type: ignore[arg-type]
    assert dep.feature_key == FeatureKey(["parent"])


def test_field_dep_feature_key_from_list_str() -> None:
    """Test FieldDep.feature_key accepts list of strings."""
    dep = FieldDep(feature_key=["my", "parent"], fields=[FieldKey(["age"])])  # type: ignore[arg-type]
    assert dep.feature_key == FeatureKey(["my", "parent"])


def test_field_dep_feature_key_from_feature_key() -> None:
    """Test FieldDep.feature_key accepts FeatureKey."""
    key = FeatureKey(["my", "parent"])
    dep = FieldDep(feature_key=key, fields=[FieldKey(["age"])])
    assert dep.feature_key == key


def test_field_dep_feature_key_from_feature_class() -> None:
    """Test FieldDep.feature_key accepts Feature class."""
    graph = FeatureGraph()
    with graph.use():

        class ParentFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["examples", "parent"]),
                deps=None,
            ),
        ):
            pass

        # Create FieldDep using Feature class
        dep = FieldDep(feature_key=ParentFeature, fields=[FieldKey(["age"])])  # type: ignore[arg-type]
        assert dep.feature_key == FeatureKey(["examples", "parent"])


def test_field_dep_feature_key_from_feature_spec() -> None:
    """Test FieldDep.feature_key accepts FeatureSpec."""
    parent_spec = FeatureSpec(
        key=FeatureKey(["examples", "parent"]),
        deps=None,
    )

    # Create FieldDep using FeatureSpec
    dep = FieldDep(feature_key=parent_spec, fields=[FieldKey(["age"])])  # type: ignore[arg-type]
    assert dep.feature_key == FeatureKey(["examples", "parent"])


def test_full_feature_spec_with_coercion() -> None:
    """Test complete FeatureSpec using type coercion."""
    graph = FeatureGraph()
    with graph.use():

        class ParentFeature(
            Feature,
            spec=FeatureSpec(
                key="parent",  # type: ignore[arg-type]  # str coercion
                deps=None,
            ),
        ):
            pass

        class ChildFeature(
            Feature,
            spec=FeatureSpec(
                key=["child", "feature"],  # type: ignore[arg-type]  # list[str] coercion
                deps=[
                    FeatureDep(key=ParentFeature),  # type: ignore[arg-type]  # Feature class coercion
                ],
                fields=[
                    FieldSpec(key="age", code_version=1),  # type: ignore[arg-type]  # str coercion
                    FieldSpec(
                        key=["user", "name"],  # type: ignore[arg-type]  # list[str] coercion
                        code_version=1,
                    ),
                ],
            ),
        ):
            pass

        # Verify the spec was constructed correctly
        assert ChildFeature.spec.key == FeatureKey(["child", "feature"])
        assert len(ChildFeature.spec.deps or []) == 1
        assert ChildFeature.spec.deps[0].key == FeatureKey(["parent"])  # type: ignore[index]
        assert ChildFeature.spec.fields[0].key == FieldKey(["age"])
        assert ChildFeature.spec.fields[1].key == FieldKey(["user", "name"])


def test_feature_key_validation_errors() -> None:
    """Test FeatureKey validation with invalid inputs."""
    # Forward slash not allowed
    with pytest.raises(ValueError, match="cannot contain forward slashes"):
        FeatureSpec(key="parent/child", deps=None)  # type: ignore[arg-type]

    # Double underscore not allowed
    with pytest.raises(ValueError, match="cannot contain double underscores"):
        FeatureSpec(key="parent__child", deps=None)  # type: ignore[arg-type]

    # These also apply to list[str]
    with pytest.raises(ValueError, match="cannot contain forward slashes"):
        FeatureSpec(key=["parent/child"], deps=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cannot contain double underscores"):
        FeatureSpec(key=["parent__child"], deps=None)  # type: ignore[arg-type]


def test_field_key_validation_errors() -> None:
    """Test FieldKey validation with invalid inputs."""
    # Forward slash not allowed
    with pytest.raises(ValueError, match="cannot contain forward slashes"):
        FieldSpec(key="age/value", code_version=1)  # type: ignore[arg-type]

    # Double underscore not allowed
    with pytest.raises(ValueError, match="cannot contain double underscores"):
        FieldSpec(key="age__value", code_version=1)  # type: ignore[arg-type]


def test_backward_compatibility() -> None:
    """Test that existing code using explicit FeatureKey/FieldKey still works."""
    graph = FeatureGraph()
    with graph.use():

        class OldStyleFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["old", "style"]),
                deps=[
                    FeatureDep(key=FeatureKey(["parent"])),
                ],
                fields=[
                    FieldSpec(key=FieldKey(["default"]), code_version=1),
                ],
            ),
        ):
            pass

        # Should work exactly as before
        assert OldStyleFeature.spec.key == FeatureKey(["old", "style"])
        assert OldStyleFeature.spec.deps[0].key == FeatureKey(["parent"])  # type: ignore[index]
        assert OldStyleFeature.spec.fields[0].key == FieldKey(["default"])
