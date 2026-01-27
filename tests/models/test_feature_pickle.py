"""Tests for pickling BaseFeature instances."""

import pickle

from metaxy_testing.models import SampleFeatureSpec

from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey

# Create dedicated graph for pickle test features to avoid polluting global graph
_pickle_test_graph = FeatureGraph()

# Define test feature classes at module level (required for pickle)
# but in their own dedicated graph to avoid pollution
with _pickle_test_graph.use():

    class SimpleFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "pickle"]),
            id_columns=["sample_uid"],
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        sample_uid: str
        value: int

    class OptionalFieldsFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "pickle_optional"]),
            id_columns=["sample_uid"],
            fields=[
                FieldSpec(key=FieldKey(["value"]), code_version="1"),
                FieldSpec(key=FieldKey(["optional"]), code_version="1"),
            ],
        ),
    ):
        sample_uid: str
        value: int
        optional: str | None = None

    class ComplexTypesFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "pickle_complex"]),
            id_columns=["sample_uid"],
            fields=[
                FieldSpec(key=FieldKey(["tags"]), code_version="1"),
                FieldSpec(key=FieldKey(["metadata"]), code_version="1"),
            ],
        ),
    ):
        sample_uid: str
        tags: list[str]
        metadata: dict[str, int]


def test_base_feature_instance_can_be_pickled():
    """Test that BaseFeature instances can be pickled and unpickled."""
    # Create instance
    instance = SimpleFeature(sample_uid="123", value=42)

    # Pickle and unpickle
    pickled = pickle.dumps(instance)
    unpickled = pickle.loads(pickled)

    # Verify data is preserved
    assert unpickled.sample_uid == "123"
    assert unpickled.value == 42


def test_base_feature_instance_with_optional_fields_can_be_pickled():
    """Test that BaseFeature instances with optional fields can be pickled."""
    # Create instance with and without optional field
    instance1 = OptionalFieldsFeature(sample_uid="123", value=42, optional="test")
    instance2 = OptionalFieldsFeature(sample_uid="456", value=99)

    # Pickle and unpickle both
    unpickled1 = pickle.loads(pickle.dumps(instance1))
    unpickled2 = pickle.loads(pickle.dumps(instance2))

    # Verify data is preserved
    assert unpickled1.sample_uid == "123"
    assert unpickled1.value == 42
    assert unpickled1.optional == "test"

    assert unpickled2.sample_uid == "456"
    assert unpickled2.value == 99
    assert unpickled2.optional is None


def test_base_feature_instance_with_complex_types_can_be_pickled():
    """Test that BaseFeature instances with complex types can be pickled."""
    # Create instance with complex types
    instance = ComplexTypesFeature(sample_uid="123", tags=["tag1", "tag2"], metadata={"key1": 1, "key2": 2})

    # Pickle and unpickle
    unpickled = pickle.loads(pickle.dumps(instance))

    # Verify data is preserved
    assert unpickled.sample_uid == "123"
    assert unpickled.tags == ["tag1", "tag2"]
    assert unpickled.metadata == {"key1": 1, "key2": 2}


def test_frozen_feature_cannot_be_mutated():
    """Test that BaseFeature instances are frozen (immutable)."""
    instance = SimpleFeature(sample_uid="123", value=42)

    # Try to modify field - should raise ValidationError
    try:
        instance.value = 99
        assert False, "Expected ValidationError for frozen model"
    except Exception as e:
        # Pydantic raises ValidationError for frozen models
        assert "frozen" in str(e).lower() or "immutable" in str(e).lower()
