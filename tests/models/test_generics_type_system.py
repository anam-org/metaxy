"""Tests for the generics type system in BaseFeature and BaseFeatureSpec.

This module tests the generic type parameter propagation through the system,
ensuring that IDColumnsT is properly preserved and type-checked across the codebase.
"""

from __future__ import annotations

from typing import Literal

import pydantic
import pytest

from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import (
    BaseFeatureSpec,
    FeatureDep,
    FeatureSpec,
    IDColumns,
    TestingFeatureSpec,
)
from metaxy.models.field import FieldSpec
from metaxy.models.types import FeatureKey, FieldKey


def test_base_feature_spec_generic_parameter():
    """Test that BaseFeatureSpec accepts and preserves generic type parameter."""

    # Create a custom ID columns type
    list[str]

    # Create a spec with the custom type
    class CustomSpec(BaseFeatureSpec):
        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=["custom_id_1", "custom_id_2"]
        )

    spec = CustomSpec(
        key=FeatureKey(["test"]),
    )

    # Verify the id_columns are preserved
    assert spec.id_columns == ["custom_id_1", "custom_id_2"]


def test_base_feature_generic_parameter():
    """Test that BaseFeature accepts and preserves generic type parameter from spec."""

    # Create a custom spec type
    class CustomSpec(BaseFeatureSpec):
        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=["entity_id", "version_id"]
        )

    # Create a feature using the custom spec
    class CustomFeature(
        BaseFeature,
        spec=CustomSpec(
            key=FeatureKey(["custom"]),
        ),
    ):
        pass

    # Verify id_columns property returns correct value
    assert CustomFeature.spec().id_columns == ["entity_id", "version_id"]


def test_testing_feature_spec_default_id_columns():
    """Test that TestingFeatureSpec has correct default id_columns."""
    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
    )

    # TestingFeatureSpec should default to ["sample_uid"]
    assert spec.id_columns == ["sample_uid"]


def test_feature_spec_default_id_columns():
    """Test that FeatureSpec (production) has correct default id_columns."""
    spec = FeatureSpec(
        key=FeatureKey(["test"]),
    )

    # FeatureSpec should default to ("sample_uid",) - a tuple
    assert spec.id_columns == ("sample_uid",)


def test_custom_literal_type_id_columns():
    """Test using Literal types for compile-time type safety of ID columns."""

    # Define a spec with Literal type for exact column names
    tuple[Literal["user_id"], Literal["session_id"]]

    class StrictSpec(BaseFeatureSpec):
        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=("user_id", "session_id"),
        )

    spec = StrictSpec(
        key=FeatureKey(["strict"]),
    )

    # Verify the literal values are preserved
    assert spec.id_columns == ("user_id", "session_id")


def test_spec_version_stability_across_generic_types():
    """Test that feature_spec_version is stable for equivalent specs with different generic types.

    When Pydantic serializes to JSON (mode="json"), lists and tuples with the same values
    are normalized to the same format, so they produce the same spec_version.
    This is expected behavior - the JSON representation is deterministic.
    """

    # Spec 1: Using list[str]
    class Spec1(BaseFeatureSpec):
        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=["user_id"]
        )

    # Spec 2: Using tuple[str, ...]
    class Spec2(BaseFeatureSpec):
        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=("user_id",)
        )

    spec1 = Spec1(
        key=FeatureKey(["test"]),
    )

    spec2 = Spec2(
        key=FeatureKey(["test"]),
    )

    # Pydantic's JSON serialization normalizes list and tuple to the same format,
    # so feature_spec_version should be the SAME despite different Python types
    assert spec1.feature_spec_version == spec2.feature_spec_version


def test_classmethod_spec_returns_correct_type():
    """Test that the .spec() classmethod returns the correct spec instance."""

    class MyFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["my_feature"]),
            id_columns=["custom_id"],
        ),
    ):
        pass

    # Verify .spec() returns the spec instance
    spec = MyFeature.spec()
    assert isinstance(spec, TestingFeatureSpec)
    assert spec.id_columns == ["custom_id"]


def test_multiple_features_with_different_id_columns():
    """Test creating multiple features with different ID column configurations."""

    class Feature1(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["feature1"]),
            id_columns=["sample_uid"],
        ),
    ):
        pass

    class Feature2(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["feature2"]),
            id_columns=["user_id", "session_id"],
        ),
    ):
        pass

    class Feature3(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["feature3"]),
            id_columns=["entity_id"],
        ),
    ):
        pass

    # Each feature should have its own ID columns
    assert Feature1.spec().id_columns == ["sample_uid"]
    assert Feature2.spec().id_columns == ["user_id", "session_id"]
    assert Feature3.spec().id_columns == ["entity_id"]


def test_feature_with_deps_preserves_id_columns():
    """Test that features with dependencies preserve their own ID columns."""

    class UpstreamFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["upstream"]),
            id_columns=["user_id"],
        ),
    ):
        pass

    class DownstreamFeature(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["downstream"]),
            deps=[FeatureDep(feature=FeatureKey(["upstream"]))],
            id_columns=["user_id", "session_id"],  # Different from upstream
        ),
    ):
        pass

    # Each feature maintains its own ID columns
    assert UpstreamFeature.spec().id_columns == ["user_id"]
    assert DownstreamFeature.spec().id_columns == ["user_id", "session_id"]


def test_empty_id_columns_validation():
    """Test that empty id_columns raises a validation error."""

    with pytest.raises(ValueError, match="id_columns must be non-empty"):
        TestingFeatureSpec(
            key=FeatureKey(["test"]),
            id_columns=[],  # Empty list should raise error
        )


def test_none_id_columns_uses_subclass_default():
    """Test that not specifying id_columns uses the subclass default."""

    # TestingFeatureSpec has default=["sample_uid"]
    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        # No id_columns specified
    )

    assert spec.id_columns == ["sample_uid"]


def test_feature_with_custom_spec_class():
    """Test creating a feature with a completely custom spec class."""

    # Define a custom spec class with its own ID column type
    class VideoIDColumns:
        """Custom ID columns type for video features."""

        video_id: str = ""
        frame_number: int = 0

    class VideoSpec(BaseFeatureSpec):
        """Custom spec for video features."""

        id_columns: pydantic.SkipValidation[IDColumns] = pydantic.Field(
            default=["video_id", "frame_number"],
        )

    class VideoFeature(
        BaseFeature,
        spec=VideoSpec(
            key=FeatureKey(["video"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Verify custom spec is used
    assert VideoFeature.spec().id_columns == ["video_id", "frame_number"]
    assert isinstance(VideoFeature.spec(), VideoSpec)


def test_base_feature_spec_without_default_requires_id_columns():
    """Test that BaseFeatureSpec without a default requires id_columns to be provided by subclass."""

    # BaseFeatureSpec itself doesn't have a default for id_columns
    # So direct instantiation should fail unless the subclass provides a default

    # This should work because TestingFeatureSpec provides a default
    spec1 = TestingFeatureSpec(
        key=FeatureKey(["test"]),
    )
    assert spec1.id_columns == ["sample_uid"]

    # This should also work - explicit value
    spec2 = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["custom"],
    )
    assert spec2.id_columns == ["custom"]


def test_feature_spec_model_dump_includes_id_columns():
    """Test that model_dump includes id_columns in the output."""

    spec = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["user_id", "session_id"],
    )

    dumped = spec.model_dump()

    # id_columns should be in the dumped dict
    assert "id_columns" in dumped
    assert dumped["id_columns"] == ["user_id", "session_id"]


def test_feature_spec_serialization_roundtrip():
    """Test that a spec can be serialized and deserialized correctly."""

    original = TestingFeatureSpec(
        key=FeatureKey(["test"]),
        id_columns=["entity_id", "timestamp"],
        fields=[
            FieldSpec(key=FieldKey(["data"]), code_version="1"),
        ],
    )

    # Serialize
    dumped = original.model_dump()

    # Deserialize
    restored = TestingFeatureSpec.model_validate(dumped)

    # Verify all fields match
    assert restored.key == original.key
    assert restored.deps == original.deps
    assert restored.id_columns == original.id_columns
    assert len(restored.fields) == len(original.fields)
    assert restored.feature_spec_version == original.feature_spec_version


def test_backward_compatibility_with_existing_features():
    """Test that existing features without explicit generic types still work."""

    # This mimics old code that doesn't specify generic types
    class LegacyFeature(
        BaseFeature,  # Using default list[str] for backward compatibility
        spec=TestingFeatureSpec(
            key=FeatureKey(["legacy"]),
            # Uses default id_columns from TestingFeatureSpec
        ),
    ):
        pass

    # Should still work with default sample_uid
    assert LegacyFeature.spec().id_columns == ["sample_uid"]


def test_multiple_spec_instances_are_independent():
    """Test that multiple spec instances don't interfere with each other."""

    spec1 = TestingFeatureSpec(
        key=FeatureKey(["test1"]),
        id_columns=["id1"],
    )

    spec2 = TestingFeatureSpec(
        key=FeatureKey(["test2"]),
        id_columns=["id2", "id3"],
    )

    # Modifying one shouldn't affect the other
    assert spec1.id_columns == ["id1"]
    assert spec2.id_columns == ["id2", "id3"]


def test_feature_graph_tracks_features_with_different_id_columns(
    graph: FeatureGraph,
):
    """Test that FeatureGraph correctly tracks features with different ID columns."""

    class Feature1(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f1"]),
            id_columns=["sample_uid"],
        ),
    ):
        pass

    class Feature2(
        BaseFeature,
        spec=TestingFeatureSpec(
            key=FeatureKey(["f2"]),
            id_columns=["user_id"],
        ),
    ):
        pass

    # Both features should be in the graph
    assert FeatureKey(["f1"]) in graph.features_by_key
    assert FeatureKey(["f2"]) in graph.features_by_key

    # Each should have correct ID columns
    f1 = graph.features_by_key[FeatureKey(["f1"])]
    f2 = graph.features_by_key[FeatureKey(["f2"])]

    assert f1.spec().id_columns == ["sample_uid"]
    assert f2.spec().id_columns == ["user_id"]
