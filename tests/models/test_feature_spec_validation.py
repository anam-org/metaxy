"""Tests for FeatureSpec validation, especially duplicate field keys."""

import pytest

from metaxy.models.feature import BaseFeature, FeatureGraph
from metaxy.models.feature_spec import BaseFeatureSpec, FeatureSpec, IDColumns
from metaxy.models.field import FieldSpec
from metaxy.models.types import FieldKey


def test_duplicate_field_keys_raises_error():
    """Test that duplicate field keys in a FeatureSpec raise a validation error."""
    with pytest.raises(ValueError, match="Duplicate field key found: .*predictions.*"):
        _ = FeatureSpec(
            key="test/feature",
            fields=[
                FieldSpec(key=FieldKey(["predictions"])),
                FieldSpec(key=FieldKey(["embeddings"])),
                FieldSpec(key=FieldKey(["predictions"])),  # Duplicate!
            ],
        )


def test_duplicate_field_keys_with_different_code_versions_still_fails():
    """Test that duplicate field keys fail even with different code versions."""
    with pytest.raises(ValueError, match="Duplicate field key found: .*analysis.*"):
        _ = FeatureSpec(
            key="test/feature",
            fields=[
                FieldSpec(key=FieldKey(["analysis"]), code_version="v1"),
                FieldSpec(
                    key=FieldKey(["analysis"]), code_version="v2"
                ),  # Still duplicate!
            ],
        )


def test_duplicate_nested_field_keys_raises_error():
    """Test that duplicate nested field keys raise a validation error."""
    with pytest.raises(ValueError, match="Duplicate field key found: .*model.output.*"):
        _ = FeatureSpec(
            key="test/feature",
            fields=[
                FieldSpec(key=FieldKey(["model", "output"])),
                FieldSpec(key=FieldKey(["model", "input"])),
                FieldSpec(key=FieldKey(["model", "output"])),  # Duplicate!
            ],
        )


def test_unique_field_keys_pass_validation():
    """Test that unique field keys pass validation successfully."""
    # This should not raise any errors
    spec = FeatureSpec(
        key="test/feature",
        fields=[
            FieldSpec(key=FieldKey(["predictions"])),
            FieldSpec(key=FieldKey(["embeddings"])),
            FieldSpec(key=FieldKey(["metadata"])),
        ],
    )
    assert len(spec.fields) == 3
    assert spec.fields[0].key == FieldKey(["predictions"])
    assert spec.fields[1].key == FieldKey(["embeddings"])
    assert spec.fields[2].key == FieldKey(["metadata"])


def test_default_field_is_unique():
    """Test that the default field doesn't conflict with itself."""
    # Using default fields (only one "default" field)
    spec = FeatureSpec(key="test/feature")
    assert len(spec.fields) == 1
    assert spec.fields[0].key == FieldKey(["default"])


def test_duplicate_field_keys_in_base_feature_spec():
    """Test that BaseFeatureSpec also validates unique field keys."""
    with pytest.raises(ValueError, match="Duplicate field key found: .*data.*"):
        _ = BaseFeatureSpec[IDColumns](
            key="test/feature",
            id_columns=["sample_uid", "chunk_id"],
            fields=[
                FieldSpec(key=FieldKey(["data"])),
                FieldSpec(key=FieldKey(["processed"])),
                FieldSpec(key=FieldKey(["data"])),  # Duplicate!
            ],
        )


def test_duplicate_field_keys_in_feature_class_definition():
    """Test that duplicate field keys are caught when defining a Feature class."""
    test_graph = FeatureGraph()

    with test_graph.use():
        with pytest.raises(ValueError, match="Duplicate field key found"):

            class _TestFeature(  # pyright: ignore[reportUnusedClass]
                BaseFeature[IDColumns],
                spec=BaseFeatureSpec(
                    key="test/duplicate_fields",
                    id_columns=["sample_uid"],
                    fields=[
                        FieldSpec(key=FieldKey(["output"]), code_version="1"),
                        FieldSpec(key=FieldKey(["intermediate"])),
                        FieldSpec(
                            key=FieldKey(["output"]), code_version="2"
                        ),  # Duplicate!
                    ],
                ),
            ):
                pass


def test_field_keys_case_sensitive():
    """Test that field keys are case-sensitive (different cases are not duplicates)."""
    # This should not raise any errors - case matters
    spec = FeatureSpec(
        key="test/feature",
        fields=[
            FieldSpec(key=FieldKey(["Data"])),
            FieldSpec(key=FieldKey(["data"])),  # Different case, not a duplicate
            FieldSpec(key=FieldKey(["DATA"])),  # Different case, not a duplicate
        ],
    )
    assert len(spec.fields) == 3
