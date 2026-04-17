"""Tests for SampleFeatureSpec validation, especially duplicate field keys."""

import pytest
from metaxy import BaseFeature
from metaxy.models.feature_spec import FeatureDep, Unique
from metaxy.models.field import FieldSpec
from metaxy.models.types import FieldKey
from metaxy_testing.models import SampleFeatureSpec
from pydantic import ValidationError


def test_duplicate_field_keys_raises_error():
    """Test that duplicate field keys in a SampleFeatureSpec raise a validation error."""
    with pytest.raises(ValueError, match="Duplicate field key found: .*predictions.*"):
        _ = SampleFeatureSpec(
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
        _ = SampleFeatureSpec(
            key="test/feature",
            fields=[
                FieldSpec(key=FieldKey(["analysis"]), code_version="v1"),
                FieldSpec(key=FieldKey(["analysis"]), code_version="v2"),  # Still duplicate!
            ],
        )


def test_duplicate_nested_field_keys_raises_error():
    """Test that duplicate nested field keys raise a validation error."""
    with pytest.raises(ValueError, match="Duplicate field key found: .*model.output.*"):
        _ = SampleFeatureSpec(
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
    spec = SampleFeatureSpec(
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
    spec = SampleFeatureSpec(key="test/feature")
    assert len(spec.fields) == 1
    assert spec.fields[0].key == FieldKey(["default"])


def test_duplicate_field_keys_in_base_feature_spec():
    """Test that SampleFeatureSpec also validates unique field keys."""
    with pytest.raises(ValueError, match="Duplicate field key found: .*data.*"):
        _ = SampleFeatureSpec(
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
    with pytest.raises(ValueError, match="Duplicate field key found"):

        class _TestFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="test/duplicate_fields",
                id_columns=["sample_uid"],
                fields=[
                    FieldSpec(key=FieldKey(["output"]), code_version="1"),
                    FieldSpec(key=FieldKey(["intermediate"])),
                    FieldSpec(key=FieldKey(["output"]), code_version="2"),  # Duplicate!
                ],
            ),
        ):
            pass


def test_field_keys_with_similar_names():
    """Test that field keys with similar but different names are not duplicates."""
    # This should not raise any errors - similar names are distinct
    spec = SampleFeatureSpec(
        key="test/feature",
        fields=[
            FieldSpec(key=FieldKey(["data_v1"])),
            FieldSpec(key=FieldKey(["data_v2"])),  # Different suffix, not a duplicate
            FieldSpec(key=FieldKey(["data_raw"])),  # Different suffix, not a duplicate
        ],
    )
    assert len(spec.fields) == 3


def test_feature_spec_requires_id_columns():
    """Test that FeatureSpec (production API) requires id_columns parameter."""
    from metaxy.models.feature_spec import FeatureSpec
    from pydantic import ValidationError

    # This should fail - id_columns is required
    with pytest.raises(ValidationError, match="id_columns"):
        FeatureSpec(key="test/feature")  # Missing id_columns  # ty: ignore[no-matching-overload]

    # This should work
    spec = FeatureSpec(key="test/feature", id_columns=["sample_uid"])
    assert spec.id_columns == ("sample_uid",)


def test_feature_spec_empty_id_columns_raises_validation_error():
    """Test that FeatureSpec uses field constraints for empty id_columns."""
    from metaxy.models.feature_spec import FeatureSpec

    with pytest.raises(ValidationError, match="at least 1 item"):
        FeatureSpec(key="test/feature", id_columns=[])


def test_feature_dep_from_feature_class():
    """Test that FeatureDep can be created directly from a Feature class."""
    from metaxy.models.types import FeatureKey

    # Create a parent feature
    class ParentFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "parent_for_dep"]),
            fields=[FieldSpec(key=FieldKey(["value"]), code_version="1")],
        ),
    ):
        pass

    # Test 1: Create FeatureDep directly from Feature class
    dep = FeatureDep(feature=ParentFeature)
    assert dep.feature == ParentFeature.spec().key
    assert dep.select is None
    assert dep.rename is None

    # Test 2: Create FeatureDep from Feature class with options
    dep_with_columns = FeatureDep(
        feature=ParentFeature,
        rename={"value": "parent_value"},
        select=("parent_value",),
    )
    assert dep_with_columns.feature == ParentFeature.spec().key
    assert dep_with_columns.select == ("parent_value",)
    assert dep_with_columns.rename == {"value": "parent_value"}

    # Test 3: Verify FeatureDep works in FeatureSpec.deps
    child_spec = SampleFeatureSpec(
        key="test/child_with_dep",
        deps=[dep],
        fields=[FieldSpec(key=FieldKey(["result"]), code_version="1")],
    )
    assert len(child_spec.deps) == 1
    assert child_spec.deps[0].feature == ParentFeature.spec().key


def test_feature_spec_deps_mixed_types():
    """Test that FeatureSpec.deps accepts all coercible types in a single list."""
    from metaxy.models.feature_spec import FeatureSpec
    from metaxy.models.types import FeatureKey

    # Create a Feature class
    class MyFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["my", "feature"]),
            fields=[FieldSpec(key=FieldKey(["data"]), code_version="1")],
        ),
    ):
        pass

    # Test all coercible types in a single deps list
    spec = FeatureSpec(
        key=FeatureKey(["test", "mixed_deps"]),
        id_columns=["sample_uid"],
        deps=[
            MyFeature,  # Feature class
            FeatureDep(feature=["my", "feature", "key"]),  # FeatureDep with list key
            ["another", "key"],  # List of strings
            "very/nice",  # String with separator
        ],
        fields=[FieldSpec(key=FieldKey(["output"]), code_version="1")],
    )

    # Verify all were converted to FeatureDep
    assert len(spec.deps) == 4
    assert all(isinstance(dep, FeatureDep) for dep in spec.deps)

    # Verify the feature keys are correct
    assert spec.deps[0].feature == FeatureKey(["my", "feature"])
    assert spec.deps[1].feature == FeatureKey(["my", "feature", "key"])
    assert spec.deps[2].feature == FeatureKey(["another", "key"])
    assert spec.deps[3].feature == FeatureKey(["very", "nice"])


class TestUniqueValidation:
    """Tests for the unique field on FeatureSpec."""

    def test_unique_default_is_none(self) -> None:
        spec = SampleFeatureSpec(key="test/feature")
        assert spec.unique is None

    def test_unique_model_round_trips(self) -> None:
        spec = SampleFeatureSpec(key="test/feature", unique=Unique(subset=("col1", "col2")))
        assert spec.unique == Unique(subset=("col1", "col2"))

    def test_unique_dict_list_is_coerced(self) -> None:
        spec = SampleFeatureSpec(key="test/feature", unique={"subset": ["col1"]})
        assert spec.unique == Unique(subset=("col1",))

    def test_unique_subset_bare_string_raises(self) -> None:
        with pytest.raises(ValidationError):
            SampleFeatureSpec(key="test/feature", unique={"subset": "col1"})

    def test_unique_subset_mapping_raises(self) -> None:
        with pytest.raises(ValidationError):
            Unique(subset={"col1": True})  # ty: ignore[invalid-argument-type]

    def test_unique_subset_non_sequence_raises(self) -> None:
        with pytest.raises(ValidationError):
            Unique(subset=42)  # ty: ignore[invalid-argument-type]

    def test_unique_subset_empty_raises(self) -> None:
        with pytest.raises(ValidationError):
            Unique(subset=())

    def test_unique_subset_duplicate_columns_deduplicates(self) -> None:
        u = Unique(subset=("col2", "col1", "col2"))
        assert u.subset == ("col2", "col1")

    def test_unique_serializes_correctly(self) -> None:
        spec = SampleFeatureSpec(key="test/feature", unique=Unique(subset=("content_hash_2", "content_hash_1")))
        dumped = spec.model_dump(mode="json")
        assert dumped["unique"] == {"subset": ["content_hash_2", "content_hash_1"], "keep": "any"}

    def test_unique_none_serializes_correctly(self) -> None:
        spec = SampleFeatureSpec(key="test/feature")
        dumped = spec.model_dump(mode="json")
        assert dumped["unique"] is None

    def test_unique_missing_column_warns_on_feature(self) -> None:
        with pytest.warns(UserWarning, match="unique.subset columns.*not found"):

            class _BadFeature(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="test/bad_dedup",
                    unique=Unique(subset=("missing",)),
                ),
            ):
                pass

    def test_unique_current_feature_column_allowed(self) -> None:
        class _Feature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="test/current_feature_dedup",
                unique=Unique(subset=("content_hash",)),
            ),
        ):
            content_hash: str | None = None

        assert _Feature.spec().unique == Unique(subset=("content_hash",))

    def test_unique_system_column_allowed(self) -> None:
        class _SysColFeature(
            BaseFeature,
            spec=SampleFeatureSpec(
                key="test/sys_dedup",
                unique=Unique(subset=("metaxy_data_version",)),
            ),
        ):
            pass

        assert _SysColFeature.spec().unique == Unique(subset=("metaxy_data_version",))

    def test_unique_upstream_column_warns(self) -> None:
        class _Parent(BaseFeature, spec=SampleFeatureSpec(key="test/dedup_parent")):
            content_hash: str | None = None

        with pytest.warns(UserWarning, match="unique.subset columns.*not found"):

            class _Child(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="test/dedup_child",
                    deps=[FeatureDep(feature="test/dedup_parent", select=("content_hash",))],
                    unique=Unique(subset=("content_hash",)),
                ),
            ):
                pass

    def test_unique_renamed_upstream_column_warns(self) -> None:
        class _Parent(BaseFeature, spec=SampleFeatureSpec(key="test/dedup_rename_parent")):
            content_hash: str | None = None

        with pytest.warns(UserWarning, match="unique.subset columns.*not found"):

            class _Child(
                BaseFeature,
                spec=SampleFeatureSpec(
                    key="test/dedup_rename_child",
                    deps=[FeatureDep(feature="test/dedup_rename_parent", rename={"content_hash": "chash"})],
                    unique=Unique(subset=("chash",)),
                ),
            ):
                pass
