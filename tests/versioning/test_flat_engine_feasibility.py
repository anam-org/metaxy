"""Feasibility tests for the flat versioning engine."""

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature_spec import FieldSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.flat_engine import FlatVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class MockFlatEngine(FlatVersioningEngine):
    """Mock engine for testing flat column operations."""

    @classmethod
    def implementation(cls) -> nw.Implementation:
        return nw.Implementation.POLARS

    def hash_string_column(
        self, df, source_column, target_column, hash_algo, truncate_length=None
    ):
        """Simple mock hashing using native Polars."""
        import polars as pl

        native_df = df.to_native()
        if isinstance(native_df, pl.LazyFrame):
            native_df = native_df.collect()

        hashed_expr = pl.col(source_column).hash().cast(pl.String)
        if truncate_length is not None:
            hashed_expr = hashed_expr.str.slice(0, truncate_length)

        hashed = native_df.with_columns(hashed_expr.alias(target_column))
        return nw.from_native(hashed, eager_only=True)

    def concat_strings_over_groups(
        self,
        df,
        source_column,
        target_column,
        group_by_columns,
        order_by_columns,
        separator="|",
    ):
        """Simple mock aggregation using native Polars window functions."""
        import polars as pl

        native_df = df.to_native()
        if isinstance(native_df, pl.LazyFrame):
            native_df = native_df.collect()

        # Use sort_by within the window to ensure deterministic ordering
        effective_order_by = order_by_columns if order_by_columns else group_by_columns
        concat_expr = (
            pl.col(source_column)
            .sort_by(*effective_order_by)
            .str.join(separator)
            .over(group_by_columns)
        )
        result = native_df.with_columns(concat_expr.alias(target_column))
        return nw.from_native(result, eager_only=True)

    @staticmethod
    def keep_latest_by_group(df, group_columns, timestamp_column):
        """Simple mock keep_latest using native Polars."""
        import polars as pl

        native_df = df.to_native()
        if isinstance(native_df, pl.LazyFrame):
            native_df = native_df.collect()

        result = native_df.sort(timestamp_column).group_by(group_columns).last()
        return nw.from_native(result, eager_only=True)


def test_record_field_versions_flattens_columns():
    """record_field_versions should create flattened columns instead of structs."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version="1"),
            FieldSpec(key=FieldKey(["field2"]), code_version="2"),
        ],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockFlatEngine(plan)

    # Create DataFrame with hash columns
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "__hash_field1": ["hash1_1", "hash1_2", "hash1_3"],
            "__hash_field2": ["hash2_1", "hash2_2", "hash2_3"],
        }
    )
    df_nw = nw.from_native(df, eager_only=True)

    # Build "virtual struct" by renaming columns
    result = engine.record_field_versions(
        df_nw,
        "metaxy_provenance_by_field",
        {"field1": "__hash_field1", "field2": "__hash_field2"},
    )

    # Check that columns are renamed, not packed into a struct
    assert "metaxy_provenance_by_field__field1" in result.columns
    assert "metaxy_provenance_by_field__field2" in result.columns
    assert "__hash_field1" not in result.columns
    assert "__hash_field2" not in result.columns

    # Verify values are preserved
    result_native = result.to_native()
    assert result_native["metaxy_provenance_by_field__field1"].to_list() == [
        "hash1_1",
        "hash1_2",
        "hash1_3",
    ]
    assert result_native["metaxy_provenance_by_field__field2"].to_list() == [
        "hash2_1",
        "hash2_2",
        "hash2_3",
    ]


def test_access_provenance_field_returns_column_reference():
    """Test that access_provenance_field returns direct column reference."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockFlatEngine(plan)

    # Create DataFrame with flattened columns
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field__field1": ["hash1", "hash2", "hash3"],
        }
    )
    df_nw = nw.from_native(df, eager_only=True)

    # Access field using engine method
    field_expr = engine.access_provenance_field("metaxy_provenance_by_field", "field1")

    # Verify it returns a column expression (not struct.field())
    result = df_nw.select(field_expr.alias("extracted"))
    result_native = result.to_native()

    assert result_native["extracted"].to_list() == ["hash1", "hash2", "hash3"]


def test_no_struct_type_in_result():
    """Flat engine should never create actual Struct columns."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version="1"),
            FieldSpec(key=FieldKey(["field2"]), code_version="2"),
        ],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockFlatEngine(plan)

    # Create DataFrame with hash columns
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "__hash_field1": ["hash1_1", "hash1_2", "hash1_3"],
            "__hash_field2": ["hash2_1", "hash2_2", "hash2_3"],
        }
    )
    df_nw = nw.from_native(df, eager_only=True)

    # Build "struct"
    result = engine.record_field_versions(
        df_nw,
        "metaxy_provenance_by_field",
        {"field1": "__hash_field1", "field2": "__hash_field2"},
    )

    # Verify schema has no Struct types
    schema = result.collect_schema()
    for col_name, dtype in schema.items():
        # Check dtype class name doesn't contain "Struct"
        dtype_class_name = dtype.__class__.__name__
        assert "Struct" not in dtype_class_name, (
            f"Column {col_name} has Struct type {dtype_class_name}, but flat engine should not create Struct columns"
        )


def test_flattened_naming_convention():
    """Test that flattened column names follow the expected convention."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["my_field"]), code_version="1")],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockFlatEngine(plan)

    # Test the naming convention
    flattened_name = engine._get_flattened_column_name(
        "metaxy_provenance_by_field", "my_field"
    )
    assert flattened_name == "metaxy_provenance_by_field__my_field"

    # Test with different struct names
    assert (
        engine._get_flattened_column_name("metaxy_data_version_by_field", "field1")
        == "metaxy_data_version_by_field__field1"
    )


def test_flat_engine_compatible_with_base_engine_interface():
    """Flat engine should satisfy the VersioningEngine interface."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockFlatEngine(plan)

    # Verify engine has all required abstract methods implemented
    assert hasattr(engine, "hash_string_column")
    assert hasattr(engine, "record_field_versions")
    assert hasattr(engine, "concat_strings_over_groups")
    assert hasattr(engine, "keep_latest_by_group")
    assert hasattr(engine, "access_provenance_field")

    # Verify methods are callable
    df = pl.DataFrame({"col": ["a", "b", "c"]})
    df_nw = nw.from_native(df, eager_only=True)

    # Test hash_string_column
    result = engine.hash_string_column(df_nw, "col", "hashed", HashAlgorithm.XXHASH64)
    assert "hashed" in result.columns

    # Test record_field_versions
    df2 = pl.DataFrame({"hash1": ["a", "b"], "hash2": ["c", "d"]})
    df2_nw = nw.from_native(df2, eager_only=True)
    result = engine.record_field_versions(
        df2_nw, "prov", {"f1": "hash1", "f2": "hash2"}
    )
    assert "prov__f1" in result.columns
    assert "prov__f2" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
