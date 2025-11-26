"""Feasibility tests for dict-based versioning engine.

These tests verify that the dict-based approach can replace struct operations
for databases that don't support struct field access (PostgreSQL, SQLite).
"""

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.feature_spec import FieldSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.dict_based import DictBasedVersioningEngine
from metaxy.versioning.types import HashAlgorithm


class MockDictBasedEngine(DictBasedVersioningEngine):
    """Mock engine for testing dict-based operations."""

    @classmethod
    def implementation(cls) -> nw.Implementation:
        return nw.Implementation.POLARS

    def hash_string_column(self, df, source_column, target_column, hash_algo):
        """Simple mock hashing using native Polars."""
        import polars as pl

        native_df = df.to_native()
        if isinstance(native_df, pl.LazyFrame):
            native_df = native_df.collect()

        hashed = native_df.with_columns(
            pl.col(source_column).hash().cast(pl.String).alias(target_column)
        )
        return nw.from_native(hashed, eager_only=True)

    @staticmethod
    def aggregate_with_string_concat(
        df, group_by_columns, concat_column, concat_separator, exclude_columns
    ):
        """Simple mock aggregation using native Polars."""
        import polars as pl

        native_df = df.to_native()
        if isinstance(native_df, pl.LazyFrame):
            native_df = native_df.collect()

        # Build aggregation expressions
        agg_exprs = [
            pl.col(concat_column).str.concat(concat_separator).alias(concat_column)
        ]

        # Add first() for all other columns
        all_columns = set(native_df.columns)
        columns_to_aggregate = (
            all_columns - set(group_by_columns) - {concat_column} - set(exclude_columns)
        )
        for col in columns_to_aggregate:
            agg_exprs.append(pl.col(col).first().alias(col))

        result = native_df.group_by(group_by_columns).agg(agg_exprs)
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


def test_build_struct_column_flattens_columns():
    """Test that build_struct_column creates flattened columns instead of structs."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version="1"),
            FieldSpec(key=FieldKey(["field2"]), code_version="2"),
        ],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockDictBasedEngine(plan)

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
    result = engine.build_struct_column(
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
    engine = MockDictBasedEngine(plan)

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
    """Test that dict-based engine never creates actual Struct columns."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version="1"),
            FieldSpec(key=FieldKey(["field2"]), code_version="2"),
        ],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockDictBasedEngine(plan)

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
    result = engine.build_struct_column(
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
            f"Column {col_name} has Struct type {dtype_class_name}, but dict-based engine should not create Struct columns"
        )


def test_flattened_naming_convention():
    """Test that flattened column names follow the expected convention."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["my_field"]), code_version="1")],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockDictBasedEngine(plan)

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


def test_dict_based_engine_compatible_with_base_engine_interface():
    """Test that dict-based engine satisfies VersioningEngine interface."""
    # Create a simple feature plan
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[FieldSpec(key=FieldKey(["field1"]), code_version="1")],
    )
    plan = FeaturePlan(feature=spec, deps=None)
    engine = MockDictBasedEngine(plan)

    # Verify engine has all required abstract methods implemented
    assert hasattr(engine, "hash_string_column")
    assert hasattr(engine, "build_struct_column")
    assert hasattr(engine, "aggregate_with_string_concat")
    assert hasattr(engine, "keep_latest_by_group")
    assert hasattr(engine, "access_provenance_field")

    # Verify methods are callable
    df = pl.DataFrame({"col": ["a", "b", "c"]})
    df_nw = nw.from_native(df, eager_only=True)

    # Test hash_string_column
    result = engine.hash_string_column(df_nw, "col", "hashed", HashAlgorithm.XXHASH64)
    assert "hashed" in result.columns

    # Test build_struct_column
    df2 = pl.DataFrame({"hash1": ["a", "b"], "hash2": ["c", "d"]})
    df2_nw = nw.from_native(df2, eager_only=True)
    result = engine.build_struct_column(df2_nw, "prov", {"f1": "hash1", "f2": "hash2"})
    assert "prov__f1" in result.columns
    assert "prov__f2" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
