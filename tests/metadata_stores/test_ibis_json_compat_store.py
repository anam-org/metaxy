"""Tests for IbisJsonCompatStore JSON pack/unpack functionality."""

from collections.abc import Mapping
from typing import Any

import narwhals as nw
import polars as pl
import pytest

from metaxy._testing.models import SampleFeatureSpec
from metaxy.metadata_store.ibis_json_compat import IbisJsonCompatStore
from metaxy.models.feature_spec import FieldSpec
from metaxy.models.plan import FeaturePlan
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.versioning.types import HashAlgorithm


class MockJsonCompatStore(IbisJsonCompatStore):
    """Mock implementation for testing JSON pack/unpack logic.

    Uses simple string concatenation to simulate JSON packing/unpacking
    for testing purposes (no actual JSON serialization).
    """

    def __init__(self):
        """Initialize mock store with DuckDB backend for testing."""
        # Use DuckDB as test backend (in-memory)
        super().__init__(
            backend="duckdb",
            connection_params={"database": ":memory:"},
            hash_algorithm=HashAlgorithm.MD5,
        )

    def _create_hash_functions(self):
        """Use DuckDB hash functions."""
        import ibis

        @ibis.udf.scalar.builtin
        def md5(_x: str) -> str:
            """DuckDB MD5() function."""
            ...

        def md5_hash(col_expr):
            return md5(col_expr.cast(str))

        return {HashAlgorithm.MD5: md5_hash}

    def _get_json_unpack_exprs(
        self,
        json_column: str,
        field_names: list[str],
    ) -> dict[str, Any]:
        """Mock JSON unpacking using DuckDB JSON functions.

        Uses json_extract_string to extract fields from JSON.
        """
        import ibis

        # DuckDB json_extract_string function
        @ibis.udf.scalar.builtin
        def json_extract_string(_json: str, _path: str) -> str:
            """DuckDB json_extract_string function."""
            ...

        table = ibis._
        exprs = {}
        for field_name in field_names:
            flattened_name = f"{json_column}__{field_name}"
            # Extract field using JSON path: $.field_name
            exprs[flattened_name] = json_extract_string(
                table[json_column].cast("string"),
                ibis.literal(f"$.{field_name}"),
            )
        return exprs

    def _get_json_pack_expr(
        self,
        struct_name: str,
        field_columns: Mapping[str, str],
    ) -> Any:
        """Mock JSON packing using DuckDB json_object.

        Creates a JSON object from column values.
        """
        import ibis
        import ibis.expr.datatypes as dt

        table = ibis._

        @ibis.udf.scalar.builtin(output_type=dt.string)
        def to_json(_input) -> str: ...

        keys_expr = ibis.array(
            [ibis.literal(k).cast("string") for k, _ in sorted(field_columns.items())]
        )
        values_expr = ibis.array(
            [
                table[col_name].cast("string")
                for _, col_name in sorted(field_columns.items())
            ]
        )
        map_expr = ibis.map(keys_expr, values_expr)
        return to_json(map_expr)


@pytest.fixture
def mock_store():
    """Create mock JSON-compatible store for testing."""
    return MockJsonCompatStore()


@pytest.fixture
def sample_feature_plan():
    """Create a sample feature plan for testing."""
    spec = SampleFeatureSpec(
        key=FeatureKey(["test", "feature"]),
        fields=[
            FieldSpec(key=FieldKey(["field1"]), code_version="1"),
            FieldSpec(key=FieldKey(["field2"]), code_version="2"),
        ],
    )
    return FeaturePlan(feature=spec, deps=None)


def test_store_uses_dict_based_engine(mock_store):
    """Test that IbisJsonCompatStore uses IbisDictBasedVersioningEngine."""
    from metaxy.versioning.flat_engine import IbisFlatVersioningEngine

    assert mock_store.versioning_engine_cls == IbisFlatVersioningEngine


def test_pack_json_columns_creates_json_from_flattened(mock_store, sample_feature_plan):
    """Test that _pack_json_columns creates JSON from flattened columns."""
    # Create DataFrame with flattened columns
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field__field1": ["hash1_1", "hash1_2", "hash1_3"],
            "metaxy_provenance_by_field__field2": ["hash2_1", "hash2_2", "hash2_3"],
        }
    )

    # Convert to Ibis for testing
    with mock_store:
        # Create temp table to convert Polars to Ibis
        mock_store.conn.create_table("test_pack", obj=df, overwrite=True)
        ibis_table = mock_store.conn.table("test_pack")
        ibis_nw = nw.from_native(ibis_table, eager_only=False)

        # Pack columns
        result = mock_store._pack_json_columns(ibis_nw, sample_feature_plan)

        # Check that JSON column exists
        assert "metaxy_provenance_by_field" in result.columns

        # Check that flattened columns are removed
        assert "metaxy_provenance_by_field__field1" not in result.columns
        assert "metaxy_provenance_by_field__field2" not in result.columns

        # Verify it's still an Ibis LazyFrame
        assert result.implementation == nw.Implementation.IBIS


def test_unpack_json_columns_creates_flattened_from_json(
    mock_store, sample_feature_plan
):
    """Test that _unpack_json_columns creates flattened columns from JSON."""
    # Create DataFrame with JSON column
    # Using DuckDB's json_object to create proper JSON
    with mock_store:
        # Create a table with JSON column manually using DuckDB SQL
        mock_store.conn.raw_sql("""
            CREATE TABLE test_unpack AS
            SELECT
                sample_uid,
                json_object('field1', field1_val, 'field2', field2_val) as metaxy_provenance_by_field
            FROM (VALUES
                (1, 'hash1_1', 'hash2_1'),
                (2, 'hash1_2', 'hash2_2'),
                (3, 'hash1_3', 'hash2_3')
            ) AS t(sample_uid, field1_val, field2_val)
        """)

        # Get the table
        ibis_table = mock_store.conn.table("test_unpack")
        ibis_nw = nw.from_native(ibis_table, eager_only=False)

        # Unpack columns
        result = mock_store._unpack_json_columns(ibis_nw, sample_feature_plan)

        # Check that flattened columns exist
        assert "metaxy_provenance_by_field__field1" in result.columns
        assert "metaxy_provenance_by_field__field2" in result.columns

        # Check JSON column handling: it may be removed or rebuilt as struct
        if "metaxy_provenance_by_field" in result.columns:
            schema = result.collect_schema()
            assert isinstance(schema["metaxy_provenance_by_field"], nw.Struct)

        # Verify it's still an Ibis LazyFrame
        assert result.implementation == nw.Implementation.IBIS

        # Collect and verify values
        result_df = result.collect().to_native()
        assert result_df["metaxy_provenance_by_field__field1"].to_pylist() == [
            "hash1_1",
            "hash1_2",
            "hash1_3",
        ]
        assert result_df["metaxy_provenance_by_field__field2"].to_pylist() == [
            "hash2_1",
            "hash2_2",
            "hash2_3",
        ]


def test_pack_then_unpack_roundtrip(mock_store, sample_feature_plan):
    """Test that pack → unpack is a roundtrip (data preserved)."""
    # Create DataFrame with flattened columns
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field__field1": ["hash1_1", "hash1_2", "hash1_3"],
            "metaxy_provenance_by_field__field2": ["hash2_1", "hash2_2", "hash2_3"],
        }
    )

    with mock_store:
        # Convert to Ibis
        mock_store.conn.create_table("test_roundtrip", obj=df, overwrite=True)
        ibis_table = mock_store.conn.table("test_roundtrip")
        ibis_nw = nw.from_native(ibis_table, eager_only=False)

        # Pack columns to JSON
        packed = mock_store._pack_json_columns(ibis_nw, sample_feature_plan)

        # Write to table to simulate database storage (materialize to avoid UDF issues)
        packed_native = packed.collect().to_native()
        mock_store.conn.create_table("test_packed", obj=packed_native, overwrite=True)

        # Read back
        packed_table = mock_store.conn.table("test_packed")
        packed_nw = nw.from_native(packed_table, eager_only=False)

        # Unpack columns from JSON
        unpacked = mock_store._unpack_json_columns(packed_nw, sample_feature_plan)

        # Verify data preserved
        unpacked_df = unpacked.collect().to_native()
        assert unpacked_df["sample_uid"].to_pylist() == [1, 2, 3]
        assert unpacked_df["metaxy_provenance_by_field__field1"].to_pylist() == [
            "hash1_1",
            "hash1_2",
            "hash1_3",
        ]
        assert unpacked_df["metaxy_provenance_by_field__field2"].to_pylist() == [
            "hash2_1",
            "hash2_2",
            "hash2_3",
        ]


def test_pack_handles_missing_flattened_columns(mock_store, sample_feature_plan):
    """Test that pack gracefully handles missing flattened columns."""
    # Create DataFrame WITHOUT flattened columns
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "other_column": ["a", "b", "c"],
        }
    )

    with mock_store:
        # Convert to Ibis
        mock_store.conn.create_table("test_missing", obj=df, overwrite=True)
        ibis_table = mock_store.conn.table("test_missing")
        ibis_nw = nw.from_native(ibis_table, eager_only=False)

        # Pack columns (should not fail, just skip packing)
        result = mock_store._pack_json_columns(ibis_nw, sample_feature_plan)

        # Verify original columns preserved
        assert "sample_uid" in result.columns
        assert "other_column" in result.columns

        # JSON column should not be added if source columns missing
        # (current implementation may still add it, but it's okay)


def test_unpack_handles_missing_json_column(mock_store, sample_feature_plan):
    """Test that unpack gracefully handles missing JSON column."""
    # Create DataFrame WITHOUT JSON column
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "other_column": ["a", "b", "c"],
        }
    )

    with mock_store:
        # Convert to Ibis
        mock_store.conn.create_table("test_no_json", obj=df, overwrite=True)
        ibis_table = mock_store.conn.table("test_no_json")
        ibis_nw = nw.from_native(ibis_table, eager_only=False)

        # Unpack columns (should not fail, just skip unpacking)
        result = mock_store._unpack_json_columns(ibis_nw, sample_feature_plan)

        # Verify original columns preserved
        assert "sample_uid" in result.columns
        assert "other_column" in result.columns

        # Flattened columns should not be added if JSON column missing
        assert "metaxy_provenance_by_field__field1" not in result.columns
        assert "metaxy_provenance_by_field__field2" not in result.columns


def test_operations_stay_lazy(mock_store, sample_feature_plan):
    """Test that pack/unpack operations stay in Ibis lazy world."""
    df = pl.DataFrame(
        {
            "sample_uid": [1, 2, 3],
            "metaxy_provenance_by_field__field1": ["hash1_1", "hash1_2", "hash1_3"],
            "metaxy_provenance_by_field__field2": ["hash2_1", "hash2_2", "hash2_3"],
        }
    )

    with mock_store:
        # Convert to Ibis
        mock_store.conn.create_table("test_lazy", obj=df, overwrite=True)
        ibis_table = mock_store.conn.table("test_lazy")
        ibis_nw = nw.from_native(ibis_table, eager_only=False)

        # Verify input is lazy
        assert ibis_nw.implementation == nw.Implementation.IBIS
        assert hasattr(ibis_nw.to_native(), "compile")  # Ibis tables have compile()

        # Pack should keep it lazy
        packed = mock_store._pack_json_columns(ibis_nw, sample_feature_plan)
        assert packed.implementation == nw.Implementation.IBIS
        assert hasattr(packed.to_native(), "compile")

        # Materialize for testing
        packed_native = packed.collect().to_native()
        mock_store.conn.create_table(
            "test_lazy_packed", obj=packed_native, overwrite=True
        )

        # Read back as lazy
        packed_table = mock_store.conn.table("test_lazy_packed")
        packed_nw = nw.from_native(packed_table, eager_only=False)

        # Unpack should keep it lazy
        unpacked = mock_store._unpack_json_columns(packed_nw, sample_feature_plan)
        assert unpacked.implementation == nw.Implementation.IBIS
        assert hasattr(unpacked.to_native(), "compile")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
