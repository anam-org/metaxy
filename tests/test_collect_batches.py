"""Tests for the collect_batches utility function."""

import ibis
import narwhals as nw
import polars as pl
import pytest

from metaxy._utils import collect_batches


class TestCollectBatchesPolars:
    """Tests for collect_batches with Polars backend."""

    def test_eager_no_chunk_size(self):
        """Without chunk_size, yields the whole frame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        nw_df = nw.from_native(df)

        batches = list(collect_batches(nw_df))

        assert len(batches) == 1
        assert batches[0].to_polars().equals(df)

    def test_lazy_with_chunk_size(self):
        """LazyFrame is split into batches of the requested size."""
        df = pl.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
        nw_lf = nw.from_native(df.lazy())

        batches = list(collect_batches(nw_lf, chunk_size=3))

        assert len(batches) == 4  # 3+3+3+1
        # Verify all data is present
        combined = pl.concat([b.to_polars() for b in batches])
        assert combined.sort("a").equals(df)

    def test_maintains_order_when_order_by_specified(self):
        """When order_by is provided, batch order is preserved."""
        df = pl.DataFrame({"id": [3, 1, 4, 1, 5, 9, 2, 6], "val": list(range(8))})
        sorted_df = df.sort("id")
        nw_lf = nw.from_native(sorted_df.lazy())

        batches = list(collect_batches(nw_lf, chunk_size=3, order_by=["id"]))

        # Combine and check ordering is preserved
        combined = pl.concat([b.to_polars() for b in batches])
        assert combined["id"].to_list() == sorted_df["id"].to_list()


class TestCollectBatchesIbis:
    """Tests for collect_batches with Ibis (DuckDB) backend."""

    @pytest.fixture
    def ibis_connection(self, tmp_path):
        """Create an in-memory DuckDB connection via Ibis."""
        return ibis.duckdb.connect(tmp_path / "test.duckdb")

    def test_eager_no_chunk_size(self, ibis_connection):
        """Without chunk_size, yields the whole frame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ibis_connection.create_table("test_data", df.to_pandas(), overwrite=True)
        ibis_table = ibis_connection.table("test_data")
        nw_frame = nw.from_native(ibis_table)

        batches = list(collect_batches(nw_frame))

        assert len(batches) == 1
        result = batches[0].to_polars().sort("a")
        assert result["a"].to_list() == [1, 2, 3]

    def test_lazy_with_chunk_size_requires_order_by(self, ibis_connection):
        """Ibis backend needs to add row index for chunking."""
        df = pl.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
        ibis_connection.create_table("test_data", df.to_pandas(), overwrite=True)
        ibis_table = ibis_connection.table("test_data")
        nw_frame = nw.from_native(ibis_table)

        # Without order_by, a dummy column is used internally
        batches = list(collect_batches(nw_frame, chunk_size=3))

        # Should have all data across batches
        combined = pl.concat([b.to_polars() for b in batches])
        assert len(combined) == 10

    def test_ordering_preserved_with_order_by(self, ibis_connection):
        """When order_by is specified, batches maintain the order."""
        df = pl.DataFrame({"id": [3, 1, 4, 1, 5, 9, 2, 6], "val": list(range(8))})
        sorted_df = df.sort("id")
        ibis_connection.create_table("test_data", sorted_df.to_pandas(), overwrite=True)
        ibis_table = ibis_connection.table("test_data")
        nw_frame = nw.from_native(ibis_table)

        batches = list(collect_batches(nw_frame, chunk_size=3, order_by=["id"]))

        # Combine batches in order and verify sequence is preserved
        combined = pl.concat([b.to_polars() for b in batches])
        assert combined["id"].to_list() == sorted_df["id"].to_list()
