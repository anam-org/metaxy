"""Tests for the collect_batches utility function."""

import ibis
import narwhals as nw
import polars as pl
import pytest

from metaxy.utils import collect_batches


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

    def test_lazy_no_chunk_size(self):
        """LazyFrame without chunk_size yields the whole frame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        nw_lf = nw.from_native(df.lazy())

        batches = list(collect_batches(nw_lf))

        assert len(batches) == 1
        assert batches[0].to_polars().equals(df)


class TestCollectBatchesIbis:
    """Tests for collect_batches with Ibis (DuckDB) backend."""

    @pytest.fixture
    def ibis_connection(self, tmp_path):
        """Create an in-memory DuckDB connection via Ibis."""
        return ibis.duckdb.connect(tmp_path / "test.duckdb")

    def test_no_chunk_size(self, ibis_connection):
        """Without chunk_size, yields the whole frame."""
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        ibis_connection.create_table("test_data", df.to_pandas(), overwrite=True)
        ibis_table = ibis_connection.table("test_data")
        nw_frame = nw.from_native(ibis_table)

        batches = list(collect_batches(nw_frame))

        assert len(batches) == 1
        result = batches[0].to_polars().sort("a")
        assert result["a"].to_list() == [1, 2, 3]

    def test_with_chunk_size(self, ibis_connection):
        """Ibis backend uses native to_pyarrow_batches for chunking."""
        df = pl.DataFrame({"a": list(range(10)), "b": list(range(10, 20))})
        ibis_connection.create_table("test_data", df.to_pandas(), overwrite=True)
        ibis_table = ibis_connection.table("test_data")
        nw_frame = nw.from_native(ibis_table)

        batches = list(collect_batches(nw_frame, chunk_size=3))

        # Should have all data across batches
        combined = pl.concat([b.to_polars() for b in batches])
        assert len(combined) == 10
        # Verify all values are present
        assert set(combined["a"].to_list()) == set(range(10))

    def test_preserves_sorted_order(self, ibis_connection):
        """When data is sorted, batches maintain that order."""
        df = pl.DataFrame({"id": list(range(10)), "val": list(range(10, 20))})
        ibis_connection.create_table("test_data", df.to_pandas(), overwrite=True)
        ibis_table = ibis_connection.table("test_data").order_by("id")
        nw_frame = nw.from_native(ibis_table)

        batches = list(collect_batches(nw_frame, chunk_size=3))

        # Combine batches in order and verify sequence is preserved
        combined = pl.concat([b.to_polars() for b in batches])
        assert combined["id"].to_list() == list(range(10))


class TestCollectBatchesUnsupported:
    """Tests for unsupported backends."""

    def test_unsupported_backend_raises(self):
        """Unsupported backends raise NotImplementedError."""
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2, 3]})
        nw_df = nw.from_native(df)

        with pytest.raises(NotImplementedError, match="not supported"):
            list(collect_batches(nw_df))
