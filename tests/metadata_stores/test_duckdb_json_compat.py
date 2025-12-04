"""DuckDB JSON compatibility sanity test using the real IbisJsonCompatStore."""

from __future__ import annotations

from typing import Any, cast

import narwhals as nw
import polars as pl

from metaxy._testing import DuckDBJsonCompatStore, SampleFeatureSpec
from metaxy.models.constants import (
    METAXY_DATA_VERSION_BY_FIELD,
    METAXY_PROVENANCE_BY_FIELD,
)
from metaxy.models.feature import BaseFeature
from metaxy.models.types import FeatureKey


def test_duckdb_json_pack_unpack_round_trip(tmp_path, graph) -> None:
    """Ensure DuckDB JSON path matches standard DuckDB behavior end-to-end."""

    class SimpleFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["duckdb", "json_test"]), fields=["value"]
        ),
    ):
        pass

    feature_key = SimpleFeature.spec().key

    df = pl.DataFrame(
        {
            "sample_uid": [1, 2],
            "value": ["a", "b"],
            "metaxy_provenance_by_field__value": ["p1", "p2"],
            "metaxy_data_version_by_field__value": ["p1", "p2"],
        }
    )

    db_path = tmp_path / "duckdb_json.duckdb"
    with graph.use(), DuckDBJsonCompatStore(database=str(db_path)) as store:
        raw_conn = cast(Any, store.conn).con

        # Ensure JSON functionality is available
        raw_conn.execute("INSTALL json")
        raw_conn.execute("LOAD json")

        table_name = store.get_table_name(feature_key)
        raw_conn.execute(
            f"""
            CREATE TABLE "{table_name}" (
                sample_uid BIGINT,
                value TEXT,
                {METAXY_PROVENANCE_BY_FIELD} JSON,
                {METAXY_DATA_VERSION_BY_FIELD} JSON
            )
            """
        )

        store.write_metadata_to_store(feature_key, nw.from_native(df))

        lazy = store.read_metadata_in_store(feature_key)
        assert lazy is not None

        result_native = lazy.collect().to_native()
        result_pl: pl.DataFrame
        if isinstance(result_native, pl.DataFrame):
            result_pl = result_native
        else:
            import pyarrow as pa

            if isinstance(result_native, pa.Table):
                result_pl = cast(pl.DataFrame, pl.from_arrow(result_native))
            else:
                result_pl = pl.DataFrame(result_native)
        assert "metaxy_provenance_by_field__value" in result_pl.columns
        assert "metaxy_data_version_by_field__value" in result_pl.columns

        if METAXY_PROVENANCE_BY_FIELD in result_pl.columns:
            assert result_pl.schema[METAXY_PROVENANCE_BY_FIELD] == pl.Struct
        if METAXY_DATA_VERSION_BY_FIELD in result_pl.columns:
            assert result_pl.schema[METAXY_DATA_VERSION_BY_FIELD] == pl.Struct

        assert result_pl["metaxy_provenance_by_field__value"].to_list() == ["p1", "p2"]
        assert result_pl["metaxy_data_version_by_field__value"].to_list() == [
            "p1",
            "p2",
        ]
