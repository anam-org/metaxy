from __future__ import annotations

import ibis
import narwhals as nw
import polars as pl
import pyarrow as pa
from metaxy.ext.ibis.versioning import IbisVersioningEngine
from metaxy.ext.polars.versioning import PolarsVersioningEngine
from metaxy.models.feature import FeatureGraph
from metaxy.models.feature_definition import FeatureDefinition
from polars_map import Map


def test_build_map_column_for_polars(
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    plan = test_graph.get_feature_plan(test_features["UpstreamFeatureA"].key)
    engine = PolarsVersioningEngine(plan)
    df = nw.from_native(pl.DataFrame({"_frames": ["f1"], "_audio": ["a1"]}))

    result = engine.build_map_column(
        df,
        "versions",
        {"frames": "_frames", "audio": "_audio"},
    )

    result_pl = result.to_native()
    assert result_pl.schema["versions"] == Map(pl.String(), pl.String())
    assert result_pl["versions"].map.get("frames").to_list() == ["f1"]
    assert result_pl["versions"].map.get("audio").to_list() == ["a1"]


def test_build_map_column_for_ibis(
    test_graph: FeatureGraph,
    test_features: dict[str, FeatureDefinition],
) -> None:
    plan = test_graph.get_feature_plan(test_features["UpstreamFeatureA"].key)
    engine = IbisVersioningEngine(plan, hash_functions={})
    df = nw.from_native(ibis.memtable({"_frames": ["f1"], "_audio": ["a1"]}), eager_only=False)
    assert isinstance(df, nw.LazyFrame)

    result = engine.build_map_column(
        df,
        "versions",
        {"frames": "_frames", "audio": "_audio"},
    )
    assert isinstance(result, nw.LazyFrame)

    table = result.collect().to_arrow()
    assert pa.types.is_map(table.schema.field("versions").type)
    assert dict(table.column("versions")[0].as_py()) == {"frames": "f1", "audio": "a1"}
