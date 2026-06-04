"""Minimal Metaxy pipeline backed by DuckLake."""

import polars as pl

import metaxy as mx
from example_ducklake.definitions import DuckLakeDemoFeature
from metaxy.ext.duckdb import DuckDBMetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD


def build_demo_rows() -> pl.DataFrame:
    """Create deterministic sample metadata for the example pipeline."""
    return pl.DataFrame(
        {
            "sample_uid": ["clip_001", "clip_002"],
            "path": [
                "s3://demo-bucket/processed/clip_001.parquet",
                "s3://demo-bucket/processed/clip_002.parquet",
            ],
            METAXY_PROVENANCE_BY_FIELD: [
                {"path": "path_hash_clip_001_v1"},
                {"path": "path_hash_clip_002_v1"},
            ],
        }
    )


def load_feature_rows(store: DuckDBMetadataStore) -> list[tuple[str, str]]:
    """Read back a few rows through Metaxy's public read API."""
    feature_df = (
        store.read(DuckLakeDemoFeature, columns=["sample_uid", "path"])
        .collect()
        .to_polars()
        .sort("sample_uid")
    )
    return [
        (str(row["sample_uid"]), str(row["path"]))
        for row in feature_df.iter_rows(named=True)
    ]


def discover_feature_keys() -> list[str]:
    """List feature keys discovered in the active Metaxy project."""
    graph = mx.current_graph()
    return sorted(feature_key.to_string() for feature_key in graph.list_features())


if __name__ == "__main__":
    config = mx.init()
    store = config.get_store()
    assert isinstance(store, DuckDBMetadataStore), (
        "DuckLake example misconfigured: expected DuckDBMetadataStore."
    )
    demo_rows = build_demo_rows()
    feature_table_name = store.get_table_name(DuckLakeDemoFeature.spec().key)
    feature_rows: list[tuple[str, str]] = []

    print("DuckLake pipeline")
    print(f"  Store class: {store.__class__.__name__}")
    print(f"  Database: {store.database}")

    with store.open("w"):
        store.write(DuckLakeDemoFeature, demo_rows)
        feature_rows = load_feature_rows(store)

    print(f"  Wrote {len(demo_rows)} rows for {DuckLakeDemoFeature.spec().key}")
    print()
    print("Discovered Metaxy features:")
    for feature_key in discover_feature_keys():
        print(f"  {feature_key}")
    print()
    print(f"Created feature table: {feature_table_name}")
    print("Files:")
    for sample_uid, path in feature_rows:
        print(f"  {sample_uid}: {path}")
