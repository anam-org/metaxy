"""Pipeline that writes and reads feature data using Alembic-managed tables."""

import metaxy as mx
import polars as pl
from metaxy.ext.metadata_stores.duckdb import DuckDBMetadataStore
from metaxy.models.constants import METAXY_PROVENANCE_BY_FIELD

from example_alembic.definitions import AlembicDemoFeature


def build_demo_rows() -> pl.DataFrame:
    """Create deterministic sample metadata."""
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


if __name__ == "__main__":
    config = mx.init()
    store = config.get_store()
    assert isinstance(store, DuckDBMetadataStore)

    demo_rows = build_demo_rows()

    print("Alembic + DuckLake pipeline")
    print(f"  auto_create_tables: {store.auto_create_tables}")

    with store.open("w"):
        store.write(AlembicDemoFeature, demo_rows)
        feature_df = (
            store.read(AlembicDemoFeature, columns=["sample_uid", "path"])
            .collect()
            .to_polars()
            .sort("sample_uid")
        )

    print(f"  Wrote {len(demo_rows)} rows for {AlembicDemoFeature.spec().key}")
    print(f"  Feature table: {store.get_table_name(AlembicDemoFeature.spec().key)}")
    print("  Read back:")
    for row in feature_df.iter_rows(named=True):
        print(f"    {row['sample_uid']}: {row['path']}")
