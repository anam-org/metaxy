"""Compute child feature (predictions) - code_version changes between v1 and v2."""

from pathlib import Path

import polars as pl

from metaxy import FeatureKey, MetaxyConfig, get_feature_by_key, load_features

# Load features
load_features()

# Get feature class
ChildFeature = get_feature_by_key(FeatureKey(["examples", "child"]))

# Load upstream data for sample_uids
data_dir = Path("/tmp/migration_example_data")
upstream_data = pl.read_parquet(data_dir / "upstream_data.parquet")

# Get metadata store
config = MetaxyConfig.load(search_parents=True)
with config.get_store() as store:
    print(f"📊 Computing {ChildFeature.spec().key.to_string()}...")
    print(f"  feature_version: {ChildFeature.feature_version()[:16]}...")

    # Use resolve_update to calculate what needs computing
    child_samples = upstream_data.select("sample_uid")
    diff_result = store.resolve_update(ChildFeature, sample_df=child_samples)

    print(
        f"Identified: {len(diff_result.added)} new samples, {len(diff_result.changed)} samples with new data_version"
    )

    if len(diff_result.added) > 0:
        store.write_metadata(ChildFeature, diff_result.added)
        print(f"✓ Materialized {len(diff_result.added)} new samples")

    if len(diff_result.changed) > 0:
        # This should NOT happen after migration!
        store.write_metadata(ChildFeature, diff_result.changed)
        print(f"⚠️  Recomputed {len(diff_result.changed)} changed samples")

    # Show current data
    import narwhals as nw

    child_result = store.read_metadata(ChildFeature, current_only=True)
    child_eager = nw.from_native(child_result.collect())
    print("\n📋 Child data_versions:")
    for row in child_eager.iter_rows(named=True):
        dv = row["data_version"]["predictions"]
        print(f"  sample_uid={row['sample_uid']}: {dv[:16]}...")
