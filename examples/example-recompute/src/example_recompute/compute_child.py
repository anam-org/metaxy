"""Pipeline for recompute example.

This script represents the actual user pipeline that:
1. Loads upstream data
2. Computes parent feature (embeddings)
3. Computes child feature (predictions) using parent embeddings

The VERSION environment variable determines which feature versions are loaded.
Run with VERSION=1 initially, then VERSION=2 to see recomputation.
"""

from metaxy import (
    FeatureKey,
    get_feature_by_key,
    init_metaxy,
    load_features,
)
from metaxy.metadata_store.system import SystemTableStorage

load_features()

# feature showcase: get feature classes by key
child_key = FeatureKey(["examples", "child"])
parent_key = FeatureKey(["examples", "parent"])
ChildFeature = get_feature_by_key(child_key)
ParentFeature = get_feature_by_key(parent_key)

# Get metadata store from metaxy.toml config
config = init_metaxy()
with config.get_store() as store:
    # Save feature graph snapshot, normally this should be done in CI/CD before running the pipeline
    result = SystemTableStorage(store).push_graph_snapshot()

    snapshot_version = result.snapshot_version

    _ = result.already_recorded

    print(f"Graph snapshot_version: {snapshot_version}")

    # Compute child feature (e.g., generate predictions from embeddings)
    print(f"\n📊 Computing {ChildFeature.spec().key.to_string()}...")
    print(f"  feature_version: {ChildFeature.feature_version()}")

    ids_lazy = store.read_metadata(ParentFeature, columns=["sample_uid"])
    # Materialize for now (samples parameter currently expects eager frames)
    ids = ids_lazy.collect().to_polars()

    diff = store.resolve_update(ChildFeature)

    print(
        f"Identified: {len(diff.added)} new samples, {len(diff.changed)} samples with new provenance_by_field"
    )

    if len(diff.added) > 0:
        # diff.added is a Narwhals DataFrame - can pass directly to write_metadata
        store.write_metadata(ChildFeature, diff.added)
        print(f"✓ Materialized {len(diff.added)} new samples")

    if len(diff.changed) > 0:
        # diff.changed is a Narwhals DataFrame
        store.write_metadata(ChildFeature, diff.changed)
        print(f"✓ Recomputed {len(diff.changed)} changed samples")

    # Show child provenance_by_field
    child_result = store.read_metadata(ChildFeature, current_only=True)
    print("\n📋 Child provenance_by_field:")
    # Materialize Narwhals LazyFrame to Polars DataFrame
    child_df = child_result.collect().to_polars()
    for row in child_df.iter_rows(named=True):
        dv = row["metaxy_provenance_by_field"]
        print(f"  sample_uid={row['sample_uid']}: {dv}")
