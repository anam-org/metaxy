"""Pipeline for recompute example.

This script represents the actual user pipeline that:
1. Loads upstream data
2. Computes parent feature (embeddings)
3. Computes child feature (predictions) using parent embeddings

The VERSION environment variable determines which feature versions are loaded.
Run with VERSION=1 initially, then VERSION=2 to see recomputation.
"""

import metaxy as mx
from metaxy.metadata_store.system import SystemTableStorage

# Initialize metaxy (loads config and discovers features)
config = mx.init_metaxy()

# feature showcase: get feature definitions by key. Of course, feature classes can be just imported instead.
child_key = mx.FeatureKey(["examples", "child"])
child_def = mx.get_feature_by_key(child_key)
parent_key = mx.FeatureKey(["examples", "parent"])

with config.get_store() as store:
    result = SystemTableStorage(store).push_graph_snapshot()

    snapshot_version = result.snapshot_version

    print(f"Graph snapshot_version: {snapshot_version}")

    # Compute child feature (e.g., generate predictions from embeddings)
    print(f"\n📊 Computing {child_key.to_string()}...")
    print(f"  feature_version: {mx.current_graph().get_feature_version(child_key)}")

    ids_lazy = store.read_metadata(parent_key, columns=["sample_uid"])
    ids = ids_lazy.collect().to_polars()

    diff = store.resolve_update(child_key)

    print(f"Identified: {len(diff.added)} new samples, {len(diff.changed)} samples with new provenance_by_field")

    if len(diff.added) > 0:
        # diff.added is a Narwhals DataFrame - can pass directly to write_metadata
        store.write_metadata(child_key, diff.added)
        print(f"✓ Materialized {len(diff.added)} new samples")

    if len(diff.changed) > 0:
        # diff.changed is a Narwhals DataFrame
        store.write_metadata(child_key, diff.changed)
        print(f"✓ Recomputed {len(diff.changed)} changed samples")

    # Show child provenance_by_field
    child_result = store.read_metadata(child_key, current_only=True)
    print("\n📋 Child provenance_by_field:")
    # Materialize Narwhals LazyFrame to Polars DataFrame
    child_df = child_result.collect().to_polars()
    for row in child_df.iter_rows(named=True):
        dv = row["metaxy_provenance_by_field"]
        print(f"  sample_uid={row['sample_uid']}: {dv}")
