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
config = mx.init()

# feature showcase: get feature definitions by key. Of course, feature classes can be just imported instead.
child_key = mx.FeatureKey(["examples", "child"])
child_def = mx.get_feature_by_key(child_key)
parent_key = mx.FeatureKey(["examples", "parent"])

with config.get_store() as store:
    result = SystemTableStorage(store).push_graph_snapshot()

    project_version = result.project_version

    print(f"Graph project_version: {project_version}")

    # Compute child feature (e.g., generate predictions from embeddings)
    print(f"\n📊 Computing {child_key.to_string()}...")
    print(f"  feature_version: {mx.current_graph().get_feature_version(child_key)}")

    ids_lazy = store.read(parent_key, columns=["sample_uid"])
    ids = ids_lazy.collect().to_polars()

    increment = store.resolve_update(child_key)

    print(
        f"Identified: {len(increment.new)} new samples, {len(increment.stale)} samples with new provenance_by_field"
    )

    if len(increment.new) > 0:
        # increment.new is a Narwhals DataFrame - can pass directly to write
        store.write(child_key, increment.new)
        print(f"✓ Materialized {len(increment.new)} new samples")

    if len(increment.stale) > 0:
        # increment.stale is a Narwhals DataFrame
        store.write(child_key, increment.stale)
        print(f"✓ Recomputed {len(increment.stale)} changed samples")

    # Show child provenance_by_field
    child_result = store.read(child_key, with_feature_history=False)
    print("\n📋 Child provenance_by_field:")
    # Materialize Narwhals LazyFrame to Polars DataFrame
    child_df = child_result.collect().to_polars()
    for row in child_df.iter_rows(named=True):
        dv = row["metaxy_provenance_by_field"]
        print(f"  sample_uid={row['sample_uid']}: {dv}")
