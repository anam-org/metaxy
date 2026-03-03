"""Compute child feature (predictions) - code_version changes between v1 and v2."""

import tempfile
from pathlib import Path

import metaxy as mx
import polars as pl

# Initialize metaxy (loads config and discovers features)
config = mx.init()

# Get feature key
child_key = mx.FeatureKey(["examples", "child"])

# Load upstream data for sample_uids (use system temp dir for cross-platform compatibility)
data_dir = Path(tempfile.gettempdir()) / "migration_example_data"
upstream_data = pl.read_parquet(data_dir / "upstream_data.parquet")
with config.get_store().open("w") as store:
    feature_version = mx.current_graph().get_feature_version(child_key)
    print(f"Computing {child_key.to_string()}...")
    print(f"  feature_version: {feature_version[:16]}...")

    # Use resolve_update to calculate what needs computing
    # Don't pass samples - let system auto-load upstream and calculate provenance_by_field
    diff_result = store.resolve_update(child_key)

    print(
        f"Identified: {len(diff_result.new)} new samples, {len(diff_result.stale)} samples with new provenance_by_field"
    )

    if len(diff_result.new) > 0:
        store.write(child_key, diff_result.new)
        print(f"[OK] Materialized {len(diff_result.new)} new samples")

    if len(diff_result.stale) > 0:
        # This should NOT happen after migration!
        store.write(child_key, diff_result.stale)
        print(f"[WARN] Recomputed {len(diff_result.stale)} changed samples")

    # Show current data
    import narwhals as nw

    child_result = store.read(child_key, with_feature_history=False)
    child_eager = nw.from_native(child_result.collect())
    print("\nChild provenance_by_field:")
    for row in child_eager.iter_rows(named=True):
        dv = row["metaxy_provenance_by_field"]["predictions"]
        print(f"  sample_uid={row['sample_uid']}: {dv[:16]}...")
