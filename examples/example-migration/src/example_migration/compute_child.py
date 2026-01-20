"""Compute child feature (predictions) - code_version changes between v1 and v2."""

import tempfile
from pathlib import Path

import polars as pl

from metaxy import FeatureKey, get_feature_by_key, init_metaxy

# Initialize metaxy (loads config and discovers features)
config = init_metaxy()

# Get feature class
ChildFeature = get_feature_by_key(FeatureKey(["examples", "child"]))

# Load upstream data for sample_uids (use system temp dir for cross-platform compatibility)
data_dir = Path(tempfile.gettempdir()) / "migration_example_data"
upstream_data = pl.read_parquet(data_dir / "upstream_data.parquet")
with config.get_store() as store:
    print(f"Computing {ChildFeature.spec().key.to_string()}...")
    print(f"  feature_version: {ChildFeature.feature_version()[:16]}...")

    # Use resolve_update to calculate what needs computing
    # Don't pass samples - let system auto-load upstream and calculate provenance_by_field
    diff_result = store.resolve_update(ChildFeature)

    print(
        f"Identified: {len(diff_result.added)} new samples, {len(diff_result.changed)} samples with new provenance_by_field"
    )

    if len(diff_result.added) > 0:
        store.write_metadata(ChildFeature, diff_result.added)
        print(f"[OK] Materialized {len(diff_result.added)} new samples")

    if len(diff_result.changed) > 0:
        # This should NOT happen after migration!
        store.write_metadata(ChildFeature, diff_result.changed)
        print(f"[WARN] Recomputed {len(diff_result.changed)} changed samples")

    # Show current data
    import narwhals as nw

    child_result = store.read_metadata(ChildFeature, current_only=True)
    child_eager = nw.from_native(child_result.collect())
    print("\nChild provenance_by_field:")
    for row in child_eager.iter_rows(named=True):
        dv = row["metaxy_provenance_by_field"]["predictions"]
        print(f"  sample_uid={row['sample_uid']}: {dv[:16]}...")
