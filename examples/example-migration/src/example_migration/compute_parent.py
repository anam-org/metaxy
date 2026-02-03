"""Compute parent feature (embeddings) - unchanged between versions."""

import tempfile
from pathlib import Path

import metaxy as mx
import polars as pl

# Initialize metaxy (loads config and discovers features)
config = mx.init_metaxy()

# Get feature key
parent_key = mx.FeatureKey(["examples", "parent"])

# Load upstream data (use system temp dir for cross-platform compatibility)
data_dir = Path(tempfile.gettempdir()) / "migration_example_data"
upstream_data = pl.read_parquet(data_dir / "upstream_data.parquet")
with config.get_store() as store:
    feature_version = mx.current_graph().get_feature_version(parent_key)
    print(f"Computing {parent_key.to_string()}...")
    print(f"  feature_version: {feature_version[:16]}...")

    # Simulate computing embeddings - same computation in both versions
    parent_data = pl.DataFrame(
        {
            "sample_uid": upstream_data["sample_uid"],
            "metaxy_provenance_by_field": [
                {"embeddings": f"embed_{sid}"} for sid in upstream_data["sample_uid"]
            ],
        }
    )

    store.write_metadata(parent_key, parent_data)
    print(f"[OK] Materialized {len(parent_data)} samples")
