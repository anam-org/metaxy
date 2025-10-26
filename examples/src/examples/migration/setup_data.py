"""Setup upstream data for migration example."""

from pathlib import Path

import polars as pl

# Create data directory
data_dir = Path("/tmp/migration_example_data")
data_dir.mkdir(exist_ok=True)

# Create upstream data
upstream_data = pl.DataFrame(
    {
        "sample_uid": ["video1", "video2", "video3"],
        "data_version": [
            {"frames": "upstream_v1_frames"},
            {"frames": "upstream_v2_frames"},
            {"frames": "upstream_v3_frames"},
        ],
    }
)

# Save to parquet
upstream_data.write_parquet(data_dir / "upstream_data.parquet")

print(f"✓ Created upstream data: {len(upstream_data)} samples")
print(f"  Saved to: {data_dir / 'upstream_data.parquet'}")
