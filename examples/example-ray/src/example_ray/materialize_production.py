"""Materialize Example 3 production Dagster assets programmatically.

Run with: python -m example_ray.materialize_production
"""

import sys

import dagster as dg
import metaxy as mx
import metaxy.ext.dagster as mxd

if __name__ == "__main__":
    mx.init()

    from example_ray.defs.assets import video_metadata
    from example_ray.defs.assets.production import video_processing
    from example_ray.features import Crop, SpeechToText, Video

    # Verify production definitions module can be imported
    from example_ray.production_definitions import definitions  # noqa: F401

    store = mxd.MetaxyStoreFromConfigResource(name="dev")

    print("Materializing Example 3 assets: video -> video_processing (branching IO)")
    result = dg.materialize(
        assets=[  # ty: ignore[invalid-argument-type]
            video_metadata,
            video_processing,
        ],
        resources={"store": store},
    )

    if not result.success:
        raise RuntimeError("Production asset materialization failed")

    for event in result.get_asset_materialization_events():
        print(f"  Materialized: {event.asset_key}")

    # Verify branching IO: both SpeechToText and Crop written by video_processing
    config = mx.MetaxyConfig.get()
    verify_store = config.get_store()
    with verify_store:
        for feature, name, expected in [
            (Video, "video", 4),
            (SpeechToText, "stt", 4),
            (Crop, "crop", 4),
        ]:
            rows = verify_store.read(feature).collect().to_polars()
            if len(rows) != expected:
                print(
                    f"  FAIL {name}: expected {expected} rows, got {len(rows)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"  Verified {name}: {len(rows)} rows")

    print("Example 3: production assets materialized successfully")
