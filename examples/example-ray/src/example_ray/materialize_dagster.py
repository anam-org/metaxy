"""Materialize Example 2 Dagster assets programmatically.

Run with: python -m example_ray.materialize_dagster
"""

import sys

import dagster as dg
import metaxy as mx
import metaxy.ext.dagster as mxd

if __name__ == "__main__":
    mx.init()

    # Verify definitions module can be imported
    from example_ray.definitions import definitions  # noqa: F401
    from example_ray.defs.assets import (
        crop_metadata,
        face_detection,
        speech_to_text,
        video_metadata,
    )
    from example_ray.features import Crop, FaceDetection, SpeechToText, Video

    store = mxd.MetaxyStoreFromConfigResource(name="dev")

    print("Materializing Example 2 assets: video -> crop, stt -> face_detection")
    result = dg.materialize(
        assets=[  # ty: ignore[invalid-argument-type]
            video_metadata,
            crop_metadata,
            speech_to_text,
            face_detection,
        ],
        resources={"store": store},
    )

    if not result.success:
        raise RuntimeError("Asset materialization failed")

    for event in result.get_asset_materialization_events():
        print(f"  Materialized: {event.asset_key}")

    # Verify data was actually written to the store
    config = mx.MetaxyConfig.get()
    verify_store = config.get_store()
    with verify_store:
        for feature, name, expected in [
            (Video, "video", 4),
            (Crop, "crop", 4),
            (SpeechToText, "stt", 4),
            (FaceDetection, "face_detection", 4),
        ]:
            rows = verify_store.read(feature).collect().to_polars()
            if len(rows) != expected:
                print(
                    f"  FAIL {name}: expected {expected} rows, got {len(rows)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"  Verified {name}: {len(rows)} rows")

    print("Example 2: all assets materialized successfully")
