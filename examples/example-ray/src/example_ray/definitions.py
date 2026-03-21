"""Dagster Definitions for the Ray example."""

import dagster as dg
import metaxy as mx
import metaxy.ext.dagster as mxd

from example_ray.defs.assets import (
    crop_metadata,
    face_detection,
    speech_to_text,
    video_metadata,
)

# --8<-- [start:definitions]

store = mxd.MetaxyStoreFromConfigResource(name="dev")


@dg.definitions
def definitions():
    mx.init()

    return dg.Definitions(
        assets=[  # ty: ignore[invalid-argument-type]
            video_metadata,
            crop_metadata,
            speech_to_text,
            face_detection,
        ],
        resources={
            "store": store,
        },
    )


# --8<-- [end:definitions]
