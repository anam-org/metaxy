"""Dagster Definitions for the production Ray example (Example 3)."""

import dagster as dg
import metaxy as mx
import metaxy.ext.dagster as mxd

from example_ray.defs.assets import video_metadata
from example_ray.defs.assets.production import video_processing

# --8<-- [start:production_definitions]

store = mxd.MetaxyStoreFromConfigResource(name="dev")


@dg.definitions
def definitions():
    mx.init()

    return dg.Definitions(
        assets=[  # ty: ignore[invalid-argument-type]
            video_metadata,
            video_processing,
        ],
        resources={
            "store": store,
        },
    )


# --8<-- [end:production_definitions]
