# Feature Dependencies

[**Back to quickstart**](./quickstart.md)

---

Now let's add a downstream feature. We can use `deps` field on [`FeatureSpec`][metaxy.FeatureSpec] in order to do that.

```py {title="features.py, hl_lines="13"}
import metaxy as mx
from pydantic import Field


class CroppedVideo(
    Video,  # inheritance is a good way to automatically get matching DB columns
    spec=mx.FeatureSpec(
        key="cropped_video",
        id_columns=["video_id"],
        fields=[
            "audio",
            "frames",
        ],
        deps=[Video],
    ),
):
    # additional columns
    height: int = Field(description="Height of the video in pixels")
    width: int = Field(description="Width of the video in pixels")
```
