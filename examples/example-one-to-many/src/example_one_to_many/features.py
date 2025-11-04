from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Literal,
    TypeAlias,
    overload,
)

from pydantic.types import JsonValue

import metaxy as mx
from metaxy.models.feature_spec import CoercibleToFieldSpec, FeatureDep
from metaxy.models.types import (
    CoercibleToFeatureKey,
)

VideoIDColumns: TypeAlias = tuple[Literal["video_id"]]
VideoChunkIDColumns: TypeAlias = tuple[Literal["video_chunk_id"]]


class VideoFeatureSpec(mx.BaseFeatureSpec[VideoIDColumns]):
    id_columns: VideoIDColumns = ("video_id",)

    @overload
    def __init__(  # pyright: ignore[reportNoOverloadImplementation,reportInconsistentOverload]
        self,
        key: CoercibleToFeatureKey,
        *,
        deps: list[FeatureDep] | None = None,
        fields: Sequence[CoercibleToFieldSpec] | None = None,
        id_columns: VideoIDColumns | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        **kwargs: Any,
    ) -> None: ...


class Video(
    mx.BaseFeature[VideoIDColumns],
    spec=VideoFeatureSpec(key="video/raw", fields=["audio", "frames"]),
):
    pass


class VideoChunkFeatureSpec(mx.BaseFeatureSpec[VideoChunkIDColumns]):
    id_columns: VideoChunkIDColumns = ("video_chunk_id",)

    @overload
    def __init__(  # pyright: ignore[reportNoOverloadImplementation,reportInconsistentOverload]
        self,
        key: CoercibleToFeatureKey,
        *,
        deps: list[FeatureDep] | None = None,
        fields: Sequence[CoercibleToFieldSpec] | None = None,
        id_columns: VideoChunkIDColumns | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        **kwargs: Any,
    ) -> None: ...


class VideoChunk(
    mx.BaseFeature[VideoChunkIDColumns],
    spec=VideoChunkFeatureSpec(
        key="video/chunk",
        fields=["audio", "frames"],
        deps=[
            mx.FeatureDep(
                feature=Video, id_columns_mapping={"video_id": "video_chunk_id"}
            )
        ],
    ),
):
    pass
