import metaxy as mx


class Video(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="example/video",
        fields=[
            mx.FieldSpec(
                key="audio",
                code_version="1",
            ),
            mx.FieldSpec(
                key="frames",
                code_version="1",
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    """Video metadata feature (root)."""

    frames: int
    duration: float
    size: int


class Crop(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="example/crop",
        deps=[mx.FeatureDep(feature=Video)],
        fields=[
            mx.FieldSpec(
                key="audio",
                code_version="1",
                deps=[
                    mx.FieldDep(
                        feature=Video,
                        fields=["audio"],
                    )
                ],
            ),
            mx.FieldSpec(
                key="frames",
                code_version="1",
                deps=[
                    mx.FieldDep(
                        feature=Video,
                        fields=["frames"],
                    )
                ],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    pass  # omit columns for the sake of simplicity


class FaceDetection(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="example/face_detection",
        deps=[
            mx.FeatureDep(
                feature=Crop,
            )
        ],
        fields=[
            mx.FieldSpec(
                key="faces",
                code_version="1",
                deps=[
                    mx.FieldDep(
                        feature=Crop,
                        fields=["frames"],
                    )
                ],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    pass


class SpeechToText(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="example/stt",
        deps=[
            mx.FeatureDep(
                feature=Video,
            )
        ],
        fields=[
            mx.FieldSpec(
                key="transcription",
                code_version="1",
                deps=[
                    mx.FieldDep(
                        feature=Video,
                        fields=["audio"],
                    )
                ],
            ),
        ],
        id_columns=("sample_uid",),
    ),
):
    pass
