from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureSpec,
    FieldDep,
    FieldSpec,
)


class Video(
    BaseFeature,
    spec=FeatureSpec(
        key="example/video",
        fields=[
            FieldSpec(
                key="audio",
                code_version="2",
            ),
            FieldSpec(
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
    BaseFeature,
    spec=FeatureSpec(
        key="example/crop",
        deps=[FeatureDep(feature=Video)],
        fields=[
            FieldSpec(
                key="audio",
                code_version="1",
                deps=[
                    FieldDep(
                        feature=Video,
                        fields=["audio"],
                    )
                ],
            ),
            FieldSpec(
                key="frames",
                code_version="1",
                deps=[
                    FieldDep(
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
    BaseFeature,
    spec=FeatureSpec(
        key="example/face_detection",
        deps=[
            FeatureDep(
                feature=Crop,
            )
        ],
        fields=[
            FieldSpec(
                key="faces",
                code_version="1",
                deps=[
                    FieldDep(
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
    BaseFeature,
    spec=FeatureSpec(
        key="example/stt",
        deps=[
            FeatureDep(
                feature=Video,
            )
        ],
        fields=[
            FieldSpec(
                key="transcription",
                code_version="1",
                deps=[
                    FieldDep(
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
