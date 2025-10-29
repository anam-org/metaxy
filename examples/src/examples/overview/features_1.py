from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)


class Video(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["example", "video"]),
        deps=None,  # Root feature
        fields=[
            FieldSpec(
                key=FieldKey(["audio"]),
                code_version="1",
            ),
            FieldSpec(
                key=FieldKey(["frames"]),
                code_version="1",
            ),
        ],
    ),
):
    """Video metadata feature (root)."""

    frames: int
    duration: float
    size: int


class Crop(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["example", "crop"]),
        deps=[FeatureDep(key=Video.spec.key)],
        fields=[
            FieldSpec(
                key=FieldKey(["audio"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature_key=Video.spec.key,
                        fields=[FieldKey(["audio"])],
                    )
                ],
            ),
            FieldSpec(
                key=FieldKey(["frames"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature_key=Video.spec.key,
                        fields=[FieldKey(["frames"])],
                    )
                ],
            ),
        ],
    ),
):
    pass  # omit columns for the sake of simplicity


class FaceDetection(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["example", "face_detection"]),
        deps=[
            FeatureDep(
                key=Crop.spec.key,
            )
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["faces"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature_key=Crop.spec.key,
                        fields=[FieldKey(["frames"])],
                    )
                ],
            ),
        ],
    ),
):
    pass


class SpeechToText(
    Feature,
    spec=FeatureSpec(
        key=FeatureKey(["example", "stt"]),
        deps=[
            FeatureDep(
                key=Video.spec.key,
            )
        ],
        fields=[
            FieldSpec(
                key=FieldKey(["transcription"]),
                code_version="1",
                deps=[
                    FieldDep(
                        feature_key=Video.spec.key,
                        fields=[FieldKey(["audio"])],
                    )
                ],
            ),
        ],
    ),
):
    pass
