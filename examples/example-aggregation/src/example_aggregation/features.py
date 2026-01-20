import metaxy as mx


class Audio(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="audio",
        id_columns=["audio_id"],
        fields=["default"],
    ),
):
    """Audio recordings of different speakers."""

    audio_id: str
    speaker_id: str
    duration_seconds: float
    path: str


class SpeakerEmbedding(
    mx.BaseFeature,
    spec=mx.FeatureSpec(
        key="speaker/embedding",
        id_columns=["speaker_id"],
        deps=[
            mx.FeatureDep(
                feature=Audio,
                lineage=mx.LineageRelationship.aggregation(on=["speaker_id"]),
            )
        ],
        fields=[
            mx.FieldSpec(key="embedding", code_version="1"),
        ],
    ),
):
    """Speaker embedding aggregated from all their audio recordings.

    This demonstrates N:1 aggregation lineage where multiple audio recordings
    from the same speaker are aggregated into a single speaker embedding.
    """

    speaker_id: str
    n_dim: int
    path: str
