"""Shared fixtures for provenance tracking tests."""

from __future__ import annotations

import narwhals as nw
import polars as pl
import pytest

from metaxy.models.feature import FeatureGraph, TestingFeature
from metaxy.models.feature_spec import FeatureDep, SampleFeatureSpec
from metaxy.models.field import FieldDep, FieldSpec, SpecialFieldDep
from metaxy.models.types import FeatureKey, FieldKey
from metaxy.provenance.types import HashAlgorithm


@pytest.fixture
def upstream_video_metadata() -> nw.LazyFrame[pl.LazyFrame]:
    """Sample upstream video metadata with provenance."""
    return nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"frames": "video_frame_hash_1", "audio": "video_audio_hash_1"},
                    {"frames": "video_frame_hash_2", "audio": "video_audio_hash_2"},
                    {"frames": "video_frame_hash_3", "audio": "video_audio_hash_3"},
                ],
                "metaxy_provenance": ["video_prov_1", "video_prov_2", "video_prov_3"],
            }
        ).lazy()
    )


@pytest.fixture
def upstream_audio_metadata() -> nw.LazyFrame[pl.LazyFrame]:
    """Sample upstream audio metadata with provenance."""
    return nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                "metaxy_provenance_by_field": [
                    {"waveform": "audio_waveform_hash_1"},
                    {"waveform": "audio_waveform_hash_2"},
                    {"waveform": "audio_waveform_hash_3"},
                ],
                "metaxy_provenance": ["audio_prov_1", "audio_prov_2", "audio_prov_3"],
            }
        ).lazy()
    )


@pytest.fixture
def simple_features(graph: FeatureGraph) -> dict[str, type[TestingFeature]]:
    """Create simple test features for basic provenance testing.

    Structure:
    - VideoRoot: Root feature with frames and audio fields
    - ProcessedVideo: Single upstream (VideoRoot), single field (default)
    """

    class VideoRoot(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    class ProcessedVideo(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["processed"]),
            deps=[FeatureDep(feature=FeatureKey(["video"]))],
            fields=[
                FieldSpec(key=FieldKey(["default"]), code_version="1"),
            ],
        ),
    ):
        pass

    return {
        "VideoRoot": VideoRoot,
        "ProcessedVideo": ProcessedVideo,
    }


@pytest.fixture
def multi_upstream_features(graph: FeatureGraph) -> dict[str, type[TestingFeature]]:
    """Create features with multiple upstream dependencies.

    Structure:
    - VideoRoot: frames, audio fields
    - AudioRoot: waveform field
    - MultiUpstreamFeature: Depends on both VideoRoot and AudioRoot
    """

    class VideoRoot(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["video"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    class AudioRoot(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["audio"]),
            fields=[
                FieldSpec(key=FieldKey(["waveform"]), code_version="1"),
            ],
        ),
    ):
        pass

    class MultiUpstreamFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["multi"]),
            deps=[
                FeatureDep(feature=FeatureKey(["video"])),
                FeatureDep(feature=FeatureKey(["audio"])),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["fusion"]),
                    code_version="1",
                    deps=SpecialFieldDep.ALL,
                ),
                FieldSpec(
                    key=FieldKey(["analysis"]),
                    code_version="2",
                    deps=SpecialFieldDep.ALL,
                ),
            ],
        ),
    ):
        pass

    return {
        "VideoRoot": VideoRoot,
        "AudioRoot": AudioRoot,
        "MultiUpstreamFeature": MultiUpstreamFeature,
    }


@pytest.fixture
def selective_field_dep_features(
    graph: FeatureGraph,
) -> dict[str, type[TestingFeature]]:
    """Create features with selective field-level dependencies.

    Structure:
    - MultiFieldRoot: frames, audio, text fields
    - SelectiveFeature: Different fields depend on different subsets of upstream fields
    """

    class MultiFieldRoot(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["multi_field"]),
            fields=[
                FieldSpec(key=FieldKey(["frames"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio"]), code_version="2"),
                FieldSpec(key=FieldKey(["text"]), code_version="3"),
            ],
        ),
    ):
        pass

    class SelectiveFeature(
        TestingFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["selective"]),
            deps=[FeatureDep(feature=FeatureKey(["multi_field"]))],
            fields=[
                # Only depends on frames
                FieldSpec(
                    key=FieldKey(["visual"]),
                    code_version="10",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["multi_field"]),
                            fields=[FieldKey(["frames"])],
                        )
                    ],
                ),
                # Only depends on audio
                FieldSpec(
                    key=FieldKey(["audio_only"]),
                    code_version="11",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["multi_field"]),
                            fields=[FieldKey(["audio"])],
                        )
                    ],
                ),
                # Depends on frames and text
                FieldSpec(
                    key=FieldKey(["mixed"]),
                    code_version="12",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["multi_field"]),
                            fields=[
                                FieldKey(["frames"]),
                                FieldKey(["text"]),
                            ],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    return {
        "MultiFieldRoot": MultiFieldRoot,
        "SelectiveFeature": SelectiveFeature,
    }


@pytest.fixture
def upstream_metadata_multi_field() -> nw.LazyFrame[pl.LazyFrame]:
    """Multi-field upstream metadata for selective dependency testing."""
    return nw.from_native(
        pl.DataFrame(
            {
                "sample_uid": [1, 2],
                "metaxy_provenance_by_field": [
                    {
                        "frames": "frame_hash_1",
                        "audio": "audio_hash_1",
                        "text": "text_hash_1",
                    },
                    {
                        "frames": "frame_hash_2",
                        "audio": "audio_hash_2",
                        "text": "text_hash_2",
                    },
                ],
                "metaxy_provenance": ["multi_prov_1", "multi_prov_2"],
            }
        ).lazy()
    )


@pytest.fixture(
    params=[
        HashAlgorithm.XXHASH64,
        HashAlgorithm.XXHASH32,
        HashAlgorithm.WYHASH,
        HashAlgorithm.SHA256,
        HashAlgorithm.MD5,
    ],
    ids=["xxhash64", "xxhash32", "wyhash", "sha256", "md5"],
)
def hash_algorithm(request: pytest.FixtureRequest) -> HashAlgorithm:
    """Parametrize tests across all supported hash algorithms."""
    return request.param  # type: ignore[no-any-return]
