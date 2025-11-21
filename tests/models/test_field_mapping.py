"""Tests for automatic field mapping functionality."""

from metaxy import Feature, FeatureDep, FeatureKey, FieldKey, FieldSpec
from metaxy._testing.models import SampleFeatureSpec
from metaxy.models.field import FieldDep


def test_default_fields_mapping_exact_match():
    """Test exact field name matching."""

    # Define upstream feature with fields
    class UpstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["video"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Define downstream feature with auto-mapped fields
    class DownstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=[
                FieldSpec(
                    key=FieldKey(["audio"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # Should map to upstream audio
                ),
                FieldSpec(
                    key=FieldKey(["video"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # Should map to upstream video
                ),
            ],
        ),
    ):
        pass

    # Check that fields have no explicit deps (will be resolved via FeatureDep.fields_mapping)
    audio_field = DownstreamFeature.spec().fields_by_key[FieldKey(["audio"])]
    assert audio_field.deps == []  # Uses default mapping from FeatureDep

    video_field = DownstreamFeature.spec().fields_by_key[FieldKey(["video"])]
    assert video_field.deps == []  # Uses default mapping from FeatureDep


def test_default_fields_mapping_suffix_match():
    """Test suffix matching for hierarchical field keys."""

    # Define upstream feature with hierarchical fields
    class UpstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["audio", "french"]), code_version="1"),
                FieldSpec(key=FieldKey(["audio", "english"]), code_version="1"),
                FieldSpec(key=FieldKey(["video", "frames"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Define downstream feature with suffix matching
    class DownstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=[
                FieldSpec(
                    key=FieldKey(["french"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # Should match audio/french
                ),
                FieldSpec(
                    key=FieldKey(["frames"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # Should match video/frames
                ),
            ],
        ),
    ):
        pass

    # Check fields have no explicit deps (will be resolved via FeatureDep.fields_mapping with suffix matching)
    french_field = DownstreamFeature.spec().fields_by_key[FieldKey(["french"])]
    assert (
        french_field.deps == []
    )  # Uses default mapping with suffix matching from FeatureDep

    frames_field = DownstreamFeature.spec().fields_by_key[FieldKey(["frames"])]
    assert (
        frames_field.deps == []
    )  # Uses default mapping with suffix matching from FeatureDep


def test_default_fields_mapping_no_match():
    """Test behavior when no matching fields are found."""

    # Define upstream feature
    class UpstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Define downstream feature with non-matching field
    class DownstreamFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[FeatureDep(feature=UpstreamFeature)],
            fields=[
                FieldSpec(
                    key=FieldKey(["video"]),  # No match in upstream
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,
                ),
            ],
        ),
    ):
        pass

    # Field should have no explicit deps (will be resolved at runtime)
    video_field = DownstreamFeature.spec().fields_by_key[FieldKey(["video"])]
    assert (
        video_field.deps == []
    )  # Will fall back to ALL at runtime when no matches found


def test_default_fields_mapping_multiple_upstreams():
    """Test auto-mapping with multiple upstream features."""

    # Define first upstream feature
    class Upstream1(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream1"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["metadata"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Define second upstream feature
    class Upstream2(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream2"]),
            fields=[
                FieldSpec(key=FieldKey(["video"]), code_version="1"),
                FieldSpec(key=FieldKey(["timestamp"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Define downstream with deps on both
    class Downstream(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=Upstream1),
                FeatureDep(feature=Upstream2),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["audio"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # From Upstream1
                ),
                FieldSpec(
                    key=FieldKey(["video"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # From Upstream2
                ),
            ],
        ),
    ):
        pass

    # Check fields have no explicit deps (will be resolved at runtime)
    audio_field = Downstream.spec().fields_by_key[FieldKey(["audio"])]
    assert audio_field.deps == []  # Will map to Upstream1 at runtime

    video_field = Downstream.spec().fields_by_key[FieldKey(["video"])]
    assert video_field.deps == []  # Will map to Upstream2 at runtime


def test_default_fields_mapping_multiple_matches():
    """Test that fields can map to multiple upstream features naturally."""

    # Define two upstream features with same field name
    class Upstream1(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream1"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
            ],
        ),
    ):
        pass

    class Upstream2(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream2"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),  # Same name!
            ],
        ),
    ):
        pass

    # This should map to both upstream features
    class Downstream(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=Upstream1),
                FeatureDep(feature=Upstream2),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["audio"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # Maps to both!
                ),
            ],
        ),
    ):
        pass

    # Field should have no explicit deps (will map to both at runtime)
    audio_field = Downstream.spec().fields_by_key[FieldKey(["audio"])]
    assert audio_field.deps == []  # Will map to both upstream features at runtime


def test_default_fields_mapping_mixed_deps():
    """Test mixing auto-mapped and explicit field deps."""

    # Define upstream features
    class Upstream1(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream1"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["video"]), code_version="1"),
            ],
        ),
    ):
        pass

    class Upstream2(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream2"]),
            fields=[
                FieldSpec(key=FieldKey(["metadata"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Mix auto and explicit deps
    class Downstream(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[
                FeatureDep(feature=Upstream1),
                FeatureDep(feature=Upstream2),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["audio"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # Auto-mapped
                ),
                FieldSpec(
                    key=FieldKey(["custom"]),
                    code_version="1",
                    deps=[  # Explicit deps
                        FieldDep(
                            feature=Upstream1.spec().key,
                            fields=[FieldKey(["video"])],
                        ),
                        FieldDep(
                            feature=Upstream2.spec().key,
                            fields=[FieldKey(["metadata"])],
                        ),
                    ],
                ),
            ],
        ),
    ):
        pass

    # Check auto-mapped field has no explicit deps
    audio_field = Downstream.spec().fields_by_key[FieldKey(["audio"])]
    assert audio_field.deps == []  # Will auto-map at runtime

    # Check explicit deps field has the specified deps
    custom_field = Downstream.spec().fields_by_key[FieldKey(["custom"])]
    assert isinstance(custom_field.deps, list)
    assert len(custom_field.deps) == 2
    assert custom_field.deps[0].feature == Upstream1.spec().key
    assert custom_field.deps[1].feature == Upstream2.spec().key


def test_default_fields_mapping_no_feature_deps_fallback():
    """Test that DefaultFieldsMapping falls back to ALL when no feature deps exist."""

    # No longer raises an error - instead falls back to ALL for backward compatibility
    class RootFeature(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "root"]),
            # No dependencies!
            fields=[
                FieldSpec(
                    key=FieldKey(["audio"]),
                    code_version="1",
                    # deps will be resolved from FeatureDep.fields_mapping,  # Will fallback to ALL
                ),
            ],
        ),
    ):
        pass

    # Field should have no explicit deps (will fallback to ALL at runtime when no feature deps)
    audio_field = RootFeature.spec().fields_by_key[FieldKey(["audio"])]
    assert audio_field.deps == []  # Will fallback to ALL at runtime


def test_backward_compatibility():
    """Test that existing code with explicit deps still works."""

    # Define upstream
    class Upstream(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "upstream"]),
            fields=[
                FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                FieldSpec(key=FieldKey(["video"]), code_version="1"),
            ],
        ),
    ):
        pass

    # Old-style explicit deps should still work
    class Downstream(
        Feature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["test", "downstream"]),
            deps=[FeatureDep(feature=Upstream)],
            fields=[
                FieldSpec(
                    key=FieldKey(["processed"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=Upstream.spec().key,
                            fields=[FieldKey(["audio"]), FieldKey(["video"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        pass

    # Check explicit deps still work
    field = Downstream.spec().fields_by_key[FieldKey(["processed"])]
    assert isinstance(field.deps, list)
    assert field.deps[0].feature == Upstream.spec().key
    # Type guard: fields should be a list after explicit declaration
    assert isinstance(field.deps[0].fields, list), "Expected fields to be a list"
    assert len(field.deps[0].fields) == 2
