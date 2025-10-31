"""Tests for automatic field mapping functionality."""

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FeatureSpec,
    FieldKey,
    FieldSpec,
)
from metaxy.models.field import DefaultFieldsMapping, FieldDep, SpecialFieldDep


def test_default_fields_mapping_exact_match():
    """Test exact field name matching."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream feature with fields
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                    FieldSpec(key=FieldKey(["video"]), code_version=1),
                ],
            ),
        ):
            pass

        # Define downstream feature with auto-mapped fields
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["audio"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(),  # Should map to upstream audio
                    ),
                    FieldSpec(
                        key=FieldKey(["video"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(),  # Should map to upstream video
                    ),
                ],
            ),
        ):
            pass

        # Check that fields were properly mapped
        audio_field = DownstreamFeature.spec().fields_by_key[FieldKey(["audio"])]
        assert isinstance(audio_field.deps, list)
        assert len(audio_field.deps) == 1
        assert audio_field.deps[0].feature == UpstreamFeature.spec().key
        assert audio_field.deps[0].fields == [FieldKey(["audio"])]

        video_field = DownstreamFeature.spec().fields_by_key[FieldKey(["video"])]
        assert isinstance(video_field.deps, list)
        assert len(video_field.deps) == 1
        assert video_field.deps[0].feature == UpstreamFeature.spec().key
        assert video_field.deps[0].fields == [FieldKey(["video"])]


def test_default_fields_mapping_suffix_match():
    """Test suffix matching for hierarchical field keys."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream feature with hierarchical fields
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio", "french"]), code_version=1),
                    FieldSpec(key=FieldKey(["audio", "english"]), code_version=1),
                    FieldSpec(key=FieldKey(["video", "frames"]), code_version=1),
                ],
            ),
        ):
            pass

        # Define downstream feature with suffix matching
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["french"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(
                            match_suffix=True
                        ),  # Should match audio/french
                    ),
                    FieldSpec(
                        key=FieldKey(["frames"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(
                            match_suffix=True
                        ),  # Should match video/frames
                    ),
                ],
            ),
        ):
            pass

        # Check suffix matching worked
        french_field = DownstreamFeature.spec().fields_by_key[FieldKey(["french"])]
        assert isinstance(french_field.deps, list)
        assert len(french_field.deps) == 1
        assert french_field.deps[0].fields == [FieldKey(["audio", "french"])]

        frames_field = DownstreamFeature.spec().fields_by_key[FieldKey(["frames"])]
        assert isinstance(frames_field.deps, list)
        assert len(frames_field.deps) == 1
        assert frames_field.deps[0].fields == [FieldKey(["video", "frames"])]


def test_default_fields_mapping_no_match():
    """Test behavior when no matching fields are found."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream feature
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                ],
            ),
        ):
            pass

        # Define downstream feature with non-matching field
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["video"]),  # No match in upstream
                        code_version=1,
                        deps=DefaultFieldsMapping(),
                    ),
                ],
            ),
        ):
            pass

        # Should fall back to ALL when no matches found
        video_field = DownstreamFeature.spec().fields_by_key[FieldKey(["video"])]
        assert video_field.deps == SpecialFieldDep.ALL


def test_default_fields_mapping_multiple_upstreams():
    """Test auto-mapping with multiple upstream features."""
    graph = FeatureGraph()

    with graph.use():
        # Define first upstream feature
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                    FieldSpec(key=FieldKey(["metadata"]), code_version=1),
                ],
            ),
        ):
            pass

        # Define second upstream feature
        class Upstream2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["video"]), code_version=1),
                    FieldSpec(key=FieldKey(["timestamp"]), code_version=1),
                ],
            ),
        ):
            pass

        # Define downstream with deps on both
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(feature=Upstream1),
                    FeatureDep(feature=Upstream2),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["audio"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(),  # From Upstream1
                    ),
                    FieldSpec(
                        key=FieldKey(["video"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(),  # From Upstream2
                    ),
                ],
            ),
        ):
            pass

        # Check mappings
        audio_field = Downstream.spec().fields_by_key[FieldKey(["audio"])]
        assert isinstance(audio_field.deps, list)
        assert audio_field.deps[0].feature == Upstream1.spec().key

        video_field = Downstream.spec().fields_by_key[FieldKey(["video"])]
        assert isinstance(video_field.deps, list)
        assert video_field.deps[0].feature == Upstream2.spec().key


def test_default_fields_mapping_multiple_matches():
    """Test that fields can map to multiple upstream features naturally."""
    graph = FeatureGraph()

    with graph.use():
        # Define two upstream features with same field name
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                ],
            ),
        ):
            pass

        class Upstream2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),  # Same name!
                ],
            ),
        ):
            pass

        # This should map to both upstream features
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(feature=Upstream1),
                    FeatureDep(feature=Upstream2),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["audio"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(),  # Maps to both!
                    ),
                ],
            ),
        ):
            pass

        # Should map to both upstream features
        audio_field = Downstream.spec().fields_by_key[FieldKey(["audio"])]
        assert isinstance(audio_field.deps, list)
        assert len(audio_field.deps) == 2  # One for each upstream feature

        # Check that both upstream features are included
        feature_keys = {dep.feature for dep in audio_field.deps}
        assert Upstream1.spec().key in feature_keys
        assert Upstream2.spec().key in feature_keys

        # Check that each maps to the audio field
        for dep in audio_field.deps:
            assert dep.fields == [FieldKey(["audio"])]


def test_default_fields_mapping_exclude_features():
    """Test excluding specific features from auto-mapping."""
    graph = FeatureGraph()

    with graph.use():
        # Define two upstream features with same field
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                ],
            ),
        ):
            pass

        class Upstream2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=2),
                ],
            ),
        ):
            pass

        # Use exclude to avoid ambiguity
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(feature=Upstream1),
                    FeatureDep(feature=Upstream2),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["audio"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(
                            exclude_features=[Upstream1.spec().key]  # Exclude Upstream1
                        ),
                    ),
                ],
            ),
        ):
            pass

        # Should map to Upstream2 only
        audio_field = Downstream.spec().fields_by_key[FieldKey(["audio"])]
        assert isinstance(audio_field.deps, list)
        assert audio_field.deps[0].feature == Upstream2.spec().key


def test_default_fields_mapping_mixed_deps():
    """Test mixing auto-mapped and explicit field deps."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream features
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                    FieldSpec(key=FieldKey(["video"]), code_version=1),
                ],
            ),
        ):
            pass

        class Upstream2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["metadata"]), code_version=1),
                ],
            ),
        ):
            pass

        # Mix auto and explicit deps
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[
                    FeatureDep(feature=Upstream1),
                    FeatureDep(feature=Upstream2),
                ],
                fields=[
                    FieldSpec(
                        key=FieldKey(["audio"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(),  # Auto-mapped
                    ),
                    FieldSpec(
                        key=FieldKey(["custom"]),
                        code_version=1,
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

        # Check auto-mapped field
        audio_field = Downstream.spec().fields_by_key[FieldKey(["audio"])]
        assert isinstance(audio_field.deps, list)
        assert len(audio_field.deps) == 1
        assert audio_field.deps[0].feature == Upstream1.spec().key

        # Check explicit deps field
        custom_field = Downstream.spec().fields_by_key[FieldKey(["custom"])]
        assert isinstance(custom_field.deps, list)
        assert len(custom_field.deps) == 2


def test_default_fields_mapping_no_feature_deps_fallback():
    """Test that DefaultFieldsMapping falls back to ALL when no feature deps exist."""
    graph = FeatureGraph()

    with graph.use():
        # No longer raises an error - instead falls back to ALL for backward compatibility
        class RootFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "root"]),
                deps=None,  # No dependencies!
                fields=[
                    FieldSpec(
                        key=FieldKey(["audio"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(),  # Will fallback to ALL
                    ),
                ],
            ),
        ):
            pass

        # Should fallback to ALL when no feature deps exist
        audio_field = RootFeature.spec().fields_by_key[FieldKey(["audio"])]
        assert audio_field.deps == SpecialFieldDep.ALL


def test_backward_compatibility():
    """Test that existing code with explicit deps still works."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream
        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                    FieldSpec(key=FieldKey(["video"]), code_version=1),
                ],
            ),
        ):
            pass

        # Old-style explicit deps should still work
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=Upstream)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["processed"]),
                        code_version=1,
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
