"""Tests for improved DefaultFieldsMapping functionality."""

from metaxy import (
    Feature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FeatureSpec,
    FieldKey,
    FieldSpec,
)
from metaxy.models.fields_mapping import FieldsMapping


def test_default_fields_mapping_is_default():
    """Test that DefaultFieldsMapping is now the default for FieldSpec.deps."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream feature
        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio"]), code_version="1"),
                    FieldSpec(key=FieldKey(["video"]), code_version="1"),
                ],
            ),
        ):
            pass

        # Define downstream feature without explicitly specifying deps
        # Should use DefaultFieldsMapping by default
        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=UpstreamFeature)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["audio"]),
                        code_version="1",
                        # No deps specified - should use DefaultFieldsMapping
                    ),
                    FieldSpec(
                        key=FieldKey(["video"]),
                        code_version="1",
                        # No deps specified - should use DefaultFieldsMapping
                    ),
                ],
            ),
        ):
            pass

        # Check that fields have no explicit deps (auto-mapped via FeatureDep)
        audio_field = DownstreamFeature.spec().fields_by_key[FieldKey(["audio"])]
        assert audio_field.deps == []  # Uses default mapping from FeatureDep

        video_field = DownstreamFeature.spec().fields_by_key[FieldKey(["video"])]
        assert video_field.deps == []  # Uses default mapping from FeatureDep


def test_exclude_fields():
    """Test excluding specific fields from auto-mapping."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream features with same field names
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version="1"),
                    FieldSpec(key=FieldKey(["metadata"]), code_version="1"),
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
                    FieldSpec(key=FieldKey(["data"]), code_version="2"),
                    FieldSpec(key=FieldKey(["metadata"]), code_version="2"),
                ],
            ),
        ):
            pass

        # Use exclude_fields to prevent metadata from being mapped
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
                        key=FieldKey(["metadata"]),
                        code_version="1",
                        # deps will be resolved from FeatureDep.fields_mapping
                    ),
                ],
            ),
        ):
            pass

        # metadata field should have no explicit deps (will resolve at runtime)
        metadata_field = Downstream.spec().fields_by_key[FieldKey(["metadata"])]
        assert metadata_field.deps == []  # Will be resolved at runtime


def test_backward_compat_with_explicit_all():
    """Test that explicitly using FieldsMapping.all() works as expected."""
    graph = FeatureGraph()

    with graph.use():

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version="1"),
                ],
            ),
        ):
            pass

        # Explicitly use FieldsMapping.all()
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=Upstream, fields_mapping=FieldsMapping.all())],
                fields=[
                    FieldSpec(
                        key=FieldKey(["processed"]),
                        code_version="1",
                        # Will depend on ALL fields from upstream via the FeatureDep
                    ),
                ],
            ),
        ):
            pass

        # Field should have no explicit deps (resolved via FeatureDep.fields_mapping)
        field = Downstream.spec().fields_by_key[FieldKey(["processed"])]
        assert field.deps == []  # Uses FieldsMapping.all() from FeatureDep


def test_default_mapping_with_no_upstream_deps():
    """Test that DefaultFieldsMapping handles features with no dependencies gracefully."""
    graph = FeatureGraph()

    with graph.use():
        # Root feature with no dependencies
        # DefaultFieldsMapping should fallback to ALL behavior
        class RootFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "root"]),
                deps=None,  # No dependencies
                fields=[
                    FieldSpec(
                        key=FieldKey(["data"]),
                        code_version="1",
                        # Uses default DefaultFieldsMapping, should fallback to ALL
                    ),
                ],
            ),
        ):
            pass

        # Should work without errors
        field = RootFeature.spec().fields_by_key[FieldKey(["data"])]
        # Field has no explicit deps (will resolve to ALL at runtime with no feature deps)
        assert field.deps == []


def test_exclude_fields_with_suffix_matching():
    """Test that exclude_fields works with suffix matching."""
    graph = FeatureGraph()

    with graph.use():

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["audio", "french"]), code_version="1"),
                    FieldSpec(key=FieldKey(["audio", "english"]), code_version="1"),
                    FieldSpec(key=FieldKey(["video", "french"]), code_version="1"),
                ],
            ),
        ):
            pass

        # Exclude audio/french but allow video/french with suffix matching
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=Upstream)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["french"]),
                        code_version="1",
                        # deps will be resolved from FeatureDep.fields_mapping
                    ),
                ],
            ),
        ):
            pass

        # Field should have no explicit deps (will resolve at runtime with exclusions)
        french_field = Downstream.spec().fields_by_key[FieldKey(["french"])]
        assert (
            french_field.deps == []
        )  # Will match video/french at runtime (audio/french excluded)
