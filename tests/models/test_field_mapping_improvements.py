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
from metaxy.models.field import DefaultFieldsMapping, SpecialFieldDep


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
                    FieldSpec(key=FieldKey(["audio"]), code_version=1),
                    FieldSpec(key=FieldKey(["video"]), code_version=1),
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
                        code_version=1,
                        # No deps specified - should use DefaultFieldsMapping
                    ),
                    FieldSpec(
                        key=FieldKey(["video"]),
                        code_version=1,
                        # No deps specified - should use DefaultFieldsMapping
                    ),
                ],
            ),
        ):
            pass

        # Check that fields were auto-mapped by default
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
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                    FieldSpec(key=FieldKey(["metadata"]), code_version=1),
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
                    FieldSpec(key=FieldKey(["data"]), code_version=2),
                    FieldSpec(key=FieldKey(["metadata"]), code_version=2),
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
                        code_version=1,
                        deps=DefaultFieldsMapping(
                            exclude_fields=[FieldKey(["metadata"])]
                        ),
                    ),
                ],
            ),
        ):
            pass

        # metadata field should fallback to ALL since it was excluded
        metadata_field = Downstream.spec().fields_by_key[FieldKey(["metadata"])]
        assert metadata_field.deps == SpecialFieldDep.ALL


def test_mapping_hook():
    """Test custom mapping logic via mapping_hook."""
    graph = FeatureGraph()

    with graph.use():
        # Define upstream features
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["important", "upstream1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                ],
            ),
        ):
            pass

        class Upstream2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["regular", "upstream2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=2),
                ],
            ),
        ):
            pass

        # Define custom hook that only keeps matches from "important" features
        def important_only_hook(field_key, matches):
            return [(fk, fld) for fk, fld in matches if "important" in fk.to_string()]

        # Use the custom hook
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
                        key=FieldKey(["data"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(mapping_hook=important_only_hook),
                    ),
                ],
            ),
        ):
            pass

        # Should only map to Upstream1 due to the hook
        data_field = Downstream.spec().fields_by_key[FieldKey(["data"])]
        assert isinstance(data_field.deps, list)
        assert len(data_field.deps) == 1
        assert data_field.deps[0].feature == Upstream1.spec().key


def test_combined_exclude_and_hook():
    """Test using exclude_fields and mapping_hook together."""
    graph = FeatureGraph()

    with graph.use():
        # Define multiple upstream features
        class Upstream1(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["priority", "upstream1"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                    FieldSpec(key=FieldKey(["extra"]), code_version=1),
                ],
            ),
        ):
            pass

        class Upstream2(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["priority", "upstream2"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=2),
                    FieldSpec(key=FieldKey(["extra"]), code_version=2),
                ],
            ),
        ):
            pass

        # Hook that selects only the first upstream
        def first_only_hook(field_key, matches):
            if not matches:
                return matches
            # Group by feature key and take only the first feature
            features_seen = set()
            filtered = []
            for fk, fld in matches:
                if fk not in features_seen:
                    features_seen.add(fk)
                    filtered.append((fk, fld))
                    break  # Only take first feature
            return filtered

        # Combine exclude_fields with mapping_hook
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
                        key=FieldKey(["data"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(
                            exclude_fields=[FieldKey(["extra"])],  # Exclude extra field
                            mapping_hook=first_only_hook,  # Select first match only
                        ),
                    ),
                    FieldSpec(
                        key=FieldKey(["extra"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(
                            exclude_fields=[
                                FieldKey(["extra"])
                            ]  # This will be excluded
                        ),
                    ),
                ],
            ),
        ):
            pass

        # data field should map to Upstream1 only (due to hook)
        data_field = Downstream.spec().fields_by_key[FieldKey(["data"])]
        assert isinstance(data_field.deps, list)
        assert len(data_field.deps) == 1
        assert data_field.deps[0].feature == Upstream1.spec().key

        # extra field should fallback to ALL (excluded)
        extra_field = Downstream.spec().fields_by_key[FieldKey(["extra"])]
        assert extra_field.deps == SpecialFieldDep.ALL


def test_backward_compat_with_explicit_all():
    """Test that explicitly specifying SpecialFieldDep.ALL still works."""
    graph = FeatureGraph()

    with graph.use():

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                ],
            ),
        ):
            pass

        # Explicitly use SpecialFieldDep.ALL
        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=Upstream)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["processed"]),
                        code_version=1,
                        deps=SpecialFieldDep.ALL,  # Explicit ALL
                    ),
                ],
            ),
        ):
            pass

        # Should still have ALL deps
        field = Downstream.spec().fields_by_key[FieldKey(["processed"])]
        assert field.deps == SpecialFieldDep.ALL


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
                        code_version=1,
                        # Uses default DefaultFieldsMapping, should fallback to ALL
                    ),
                ],
            ),
        ):
            pass

        # Should work without errors
        field = RootFeature.spec().fields_by_key[FieldKey(["data"])]
        # When resolved with no feature deps, should return ALL
        assert field.deps == SpecialFieldDep.ALL


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
                    FieldSpec(key=FieldKey(["audio", "french"]), code_version=1),
                    FieldSpec(key=FieldKey(["audio", "english"]), code_version=1),
                    FieldSpec(key=FieldKey(["video", "french"]), code_version=1),
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
                        code_version=1,
                        deps=DefaultFieldsMapping(
                            match_suffix=True,
                            exclude_fields=[FieldKey(["audio", "french"])],
                        ),
                    ),
                ],
            ),
        ):
            pass

        # Should only match video/french since audio/french is excluded
        french_field = Downstream.spec().fields_by_key[FieldKey(["french"])]
        assert isinstance(french_field.deps, list)
        assert len(french_field.deps) == 1
        assert french_field.deps[0].fields == [FieldKey(["video", "french"])]


def test_mapping_hook_returns_empty():
    """Test that mapping_hook returning empty list causes fallback to ALL."""
    graph = FeatureGraph()

    with graph.use():

        class Upstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["data"]), code_version=1),
                ],
            ),
        ):
            pass

        # Hook that filters out everything
        def filter_all_hook(field_key, matches):
            return []  # Filter out all matches

        class Downstream(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["test", "downstream"]),
                deps=[FeatureDep(feature=Upstream)],
                fields=[
                    FieldSpec(
                        key=FieldKey(["data"]),
                        code_version=1,
                        deps=DefaultFieldsMapping(mapping_hook=filter_all_hook),
                    ),
                ],
            ),
        ):
            pass

        # Should fallback to ALL when hook returns empty list
        data_field = Downstream.spec().fields_by_key[FieldKey(["data"])]
        assert data_field.deps == SpecialFieldDep.ALL
