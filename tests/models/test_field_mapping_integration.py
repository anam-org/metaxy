"""Test field mapping integration with FeatureDep."""

from metaxy.models.feature import Feature, FeatureGraph
from metaxy.models.feature_spec import FeatureDep, FeatureSpec
from metaxy.models.field import FieldDep, FieldKey, FieldSpec
from metaxy.models.fields_mapping import FieldsMapping
from metaxy.models.types import FeatureKey


def test_field_mapping_on_feature_dep():
    """Test that field mapping on FeatureDep resolves correctly."""
    test_graph = FeatureGraph()

    with test_graph.use():
        # Create upstream features with various fields
        class UpstreamA(
            Feature,
            spec=FeatureSpec(
                key="upstream_a",
                fields=[
                    FieldSpec(key="audio", code_version="1"),
                    FieldSpec(key="video", code_version="1"),
                    FieldSpec(key="metadata", code_version="1"),
                ],
            ),
        ):
            pass

        class UpstreamB(
            Feature,
            spec=FeatureSpec(
                key="upstream_b",
                fields=[
                    FieldSpec(key="audio/french", code_version="1"),
                    FieldSpec(key="text", code_version="1"),
                ],
            ),
        ):
            pass

        # Test 1: Default field mapping (exact match)
        class DownstreamDefault(
            Feature,
            spec=FeatureSpec(
                key="downstream_default",
                deps=[
                    FeatureDep(
                        feature="upstream_a"
                    ),  # Uses DefaultFieldsMapping() by default
                    FeatureDep(feature="upstream_b"),
                ],
                fields=[
                    FieldSpec(key="audio"),  # Should match upstream_a.audio
                    FieldSpec(key="text"),  # Should match upstream_b.text
                    FieldSpec(key="other"),  # No match, should get ALL
                ],
            ),
        ):
            pass

        # Get the feature plan
        plan_default = test_graph.get_feature_plan(FeatureKey(["downstream_default"]))

        # Check field dependencies
        field_deps = plan_default.field_dependencies

        # Debug output
        print(f"field_deps keys: {list(field_deps.keys())}")
        print(f"field_deps: {field_deps}")

        # "audio" field should map to upstream_a
        assert FieldKey(["audio"]) in field_deps
        audio_deps = field_deps[FieldKey(["audio"])]
        assert FeatureKey(["upstream_a"]) in audio_deps
        assert FieldKey(["audio"]) in audio_deps[FeatureKey(["upstream_a"])]

        # "text" field should map to upstream_b
        assert FieldKey(["text"]) in field_deps
        text_deps = field_deps[FieldKey(["text"])]
        assert FeatureKey(["upstream_b"]) in text_deps
        assert FieldKey(["text"]) in text_deps[FeatureKey(["upstream_b"])]

        # "other" field has no match, should get ALL
        assert FieldKey(["other"]) in field_deps
        other_deps = field_deps[FieldKey(["other"])]
        # Should have all fields from all upstreams
        assert FeatureKey(["upstream_a"]) in other_deps
        assert len(other_deps[FeatureKey(["upstream_a"])]) == 3  # All 3 fields
        assert FeatureKey(["upstream_b"]) in other_deps
        assert len(other_deps[FeatureKey(["upstream_b"])]) == 2  # All 2 fields


def test_field_mapping_with_suffix_matching():
    """Test field mapping with suffix matching enabled."""
    test_graph = FeatureGraph()

    with test_graph.use():

        class UpstreamAudio(
            Feature,
            spec=FeatureSpec(
                key="upstream_audio",
                fields=[
                    FieldSpec(key="audio/english", code_version="1"),
                    FieldSpec(key="audio/french", code_version="1"),
                    FieldSpec(key="metadata", code_version="1"),
                ],
            ),
        ):
            pass

        # Use suffix matching to match "french" to "audio/french"
        class DownstreamSuffix(
            Feature,
            spec=FeatureSpec(
                key="downstream_suffix",
                deps=[
                    FeatureDep(
                        feature="upstream_audio",
                        fields_mapping=FieldsMapping.default(match_suffix=True),
                    ),
                ],
                fields=[
                    FieldSpec(
                        key="french"
                    ),  # Should match audio/french with suffix matching
                    FieldSpec(
                        key="english"
                    ),  # Should match audio/english with suffix matching
                ],
            ),
        ):
            pass

        plan = test_graph.get_feature_plan(FeatureKey(["downstream_suffix"]))
        field_deps = plan.field_dependencies

        # "french" should map to audio/french
        assert FieldKey(["french"]) in field_deps
        french_deps = field_deps[FieldKey(["french"])]
        assert FeatureKey(["upstream_audio"]) in french_deps
        assert (
            FieldKey(["audio", "french"]) in french_deps[FeatureKey(["upstream_audio"])]
        )

        # "english" should map to audio/english
        assert FieldKey(["english"]) in field_deps
        english_deps = field_deps[FieldKey(["english"])]
        assert FeatureKey(["upstream_audio"]) in english_deps
        assert (
            FieldKey(["audio", "english"])
            in english_deps[FeatureKey(["upstream_audio"])]
        )


def test_field_mapping_with_explicit_all():
    """Test field mapping with explicit ALL mapping."""
    test_graph = FeatureGraph()

    with test_graph.use():

        class UpstreamData(
            Feature,
            spec=FeatureSpec(
                key="upstream_data",
                fields=[
                    FieldSpec(key="field1", code_version="1"),
                    FieldSpec(key="field2", code_version="1"),
                ],
            ),
        ):
            pass

        # Use FieldsMapping.all() to explicitly depend on all upstream fields
        class DownstreamAll(
            Feature,
            spec=FeatureSpec(
                key="downstream_all",
                deps=[
                    FeatureDep(
                        feature="upstream_data", fields_mapping=FieldsMapping.all()
                    ),
                ],
                fields=[
                    FieldSpec(key="combined"),  # Will depend on ALL upstream fields
                ],
            ),
        ):
            pass

        plan = test_graph.get_feature_plan(FeatureKey(["downstream_all"]))
        field_deps = plan.field_dependencies

        # "combined" should depend on all fields from upstream
        assert FieldKey(["combined"]) in field_deps
        combined_deps = field_deps[FieldKey(["combined"])]
        assert FeatureKey(["upstream_data"]) in combined_deps
        assert len(combined_deps[FeatureKey(["upstream_data"])]) == 2  # All fields


def test_explicit_deps_override_field_mapping():
    """Test that explicit field deps are preserved alongside automatic mapping."""
    test_graph = FeatureGraph()

    with test_graph.use():

        class UpstreamX(
            Feature,
            spec=FeatureSpec(
                key="upstream_x",
                fields=[
                    FieldSpec(key="data", code_version="1"),
                    FieldSpec(key="metadata", code_version="1"),
                ],
            ),
        ):
            pass

        class UpstreamY(
            Feature,
            spec=FeatureSpec(
                key="upstream_y",
                fields=[
                    FieldSpec(key="data", code_version="1"),
                    FieldSpec(key="other", code_version="1"),
                ],
            ),
        ):
            pass

        class DownstreamExplicit(
            Feature,
            spec=FeatureSpec(
                key="downstream_explicit",
                deps=[
                    FeatureDep(feature="upstream_x"),  # Default mapping
                    FeatureDep(feature="upstream_y"),  # Default mapping
                ],
                fields=[
                    # Explicit deps - should be used directly
                    FieldSpec(
                        key="data",
                        deps=[
                            FieldDep(
                                feature="upstream_x", fields=[FieldKey(["metadata"])]
                            ),
                            # Explicitly depend on metadata from X, not data
                        ],
                    ),
                ],
            ),
        ):
            pass

        plan = test_graph.get_feature_plan(FeatureKey(["downstream_explicit"]))
        field_deps = plan.field_dependencies

        # "data" field should use explicit deps, not automatic mapping
        assert FieldKey(["data"]) in field_deps
        data_deps = field_deps[FieldKey(["data"])]
        assert FeatureKey(["upstream_x"]) in data_deps
        # Should only have metadata field, not data field
        assert data_deps[FeatureKey(["upstream_x"])] == [FieldKey(["metadata"])]
        # Should not have upstream_y since not in explicit deps
        assert FeatureKey(["upstream_y"]) not in data_deps


def test_field_mapping_with_exclusions():
    """Test field mapping with exclusions."""
    test_graph = FeatureGraph()

    with test_graph.use():

        class UpstreamMulti(
            Feature,
            spec=FeatureSpec(
                key="upstream_multi",
                fields=[
                    FieldSpec(key="data", code_version="1"),
                    FieldSpec(key="metadata", code_version="1"),
                    FieldSpec(key="debug", code_version="1"),
                ],
            ),
        ):
            pass

        class UpstreamExcluded(
            Feature,
            spec=FeatureSpec(
                key="upstream_excluded",
                fields=[
                    FieldSpec(key="data", code_version="1"),
                ],
            ),
        ):
            pass

        # Exclude certain fields and features from auto-mapping
        class DownstreamExclusions(
            Feature,
            spec=FeatureSpec(
                key="downstream_exclusions",
                deps=[
                    FeatureDep(
                        feature="upstream_multi",
                        fields_mapping=FieldsMapping.default(
                            exclude_fields=[
                                FieldKey(["debug"])
                            ],  # Don't auto-map debug field
                        ),
                    ),
                    FeatureDep(
                        feature="upstream_excluded",
                        fields_mapping=FieldsMapping.default(
                            exclude_features=[
                                FeatureKey(["upstream_excluded"])
                            ],  # Exclude this entire feature
                        ),
                    ),
                ],
                fields=[
                    FieldSpec(key="data"),  # Should only match upstream_multi.data
                    FieldSpec(key="debug"),  # Won't match due to exclusion
                ],
            ),
        ):
            pass

        plan = test_graph.get_feature_plan(FeatureKey(["downstream_exclusions"]))
        field_deps = plan.field_dependencies

        # "data" should only map to upstream_multi (upstream_excluded is excluded)
        data_deps = field_deps[FieldKey(["data"])]
        assert FeatureKey(["upstream_multi"]) in data_deps
        assert FieldKey(["data"]) in data_deps[FeatureKey(["upstream_multi"])]
        # upstream_excluded should not be in deps even though it has matching field
        assert FeatureKey(["upstream_excluded"]) not in data_deps

        # "debug" field doesn't match due to exclusion, falls back to ALL except excluded
        debug_deps = field_deps[FieldKey(["debug"])]
        # Should have all fields from upstream_multi except the excluded "debug" field
        assert FeatureKey(["upstream_multi"]) in debug_deps
        assert (
            len(debug_deps[FeatureKey(["upstream_multi"])]) == 2
        )  # All fields except debug
        # upstream_excluded should not contribute since the entire feature is excluded
        assert FeatureKey(["upstream_excluded"]) not in debug_deps


def test_no_field_mapping_specified():
    """Test behavior when no field mapping is specified (should still use default)."""
    test_graph = FeatureGraph()

    with test_graph.use():

        class SimpleUpstream(
            Feature,
            spec=FeatureSpec(
                key="simple_upstream",
                fields=[
                    FieldSpec(key="value", code_version="1"),
                ],
            ),
        ):
            pass

        # No fields_mapping specified - should use DefaultFieldsMapping()
        class SimpleDownstream(
            Feature,
            spec=FeatureSpec(
                key="simple_downstream",
                deps=[
                    FeatureDep(
                        feature="simple_upstream"
                    ),  # fields_mapping=None, defaults to DefaultFieldsMapping()
                ],
                fields=[
                    FieldSpec(key="value"),  # Should match simple_upstream.value
                ],
            ),
        ):
            pass

        plan = test_graph.get_feature_plan(FeatureKey(["simple_downstream"]))
        field_deps = plan.field_dependencies

        # Should still work with default mapping
        value_deps = field_deps[FieldKey(["value"])]
        assert FeatureKey(["simple_upstream"]) in value_deps
        assert FieldKey(["value"]) in value_deps[FeatureKey(["simple_upstream"])]
