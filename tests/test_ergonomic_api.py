"""Tests for ergonomic API improvements - accepting native Python types."""

from metaxy import (
    Feature,
    FeatureDep,
    FeatureKey,
    FeatureSpec,
    FieldDep,
    FieldKey,
    FieldSpec,
)
from metaxy.models.feature import FeatureGraph


def test_feature_key_from_string():
    """FeatureKey should accept a single string."""
    spec = FeatureSpec(key="my_feature", deps=None)
    assert spec.key == FeatureKey(["my_feature"])
    assert isinstance(spec.key, FeatureKey)


def test_feature_key_from_list():
    """FeatureKey should accept a list of strings."""
    spec = FeatureSpec(key=["namespace", "my_feature"], deps=None)
    assert spec.key == FeatureKey(["namespace", "my_feature"])
    assert isinstance(spec.key, FeatureKey)


def test_feature_key_backward_compat():
    """FeatureKey should still accept explicit FeatureKey objects."""
    key = FeatureKey(["my", "feature"])
    spec = FeatureSpec(key=key, deps=None)
    assert spec.key == key
    assert isinstance(spec.key, FeatureKey)


def test_field_key_from_string():
    """FieldKey should accept a single string."""
    field = FieldSpec(key="result")
    assert field.key == FieldKey(["result"])
    assert isinstance(field.key, FieldKey)


def test_field_key_from_list():
    """FieldKey should accept a list of strings."""
    field = FieldSpec(key=["nested", "field"])
    assert field.key == FieldKey(["nested", "field"])
    assert isinstance(field.key, FieldKey)


def test_field_key_backward_compat():
    """FieldKey should still accept explicit FieldKey objects."""
    key = FieldKey(["my", "field"])
    field = FieldSpec(key=key)
    assert field.key == key
    assert isinstance(field.key, FieldKey)


def test_feature_dep_from_feature_class():
    """FeatureDep should accept Feature class directly."""
    graph = FeatureGraph()
    with graph.use():

        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(key="upstream", deps=None),
        ):
            pass

        # Should be able to pass Feature class
        dep = FeatureDep(key=UpstreamFeature)
        assert dep.key == FeatureKey(["upstream"])
        assert isinstance(dep.key, FeatureKey)


def test_feature_dep_from_string():
    """FeatureDep should accept string for key."""
    dep = FeatureDep(key="upstream")
    assert dep.key == FeatureKey(["upstream"])
    assert isinstance(dep.key, FeatureKey)


def test_feature_dep_from_list():
    """FeatureDep should accept list of strings for key."""
    dep = FeatureDep(key=["namespace", "upstream"])
    assert dep.key == FeatureKey(["namespace", "upstream"])
    assert isinstance(dep.key, FeatureKey)


def test_feature_dep_from_feature_spec():
    """FeatureDep should accept FeatureSpec object."""
    spec = FeatureSpec(key="upstream", deps=None)
    dep = FeatureDep(key=spec)
    assert dep.key == FeatureKey(["upstream"])
    assert isinstance(dep.key, FeatureKey)


def test_feature_dep_backward_compat():
    """FeatureDep should still accept explicit FeatureKey."""
    key = FeatureKey(["upstream"])
    dep = FeatureDep(key=key)
    assert dep.key == key
    assert isinstance(dep.key, FeatureKey)


def test_field_dep_from_feature_class():
    """FieldDep should accept Feature class for feature_key."""
    graph = FeatureGraph()
    with graph.use():

        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(key="upstream", deps=None),
        ):
            pass

        # Should be able to pass Feature class
        dep = FieldDep(feature_key=UpstreamFeature)
        assert dep.feature_key == FeatureKey(["upstream"])
        assert isinstance(dep.feature_key, FeatureKey)


def test_field_dep_from_string():
    """FieldDep should accept string for feature_key."""
    dep = FieldDep(feature_key="upstream")
    assert dep.feature_key == FeatureKey(["upstream"])
    assert isinstance(dep.feature_key, FeatureKey)


def test_field_dep_from_list():
    """FieldDep should accept list of strings for feature_key."""
    dep = FieldDep(feature_key=["namespace", "upstream"])
    assert dep.feature_key == FeatureKey(["namespace", "upstream"])
    assert isinstance(dep.feature_key, FeatureKey)


def test_field_dep_backward_compat():
    """FieldDep should still accept explicit FeatureKey."""
    key = FeatureKey(["upstream"])
    dep = FieldDep(feature_key=key)
    assert dep.feature_key == key
    assert isinstance(dep.feature_key, FeatureKey)


def test_field_dep_with_field_list():
    """FieldDep should accept string keys for fields list."""
    dep = FieldDep(
        feature_key="upstream",
        fields=["field1", "field2"],
    )
    assert dep.feature_key == FeatureKey(["upstream"])
    assert dep.fields == [FieldKey(["field1"]), FieldKey(["field2"])]


def test_complete_ergonomic_feature_definition():
    """Test complete feature definition with all ergonomic improvements."""
    graph = FeatureGraph()
    with graph.use():

        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key="upstream",
                deps=None,
                fields=[
                    FieldSpec(key="field1", code_version="1"),
                    FieldSpec(key="field2", code_version="1"),
                ],
            ),
        ):
            pass

        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key="downstream",
                deps=[UpstreamFeature],  # Feature class directly!
                fields=[
                    FieldSpec(
                        key="result",  # String instead of FieldKey
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature_key=UpstreamFeature,  # Feature class!
                                fields=["field1"],  # String list!
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

        # Verify everything was converted correctly
        assert DownstreamFeature.spec.key == FeatureKey(["downstream"])
        assert len(DownstreamFeature.spec.deps) == 1
        assert DownstreamFeature.spec.deps[0].key == FeatureKey(["upstream"])
        assert DownstreamFeature.spec.fields[0].key == FieldKey(["result"])


def test_backward_compatibility_verbose_syntax():
    """Ensure old verbose syntax still works."""
    graph = FeatureGraph()
    with graph.use():

        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["upstream"]),
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version="1"),
                ],
            ),
        ):
            pass

        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),
                deps=[FeatureDep(key=FeatureKey(["upstream"]))],
                fields=[
                    FieldSpec(
                        key=FieldKey(["result"]),
                        code_version="1",
                        deps=[
                            FieldDep(
                                feature_key=FeatureKey(["upstream"]),
                                fields=[FieldKey(["field1"])],
                            )
                        ],
                    )
                ],
            ),
        ):
            pass

        # Should work exactly as before
        assert DownstreamFeature.spec.key == FeatureKey(["downstream"])
        assert DownstreamFeature.spec.deps[0].key == FeatureKey(["upstream"])


def test_mixed_syntax():
    """Can mix ergonomic and verbose syntax in same definition."""
    graph = FeatureGraph()
    with graph.use():

        class UpstreamFeature(
            Feature,
            spec=FeatureSpec(
                key="upstream",  # Ergonomic
                deps=None,
                fields=[
                    FieldSpec(key=FieldKey(["field1"]), code_version="1"),  # Verbose
                ],
            ),
        ):
            pass

        class DownstreamFeature(
            Feature,
            spec=FeatureSpec(
                key=FeatureKey(["downstream"]),  # Verbose
                deps=[UpstreamFeature],  # Ergonomic!
                fields=[
                    FieldSpec(key="result", code_version="1")  # Ergonomic
                ],
            ),
        ):
            pass

        # Should work fine
        assert DownstreamFeature.spec.key == FeatureKey(["downstream"])
        assert DownstreamFeature.spec.deps[0].key == FeatureKey(["upstream"])
