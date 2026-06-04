"""Integration tests for FeatureDep filters with metadata stores.

This file tests that filters are correctly applied when resolving and updating features
through the metadata store, ensuring FeatureDepTransformer is invoked properly.
"""

from __future__ import annotations

from metaxy import (
    FeatureDep,
    FeatureGraph,
    FieldsMapping,
)
from metaxy_testing.models import SampleFeature, SampleFeatureSpec


def test_feature_dep_filters_basic() -> None:
    """Test that filters are correctly serialized and deserialized."""

    with FeatureGraph().use():

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["age", "status"],
            ),
        ):
            pass

        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[
                    FeatureDep(
                        feature=UpstreamFeature,
                        filters=["age >= 25", "status = 'active'"],
                        fields_mapping=FieldsMapping.all(),
                    )
                ],
                fields=["result"],
            ),
        ):
            pass

        # Verify the spec has filters
        spec = DownstreamFeature.spec()
        assert spec.deps[0].filters is not None
        assert len(spec.deps[0].filters) == 2

        # Verify serialization (using by_alias=True to use serialization_alias)
        spec_dict = spec.model_dump(mode="json", by_alias=True)
        assert "filters" in spec_dict["deps"][0]
        assert spec_dict["deps"][0]["filters"] == ["age >= 25", "status = 'active'"]


def test_feature_dep_filters_serialization() -> None:
    """Test that filters are correctly serialized and deserialized."""

    with FeatureGraph().use():

        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["x"],
            ),
        ):
            pass

        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[
                    FeatureDep(
                        feature=UpstreamFeature,
                        filters=["x >= 5"],
                    )
                ],
                fields=["result"],
            ),
        ):
            pass

        # Get the spec and serialize it (using by_alias=True to use serialization_alias)
        spec = DownstreamFeature.spec()
        spec_dict = spec.model_dump(mode="json", by_alias=True)

        # Verify filters are serialized with the "filters" key
        assert "filters" in spec_dict["deps"][0]
        assert spec_dict["deps"][0]["filters"] == ["x >= 5"]

        # Verify we can deserialize and filters still work
        from metaxy.models.feature_spec import FeatureSpec

        restored_spec = FeatureSpec.model_validate(spec_dict)
        assert restored_spec.deps[0].filters is not None
        assert len(restored_spec.deps[0].filters) == 1


def test_feature_dep_filters_sql_strings() -> None:
    """Test that filters can be provided as SQL strings via filters parameter."""
    dep = FeatureDep(
        feature="upstream",
        filters=["age >= 25", "status = 'active'"],
    )

    # Verify filters were parsed (lazy evaluation via @cached_property)
    assert dep.filters is not None
    assert len(dep.filters) == 2
