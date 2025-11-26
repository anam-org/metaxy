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
from metaxy._testing.models import SampleFeature, SampleFeatureSpec
from metaxy.metadata_store import MetadataStore


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
                    )
                ],
                fields=["result"],
                fields_mapping=FieldsMapping.all(),
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


def test_feature_dep_filters_integration(any_store: MetadataStore) -> None:
    """Test that filters are applied when reading upstream dependencies through the store.

    This is a proper integration test that:
    1. Writes upstream feature metadata to the store
    2. Resolves a downstream feature with filters on the dependency
    3. Verifies the FeatureDepTransformer applies filters correctly
    """
    import polars as pl

    store = any_store

    with FeatureGraph().use() as graph:
        # Define upstream feature with age and status fields
        class UpstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="upstream",
                fields=["age", "status"],
            ),
        ):
            pass

        # Define downstream feature with filters on the dependency
        class DownstreamFeature(
            SampleFeature,
            spec=SampleFeatureSpec(
                key="downstream",
                deps=[
                    FeatureDep(
                        feature=UpstreamFeature,
                        filters=["age >= 25"],  # Only include samples with age >= 25
                    )
                ],
                fields=["result"],
                fields_mapping=FieldsMapping.all(),
            ),
        ):
            pass

        # Create sample upstream data with some rows that should be filtered out
        upstream_data = pl.DataFrame(
            {
                "sample_uid": ["s1", "s2", "s3", "s4"],
                "age": [20, 25, 30, 22],  # s1 and s4 should be filtered out
                "status": ["active", "active", "inactive", "active"],
                "metaxy_provenance_by_field": [
                    {"age": "h1", "status": "h1"},
                    {"age": "h2", "status": "h2"},
                    {"age": "h3", "status": "h3"},
                    {"age": "h4", "status": "h4"},
                ],
            }
        )

        # Write upstream metadata to store
        with store:
            from metaxy.metadata_store import HashAlgorithmNotSupportedError

            try:
                store.write_metadata(UpstreamFeature, upstream_data)

                # Resolve downstream feature - this should trigger FeatureDepTransformer
                # which will read upstream data and apply filters
                increment = store.resolve_update(
                    DownstreamFeature,
                    target_version=DownstreamFeature.feature_version(),
                    snapshot_version=graph.snapshot_version,
                )
            except HashAlgorithmNotSupportedError:
                import pytest

                pytest.skip(
                    f"Hash algorithm {store.hash_algorithm} not supported by {store}"
                )

            # Get the result
            result_df = increment.added.lazy().collect().to_polars()

            # Verify only samples with age >= 25 are present
            # (s2 with age=25 and s3 with age=30)
            assert len(result_df) == 2
            assert set(result_df["sample_uid"].to_list()) == {"s2", "s3"}

            # Verify the age values are correct
            ages = result_df.sort("sample_uid")["age"].to_list()
            assert ages == [25, 30]
