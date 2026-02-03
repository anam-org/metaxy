"""Test writing proper dataframes with all required columns.

These tests ensure that writing dataframes with the correct columns
does NOT emit any warnings.

For root features (no dependencies): only metaxy_provenance_by_field is required.
For non-root features (with dependencies): metaxy_provenance_by_field AND metaxy_provenance are required.
"""

import warnings

import polars as pl
import pytest
from metaxy_testing.models import SampleFeatureSpec
from pytest_cases import fixture, parametrize_with_cases

from metaxy import (
    BaseFeature,
    FeatureDep,
    FeatureGraph,
    FeatureKey,
    FieldDep,
    FieldKey,
    FieldSpec,
)
from metaxy._utils import collect_to_polars
from metaxy.metadata_store import MetadataStore
from metaxy.metadata_store.duckdb import DuckDBMetadataStore
from metaxy.metadata_store.warnings import MetaxyColumnMissingWarning
from metaxy.models.constants import (
    METAXY_PROVENANCE,
    METAXY_PROVENANCE_BY_FIELD,
)

from .conftest import AllStoresCases


@fixture
@parametrize_with_cases("store", cases=AllStoresCases)
def any_store(store: MetadataStore) -> MetadataStore:
    """Parametrized store (all store types)."""
    return store


@pytest.fixture
def features(graph: FeatureGraph):
    """Define test features: a root feature and a downstream feature."""

    class RootFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["root", "feature"]),
            fields=[
                FieldSpec(key=FieldKey(["field_a"]), code_version="1"),
                FieldSpec(key=FieldKey(["field_b"]), code_version="1"),
            ],
        ),
    ):
        """A root feature with no dependencies."""

        pass

    class DownstreamFeature(
        BaseFeature,
        spec=SampleFeatureSpec(
            key=FeatureKey(["downstream", "feature"]),
            deps=[
                FeatureDep(feature=FeatureKey(["root", "feature"])),
            ],
            fields=[
                FieldSpec(
                    key=FieldKey(["output"]),
                    code_version="1",
                    deps=[
                        FieldDep(
                            feature=FeatureKey(["root", "feature"]),
                            fields=[FieldKey(["field_a"]), FieldKey(["field_b"])],
                        )
                    ],
                ),
            ],
        ),
    ):
        """A downstream feature that depends on the root feature."""

        pass

    return {
        "RootFeature": RootFeature,
        "DownstreamFeature": DownstreamFeature,
    }


@pytest.fixture
def RootFeature(features: dict[str, type[BaseFeature]]):
    return features["RootFeature"]


@pytest.fixture
def DownstreamFeature(features: dict[str, type[BaseFeature]]):
    return features["DownstreamFeature"]


class TestProperDataframeWrite:
    """Test that writing proper dataframes does not emit warnings."""

    def test_root_feature_with_provenance_by_field_only(self, any_store: MetadataStore, RootFeature):
        """Root features should only require metaxy_provenance_by_field.

        When writing a root feature (no dependencies) with only
        metaxy_provenance_by_field, no warnings should be emitted since
        the provenance can be computed from provenance_by_field.
        """
        df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"field_a": "hash_a1", "field_b": "hash_b1"},
                    {"field_a": "hash_a2", "field_b": "hash_b2"},
                    {"field_a": "hash_a3", "field_b": "hash_b3"},
                ],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with any_store.open("w"):
                any_store.write(RootFeature, df)

                # Verify write succeeded
                result = collect_to_polars(any_store.read(RootFeature))
                assert len(result) == 3

            # Check for MetaxyColumnMissingWarning
            metaxy_warnings = [warning for warning in w if issubclass(warning.category, MetaxyColumnMissingWarning)]
            assert len(metaxy_warnings) == 0, (
                f"Unexpected MetaxyColumnMissingWarning: {[str(warning.message) for warning in metaxy_warnings]}"
            )

    def test_non_root_feature_with_both_provenance_columns(
        self, any_store: MetadataStore, RootFeature, DownstreamFeature
    ):
        """Non-root features require both metaxy_provenance_by_field AND metaxy_provenance.

        When writing a non-root feature (has dependencies) with both
        provenance columns, no warnings should be emitted.
        """
        # First, write root feature data
        root_df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"field_a": "hash_a1", "field_b": "hash_b1"},
                    {"field_a": "hash_a2", "field_b": "hash_b2"},
                    {"field_a": "hash_a3", "field_b": "hash_b3"},
                ],
            }
        )

        # Downstream feature with BOTH provenance columns
        downstream_df = pl.DataFrame(
            {
                "sample_uid": [1, 2, 3],
                METAXY_PROVENANCE_BY_FIELD: [
                    {"output": "hash_out1"},
                    {"output": "hash_out2"},
                    {"output": "hash_out3"},
                ],
                METAXY_PROVENANCE: ["prov_hash1", "prov_hash2", "prov_hash3"],
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with any_store.open("w"):
                any_store.write(RootFeature, root_df)
                any_store.write(DownstreamFeature, downstream_df)

                # Verify write succeeded
                result = collect_to_polars(any_store.read(DownstreamFeature))
                assert len(result) == 3

            # Check for MetaxyColumnMissingWarning
            metaxy_warnings = [warning for warning in w if issubclass(warning.category, MetaxyColumnMissingWarning)]
            # Filter to only warnings about the downstream feature write
            # (the root feature write may also emit warnings that we'll test separately)
            assert len(metaxy_warnings) == 0, (
                f"Unexpected MetaxyColumnMissingWarning: {[str(warning.message) for warning in metaxy_warnings]}"
            )


def test_sql_dialect_uses_connection(ibis_store: DuckDBMetadataStore) -> None:
    with ibis_store.open("w"):
        expected = ibis_store.conn.name
        assert ibis_store._sql_dialect == expected
